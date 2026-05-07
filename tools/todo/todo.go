package todo

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"github.com/sausheong/harness/tool"
)

// TodoStatus is the status of a todo item.
type TodoStatus string

const (
	TodoPending    TodoStatus = "pending"
	TodoInProgress TodoStatus = "in_progress"
	TodoCompleted  TodoStatus = "completed"
)

// TodoItem is one entry in the per-workspace todo list.
type TodoItem struct {
	ID        string     `json:"id"`
	Content   string     `json:"content"`
	Status    TodoStatus `json:"status"`
	CreatedAt int64      `json:"createdAt"`
	UpdatedAt int64      `json:"updatedAt"`
}

// TodoWriteTool maintains a per-workspace todo list the agent uses to
// track multi-step work. Persisted to <Workspace>/.harness/todos.json so
// it survives across turns within a session and across sessions for
// the same workspace.
//
// State is loaded lazily on first call and re-read from disk before
// each operation — so external edits (a human deleting the file, a
// concurrent agent in another process) are picked up. The mutex only
// serialises calls within this process.
type TodoWriteTool struct {
	WorkDir string

	mu sync.Mutex
}

func (t *TodoWriteTool) Name() string { return "todo_write" }

func (t *TodoWriteTool) Description() string {
	return "Persistent todo list for tracking long, multi-stage work. DO NOT use this as a planning scratchpad before starting a task — start working directly and call your real tools. Reserve todo_write for genuinely long-running work with roughly 5+ independent subtasks that will span many turns. When initializing a list, emit every `add` call as a parallel tool call in the SAME assistant response (the runtime serialises them safely) — never call `add` once per turn, that wastes round trips. Operations: list (default), add, update, complete, remove. The full current list is returned every call."
}

func (t *TodoWriteTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"op": {
				"type": "string",
				"enum": ["list", "add", "update", "complete", "remove"],
				"description": "Operation to perform. Default: list."
			},
			"id": {
				"type": "string",
				"description": "Todo id, required for update/complete/remove. Returned by add."
			},
			"content": {
				"type": "string",
				"description": "Todo body, required for add and optional for update."
			},
			"status": {
				"type": "string",
				"enum": ["pending", "in_progress", "completed"],
				"description": "Status, used by update. Add starts items as pending; complete is shorthand for status=completed."
			}
		}
	}`)
}

// IsConcurrencySafe is FALSE: the tool reads-modify-writes a shared
// file and modifies in-process state under a mutex. Marking it safe
// would let the agent loop dispatch parallel todo_write calls, all
// racing the same file. Better to serialise.
func (t *TodoWriteTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

type todoInput struct {
	Op      string     `json:"op"`
	ID      string     `json:"id"`
	Content string     `json:"content"`
	Status  TodoStatus `json:"status"`
}

func (t *TodoWriteTool) Execute(_ context.Context, input json.RawMessage) (tool.ToolResult, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	var in todoInput
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
		}
	}
	if in.Op == "" {
		in.Op = "list"
	}

	items, err := t.load()
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("load todos: %v", err)}, nil
	}

	now := time.Now().Unix()
	switch in.Op {
	case "list":
		// nothing to do
	case "add":
		if in.Content == "" {
			return tool.ToolResult{Error: "content is required for add"}, nil
		}
		items = append(items, TodoItem{
			ID:        nextTodoID(items),
			Content:   in.Content,
			Status:    TodoPending,
			CreatedAt: now,
			UpdatedAt: now,
		})
	case "update":
		if in.ID == "" {
			return tool.ToolResult{Error: "id is required for update"}, nil
		}
		idx := findTodo(items, in.ID)
		if idx < 0 {
			return tool.ToolResult{Error: fmt.Sprintf("todo not found: %q", in.ID)}, nil
		}
		if in.Content != "" {
			items[idx].Content = in.Content
		}
		if in.Status != "" {
			if !validStatus(in.Status) {
				return tool.ToolResult{Error: fmt.Sprintf("invalid status: %q", in.Status)}, nil
			}
			items[idx].Status = in.Status
		}
		items[idx].UpdatedAt = now
	case "complete":
		if in.ID == "" {
			return tool.ToolResult{Error: "id is required for complete"}, nil
		}
		idx := findTodo(items, in.ID)
		if idx < 0 {
			return tool.ToolResult{Error: fmt.Sprintf("todo not found: %q", in.ID)}, nil
		}
		items[idx].Status = TodoCompleted
		items[idx].UpdatedAt = now
	case "remove":
		if in.ID == "" {
			return tool.ToolResult{Error: "id is required for remove"}, nil
		}
		idx := findTodo(items, in.ID)
		if idx < 0 {
			return tool.ToolResult{Error: fmt.Sprintf("todo not found: %q", in.ID)}, nil
		}
		items = append(items[:idx], items[idx+1:]...)
	default:
		return tool.ToolResult{Error: fmt.Sprintf("unknown op: %q", in.Op)}, nil
	}

	if in.Op != "list" {
		if err := t.save(items); err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("save todos: %v", err)}, nil
		}
	}

	return tool.ToolResult{Output: formatTodos(items)}, nil
}

// load reads the persisted todos from <WorkDir>/.harness/todos.json.
// Missing file → empty list (not an error). Corrupt JSON → error so
// the agent knows the file needs manual repair rather than silently
// dropping work.
func (t *TodoWriteTool) load() ([]TodoItem, error) {
	if t.WorkDir == "" {
		return nil, nil // headless / test mode — no persistence
	}
	path := t.todosPath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	var items []TodoItem
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, fmt.Errorf("corrupt todos.json: %w", err)
	}
	return items, nil
}

func (t *TodoWriteTool) save(items []TodoItem) error {
	if t.WorkDir == "" {
		return nil
	}
	path := t.todosPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (t *TodoWriteTool) todosPath() string {
	return filepath.Join(t.WorkDir, ".harness", "todos.json")
}

// nextTodoID returns the next sequential id (t1, t2, …) — short and
// memorable for the model to reference. Numeric suffix is one past
// the highest existing numeric id; non-numeric ids are ignored when
// computing the next number.
func nextTodoID(items []TodoItem) string {
	max := 0
	for _, it := range items {
		var n int
		if _, err := fmt.Sscanf(it.ID, "t%d", &n); err == nil && n > max {
			max = n
		}
	}
	return fmt.Sprintf("t%d", max+1)
}

func findTodo(items []TodoItem, id string) int {
	for i, it := range items {
		if it.ID == id {
			return i
		}
	}
	return -1
}

func validStatus(s TodoStatus) bool {
	return s == TodoPending || s == TodoInProgress || s == TodoCompleted
}

// formatTodos renders the list as a checklist the model can scan.
// Sorted: in_progress → pending → completed, then by id within each
// bucket, so the model's eye lands on what's currently active first.
func formatTodos(items []TodoItem) string {
	if len(items) == 0 {
		return "(no todos)"
	}
	statusOrder := map[TodoStatus]int{
		TodoInProgress: 0,
		TodoPending:    1,
		TodoCompleted:  2,
	}
	sorted := make([]TodoItem, len(items))
	copy(sorted, items)
	sort.SliceStable(sorted, func(i, j int) bool {
		si, sj := statusOrder[sorted[i].Status], statusOrder[sorted[j].Status]
		if si != sj {
			return si < sj
		}
		return sorted[i].ID < sorted[j].ID
	})
	var b strings.Builder
	for _, it := range sorted {
		var marker string
		switch it.Status {
		case TodoCompleted:
			marker = "[x]"
		case TodoInProgress:
			marker = "[~]"
		default:
			marker = "[ ]"
		}
		fmt.Fprintf(&b, "%s %s — %s\n", marker, it.ID, it.Content)
	}
	return strings.TrimRight(b.String(), "\n")
}
