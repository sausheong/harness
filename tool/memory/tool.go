package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sausheong/harness/tool"
)

// MemoryTool is the action-discriminated tool the agent calls to read
// or write memory entries. Wraps any MemoryStore; the default JSONL
// implementation is in tool/memory/jsonl/.
//
// Schema is one tool with an "action" enum rather than five separate
// tools, to keep the request prefix small (less schema bloat -> better
// prompt-cache hits).
type MemoryTool struct {
	Store MemoryStore
}

// Name returns "memory".
func (t *MemoryTool) Name() string { return "memory" }

// Description is shown to the model in the tool list.
func (t *MemoryTool) Description() string {
	return "Save, update, remove, or list memory entries — durable facts " +
		"and preferences that survive across sessions. Use to capture user " +
		"preferences, project context, or workflow conventions worth " +
		"remembering."
}

// Parameters returns the JSON-Schema for the tool input.
func (t *MemoryTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["save", "update", "remove", "list", "get"],
				"description": "Operation to perform."
			},
			"id": {
				"type": "string",
				"description": "Required for update/remove/get. Optional for save (auto-generated when omitted)."
			},
			"content": {
				"type": "string",
				"maxLength": 4000,
				"description": "Required for save/update. The memory text. Max 4000 characters."
			},
			"tags": {
				"type": "array",
				"items": {"type": "string"},
				"maxItems": 8,
				"description": "Optional categorization for save. Up to 8 tags."
			}
		},
		"required": ["action"]
	}`)
}

// IsConcurrencySafe returns false — Save/Update/Remove all mutate the
// underlying store. Even action=list/get touch shared state we don't
// want racing with writes.
func (t *MemoryTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

// memoryInput is the parsed tool input.
type memoryInput struct {
	Action  string   `json:"action"`
	ID      string   `json:"id"`
	Content string   `json:"content"`
	Tags    []string `json:"tags"`
}

// actionResult is the canonical envelope every successful or failed
// MemoryTool call returns. runtime.Review reads this shape to extract
// human-readable action summaries.
type actionResult struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
	Target  string `json:"target,omitempty"`
	ID      string `json:"id,omitempty"`
}

func (r actionResult) toToolResult() tool.ToolResult {
	b, _ := json.Marshal(r)
	return tool.ToolResult{Output: string(b)}
}

func successResult(message, id string) tool.ToolResult {
	return actionResult{
		Success: true,
		Message: message,
		Target:  "memory",
		ID:      id,
	}.toToolResult()
}

func errorResult(errMsg, id string) tool.ToolResult {
	return actionResult{
		Success: false,
		Error:   errMsg,
		Target:  "memory",
		ID:      id,
	}.toToolResult()
}

// truncatePreview returns content suitable for inclusion in a success
// message — truncated to ~60 chars with an ellipsis.
func truncatePreview(s string) string {
	const max = 60
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

const maxContentBytes = 4000

// originFromContext returns the OriginKey value or "agent" as default.
func originFromContext(ctx context.Context) string {
	if v := ctx.Value(OriginKey); v != nil {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	return "agent"
}

// Execute dispatches on the action enum.
func (t *MemoryTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	if t.Store == nil {
		return errorResult("memory tool: no store configured", ""), nil
	}
	var in memoryInput
	if err := json.Unmarshal(input, &in); err != nil {
		return errorResult(fmt.Sprintf("invalid input: %v", err), ""), nil
	}
	switch in.Action {
	case "save":
		return t.executeSave(ctx, in), nil
	case "update":
		return t.executeUpdate(ctx, in), nil
	case "remove":
		return t.executeRemove(ctx, in), nil
	case "list":
		return t.executeList(ctx, in), nil
	case "get":
		return t.executeGet(ctx, in), nil
	case "":
		return errorResult("action is required", ""), nil
	default:
		return errorResult(fmt.Sprintf("unknown action %q", in.Action), ""), nil
	}
}

func (t *MemoryTool) executeSave(ctx context.Context, in memoryInput) tool.ToolResult {
	content := strings.TrimSpace(in.Content)
	if content == "" {
		return errorResult("content is required for save", "")
	}
	if len(content) > maxContentBytes {
		return errorResult(fmt.Sprintf("content exceeds max length (%d > %d)", len(content), maxContentBytes), "")
	}
	if len(in.Tags) > 8 {
		return errorResult(fmt.Sprintf("too many tags (max 8, got %d)", len(in.Tags)), "")
	}

	saved, err := t.Store.Save(ctx, Entry{
		Content: content,
		Tags:    in.Tags,
		Origin:  originFromContext(ctx),
	})
	if err != nil {
		return errorResult(err.Error(), "")
	}
	return successResult(fmt.Sprintf("saved memory: %s", truncatePreview(saved.Content)), saved.ID)
}

func (t *MemoryTool) executeUpdate(ctx context.Context, in memoryInput) tool.ToolResult {
	if in.ID == "" {
		return errorResult("id is required for update", "")
	}
	content := strings.TrimSpace(in.Content)
	if content == "" {
		return errorResult("content is required for update", "")
	}
	if len(content) > maxContentBytes {
		return errorResult(fmt.Sprintf("content exceeds max length (%d > %d)", len(content), maxContentBytes), in.ID)
	}
	updated, err := t.Store.Update(ctx, in.ID, content)
	if err != nil {
		return errorResult(err.Error(), in.ID)
	}
	return successResult(fmt.Sprintf("updated memory: %s", truncatePreview(updated.Content)), updated.ID)
}

// executeList returns the entries as a raw JSON array (not the
// action-result envelope) so the model receives a list-shaped value
// it can iterate over directly. runtime.Review's action extractor
// ignores list/get results — they're not "actions taken", they're
// reads.
func (t *MemoryTool) executeList(ctx context.Context, in memoryInput) tool.ToolResult {
	tag := ""
	if len(in.Tags) > 0 {
		tag = in.Tags[0] // first tag is the filter; we don't intersect multi-tag
	}
	entries, err := t.Store.List(ctx, tag)
	if err != nil {
		return errorResult(err.Error(), "")
	}
	if entries == nil {
		entries = []Entry{}
	}
	b, err := json.Marshal(entries)
	if err != nil {
		return errorResult(err.Error(), "")
	}
	return tool.ToolResult{Output: string(b)}
}

// executeGet returns the raw Entry as JSON on success; the action-result
// envelope on failure.
func (t *MemoryTool) executeGet(ctx context.Context, in memoryInput) tool.ToolResult {
	if in.ID == "" {
		return errorResult("id is required for get", "")
	}
	e, ok, err := t.Store.Get(ctx, in.ID)
	if err != nil {
		return errorResult(err.Error(), in.ID)
	}
	if !ok {
		return errorResult("not found", in.ID)
	}
	b, err := json.Marshal(e)
	if err != nil {
		return errorResult(err.Error(), in.ID)
	}
	return tool.ToolResult{Output: string(b)}
}

func (t *MemoryTool) executeRemove(ctx context.Context, in memoryInput) tool.ToolResult {
	if in.ID == "" {
		return errorResult("id is required for remove", "")
	}
	if err := t.Store.Remove(ctx, in.ID); err != nil {
		return errorResult(err.Error(), in.ID)
	}
	return successResult(fmt.Sprintf("removed memory: %s", in.ID), in.ID)
}

// Compile-time assertion that *MemoryTool satisfies tool.Tool.
var _ tool.Tool = (*MemoryTool)(nil)
