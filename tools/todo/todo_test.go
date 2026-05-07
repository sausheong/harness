package todo

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- single-call ops ---

func TestTodoWriteListEmpty(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	res, err := tool.Execute(context.Background(), json.RawMessage(`{"op":"list"}`))
	require.NoError(t, err)
	assert.Empty(t, res.Error)
	assert.Equal(t, "(no todos)", res.Output)
}

func TestTodoWriteAddReturnsItemAndPersists(t *testing.T) {
	dir := t.TempDir()
	tool := &TodoWriteTool{WorkDir: dir}
	res, err := tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"first task"}`))
	require.NoError(t, err)
	assert.Empty(t, res.Error)
	assert.Contains(t, res.Output, "[ ] t1 — first task")

	// Persisted to disk so a fresh tool instance picks it up.
	path := filepath.Join(dir, ".harness", "todos.json")
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(data), "first task")

	tool2 := &TodoWriteTool{WorkDir: dir}
	res2, _ := tool2.Execute(context.Background(), json.RawMessage(`{"op":"list"}`))
	assert.Contains(t, res2.Output, "first task")
}

func TestTodoWriteAddMissingContentRejected(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"add"}`))
	assert.Contains(t, res.Error, "content is required")
}

func TestTodoWriteCompleteShortcut(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"x"}`))
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"complete","id":"t1"}`))
	assert.Empty(t, res.Error)
	assert.Contains(t, res.Output, "[x] t1 — x")
}

func TestTodoWriteUpdateChangesStatusAndContent(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"orig"}`))
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"update","id":"t1","content":"revised","status":"in_progress"}`))
	assert.Empty(t, res.Error)
	assert.Contains(t, res.Output, "[~] t1 — revised")
}

func TestTodoWriteUpdateInvalidStatusRejected(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"x"}`))
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"update","id":"t1","status":"frobnicated"}`))
	assert.Contains(t, res.Error, "invalid status")
}

func TestTodoWriteRemove(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"x"}`))
	tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"y"}`))
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"remove","id":"t1"}`))
	assert.Empty(t, res.Error)
	assert.NotContains(t, res.Output, "t1")
	assert.Contains(t, res.Output, "t2 — y")
}

func TestTodoWriteRemoveUnknownID(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"remove","id":"ghost"}`))
	assert.Contains(t, res.Error, "todo not found")
}

func TestTodoWriteUnknownOpRejected(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"frobnicate"}`))
	assert.Contains(t, res.Error, "unknown op")
}

func TestTodoWriteEmptyInputDefaultsToList(t *testing.T) {
	tool := &TodoWriteTool{WorkDir: t.TempDir()}
	res, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, res.Error)
	assert.Equal(t, "(no todos)", res.Output)
}

// --- ID generation ---

func TestNextTodoIDGapsAreSafe(t *testing.T) {
	// id allocator must keep monotonically increasing even when items
	// have been removed in the middle. Otherwise an old removed id
	// could get re-issued and the model's earlier reference would
	// silently target a different todo.
	items := []TodoItem{
		{ID: "t1"},
		{ID: "t3"},
	}
	assert.Equal(t, "t4", nextTodoID(items))
}

func TestNextTodoIDIgnoresNonNumeric(t *testing.T) {
	items := []TodoItem{
		{ID: "legacy"},
		{ID: "t2"},
	}
	assert.Equal(t, "t3", nextTodoID(items))
}

// --- formatTodos ordering ---

func TestFormatTodosSortsByStatusThenID(t *testing.T) {
	items := []TodoItem{
		{ID: "t3", Content: "c3", Status: TodoCompleted},
		{ID: "t1", Content: "c1", Status: TodoPending},
		{ID: "t2", Content: "c2", Status: TodoInProgress},
		{ID: "t4", Content: "c4", Status: TodoInProgress},
	}
	got := formatTodos(items)
	// Order: in_progress (t2, t4), then pending (t1), then completed (t3).
	t2Idx := indexOf(got, "t2")
	t4Idx := indexOf(got, "t4")
	t1Idx := indexOf(got, "t1")
	t3Idx := indexOf(got, "t3")
	require.True(t, t2Idx < t4Idx, "in_progress sorted by id")
	require.True(t, t4Idx < t1Idx, "in_progress before pending")
	require.True(t, t1Idx < t3Idx, "pending before completed")
}

// --- corrupt store ---

func TestTodoWriteCorruptStoreSurfacesError(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(dir, ".harness"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, ".harness", "todos.json"), []byte("{not valid json"), 0o644))

	tool := &TodoWriteTool{WorkDir: dir}
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"list"}`))
	// Corrupt store must not return "(no todos)" — that would be
	// silent data loss. Surface the error so a human can intervene.
	assert.Contains(t, res.Error, "load todos")
}

// --- headless mode ---

func TestTodoWriteHeadlessNoWorkDir(t *testing.T) {
	// No WorkDir → no persistence. Operations still succeed but
	// state lives only in the current call (not even between calls
	// on the same tool — that's intentional; the persistence layer
	// IS the WorkDir file).
	tool := &TodoWriteTool{}
	res, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"add","content":"x"}`))
	assert.Empty(t, res.Error)
	assert.Contains(t, res.Output, "[ ] t1 — x")

	res2, _ := tool.Execute(context.Background(), json.RawMessage(`{"op":"list"}`))
	// No persistence, so the second list call sees nothing.
	assert.Equal(t, "(no todos)", res2.Output)
}

func TestTodoWriteIsConcurrencySafeFalse(t *testing.T) {
	tool := &TodoWriteTool{}
	assert.False(t, tool.IsConcurrencySafe(nil),
		"todo_write is stateful read-modify-write; concurrent dispatch would race")
}

// helper
func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
