package memory

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryTool_NameDescriptionSchema(t *testing.T) {
	tool := &MemoryTool{}

	assert.Equal(t, "memory", tool.Name())
	assert.NotEmpty(t, tool.Description())

	var schema struct {
		Type       string                 `json:"type"`
		Properties map[string]interface{} `json:"properties"`
		Required   []string               `json:"required"`
	}
	require.NoError(t, json.Unmarshal(tool.Parameters(), &schema))
	assert.Equal(t, "object", schema.Type)
	assert.Contains(t, schema.Properties, "action")
	assert.Contains(t, schema.Properties, "id")
	assert.Contains(t, schema.Properties, "content")
	assert.Contains(t, schema.Properties, "tags")
	assert.Equal(t, []string{"action"}, schema.Required)
}

func TestMemoryTool_IsConcurrencySafeFalse(t *testing.T) {
	tool := &MemoryTool{}
	assert.False(t, tool.IsConcurrencySafe(nil))
}

// fakeStore is an in-memory MemoryStore for tool tests.
type fakeStore struct {
	entries []Entry
	failOn  string // method name to fail on, e.g. "Save"
}

func (f *fakeStore) Save(_ context.Context, e Entry) (Entry, error) {
	if f.failOn == "Save" {
		return Entry{}, assertErr("save failed")
	}
	if e.ID == "" {
		e.ID = "fake_id"
	}
	e.CreatedAt = time.Now()
	e.UpdatedAt = e.CreatedAt
	f.entries = append(f.entries, e)
	return e, nil
}

func (f *fakeStore) Update(_ context.Context, id, content string) (Entry, error) {
	for i, e := range f.entries {
		if e.ID == id {
			e.Content = content
			e.ID = "fake_id_updated"
			f.entries[i] = e
			return e, nil
		}
	}
	return Entry{}, ErrNotFound
}

func (f *fakeStore) Remove(_ context.Context, id string) error {
	out := f.entries[:0]
	for _, e := range f.entries {
		if e.ID != id {
			out = append(out, e)
		}
	}
	f.entries = out
	return nil
}

func (f *fakeStore) List(_ context.Context, tag string) ([]Entry, error) {
	if tag == "" {
		return f.entries, nil
	}
	out := []Entry{}
	for _, e := range f.entries {
		for _, t := range e.Tags {
			if t == tag {
				out = append(out, e)
				break
			}
		}
	}
	return out, nil
}

func (f *fakeStore) Get(_ context.Context, id string) (Entry, bool, error) {
	for _, e := range f.entries {
		if e.ID == id {
			return e, true, nil
		}
	}
	return Entry{}, false, nil
}

type assertErr string

func (a assertErr) Error() string { return string(a) }

func TestMemoryTool_SaveHappyPath(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}

	in, _ := json.Marshal(memoryInput{Action: "save", Content: "user likes terse output"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, "memory", ar.Target)
	assert.Equal(t, "fake_id", ar.ID)
	assert.Contains(t, ar.Message, "user likes terse output")

	require.Len(t, store.entries, 1)
}

func TestMemoryTool_SaveRejectsEmptyContent(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "save", Content: ""})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "content")
	assert.Empty(t, store.entries)
}

func TestMemoryTool_SaveRejectsOversizedContent(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	big := make([]byte, 4001)
	for i := range big {
		big[i] = 'x'
	}
	in, _ := json.Marshal(memoryInput{Action: "save", Content: string(big)})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "4000")
	assert.Empty(t, store.entries)
}

func TestMemoryTool_SaveOriginFromContext(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	ctx := context.WithValue(context.Background(), OriginKey, "review")
	in, _ := json.Marshal(memoryInput{Action: "save", Content: "from review"})
	_, err := tool.Execute(ctx, in)
	require.NoError(t, err)

	require.Len(t, store.entries, 1)
	assert.Equal(t, "review", store.entries[0].Origin)
}

func TestMemoryTool_SaveOriginDefault(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "save", Content: "from foreground"})
	_, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	require.Len(t, store.entries, 1)
	assert.Equal(t, "agent", store.entries[0].Origin)
}

func TestMemoryTool_UpdateHappyPath(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	saved, _ := store.Save(context.Background(), Entry{Content: "v1"})

	in, _ := json.Marshal(memoryInput{Action: "update", ID: saved.ID, Content: "v2"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, "fake_id_updated", ar.ID)
	assert.Contains(t, ar.Message, "updated")
}

func TestMemoryTool_UpdateRejectsMissingID(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "update", Content: "v2"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "id")
}

func TestMemoryTool_UpdateNotFound(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "update", ID: "nope", Content: "v2"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestMemoryTool_UpdateRejectsEmptyContent(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	saved, _ := store.Save(context.Background(), Entry{Content: "v1"})
	in, _ := json.Marshal(memoryInput{Action: "update", ID: saved.ID, Content: ""})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
}

func TestMemoryTool_RemoveHappyPath(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	saved, _ := store.Save(context.Background(), Entry{Content: "doomed"})

	in, _ := json.Marshal(memoryInput{Action: "remove", ID: saved.ID})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, saved.ID, ar.ID)
	assert.Contains(t, ar.Message, "removed")

	assert.Empty(t, store.entries)
}

func TestMemoryTool_RemoveRejectsMissingID(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "remove"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "id")
}

func TestMemoryTool_RemoveUnknownIsSuccessful(t *testing.T) {
	// Idempotent: removing a never-existed id returns success.
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "remove", ID: "ghost"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
}

func TestMemoryTool_ListReturnsAllAsJSON(t *testing.T) {
	store := &fakeStore{}
	_, _ = store.Save(context.Background(), Entry{Content: "first", Tags: []string{"a"}})
	_, _ = store.Save(context.Background(), Entry{Content: "second", Tags: []string{"b"}})
	tool := &MemoryTool{Store: store}

	in, _ := json.Marshal(memoryInput{Action: "list"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	assert.Empty(t, res.Error)

	// List output is the raw entries array, not the action-result envelope.
	var entries []Entry
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entries))
	require.Len(t, entries, 2)
	assert.Equal(t, "first", entries[0].Content)
	assert.Equal(t, "second", entries[1].Content)
}

func TestMemoryTool_ListWithTagFilter(t *testing.T) {
	store := &fakeStore{}
	_, _ = store.Save(context.Background(), Entry{Content: "general"})
	_, _ = store.Save(context.Background(), Entry{Content: "preference", Tags: []string{"pref"}})
	tool := &MemoryTool{Store: store}

	in, _ := json.Marshal(memoryInput{Action: "list", Tags: []string{"pref"}})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var entries []Entry
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entries))
	require.Len(t, entries, 1)
	assert.Equal(t, "preference", entries[0].Content)
}

func TestMemoryTool_ListEmpty(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "list"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, "[]", res.Output)
}

func TestMemoryTool_GetHappyPath(t *testing.T) {
	store := &fakeStore{}
	saved, _ := store.Save(context.Background(), Entry{Content: "hello"})
	tool := &MemoryTool{Store: store}

	in, _ := json.Marshal(memoryInput{Action: "get", ID: saved.ID})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var entry Entry
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entry))
	assert.Equal(t, saved.ID, entry.ID)
	assert.Equal(t, "hello", entry.Content)
}

func TestMemoryTool_GetUnknownReturnsError(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "get", ID: "ghost"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestMemoryTool_GetRejectsMissingID(t *testing.T) {
	store := &fakeStore{}
	tool := &MemoryTool{Store: store}
	in, _ := json.Marshal(memoryInput{Action: "get"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "id")
}
