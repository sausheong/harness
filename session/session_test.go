package session

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionAppendAndHistory(t *testing.T) {
	sess := NewSession("default", "test")

	sess.Append(UserMessageEntry("hello"))
	sess.Append(AssistantMessageEntry("hi there"))
	sess.Append(UserMessageEntry("how are you?"))

	history := sess.History()
	assert.Len(t, history, 3)

	assert.Equal(t, EntryTypeMessage, history[0].Type)
	assert.Equal(t, "user", history[0].Role)
	assert.Equal(t, "assistant", history[1].Role)
	assert.Equal(t, "user", history[2].Role)
}

func TestSessionDAGTraversal(t *testing.T) {
	sess := NewSession("default", "test")

	sess.Append(UserMessageEntry("first"))
	sess.Append(AssistantMessageEntry("second"))

	history := sess.History()
	assert.Len(t, history, 2)

	// Parent chain should be connected
	assert.Empty(t, history[0].ParentID)
	assert.Equal(t, history[0].ID, history[1].ParentID)
}

func TestStorePersistence(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	// Create and populate a session
	sess, err := store.Load("agent1", "test_peer")
	require.NoError(t, err)

	sess.Append(UserMessageEntry("hello"))
	sess.Append(AssistantMessageEntry("world"))

	// Reload from disk
	sess2, err := store.Load("agent1", "test_peer")
	require.NoError(t, err)

	history := sess2.History()
	assert.Len(t, history, 2)

	// Check file exists
	path := filepath.Join(dir, "agent1", "test_peer.jsonl")
	assert.FileExists(t, path)
}

func TestToolCallEntries(t *testing.T) {
	sess := NewSession("default", "test")

	sess.Append(UserMessageEntry("run ls"))
	sess.Append(ToolCallEntry("tc_1", "bash", []byte(`{"command":"ls"}`)))
	sess.Append(ToolResultEntry("tc_1", "file1\nfile2", "", nil))
	sess.Append(AssistantMessageEntry("Here are the files."))

	history := sess.History()
	assert.Len(t, history, 4)
	assert.Equal(t, EntryTypeToolCall, history[1].Type)
	assert.Equal(t, EntryTypeToolResult, history[2].Type)
}

// TestToolCallEntrySanitisesEmptyInput is the regression guard for
// the "data:null" bug. When the LLM emits a tool_use whose arguments
// stream produces zero bytes (or an empty/non-nil json.RawMessage,
// or invalid JSON), the unsanitised version of ToolCallEntry hit
// json.Marshal's "unexpected end of JSON input" path, swallowed the
// error via `data, _ :=`, and persisted Data: nil. On disk: `"data":null`.
// On reload, assembleMessages would build a tool_use with an empty
// ID, and the next LLM call would fail with the Anthropic 400:
// "messages.N.content.0: unexpected `tool_use_id` ... Each
// `tool_result` block must have a corresponding `tool_use` block in
// the previous message." The fix substitutes "{}" for invalid input;
// this test asserts the entry's Data is non-nil and round-trips.
func TestToolCallEntrySanitisesEmptyInput(t *testing.T) {
	cases := []struct {
		name  string
		input json.RawMessage
	}{
		{"nil_input", nil},
		{"empty_non_nil_input", json.RawMessage{}},
		{"truncated_object", json.RawMessage(`{"a":`)},
		{"plain_text", json.RawMessage(`hello`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := ToolCallEntry("toolu_x", "search", tc.input)
			require.NotNil(t, e.Data, "Data must not be nil — that's the bug")
			require.True(t, json.Valid(e.Data), "Data must be valid JSON")

			var td ToolCallData
			require.NoError(t, json.Unmarshal(e.Data, &td))
			assert.Equal(t, "toolu_x", td.ID, "ID must round-trip")
			assert.Equal(t, "search", td.Tool, "Tool must round-trip")
			assert.True(t, json.Valid(td.Input), "Input must round-trip as valid JSON")
		})
	}
}

func TestSessionBranch(t *testing.T) {
	sess := NewSession("default", "test")

	sess.Append(UserMessageEntry("first"))
	firstID := sess.LeafID()
	sess.Append(AssistantMessageEntry("response 1"))
	sess.Append(UserMessageEntry("second"))

	// Branch back to first entry
	err := sess.Branch(firstID)
	require.NoError(t, err)

	assert.Equal(t, firstID, sess.LeafID())

	// Append on the branch
	sess.Append(AssistantMessageEntry("alternate response"))

	// History should follow the branch
	history := sess.History()
	assert.Len(t, history, 2) // first + alternate response
	assert.Equal(t, "user", history[0].Role)
	assert.Equal(t, "assistant", history[1].Role)
}

func TestSessionBranchInvalidID(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("hello"))

	err := sess.Branch("nonexistent")
	assert.Error(t, err)
}

func TestSessionCompact(t *testing.T) {
	sess := NewSession("default", "test")

	// Add 10 exchanges
	for i := 0; i < 10; i++ {
		sess.Append(UserMessageEntry("question " + string(rune('0'+i))))
		sess.Append(AssistantMessageEntry("answer " + string(rune('0'+i))))
	}

	history := sess.History()
	assert.Len(t, history, 20)

	// Compact, keeping last 4 entries
	sess.Compact("Summary of conversation: discussed topics 0-7", 4)

	history = sess.History()
	// Should have: 1 summary + 4 kept entries = 5
	assert.Len(t, history, 5)

	// First entry should be the summary meta entry
	assert.Equal(t, EntryTypeMeta, history[0].Type)
	assert.Equal(t, "system", history[0].Role)
}

func TestSessionCompactNoOp(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("hello"))
	sess.Append(AssistantMessageEntry("world"))

	// Compacting with keepEntries >= history length should be a no-op
	sess.Compact("summary", 10)

	history := sess.History()
	assert.Len(t, history, 2)
}

func TestSessionCompactWithStore(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)

	sess, err := store.Load("agent1", "compact_test")
	require.NoError(t, err)

	for i := 0; i < 10; i++ {
		sess.Append(UserMessageEntry("msg " + string(rune('0'+i))))
		sess.Append(AssistantMessageEntry("reply " + string(rune('0'+i))))
	}

	sess.Compact("Summary of conversation", 4)

	// Reload and verify
	sess2, err := store.Load("agent1", "compact_test")
	require.NoError(t, err)

	history := sess2.History()
	assert.Len(t, history, 5) // 1 summary + 4 kept
	assert.Equal(t, EntryTypeMeta, history[0].Type)
}

func TestEstimateTokens(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("Hello, how are you doing today?"))
	sess.Append(AssistantMessageEntry("I'm doing well, thank you for asking!"))

	tokens := sess.EstimateTokens()
	assert.Greater(t, tokens, 0)
}

func TestSessionViewWithoutCompactionMatchesHistory(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("hi"))
	sess.Append(AssistantMessageEntry("hello"))
	sess.Append(UserMessageEntry("hello again"))

	view := sess.View()
	hist := sess.History()
	assert.Equal(t, len(hist), len(view))
	for i := range hist {
		assert.Equal(t, hist[i].ID, view[i].ID)
	}
}

func TestSessionViewWithSingleCompaction(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("u1"))
	sess.Append(AssistantMessageEntry("a1"))
	sess.Append(UserMessageEntry("u2"))
	// Simulate compaction over [u1, a1, u2]: append a CompactionEntry,
	// then continue appending normal entries after it.
	sess.Append(CompactionEntry("summary of u1/a1/u2", "", "", "ollama/qwen2.5:3b-instruct", 100, 25, 3))
	sess.Append(AssistantMessageEntry("a2 after compaction"))
	sess.Append(UserMessageEntry("u3"))

	view := sess.View()
	require.Len(t, view, 3)
	assert.Equal(t, EntryTypeCompaction, view[0].Type)
	assert.Equal(t, EntryTypeMessage, view[1].Type)
	assert.Equal(t, "assistant", view[1].Role)
	assert.Equal(t, "user", view[2].Role)
}

func TestSessionViewWithMultipleCompactions(t *testing.T) {
	sess := NewSession("default", "test")
	sess.Append(UserMessageEntry("old"))
	sess.Append(CompactionEntry("first summary", "", "", "m", 0, 0, 1))
	sess.Append(UserMessageEntry("middle"))
	sess.Append(CompactionEntry("second summary", "", "", "m", 0, 0, 1))
	sess.Append(UserMessageEntry("recent"))

	view := sess.View()
	require.Len(t, view, 2)
	// Most recent compaction supersedes the first — view starts at it.
	var cd CompactionData
	require.NoError(t, json.Unmarshal(view[0].Data, &cd))
	assert.Equal(t, "second summary", cd.Summary)
	assert.Equal(t, "user", view[1].Role)
}

func TestCompactionEntryHasCorrectFields(t *testing.T) {
	e := CompactionEntry("hello summary", "start_id", "end_id", "ollama/qwen2.5:3b", 1000, 250, 12)
	assert.Equal(t, EntryTypeCompaction, e.Type)
	assert.Equal(t, "system", e.Role)
	var cd CompactionData
	require.NoError(t, json.Unmarshal(e.Data, &cd))
	assert.Equal(t, "hello summary", cd.Summary)
	assert.Equal(t, "start_id", cd.RangeStartID)
	assert.Equal(t, "end_id", cd.RangeEndID)
	assert.Equal(t, "ollama/qwen2.5:3b", cd.Model)
	assert.Equal(t, 1000, cd.TokensBefore)
	assert.Equal(t, 250, cd.TokensEstimatedAfter)
	assert.Equal(t, 12, cd.TurnsCompacted)
}

func TestToolResultData_AbortedFieldRoundTrip(t *testing.T) {
	entry := AbortedToolResultEntry("tc_abc")
	require.Equal(t, EntryTypeToolResult, entry.Type)

	var data ToolResultData
	require.NoError(t, json.Unmarshal(entry.Data, &data))

	require.Equal(t, "tc_abc", data.ToolCallID)
	require.Equal(t, "aborted by user", data.Error)
	require.True(t, data.IsError)
	require.True(t, data.Aborted)
	require.Empty(t, data.Output)
}

func TestToolResultData_OldJSONLWithoutAbortedField(t *testing.T) {
	// Simulate an old session entry written before the Aborted field existed.
	oldJSON := []byte(`{"tool_call_id":"tc_old","output":"hello","is_error":false}`)
	var data ToolResultData
	require.NoError(t, json.Unmarshal(oldJSON, &data))

	require.Equal(t, "tc_old", data.ToolCallID)
	require.Equal(t, "hello", data.Output)
	require.False(t, data.IsError)
	require.False(t, data.Aborted, "missing field must default to false")
}

func TestSession_AppendConcurrent(t *testing.T) {
	// Race-detector test: 100 goroutines each Append a uniquely-IDed entry.
	// Run with `go test -race` to catch any unguarded mutation.
	sess := NewSession("a", "k")

	const N = 100
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		i := i
		go func() {
			defer wg.Done()
			sess.Append(SessionEntry{
				ID:   fmt.Sprintf("e_%d", i),
				Type: EntryTypeMessage,
			})
		}()
	}
	wg.Wait()

	view := sess.View()
	require.Len(t, view, N, "every Append must land")

	seen := map[string]bool{}
	for _, e := range view {
		require.False(t, seen[e.ID], "duplicate ID %s", e.ID)
		seen[e.ID] = true
	}
	require.Len(t, seen, N)
}

func TestSession_ViewReturnsCopy(t *testing.T) {
	// Mutating the returned slice must not affect subsequent View() calls.
	sess := NewSession("a", "k")
	sess.Append(SessionEntry{ID: "e_1", Type: EntryTypeMessage})

	v1 := sess.View()
	require.Len(t, v1, 1)
	v1[0] = SessionEntry{ID: "MUTATED"}

	v2 := sess.View()
	require.Len(t, v2, 1)
	require.Equal(t, "e_1", v2[0].ID, "internal state must not be mutated by caller's slice modification")
}

func TestConcurrentAppendAndViewNoRace(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir)
	require.NoError(t, store.Create("a", "k"))
	sess := NewSession("a", "k")
	sess.SetStore(store)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			sess.Append(SessionEntry{Type: EntryTypeMessage, Role: "user", Data: []byte(`{"text":"x"}`)})
		}
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			_ = sess.View()
		}
	}()
	wg.Wait()

	require.Len(t, sess.Entries(), 500)
}
