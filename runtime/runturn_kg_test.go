package runtime

import (
	"encoding/json"
	"testing"

	"github.com/sausheong/harness/session"
	"github.com/stretchr/testify/require"
)

func TestSessionThread_RendersEntries(t *testing.T) {
	sess := session.NewSession("a", "k")
	sess.Append(session.UserMessageEntry("hello"))
	sess.Append(session.AssistantMessageEntry("hi there"))
	sess.Append(session.ToolCallEntry("tc1", "echo", json.RawMessage(`{"x":1}`)))
	sess.Append(session.ToolResultEntry("tc1", "echoed", "", nil))
	sess.Append(session.ToolResultEntry("tc2", "", "boom", nil))

	thread := sessionThread(sess.View())

	require.Equal(t, []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "assistant", Content: "[tool: echo]\n{\"x\":1}"},
		{Role: "user", Content: "echoed"},
		{Role: "user", Content: "[error] boom"},
	}, thread)
}

func TestSessionThread_SkipsCompaction(t *testing.T) {
	sess := session.NewSession("a", "k")
	// Compaction is appended FIRST so the entries that follow remain on the
	// View() walk-back path. View() stops its back-walk at the most recent
	// compaction (inclusive), so appending user/assistant AFTER it yields
	// View() == [compaction, user, assistant] — the compaction is genuinely
	// present in the slice handed to sessionThread, exercising the skip path.
	// (Appending compaction last would make it the leaf, and View() would
	// return only [compaction], never reaching the messages.)
	sess.Append(session.CompactionEntry("summary", "", "", "m", 0, 0, 0))
	sess.Append(session.UserMessageEntry("hello"))
	sess.Append(session.AssistantMessageEntry("hi"))

	view := sess.View()
	require.Equal(t, session.EntryTypeCompaction, view[0].Type, "View() must include the compaction entry to exercise the skip")

	got := sessionThread(view)
	require.Equal(t, []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}, got)
}

func TestSessionThread_SkipsUndecodableEntry(t *testing.T) {
	history := []session.SessionEntry{
		{Type: session.EntryTypeToolCall, Data: []byte("not json")},
		session.UserMessageEntry("after"),
	}
	got := sessionThread(history)
	require.Equal(t, []Message{{Role: "user", Content: "after"}}, got)
}

func TestSessionThread_Empty(t *testing.T) {
	require.Nil(t, sessionThread(nil))
}
