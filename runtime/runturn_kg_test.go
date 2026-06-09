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
	sess.Append(session.UserMessageEntry("hello"))
	sess.Append(session.AssistantMessageEntry("hi"))
	// Build the expected from whatever View() returns, minus any compaction/meta.
	got := sessionThread(sess.View())
	require.Equal(t, []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}, got)
}

func TestSessionThread_Empty(t *testing.T) {
	require.Nil(t, sessionThread(nil))
}
