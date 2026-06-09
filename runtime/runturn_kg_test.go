package runtime

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
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

// fakeKG is a controllable KnowledgeGraph for RunTurn wiring tests.
type fakeKG struct {
	shouldRecall bool
	recallReturn string
	recallDelay  time.Duration
	recallCalls  int
	recallQuery  string

	ingestCalls   int
	ingestThreads [][]Message
}

func (f *fakeKG) ShouldRecall(q string) bool { return f.shouldRecall }
func (f *fakeKG) Recall(ctx context.Context, q string) string {
	f.recallCalls++
	f.recallQuery = q
	if f.recallDelay > 0 {
		select {
		case <-time.After(f.recallDelay):
		case <-ctx.Done():
			return ""
		}
	}
	return f.recallReturn
}
func (f *fakeKG) Ingest(_ context.Context, thread []Message) {
	f.ingestCalls++
	f.ingestThreads = append(f.ingestThreads, thread)
}

func TestRunTurn_RecallInjectedIntoPrompt(t *testing.T) {
	var captured []llm.SystemPromptPart
	cap := &capturingLLMStub{onChatStream: func(req llm.ChatRequest) { captured = req.SystemPromptParts }}
	kg := &fakeKG{shouldRecall: true, recallReturn: "Relevant memories:\n- the user likes tea\n"}

	rt := &Runtime{
		LLM: cap, Tools: newEchoRegistry(), Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", StaticSystemPrompt: "STATIC", KG: kg,
	}
	_, err := rt.RunTurn(context.Background(), "what do I like?", nil, nil)
	require.NoError(t, err)
	require.Equal(t, 1, kg.recallCalls)
	require.Equal(t, "what do I like?", kg.recallQuery)

	require.GreaterOrEqual(t, len(captured), 2, "expected static + recall parts")
	require.Equal(t, "STATIC", captured[0].Text)
	require.True(t, captured[0].Cache)
	require.Equal(t, "Relevant memories:\n- the user likes tea\n", captured[len(captured)-1].Text)
	require.False(t, captured[len(captured)-1].Cache, "recall hint must NOT be cached")
}

func TestRunTurn_NoRecallWhenShouldRecallFalse(t *testing.T) {
	var captured []llm.SystemPromptPart
	cap := &capturingLLMStub{onChatStream: func(req llm.ChatRequest) { captured = req.SystemPromptParts }}
	kg := &fakeKG{shouldRecall: false, recallReturn: "should not appear"}

	rt := &Runtime{
		LLM: cap, Tools: newEchoRegistry(), Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", StaticSystemPrompt: "STATIC", KG: kg,
	}
	_, err := rt.RunTurn(context.Background(), "hi", nil, nil)
	require.NoError(t, err)
	require.Equal(t, 0, kg.recallCalls, "ShouldRecall false ⇒ Recall not called")
	require.Len(t, captured, 1, "only the static part")
	require.Equal(t, "STATIC", captured[0].Text)
}

func TestRunTurn_NoRecallOnContinuationRound(t *testing.T) {
	var captured []llm.SystemPromptPart
	cap := &capturingLLMStub{onChatStream: func(req llm.ChatRequest) { captured = req.SystemPromptParts }}
	kg := &fakeKG{shouldRecall: true, recallReturn: "hint"}

	rt := &Runtime{
		LLM: cap, Tools: newEchoRegistry(), Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", StaticSystemPrompt: "STATIC", KG: kg,
	}
	_, err := rt.RunTurn(context.Background(), "", nil, nil)
	require.NoError(t, err)
	require.Equal(t, 0, kg.recallCalls, "continuation round must not recall")
	require.Len(t, captured, 1)
}

func TestRunTurn_RecallTimeoutInjectsNothing(t *testing.T) {
	var captured []llm.SystemPromptPart
	cap := &capturingLLMStub{onChatStream: func(req llm.ChatRequest) { captured = req.SystemPromptParts }}
	kg := &fakeKG{shouldRecall: true, recallReturn: "late", recallDelay: 2 * time.Second}

	rt := &Runtime{
		LLM: cap, Tools: newEchoRegistry(), Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", StaticSystemPrompt: "STATIC", KG: kg,
	}
	start := time.Now()
	_, err := rt.RunTurn(context.Background(), "slow?", nil, nil)
	require.NoError(t, err)
	require.Less(t, time.Since(start), 1500*time.Millisecond, "recall must be bounded ~800ms")
	require.Len(t, captured, 1, "timed-out recall injects nothing")
}
