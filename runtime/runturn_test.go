package runtime

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/require"
)

// echoTool is a minimal tool.Tool implementation for RunTurn tests. It
// satisfies the real tool.Tool interface (Name/Description/Parameters/
// Execute/IsConcurrencySafe) and echoes its raw input back as output.
type echoTool struct{}

func (echoTool) Name() string                { return "echo" }
func (echoTool) Description() string         { return "echoes its input" }
func (echoTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (echoTool) Execute(_ context.Context, input json.RawMessage) (tool.ToolResult, error) {
	return tool.ToolResult{Output: string(input)}, nil
}
func (echoTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }

// newEchoRegistry returns a tool.Registry containing only echoTool.
func newEchoRegistry() *tool.Registry {
	reg := tool.NewRegistry()
	reg.Register(echoTool{})
	return reg
}

// TestRunTurn_TextOnly_IsDone: provider emits a text delta then EventDone
// with NO tool calls. The turn must be terminal (Done=true, StopReason
// "completed") and append exactly two entries: the user message and the
// assistant message.
func TestRunTurn_TextOnly_IsDone(t *testing.T) {
	// scriptedStreamLLM emits its scripted events only on the FIRST call;
	// subsequent calls emit a bare EventDone. For a text-only first turn we
	// script text + done directly.
	llmProvider := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventTextDelta, text: "hello world"},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM:     llmProvider,
		Tools:   newEchoRegistry(),
		Session: session.NewSession("a", "k"),
		AgentID: "a",
		Model:   "test",
	}

	res, err := rt.RunTurn(context.Background(), "hi", nil, nil)
	require.NoError(t, err)
	require.True(t, res.Done, "text-only turn should be terminal")
	require.Equal(t, "completed", res.StopReason)
	require.Len(t, res.Entries, 2, "expected user + assistant entries")
	require.Equal(t, session.EntryTypeMessage, res.Entries[0].Type)
	require.Equal(t, "user", res.Entries[0].Role)
	require.Equal(t, session.EntryTypeMessage, res.Entries[1].Type)
	require.Equal(t, "assistant", res.Entries[1].Role)
}

// TestRunTurn_ToolCall_NotDone_ThenDone: turn 1 the model calls echo ⇒ the
// turn is NOT done and the entries include a tool_result entry. Turn 2
// (continuation, empty userMsg) the model emits text only ⇒ the turn is done.
func TestRunTurn_ToolCall_NotDone_ThenDone(t *testing.T) {
	tc := &llm.ToolCall{ID: "tc_1", Name: "echo", Input: json.RawMessage(`{"x":1}`)}

	// scriptedStreamLLM emits the scripted events on call 1 and a bare
	// EventDone on every subsequent call — which is exactly the turn-2
	// text-only/no-tool-calls behavior we need.
	llmProvider := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallStart, toolCall: tc},
		{typ: llm.EventToolCallDone, toolCall: tc},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM:     llmProvider,
		Tools:   newEchoRegistry(),
		Session: session.NewSession("a", "k"),
		AgentID: "a",
		Model:   "test",
	}

	// Turn 1: tool call ⇒ not done, has a tool_result entry.
	res1, err := rt.RunTurn(context.Background(), "use echo", nil, nil)
	require.NoError(t, err)
	require.False(t, res1.Done, "a turn that produced tool calls must not be terminal")
	require.Equal(t, "continue", res1.StopReason)

	var sawToolResult bool
	for _, e := range res1.Entries {
		if e.Type == session.EntryTypeToolResult {
			sawToolResult = true
		}
	}
	require.True(t, sawToolResult, "turn 1 entries should include a tool_result")

	// Turn 2: continuation with empty userMsg ⇒ model emits no tool calls
	// (bare EventDone) ⇒ terminal.
	res2, err := rt.RunTurn(context.Background(), "", nil, nil)
	require.NoError(t, err)
	require.True(t, res2.Done, "second turn (no tool calls) should be terminal")
	require.Equal(t, "completed", res2.StopReason)
}

// TestRunTurn_LLMError: the provider emits a mid-stream EventError. RunTurn
// does NOT use Run's stream-fallback path, so the error surfaces directly:
// the turn is terminal with StopReason "error" and a non-nil Err.
//
// Note on the fake: scriptedStreamLLM's scriptedStreamEvent cannot carry an
// error value (it has no error field), so it cannot emit a meaningful
// EventError. We reuse streamingOnlyProvider from streamfallback_test.go —
// the same-package fake the streaming tests use to simulate a mid-stream
// error (text delta then EventError with a real error).
func TestRunTurn_LLMError(t *testing.T) {
	llmProvider := &streamingOnlyProvider{}

	rt := &Runtime{
		LLM:     llmProvider,
		Tools:   newEchoRegistry(),
		Session: session.NewSession("a", "k"),
		AgentID: "a",
		Model:   "test",
	}

	res, err := rt.RunTurn(context.Background(), "do it", nil, nil)
	require.NoError(t, err, "RunTurn surfaces stream errors via TurnResult.Err, not its own error return")
	require.True(t, res.Done, "an errored turn must be terminal")
	require.Equal(t, "error", res.StopReason)
	require.Error(t, res.Err)
	require.EqualValues(t, 1, llmProvider.streamCalls.Load(), "stream hit exactly once (no fallback in RunTurn)")
}

// TestRunTurn_PreCancelledContext: a context cancelled before RunTurn runs
// must short-circuit to an aborted turn with a non-nil Err, and the LLM must
// never be called. scriptedStreamLLM exposes a `calls` counter, so we assert
// the provider was never invoked.
func TestRunTurn_PreCancelledContext(t *testing.T) {
	llmProvider := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventTextDelta, text: "should never run"},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM:     llmProvider,
		Tools:   newEchoRegistry(),
		Session: session.NewSession("a", "k"),
		AgentID: "a",
		Model:   "test",
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res, err := rt.RunTurn(ctx, "hi", nil, nil)
	require.NoError(t, err)
	require.Equal(t, "aborted", res.StopReason)
	require.Error(t, res.Err)
	require.EqualValues(t, 0, llmProvider.calls.Load(), "LLM must not be called when context is pre-cancelled")
}
