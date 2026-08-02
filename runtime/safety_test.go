package runtime

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type panickingTool struct{}

func (*panickingTool) Name() string                           { return "panic_tool" }
func (*panickingTool) Description() string                    { return "panics for testing" }
func (*panickingTool) Parameters() json.RawMessage            { return json.RawMessage(`{"type":"object"}`) }
func (*panickingTool) IsConcurrencySafe(json.RawMessage) bool { return false }
func (*panickingTool) Execute(context.Context, json.RawMessage) (tool.ToolResult, error) {
	panic("boom")
}

type panickingPermission struct{}

func (panickingPermission) Check(context.Context, string, string, json.RawMessage) tool.Decision {
	panic("permission boom")
}
func (panickingPermission) FilterToolDefs(defs []llm.ToolDef, _ string) []llm.ToolDef {
	return defs
}

func TestDispatchToolRecoversExecutePanicAndPreservesPairing(t *testing.T) {
	reg := tool.NewRegistry()
	reg.Register(&panickingTool{})
	rt := &Runtime{Tools: reg, Session: session.NewSession("agent", "key")}

	result, aborted := rt.dispatchTool(context.Background(), llm.ToolCall{
		ID: "call-1", Name: "panic_tool", Input: json.RawMessage(`{}`),
	}, nil)

	assert.False(t, aborted)
	assert.Contains(t, result.Error, "tool panic_tool panicked: boom")
	entries := rt.Session.Entries()
	require.Len(t, entries, 2)
	assert.Equal(t, session.EntryTypeToolCall, entries[0].Type)
	assert.Equal(t, session.EntryTypeToolResult, entries[1].Type)
}

func TestDispatchToolFailsClosedWhenPermissionPanics(t *testing.T) {
	reg := tool.NewRegistry()
	reg.Register(&mockTool{name: "safe", output: "must not execute"})
	rt := &Runtime{
		Tools: reg, Session: session.NewSession("agent", "key"),
		Permission: panickingPermission{},
	}

	result, aborted := rt.dispatchTool(context.Background(), llm.ToolCall{
		ID: "call-1", Name: "safe", Input: json.RawMessage(`{}`),
	}, nil)

	assert.False(t, aborted)
	assert.Contains(t, result.Error, "permission checker panicked")
	assert.NotContains(t, result.Output, "must not execute")
	require.Len(t, rt.Session.Entries(), 2)
}

func TestRunRecoversHookPanicAsEventError(t *testing.T) {
	rt := hookFixture(t, LifecycleHooks{
		OnUserPromptSubmit: func(context.Context, string, []llm.ImageContent) (string, []llm.ImageContent, error) {
			panic("hook boom")
		},
	}, nil, nil)

	events, err := rt.Run(context.Background(), "hello", nil)
	require.NoError(t, err)
	var got error
	for event := range events {
		if event.Type == EventError {
			got = event.Error
		}
	}
	require.Error(t, got)
	assert.Contains(t, got.Error(), "runtime panicked: hook boom")
}

func TestRunRejectsConcurrentCall(t *testing.T) {
	started := make(chan struct{})
	provider := &llmtest.Stub{Text: "done", Delay: 50 * time.Millisecond, Started: started}
	rt := &Runtime{
		LLM: provider, Tools: tool.NewRegistry(),
		Session: session.NewSession("agent", "key"), Model: "mock", MaxTurns: 1,
	}

	first, err := rt.Run(context.Background(), "first", nil)
	require.NoError(t, err)
	<-started
	second, err := rt.Run(context.Background(), "second", nil)
	require.ErrorContains(t, err, "already running")
	assert.Nil(t, second)
	for range first {
	}

	third, err := rt.Run(context.Background(), "third", nil)
	require.NoError(t, err, "runtime must become available after the first run ends")
	for range third {
	}
}
