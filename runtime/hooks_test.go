package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hookFixture builds a Runtime wired to a mock LLM that emits the given
// canned event stream. Returns the Runtime and the session so tests can
// assert on session entries.
func hookFixture(t *testing.T, hooks LifecycleHooks, events []llm.ChatEvent, reg *tool.Registry) *Runtime {
	t.Helper()
	if reg == nil {
		reg = tool.NewRegistry()
	}
	return &Runtime{
		LLM:       &mockLLMProvider{events: events},
		Tools:     reg,
		Session:   session.NewSession("test-agent", "test-key"),
		Model:     "mock-model",
		Workspace: t.TempDir(),
		MaxTurns:  3,
		AgentLoop: LoopConfig{Hooks: hooks},
	}
}

// drainEvents consumes the channel and returns any error events seen.
func drainEvents(t *testing.T, events <-chan AgentEvent) {
	t.Helper()
	for e := range events {
		if e.Type == EventError {
			t.Fatalf("unexpected error: %v", e.Error)
		}
	}
}

func TestHook_OnSessionStart_FiresOnce(t *testing.T) {
	var fires int32
	var sawSession *session.Session
	hooks := LifecycleHooks{
		OnSessionStart: func(_ context.Context, s *session.Session) {
			atomic.AddInt32(&fires, 1)
			sawSession = s
		},
	}
	rt := hookFixture(t, hooks, []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "hi"},
		{Type: llm.EventDone},
	}, nil)

	events, err := rt.Run(context.Background(), "ping", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, int32(1), atomic.LoadInt32(&fires))
	assert.Same(t, rt.Session, sawSession, "hook receives the runtime's own session")
}

func TestHook_OnUserPromptSubmit_RewritesPrompt(t *testing.T) {
	hooks := LifecycleHooks{
		OnUserPromptSubmit: func(_ context.Context, prompt string, imgs []llm.ImageContent) (string, []llm.ImageContent, error) {
			return "REWRITTEN: " + prompt, imgs, nil
		},
	}
	rt := hookFixture(t, hooks, []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "ok"},
		{Type: llm.EventDone},
	}, nil)

	events, err := rt.Run(context.Background(), "original", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	// First session entry is the user message — assert the rewritten one
	// landed there.
	view := rt.Session.View()
	require.NotEmpty(t, view)
	first := view[0]
	require.Equal(t, session.EntryTypeMessage, first.Type)
	require.Equal(t, "user", first.Role)
	var md session.MessageData
	require.NoError(t, json.Unmarshal(first.Data, &md))
	assert.Equal(t, "REWRITTEN: original", md.Text)
}

func TestHook_OnUserPromptSubmit_ErrorAbortsRun(t *testing.T) {
	hooks := LifecycleHooks{
		OnUserPromptSubmit: func(_ context.Context, _ string, _ []llm.ImageContent) (string, []llm.ImageContent, error) {
			return "", nil, errors.New("rejected")
		},
	}
	rt := hookFixture(t, hooks, []llm.ChatEvent{{Type: llm.EventDone}}, nil)

	events, err := rt.Run(context.Background(), "x", nil)
	require.NoError(t, err)

	var sawError bool
	for e := range events {
		if e.Type == EventError {
			sawError = true
			require.Error(t, e.Error)
			assert.Contains(t, e.Error.Error(), "rejected")
		}
	}
	assert.True(t, sawError, "expected EventError when OnUserPromptSubmit returns err")
}

func TestHook_BeforeToolUse_DenialBlocksExecute(t *testing.T) {
	var executed int32
	reg := tool.NewRegistry()
	reg.Register(&countingTool{name: "noop", executed: &executed})

	callCount := 0
	llmMock := &statefulMockLLMProvider{
		responses: [][]llm.ChatEvent{
			{
				{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{
					ID: "tc_1", Name: "noop", Input: json.RawMessage(`{}`),
				}},
				{Type: llm.EventDone},
			},
			{{Type: llm.EventTextDelta, Text: "after"}, {Type: llm.EventDone}},
		},
		callCount: &callCount,
	}

	hooks := LifecycleHooks{
		BeforeToolUse: func(_ context.Context, name string, _ json.RawMessage) (HookDecision, error) {
			return HookDecision{Allow: false, Reason: "policy: " + name + " forbidden"}, nil
		},
	}
	rt := &Runtime{
		LLM:       llmMock,
		Tools:     reg,
		Session:   session.NewSession("a", "k"),
		Model:     "m",
		Workspace: t.TempDir(),
		MaxTurns:  3,
		AgentLoop: LoopConfig{Hooks: hooks},
	}
	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, int32(0), atomic.LoadInt32(&executed),
		"tool Execute must not run when BeforeToolUse denies")

	// Walk the session for the tool_result entry — it should carry the
	// denial reason.
	var foundDenial bool
	for _, e := range rt.Session.View() {
		if e.Type != session.EntryTypeToolResult {
			continue
		}
		var tr session.ToolResultData
		require.NoError(t, json.Unmarshal(e.Data, &tr))
		if tr.Error == "policy: noop forbidden" {
			foundDenial = true
			break
		}
	}
	assert.True(t, foundDenial, "denial reason must surface in the tool_result entry")
}

func TestHook_BeforeToolUse_AllowsThroughToPermission(t *testing.T) {
	var executed int32
	reg := tool.NewRegistry()
	reg.Register(&countingTool{name: "noop", executed: &executed})

	callCount := 0
	llmMock := &statefulMockLLMProvider{
		responses: [][]llm.ChatEvent{
			{
				{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{
					ID: "tc_1", Name: "noop", Input: json.RawMessage(`{}`),
				}},
				{Type: llm.EventDone},
			},
			{{Type: llm.EventTextDelta, Text: "done"}, {Type: llm.EventDone}},
		},
		callCount: &callCount,
	}

	var permissionConsulted int32
	perm := &countingPermission{calls: &permissionConsulted}

	hooks := LifecycleHooks{
		BeforeToolUse: func(_ context.Context, _ string, _ json.RawMessage) (HookDecision, error) {
			return HookDecision{Allow: true}, nil
		},
	}
	rt := &Runtime{
		LLM:        llmMock,
		Tools:      reg,
		Session:    session.NewSession("a", "k"),
		Model:      "m",
		Workspace:  t.TempDir(),
		MaxTurns:   3,
		Permission: perm,
		AgentLoop:  LoopConfig{Hooks: hooks},
	}

	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, int32(1), atomic.LoadInt32(&permissionConsulted),
		"PermissionChecker.Check must still run when BeforeToolUse allows")
	assert.Equal(t, int32(1), atomic.LoadInt32(&executed),
		"tool must execute when both BeforeToolUse and PermissionChecker allow")
}

func TestHook_AfterToolUse_ObservesResult(t *testing.T) {
	reg := tool.NewRegistry()
	reg.Register(&mockTool{name: "noop", output: "tool-said-this"})

	callCount := 0
	llmMock := &statefulMockLLMProvider{
		responses: [][]llm.ChatEvent{
			{
				{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{
					ID: "tc_1", Name: "noop", Input: json.RawMessage(`{}`),
				}},
				{Type: llm.EventDone},
			},
			{{Type: llm.EventTextDelta, Text: "fin"}, {Type: llm.EventDone}},
		},
		callCount: &callCount,
	}

	var capturedName string
	var capturedOutput string
	hooks := LifecycleHooks{
		AfterToolUse: func(_ context.Context, name string, _ json.RawMessage, res tool.ToolResult) {
			capturedName = name
			capturedOutput = res.Output
		},
	}
	rt := &Runtime{
		LLM:       llmMock,
		Tools:     reg,
		Session:   session.NewSession("a", "k"),
		Model:     "m",
		Workspace: t.TempDir(),
		MaxTurns:  3,
		AgentLoop: LoopConfig{Hooks: hooks},
	}

	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, "noop", capturedName)
	assert.Equal(t, "tool-said-this", capturedOutput)
}

func TestHook_OnStop_FiresWithCompletedReason(t *testing.T) {
	var captured string
	hooks := LifecycleHooks{
		OnStop: func(_ context.Context, reason string) { captured = reason },
	}
	rt := hookFixture(t, hooks, []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "hi"},
		{Type: llm.EventDone},
	}, nil)

	events, err := rt.Run(context.Background(), "ping", nil)
	require.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, "completed", captured)
}

func TestHook_OnStop_FiresWithAbortedReason(t *testing.T) {
	var captured string
	hooks := LifecycleHooks{
		OnStop: func(_ context.Context, reason string) { captured = reason },
	}
	// Cancel ctx before Run starts → first turn's ctx.Err() check trips
	// immediately and we emit Aborted.
	rt := hookFixture(t, hooks, []llm.ChatEvent{{Type: llm.EventDone}}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	events, err := rt.Run(ctx, "ping", nil)
	require.NoError(t, err)
	for range events {
	}

	assert.Equal(t, "aborted", captured)
}

func TestHook_NilHooks_NoChange(t *testing.T) {
	// Sanity: zero LifecycleHooks behaves identically to current behavior.
	rt := hookFixture(t, LifecycleHooks{}, []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "hello"},
		{Type: llm.EventDone},
	}, nil)
	events, err := rt.Run(context.Background(), "x", nil)
	require.NoError(t, err)
	var got string
	for e := range events {
		if e.Type == EventTextDelta {
			got += e.Text
		}
	}
	assert.Equal(t, "hello", got)
}

// --- test fixtures ---

type countingTool struct {
	name     string
	executed *int32
}

func (t *countingTool) Name() string                            { return t.name }
func (t *countingTool) Description() string                     { return "counting" }
func (t *countingTool) Parameters() json.RawMessage             { return json.RawMessage(`{"type":"object"}`) }
func (t *countingTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *countingTool) Execute(_ context.Context, _ json.RawMessage) (tool.ToolResult, error) {
	atomic.AddInt32(t.executed, 1)
	return tool.ToolResult{Output: "ok"}, nil
}

type countingPermission struct {
	calls *int32
}

func (p *countingPermission) Check(_ context.Context, _, _ string, _ json.RawMessage) tool.Decision {
	atomic.AddInt32(p.calls, 1)
	return tool.Decision{Behavior: tool.DecisionAllow}
}

func (p *countingPermission) FilterToolDefs(defs []llm.ToolDef, _ string) []llm.ToolDef {
	return defs
}
