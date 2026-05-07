package runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedTextLLM emits a single text response then EventDone. Distinct from
// fakeLLM only in that it doesn't count an idx — every call gets the same text.
// Used as the SUBAGENT's LLM so the parent's task tool result captures it.
type scriptedTextLLM struct {
	llmtest.Base
	text string
}

func (s *scriptedTextLLM) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		// Emit per-character so event-forwarding observability tests can see
		// multiple deltas with non-empty AgentID forwarded to the parent.
		for _, r := range s.text {
			ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: string(r)}
		}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

// taskCallingLLM emits exactly one tool_call to the "task" tool on the first
// call (delegating to the configured subagent_id with the configured prompt),
// then on subsequent calls emits a single text delta + Done so the loop
// terminates.
type taskCallingLLM struct {
	llmtest.Base
	subagentID  string
	prompt      string
	finalText   string
	calls       atomic.Int32
	toolCallID  string
}

func (t *taskCallingLLM) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	n := t.calls.Add(1)
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		if n == 1 {
			id := t.toolCallID
			if id == "" {
				id = "tc_task_0"
			}
			input, _ := json.Marshal(map[string]string{
				"agent_id": t.subagentID,
				"prompt":   t.prompt,
			})
			tc := &llm.ToolCall{
				ID:    id,
				Name:  "task",
				Input: input,
			}
			ch <- llm.ChatEvent{Type: llm.EventToolCallStart, ToolCall: tc}
			ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: tc}
			ch <- llm.ChatEvent{Type: llm.EventDone}
			return
		}
		// Second call: parent has the tool result, emit final assistant text.
		final := t.finalText
		if final == "" {
			final = "ok"
		}
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: final}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

// taskOnlyExecutor is an Executor that only knows the "task" tool. Anything
// else returns a useful error (subagent tests shouldn't hit other tools).
type taskOnlyExecutor struct {
	task *tool.TaskTool
}

func (e *taskOnlyExecutor) Execute(ctx context.Context, name string, input json.RawMessage) (tool.ToolResult, error) {
	if name == e.task.Name() {
		return e.task.Execute(ctx, input)
	}
	return tool.ToolResult{}, fmt.Errorf("unknown tool: %s", name)
}

func (e *taskOnlyExecutor) ToolDefs() []llm.ToolDef {
	return []llm.ToolDef{{
		Name:        e.task.Name(),
		Description: e.task.Description(),
		Parameters:  e.task.Parameters(),
	}}
}

func (e *taskOnlyExecutor) Names() []string {
	return []string{e.task.Name()}
}

func (e *taskOnlyExecutor) Get(name string) (tool.Tool, bool) {
	if name == e.task.Name() {
		return e.task, true
	}
	return nil, false
}


// TestRun_SubagentDelegationProducesFinalText verifies that delegating to a
// subagent via the task tool surfaces the subagent's final assistant text as
// the tool's output (which the parent session captures as a tool_result).
func TestRun_SubagentDelegationProducesFinalText(t *testing.T) {
	cfg := newTwoAgentCfg(t)
	subLLM := &scriptedTextLLM{text: "research result"}
	parentLLM := &taskCallingLLM{
		subagentID: "researcher",
		prompt:     "look into X",
		finalText:  "done",
	}

	deps := RuntimeDeps{}
	parent := &Runtime{
		LLM:      parentLLM,
		Session:  session.NewSession("default", "test"),
		AgentID:  "default",
		Model:    "parent",
		MaxTurns: 5,
	}
	factory := MakeSubagentFactory(cfg.Resolve, deps, subagentBuilderForLLM(subLLM), parent)
	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	out, err := parent.RunSync(context.Background(), "go research", nil)
	require.NoError(t, err)
	// RunSync collects ALL EventTextDelta events, including those forwarded
	// from the subagent (Runtime.emit forwards subagent deltas to the parent
	// channel). So the collected output contains both the subagent's text
	// and the parent's final text.
	assert.Contains(t, out, "research result", "subagent text was forwarded")
	assert.Contains(t, out, "done", "parent's final text was emitted")

	// The session should contain a tool_result entry whose Output is the
	// subagent's text. This is the more rigorous assertion — the session is
	// the durable record of what the parent's LLM saw as the tool result.
	var foundResult bool
	for _, e := range parent.Session.Entries() {
		if e.Type != session.EntryTypeToolResult {
			continue
		}
		var data session.ToolResultData
		require.NoError(t, json.Unmarshal(e.Data, &data))
		if strings.Contains(data.Output, "research result") {
			foundResult = true
			break
		}
	}
	require.True(t, foundResult, "parent session should contain the subagent's text as a tool_result")
}

// TestRun_SubagentEventsForwardedToParent asserts that intermediate text
// deltas emitted by the subagent are forwarded to the parent's events channel
// with AgentID set to the subagent's ID.
func TestRun_SubagentEventsForwardedToParent(t *testing.T) {
	cfg := newTwoAgentCfg(t)
	subLLM := &scriptedTextLLM{text: "research result"}
	parentLLM := &taskCallingLLM{
		subagentID: "researcher",
		prompt:     "look",
		finalText:  "done",
	}

	deps := RuntimeDeps{}
	parent := &Runtime{
		LLM:      parentLLM,
		Session:  session.NewSession("default", "test"),
		AgentID:  "default",
		Model:    "parent",
		MaxTurns: 5,
	}
	factory := MakeSubagentFactory(cfg.Resolve, deps, subagentBuilderForLLM(subLLM), parent)
	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	events, err := parent.Run(context.Background(), "go", nil)
	require.NoError(t, err)

	var subagentDeltas []string
	var sawDone bool
	timeout := time.After(5 * time.Second)
loop:
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				break loop
			}
			if ev.Type == EventTextDelta && ev.AgentID == "researcher" {
				subagentDeltas = append(subagentDeltas, ev.Text)
			}
			if ev.Type == EventDone && ev.AgentID == "" {
				sawDone = true
			}
		case <-timeout:
			t.Fatal("timed out waiting for parent events")
		}
	}

	assert.True(t, sawDone, "expected parent's own EventDone (AgentID empty)")
	require.NotEmpty(t, subagentDeltas, "expected at least one text delta forwarded with AgentID=researcher")
	joined := strings.Join(subagentDeltas, "")
	assert.Contains(t, joined, "research result",
		"forwarded subagent text deltas should accumulate to the subagent's final text")
}

// TestRun_SubagentDepthCapEnforced verifies the cap stops construction past
// the configured limit. With MaxAgentDepth=2, a parent at depth 0 can call
// a subagent (depth 1), and that subagent can call another (depth 2), but
// a third-level attempt (depth 3) must fail with the cap error and the
// would-be subagent's RuntimeInputs builder must NOT be invoked.
//
// Note: this test reimplements the factory wiring rather than calling
// MakeSubagentFactory directly because production buildSubagentInputs
// closures do not register the `task` tool on subagent registries (so
// recursion past depth 1 is unreachable from production paths today).
// To exercise the cap, this test gives each level its own task tool
// pointing at the next level. If production ever registers task tool on
// subagent registries, this test should be simplified to use the production
// MakeSubagentFactory directly. See subagent.go's depth-cap comment.
func TestRun_SubagentDepthCapEnforced(t *testing.T) {
	t.Setenv("HARNESS_MAX_AGENT_DEPTH", "2")

	cfg := newTwoAgentCfg(t)

	// Parent emits a task call to "researcher".
	parentLLM := &taskCallingLLM{
		subagentID: "researcher",
		prompt:     "go",
		finalText:  "parent done",
		toolCallID: "tc_parent",
	}

	// Track build calls. With max depth 2, the parent (depth 0) can build a
	// subagent at depth 1; that subagent can build one at depth 2; a 3rd-level
	// build must be blocked BEFORE buildInputs runs.
	var buildCount atomic.Int32

	deps := RuntimeDeps{}
	parent := &Runtime{
		LLM:      parentLLM,
		Session:  session.NewSession("default", "test"),
		AgentID:  "default",
		Model:    "parent",
		MaxTurns: 5,
		Depth:    0,
	}

	// Self-referencing factory: every subagent gets a task tool whose factory
	// is THIS SAME closure, so each level recurses one step deeper. The
	// tools-executor patch happens AFTER BuildRuntime so the subagent's
	// task tool captures its own runtime as parent (matching production wiring).
	var factory tool.SubagentFactory
	buildInputs := func(spec AgentSpec) (RuntimeInputs, error) {
		buildCount.Add(1)
		return RuntimeInputs{
			Provider: &taskCallingLLM{
				subagentID: "researcher",
				prompt:     "deeper",
				finalText:  "sub done",
				toolCallID: fmt.Sprintf("tc_sub_%d", buildCount.Load()),
			},
			Session:      NewSubagentSession(spec.ID),
			Compaction:   nil,
			IngestSource: "",
		}, nil
	}
	factory = func(ctx context.Context, agentID string, parentDepth int) (tool.SubagentRunner, error) {
		if parentDepth+1 > parent.maxAgentDepth() {
			return nil, fmt.Errorf("subagent depth limit %d reached", parent.maxAgentDepth())
		}
		ss, ok := cfg.Resolve(agentID)
		if !ok {
			return nil, fmt.Errorf("subagent %s not found", agentID)
		}
		if !ss.Registered {
			return nil, fmt.Errorf("agent %s is not a subagent", agentID)
		}
		inputs, err := buildInputs(ss.Spec)
		if err != nil {
			return nil, err
		}
		rt, err := BuildRuntime(deps, inputs, ss.Spec)
		if err != nil {
			return nil, err
		}
		rt.Parent = parent
		rt.Depth = parentDepth + 1
		// Patch the subagent's Tools to include a task tool wired to THIS
		// same factory so it can recurse one step deeper.
		rt.Tools = &taskOnlyExecutor{
			task: tool.NewTaskTool(factory, rt.Depth, cfg.EligibleSubagents()),
		}
		return &subagentRunnerAdapter{rt: rt}, nil
	}

	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	_, err := parent.RunSync(context.Background(), "go", nil)
	require.NoError(t, err)

	// With max depth = 2 we expect at most 2 buildInputs calls:
	//   - depth 1 (parent calls subagent): allowed → builds
	//   - depth 2 (sub calls sub-sub): allowed → builds
	//   - depth 3 (sub-sub tries to call): blocked BEFORE buildInputs runs
	assert.LessOrEqual(t, int(buildCount.Load()), 2,
		"buildInputs must not run for a subagent that would exceed the depth cap")
	assert.GreaterOrEqual(t, int(buildCount.Load()), 1,
		"at least the first subagent (depth 1) must have been built")

	// At least one tool_result somewhere in the parent's session should be
	// the deepest subagent's output, which is the depth-limit error string.
	// The error propagates UP as the deeper subagent's tool result Error.
	// (We don't see it directly in the parent's session — it lives in the
	// inner subagent's session — but the buildCount assertion above proves
	// the cap fired correctly.)
}

// TestRun_SubagentAbortPropagatesFromParent cancels the parent's ctx mid-
// subagent execution. The subagent should terminate and the parent session
// should still satisfy the tool_call ↔ tool_result pairing invariant.
func TestRun_SubagentAbortPropagatesFromParent(t *testing.T) {
	cfg := newTwoAgentCfg(t)

	// Subagent LLM that takes its time so we can cancel during it.
	slowSub := &slowLLM{
		text:  "should not finish",
		delay: 500 * time.Millisecond,
	}
	parentLLM := &taskCallingLLM{
		subagentID: "researcher",
		prompt:     "go",
		finalText:  "shouldn't reach",
	}

	deps := RuntimeDeps{}
	parent := &Runtime{
		LLM:      parentLLM,
		Session:  session.NewSession("default", "test"),
		AgentID:  "default",
		Model:    "parent",
		MaxTurns: 5,
	}
	factory := MakeSubagentFactory(cfg.Resolve, deps, subagentBuilderForLLM(slowSub), parent)
	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel after a short delay — the subagent's slow LLM is mid-stream by then.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	events, err := parent.Run(ctx, "go", nil)
	require.NoError(t, err)
	for range events {
		// drain
	}

	// Confirm pairing: count tool_calls and tool_results.
	var calls, results int
	for _, e := range parent.Session.Entries() {
		switch e.Type {
		case session.EntryTypeToolCall:
			calls++
		case session.EntryTypeToolResult:
			results++
		}
	}
	require.Equal(t, calls, results,
		"every tool_call must have a paired tool_result (no orphans after abort)")
}

// slowLLM streams its text after a delay; used to test cancellation paths.
type slowLLM struct {
	llmtest.Base
	text  string
	delay time.Duration
}

func (s *slowLLM) ChatStream(ctx context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			ch <- llm.ChatEvent{Type: llm.EventError, Error: ctx.Err()}
			return
		}
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: s.text}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

// TestRun_SubagentNotInEligibleListReturnsError verifies that calling a
// task tool with an unknown subagent ID returns a tool result error that
// names the bad ID and mentions the eligible alternatives.
func TestRun_SubagentNotInEligibleListReturnsError(t *testing.T) {
	cfg := newTwoAgentCfg(t)

	// Parent dispatches to "ghost" — not registered as a subagent.
	parentLLM := &taskCallingLLM{
		subagentID: "ghost",
		prompt:     "go",
		finalText:  "parent done",
	}

	deps := RuntimeDeps{}
	parent := &Runtime{
		LLM:      parentLLM,
		Session:  session.NewSession("default", "test"),
		AgentID:  "default",
		Model:    "parent",
		MaxTurns: 5,
	}
	factory := MakeSubagentFactory(cfg.Resolve, deps, subagentBuilderForLLM(&scriptedTextLLM{text: "x"}), parent)
	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	out, err := parent.RunSync(context.Background(), "go", nil)
	require.NoError(t, err)
	assert.Equal(t, "parent done", out)

	// Find the tool_result and assert its Error mentions the bad ID and the
	// eligible list ("researcher" is the only opt-in subagent in cfg).
	var found bool
	for _, e := range parent.Session.Entries() {
		if e.Type != session.EntryTypeToolResult {
			continue
		}
		var data session.ToolResultData
		require.NoError(t, json.Unmarshal(e.Data, &data))
		if data.Error == "" {
			continue
		}
		assert.Contains(t, data.Error, "ghost",
			"error should name the unknown subagent")
		assert.Contains(t, data.Error, "researcher",
			"error should list the eligible subagents")
		found = true
		break
	}
	require.True(t, found, "expected a tool_result entry with an error mentioning the bad subagent")
}
