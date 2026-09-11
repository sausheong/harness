package runtime

import (
	"context"
	"encoding/json"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/require"
)

func TestStreamingToolsEnabled_Default(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "")
	if (&Runtime{}).streamingToolsEnabled() {
		t.Fatal("expected false when env unset")
	}
}

func TestStreamingToolsEnabled_Override(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")
	if !(&Runtime{}).streamingToolsEnabled() {
		t.Fatal("expected true when env=1")
	}
}

func TestStreamingToolsEnabled_InvalidFallsBack(t *testing.T) {
	cases := []string{"0", "true", "True", "garbage", " 1 ", "01", "yes"}
	for _, v := range cases {
		t.Run(v, func(t *testing.T) {
			t.Setenv("HARNESS_STREAMING_TOOLS", v)
			if (&Runtime{}).streamingToolsEnabled() {
				t.Fatalf("expected false for %q", v)
			}
		})
	}
}

func TestRuntime_StreamingTools_ConfigTrueWinsOverEnvUnset(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "")
	r := &Runtime{AgentLoop: LoopConfig{StreamingTools: true}}
	if !r.streamingToolsEnabled() {
		t.Fatal("expected true when config sets streamingTools=true")
	}
}

func TestRuntime_StreamingTools_ConfigTrueWinsOverEnvZero(t *testing.T) {
	// Env is "0" (off) but config says on — config wins.
	t.Setenv("HARNESS_STREAMING_TOOLS", "0")
	r := &Runtime{AgentLoop: LoopConfig{StreamingTools: true}}
	if !r.streamingToolsEnabled() {
		t.Fatal("expected true when config=true even with env=0")
	}
}

func TestRuntime_StreamingTools_ConfigFalseFallsBackToEnv(t *testing.T) {
	// When config bool is false (the zero value, indistinguishable from
	// "field absent"), fall back to env. Env=1 → on.
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")
	r := &Runtime{AgentLoop: LoopConfig{StreamingTools: false}}
	if !r.streamingToolsEnabled() {
		t.Fatal("expected true when config=false but env=1")
	}
}

func TestRuntime_StreamingTools_BothUnsetIsOff(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "")
	r := &Runtime{AgentLoop: LoopConfig{}}
	if r.streamingToolsEnabled() {
		t.Fatal("expected false when neither config nor env set")
	}
}

// scriptedStreamEvent is a single event for scriptedStreamLLM to emit.
type scriptedStreamEvent struct {
	typ      llm.EventType
	text     string
	toolCall *llm.ToolCall
	delay    time.Duration // sleep BEFORE emitting this event
}

// scriptedStreamLLM is a fake LLMProvider that emits a fixed sequence of
// ChatEvents with optional delays. Use to script "text → tool_use → text → done"
// streams for streaming-kickoff tests. After the first call has emitted its
// scripted events, subsequent calls just emit EventDone so the agent loop
// terminates cleanly.
type scriptedStreamLLM struct {
	llmtest.Base
	events []scriptedStreamEvent
	calls  atomic.Int32
}

func (s *scriptedStreamLLM) ChatStream(ctx context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	n := s.calls.Add(1)
	out := make(chan llm.ChatEvent, len(s.events)+1)
	go func() {
		defer close(out)
		if n != 1 {
			// Subsequent turns: answer without further tool calls.
			out <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Answer"}
			out <- llm.ChatEvent{Type: llm.EventDone}
			return
		}
		for _, e := range s.events {
			if e.delay > 0 {
				select {
				case <-ctx.Done():
					return
				case <-time.After(e.delay):
				}
			}
			ev := llm.ChatEvent{Type: e.typ, Text: e.text, ToolCall: e.toolCall}
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// timedExecutor records when each tool call's Execute starts. Each tool call
// gets a unique key (toolName_{counter}) so concurrent invocations of the same
// tool are tracked separately. Used to assert streaming-kickoff timing.
type timedExecutor struct {
	mu             sync.Mutex
	startTimes     []time.Time
	startNames     []string
	safeNames      map[string]bool // tool names that report concurrencySafe
	blockUntil     chan struct{}   // tools block until this is closed (or nil for no block)
	delayPerCall   time.Duration   // sleep inside Execute (after recording start)
	toolDefs       []llm.ToolDef
	onExecuteStart func() // optional hook fired at the start of every Execute (before blockUntil/delay)
}

func newTimedExecutor() *timedExecutor {
	return &timedExecutor{
		safeNames: map[string]bool{},
	}
}

func (e *timedExecutor) markSafe(name string) { e.safeNames[name] = true }

func (e *timedExecutor) addTool(name string) {
	e.toolDefs = append(e.toolDefs, llm.ToolDef{Name: name})
}

func (e *timedExecutor) starts() []time.Time {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]time.Time, len(e.startTimes))
	copy(out, e.startTimes)
	return out
}

func (e *timedExecutor) callCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.startTimes)
}

func (e *timedExecutor) Execute(ctx context.Context, name string, _ json.RawMessage) (tool.ToolResult, error) {
	e.mu.Lock()
	e.startTimes = append(e.startTimes, time.Now())
	e.startNames = append(e.startNames, name)
	bu := e.blockUntil
	delay := e.delayPerCall
	hook := e.onExecuteStart
	e.mu.Unlock()

	if hook != nil {
		hook()
	}

	if bu != nil {
		select {
		case <-bu:
		case <-ctx.Done():
			return tool.ToolResult{}, ctx.Err()
		}
	}
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return tool.ToolResult{}, ctx.Err()
		}
	}
	return tool.ToolResult{Output: "result for " + name}, nil
}

func (e *timedExecutor) Names() []string {
	out := make([]string, len(e.toolDefs))
	for i, td := range e.toolDefs {
		out[i] = td.Name
	}
	return out
}
func (e *timedExecutor) ToolDefs() []llm.ToolDef { return e.toolDefs }
func (e *timedExecutor) Get(name string) (tool.Tool, bool) {
	return &timedExecutorTool{name: name, safe: e.safeNames[name]}, true
}

type timedExecutorTool struct {
	name string
	safe bool
}

func (t *timedExecutorTool) Name() string                { return t.name }
func (t *timedExecutorTool) Description() string         { return "" }
func (t *timedExecutorTool) Parameters() json.RawMessage { return json.RawMessage(`{}`) }
func (t *timedExecutorTool) Execute(_ context.Context, _ json.RawMessage) (tool.ToolResult, error) {
	return tool.ToolResult{}, nil
}
func (t *timedExecutorTool) IsConcurrencySafe(_ json.RawMessage) bool { return t.safe }

// TestRun_StreamingKickoffOverlapsWithLLMStream asserts that with streaming
// enabled, a safe tool starts BEFORE the LLM stream ends. The LLM emits a
// 200ms delay between EventToolCallDone and the next text delta; if streaming
// is off, the tool would only start after EventDone. With streaming on, the
// tool starts shortly after EventToolCallDone — well before the stream ends.
func TestRun_StreamingKickoffOverlapsWithLLMStream(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe_read")
	exec.addTool("safe_read")
	exec.delayPerCall = 50 * time.Millisecond

	tcID := "tc_a"
	tc := &llm.ToolCall{ID: tcID, Name: "safe_read", Input: json.RawMessage(`{}`)}

	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventTextDelta, text: "thinking..."},
		{typ: llm.EventToolCallStart, toolCall: tc},
		{typ: llm.EventToolCallDone, toolCall: tc},
		{delay: 200 * time.Millisecond, typ: llm.EventTextDelta, text: "more text"},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM: scriptedLLM, Tools: exec,
		Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", MaxTurns: 2,
	}
	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)

	// Stream-end timestamp is when EventDone arrives on the channel. The
	// runtime emits EventDone only after the LLM stream completes (and the
	// post-stream pending-tool resolution finishes). For the overlap check
	// we want the LLM stream-end moment, which lines up with when the post-
	// stream block starts running.
	var streamEnd time.Time
	var sawToolResult bool
	for ev := range events {
		switch ev.Type {
		case EventToolResult:
			sawToolResult = true
		case EventDone, EventAborted:
			streamEnd = time.Now()
		}
	}
	require.True(t, sawToolResult, "expected an EventToolResult for the kicked-off tool")

	starts := exec.starts()
	require.Len(t, starts, 1, "exactly one tool execution expected")
	toolStart := starts[0]
	gap := streamEnd.Sub(toolStart)
	// Tool started shortly after EventToolCallDone; stream ends ~200ms later
	// (after the scripted delay before the final text delta). Gap should
	// comfortably exceed 100ms even with scheduler jitter.
	require.Greater(t, gap, 100*time.Millisecond,
		"tool should have started well before stream end (gap=%v)", gap)
}

// TestRun_StreamingStopsAtFirstUnsafe streams [safe1, safe2, unsafe1, safe3].
// Asserts: safe1 + safe2 ran in parallel mid-stream (start times within a
// small window); unsafe1 and safe3 ran post-stream via the batcher.
func TestRun_StreamingStopsAtFirstUnsafe(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe1")
	exec.markSafe("safe2")
	exec.markSafe("safe3")
	exec.addTool("safe1")
	exec.addTool("safe2")
	exec.addTool("unsafe1")
	exec.addTool("safe3")
	exec.delayPerCall = 30 * time.Millisecond

	mk := func(id, name string) *llm.ToolCall {
		return &llm.ToolCall{ID: id, Name: name, Input: json.RawMessage(`{}`)}
	}
	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallDone, toolCall: mk("tc_safe1", "safe1")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_safe2", "safe2")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_unsafe1", "unsafe1")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_safe3", "safe3")},
		// Hold the stream open briefly so the safe kickoffs definitely overlap.
		{delay: 100 * time.Millisecond, typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM: scriptedLLM, Tools: exec,
		Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", MaxTurns: 2,
	}
	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	for range events {
	}

	// All four tools must have run.
	require.Equal(t, 4, exec.callCount(), "all four tools should have executed")

	// The first two starts are safe1 and safe2 (kicked off in stream order).
	// They must overlap — both started within a window much smaller than
	// delayPerCall (30ms). 20ms is generous for scheduler jitter while still
	// proving overlap (vs. >30ms which would indicate sequential execution).
	exec.mu.Lock()
	starts := append([]time.Time(nil), exec.startTimes...)
	names := append([]string(nil), exec.startNames...)
	exec.mu.Unlock()

	require.Len(t, starts, 4)
	// First two should be the safe kickoffs (order between them is goroutine-
	// scheduling dependent, so we just check both names are in the first pair).
	firstPair := map[string]bool{names[0]: true, names[1]: true}
	require.True(t, firstPair["safe1"] && firstPair["safe2"],
		"first two starts should be safe1 and safe2 (got %v, %v)", names[0], names[1])

	gap := starts[1].Sub(starts[0])
	if gap < 0 {
		gap = -gap
	}
	require.Less(t, gap, 25*time.Millisecond,
		"safe1 and safe2 should have started concurrently (gap=%v, delayPerCall=30ms)", gap)

	// unsafe1 and safe3 ran post-stream. Their starts must be after both
	// safe kickoffs FINISHED (i.e., after starts[0]+delayPerCall).
	postStreamStart := starts[0].Add(exec.delayPerCall)
	require.True(t, starts[2].After(postStreamStart) || starts[2].Equal(postStreamStart),
		"unsafe1/safe3 should run after safe1+safe2 finished")

	// Pin the "unsafe runs alone" Phase B contract: unsafe1 must run BEFORE
	// safe3, and safe3 must wait for unsafe1 to finish (sequential, not
	// parallel). The post-stream batcher partitions [unsafe1, safe3] into
	// two batches: an unsafe singleton followed by a safe singleton.
	require.Equal(t, "unsafe1", names[2], "third tool to start should be unsafe1")
	require.Equal(t, "safe3", names[3], "fourth tool to start should be safe3")

	unsafeStart := starts[2]
	safe3Start := starts[3]
	require.GreaterOrEqual(t, safe3Start.Sub(unsafeStart), exec.delayPerCall-5*time.Millisecond,
		"safe3 should start at least delayPerCall after unsafe1 (sequential), got gap=%v",
		safe3Start.Sub(unsafeStart))
}

// TestRun_StreamingDisabledMatchesNonStreaming: same script as test 1 with
// HARNESS_STREAMING_TOOLS unset. Tool must start AFTER the LLM stream ends.
// Validates the default-off contract.
func TestRun_StreamingDisabledMatchesNonStreaming(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "")

	exec := newTimedExecutor()
	exec.markSafe("safe_read")
	exec.addTool("safe_read")
	exec.delayPerCall = 10 * time.Millisecond

	tc := &llm.ToolCall{ID: "tc_a", Name: "safe_read", Input: json.RawMessage(`{}`)}

	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallStart, toolCall: tc},
		{typ: llm.EventToolCallDone, toolCall: tc},
		{delay: 100 * time.Millisecond, typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM: scriptedLLM, Tools: exec,
		Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", MaxTurns: 2,
	}
	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)

	// We track when the LLM stream-end happened by observing when the first
	// EventToolResult lands. With streaming OFF, no tool result can land
	// until the post-stream block runs, which is strictly AFTER the stream
	// loop terminates. We use a sentinel: capture the time of the LAST text
	// delta + the time of the first tool result. Since the script has a
	// 100ms delay before EventDone (no text after toolCallDone), the tool
	// must START at least near that 100ms mark.
	scriptStart := time.Now()
	var firstToolResult time.Time
	for ev := range events {
		if ev.Type == EventToolResult && firstToolResult.IsZero() {
			firstToolResult = time.Now()
		}
	}
	require.False(t, firstToolResult.IsZero(), "expected an EventToolResult")

	starts := exec.starts()
	require.Len(t, starts, 1)
	// With streaming OFF the tool only runs after the stream completes, which
	// requires waiting through the 100ms delay before EventDone.
	elapsed := starts[0].Sub(scriptStart)
	require.Greater(t, elapsed, 80*time.Millisecond,
		"tool must start AFTER stream end with streaming disabled (elapsed=%v)", elapsed)
}

// TestRun_StreamingAbortMidKickoffPairsAllEntries: kick off 3 safe tools mid-
// stream, then cancel ctx. Asserts: session has 3 paired tool_call/tool_result
// entries; exactly one EventAborted; no goroutine leaks.
func TestRun_StreamingAbortMidKickoffPairsAllEntries(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe_read")
	exec.addTool("safe_read")
	exec.blockUntil = make(chan struct{}) // tools block until ctx cancel

	// Deterministic in-flight synchronization: cancel only after all 3
	// kickoffs have entered Execute. Avoids the timing flake of a fixed
	// sleep that can lose a race on slow CI.
	var inFlight sync.WaitGroup
	inFlight.Add(3)
	exec.onExecuteStart = func() { inFlight.Done() }

	mk := func(id string) *llm.ToolCall {
		return &llm.ToolCall{ID: id, Name: "safe_read", Input: json.RawMessage(`{}`)}
	}
	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallDone, toolCall: mk("tc_0")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_1")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_2")},
		{typ: llm.EventDone},
	}}

	ctx, cancel := context.WithCancel(context.Background())
	rt := &Runtime{
		LLM: scriptedLLM, Tools: exec,
		Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", MaxTurns: 2,
	}

	beforeGoroutines := runtime.NumGoroutine()

	events, err := rt.Run(ctx, "go", nil)
	require.NoError(t, err)

	// Cancel only once all 3 kickoffs have actually entered Execute.
	go func() {
		inFlight.Wait()
		cancel()
	}()

	var abortedEvents int
	for ev := range events {
		if ev.Type == EventAborted {
			abortedEvents++
		}
	}
	require.Equal(t, 1, abortedEvents, "exactly one EventAborted expected")

	// Session must contain 3 tool_call/tool_result pairs (paired by ID).
	entries := rt.Session.Entries()
	callIDs := map[string]bool{}
	resultIDs := map[string]bool{}
	for _, e := range entries {
		switch e.Type {
		case session.EntryTypeToolCall:
			var cd session.ToolCallData
			require.NoError(t, json.Unmarshal(e.Data, &cd))
			callIDs[cd.ID] = true
		case session.EntryTypeToolResult:
			var trd session.ToolResultData
			require.NoError(t, json.Unmarshal(e.Data, &trd))
			resultIDs[trd.ToolCallID] = true
		}
	}
	require.Equal(t, 3, len(callIDs), "expected 3 ToolCall entries")
	require.Equal(t, 3, len(resultIDs), "expected 3 ToolResult entries")
	for id := range callIDs {
		require.True(t, resultIDs[id], "call %s has no matching result", id)
	}

	// Goroutine leak check: give kickoff goroutines a grace period to finish.
	time.Sleep(100 * time.Millisecond)
	afterGoroutines := runtime.NumGoroutine()
	// Allow some tolerance — runtime/test scheduler can keep transient
	// goroutines alive briefly. A leak would show up as 3+ extras.
	require.LessOrEqual(t, afterGoroutines, beforeGoroutines+2,
		"suspected goroutine leak: before=%d after=%d", beforeGoroutines, afterGoroutines)
}

// TestRun_StreamingResultEmittedBeforeStreamEnds asserts that with streaming
// on, a tool's EventToolResult lands on the events channel BEFORE the LLM's
// EventDone arrives at the runtime — i.e., the result is live-emitted mid-
// stream rather than buffered to post-stream.
func TestRun_StreamingResultEmittedBeforeStreamEnds(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe_read")
	exec.addTool("safe_read")
	exec.delayPerCall = 10 * time.Millisecond

	tc := &llm.ToolCall{ID: "tc_a", Name: "safe_read", Input: json.RawMessage(`{}`)}

	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallStart, toolCall: tc},
		{typ: llm.EventToolCallDone, toolCall: tc},
		// LLM stream stays open for 200ms after toolCallDone — tool should
		// have completed mid-stream and emitted its result well before the
		// stream end propagates through the post-stream block.
		{delay: 200 * time.Millisecond, typ: llm.EventTextDelta, text: "tail text"},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM: scriptedLLM, Tools: exec,
		Session: session.NewSession("a", "k"),
		AgentID: "a", Model: "test", MaxTurns: 2,
	}
	events, err := rt.Run(context.Background(), "go", nil)
	require.NoError(t, err)

	// Track relative ordering of EventToolResult vs the tail EventTextDelta.
	// With streaming, EventToolResult must precede the tail "tail text"
	// delta (which is gated behind a 200ms delay). Without streaming, the
	// delta would arrive first.
	var sawResult, sawTail bool
	var resultBeforeTail bool
	for ev := range events {
		switch ev.Type {
		case EventToolResult:
			sawResult = true
			if !sawTail {
				resultBeforeTail = true
			}
		case EventTextDelta:
			if ev.Text == "tail text" {
				sawTail = true
			}
		}
	}
	require.True(t, sawResult, "expected EventToolResult")
	require.True(t, sawTail, "expected tail EventTextDelta")
	require.True(t, resultBeforeTail,
		"EventToolResult must land BEFORE the tail text delta (streaming kickoff)")
}

// TestRun_SubagentStreamingForwardsToParent asserts that with streaming on, a
// subagent's mid-stream EventToolResult forwards to the parent's events
// channel with AgentID set to the subagent's ID — proving the streaming
// kickoff inside a subagent runtime cooperates with Phase C's event forwarding.
func TestRun_SubagentStreamingForwardsToParent(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	cfg := newTwoAgentCfg(t)

	subExec := newTimedExecutor()
	subExec.markSafe("safe_read")
	subExec.addTool("safe_read")
	subExec.delayPerCall = 10 * time.Millisecond

	tc := &llm.ToolCall{ID: "tc_sub", Name: "safe_read", Input: json.RawMessage(`{}`)}
	subLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventTextDelta, text: "sub thinking"},
		{typ: llm.EventToolCallStart, toolCall: tc},
		{typ: llm.EventToolCallDone, toolCall: tc},
		{delay: 100 * time.Millisecond, typ: llm.EventTextDelta, text: "sub done"},
		{typ: llm.EventDone},
	}}

	parentLLM := &taskCallingLLM{
		subagentID: "researcher",
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

	// Custom subagent builder that gives the subagent the timedExecutor
	// instead of the default noopExecutor.
	subBuilder := func(spec AgentSpec) (RuntimeInputs, error) {
		return RuntimeInputs{
			Provider:     subLLM,
			Tools:        subExec,
			Session:      NewSubagentSession(spec.ID),
			Compaction:   nil,
			IngestSource: "",
		}, nil
	}
	factory := MakeSubagentFactory(cfg.Resolve, deps, subBuilder, parent)
	parent.Tools = &taskOnlyExecutor{
		task: tool.NewTaskTool(factory, parent.Depth, cfg.EligibleSubagents()),
	}

	events, err := parent.Run(context.Background(), "go", nil)
	require.NoError(t, err)

	var subToolResultEvent *AgentEvent
	timeout := time.After(5 * time.Second)
loop:
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				break loop
			}
			if ev.Type == EventToolResult && ev.AgentID == "researcher" {
				e := ev
				subToolResultEvent = &e
			}
		case <-timeout:
			t.Fatal("timed out waiting for parent events")
		}
	}

	require.NotNil(t, subToolResultEvent,
		"expected an EventToolResult forwarded from subagent with AgentID=researcher")
	require.NotNil(t, subToolResultEvent.ToolCall)
	require.Equal(t, "safe_read", subToolResultEvent.ToolCall.Name)
	// Sanity: subagent ran the tool exactly once.
	require.Equal(t, 1, subExec.callCount())
}

// TestRun_StreamingPreservesSessionOrderForSlowTool ensures session entries
// land in [UserMessage, AssistantMessage, ToolCall, ToolResult] order even
// when the tool takes longer than the stream. Regression for the bug where
// kickoff goroutines wrote ToolCall mid-stream (before the assistant text
// was saved), causing assembleMessages to produce a synthetic "(interrupted)"
// tool_result and break the Anthropic API on the next turn.
//
// We deterministically expose the race by:
//  1. blocking the tool inside Execute (so the kickoff goroutine has
//     already appended the ToolCall synchronously inside dispatchTool
//     before Execute is invoked),
//  2. then letting the LLM stream emit its tail assistant-text + EventDone
//     so the main goroutine appends the AssistantMessage,
//  3. then releasing the tool so the kickoff appends the ToolResult.
//
// On the unfixed code, sess.Entries() ends up [User, ToolCall, Assistant,
// ToolResult] which breaks the Anthropic invariant. After the fix, all
// session writes are deferred to the main goroutine and the order is
// [User, Assistant, ToolCall, ToolResult].
func TestRun_StreamingPreservesSessionOrderForSlowTool(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe_search")
	exec.addTool("safe_search")
	// Block inside Execute so the kickoff goroutine has reached the point
	// where (under the bug) it has already appended the ToolCall entry, and
	// hold it there until after the stream emits its assistant text.
	exec.blockUntil = make(chan struct{})

	var inFlight sync.WaitGroup
	inFlight.Add(1)
	exec.onExecuteStart = func() { inFlight.Done() }

	tcID := "tc_slow"
	tc := &llm.ToolCall{ID: tcID, Name: "safe_search", Input: json.RawMessage(`{}`)}

	sess := session.NewSession("a", "k")
	rt := &Runtime{
		LLM: &scriptedStreamLLM{events: []scriptedStreamEvent{
			{typ: llm.EventTextDelta, text: "I'll search now."},
			{typ: llm.EventToolCallStart, toolCall: tc},
			{typ: llm.EventToolCallDone, toolCall: tc},
			// Delay the tail text + EventDone so the kickoff goroutine
			// has time to enter Execute (and, under the bug, append its
			// ToolCall) before the main goroutine processes EventDone
			// and appends the AssistantMessage.
			{delay: 50 * time.Millisecond, typ: llm.EventTextDelta, text: " done."},
			{typ: llm.EventDone},
		}},
		Tools:    exec,
		Session:  sess,
		AgentID:  "a",
		Model:    "test",
		MaxTurns: 1,
	}

	// Release the tool blocker only after the main goroutine has had time
	// to process the tail text + EventDone and append the AssistantMessage.
	// This deterministically pins ToolCall+ToolResult after AssistantMessage
	// under the fix; under the bug, the ToolCall lands BEFORE the assistant
	// text (because dispatchTool appends it synchronously before Execute).
	go func() {
		inFlight.Wait()
		time.Sleep(150 * time.Millisecond) // > 50ms tail delay; main loop will save AssistantMessage
		close(exec.blockUntil)
	}()

	events, err := rt.Run(context.Background(), "search please", nil)
	require.NoError(t, err)
	for range events {
	}

	entries := sess.Entries()
	// Expected: UserMessage, AssistantMessage, ToolCall, ToolResult.
	require.GreaterOrEqual(t, len(entries), 4, "expected at least 4 entries, got %d", len(entries))
	require.Equal(t, session.EntryTypeMessage, entries[0].Type)
	require.Equal(t, "user", entries[0].Role)
	// entries[1] MUST be the AssistantMessage, NOT a ToolCall.
	require.Equal(t, session.EntryTypeMessage, entries[1].Type,
		"AssistantMessage must come before ToolCall (bug regression check); got entry type %v at index 1",
		entries[1].Type)
	require.Equal(t, "assistant", entries[1].Role)
	require.Equal(t, session.EntryTypeToolCall, entries[2].Type,
		"ToolCall must come after AssistantMessage; got entry type %v at index 2", entries[2].Type)
	require.Equal(t, session.EntryTypeToolResult, entries[3].Type,
		"ToolResult must come after ToolCall; got entry type %v at index 3", entries[3].Type)
}

// TestRun_StreamingCortexAppendIsRaceClean is the C1 regression test.
//
// Phase D introduced kickoff goroutines that call dispatchTool mid-stream.
// dispatchTool appends to the per-Run cortex thread slice under r.cortexMu.
// The main goroutine, after the LLM stream loop ends, also appends the
// assistant text to the same slice. Without locking that append, two
// writers race on the same slice header.
//
// This test exercises the race window deterministically: 3 safe tools are
// kicked off mid-stream and made to block inside Execute via blockUntil;
// the LLM emits assistant text before EventDone; the main goroutine then
// runs the assistant-text append while all 3 kickoff goroutines are still
// inside dispatchTool. With -race, the unfixed code flags a data race;
// the fix (locking the append under r.cortexMu) clears it.
//
// Cortex setup: a zero-valued *cortex.Cortex is enough — the runtime only
// uses it as a non-nil sentinel for the cortex code paths in this test.
// Recall is skipped (userMsg "ok" is a trivial phrase per ShouldRecall),
// and ingest is skipped (IngestSource="cron" disables ShouldIngest gate).
// So no methods are called on the zero-Cortex pointer.
func TestRun_StreamingCortexAppendIsRaceClean(t *testing.T) {
	t.Setenv("HARNESS_STREAMING_TOOLS", "1")

	exec := newTimedExecutor()
	exec.markSafe("safe_read")
	exec.addTool("safe_read")

	// Tools block until released. We release AFTER the assistant-text append
	// has had a chance to fire concurrently with the kickoffs being inside
	// dispatchTool's locked region.
	exec.blockUntil = make(chan struct{})

	// Synchronize on all 3 kickoffs being inside Execute (i.e. past the
	// dispatchTool locked append) before the LLM stream emits its closing
	// text + EventDone. This guarantees the race window is open when the
	// main goroutine performs its assistant-text append.
	var inFlight sync.WaitGroup
	inFlight.Add(3)
	exec.onExecuteStart = func() { inFlight.Done() }

	mk := func(id string) *llm.ToolCall {
		return &llm.ToolCall{ID: id, Name: "safe_read", Input: json.RawMessage(`{}`)}
	}
	scriptedLLM := &scriptedStreamLLM{events: []scriptedStreamEvent{
		{typ: llm.EventToolCallDone, toolCall: mk("tc_0")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_1")},
		{typ: llm.EventToolCallDone, toolCall: mk("tc_2")},
		// Hold the stream open briefly so the kickoffs all enter Execute
		// (and are blocked inside it) before assistant text + EventDone.
		{delay: 50 * time.Millisecond, typ: llm.EventTextDelta, text: "all done"},
		{typ: llm.EventDone},
	}}

	rt := &Runtime{
		LLM:          scriptedLLM,
		Tools:        exec,
		Session:      session.NewSession("a", "k"),
		AgentID:      "a",
		Model:        "test",
		MaxTurns:     2,
		KG:           neverRecallKG{}, // non-nil sentinel; never invoked
		IngestSource: "cron",          // disables deferred IngestThreadAsync
	}

	// Trivial userMsg "ok" → ShouldRecall returns false → no Recall call on
	// the zero Cortex. Combined with IngestSource="cron" (disables ingest),
	// the zero Cortex is safe to leave dangling for the duration of the run.
	events, err := rt.Run(context.Background(), "ok", nil)
	require.NoError(t, err)

	// Release tool blockers only after all 3 kickoffs have entered Execute
	// AND given the main goroutine time to process the LLM's tail text +
	// run the assistant-text append concurrently with the in-flight tool
	// goroutines (which are still holding pointers to `thread`).
	go func() {
		inFlight.Wait()
		// Small grace so the main goroutine reaches its post-stream assistant
		// append while kickoffs are still parked inside dispatchTool's
		// post-Execute locked region (or about to enter it after release).
		time.Sleep(20 * time.Millisecond)
		close(exec.blockUntil)
	}()

	// Drain.
	for range events {
	}

	require.Equal(t, 3, exec.callCount(), "all 3 kickoffs should have executed")
}
