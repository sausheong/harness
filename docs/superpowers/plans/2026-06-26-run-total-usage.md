# Run-Total Token Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Runtime.Run`'s terminal `EventDone.Usage` carry the token total accumulated across all turns, instead of only the last turn.

**Architecture:** Add a pure, non-mutating `addUsage` helper that field-wise sums two `*llm.Usage` snapshots; declare a run-scoped accumulator in `Run` outside the turn loop; fold each turn's provider usage into it; emit the accumulator (not `lastUsage`) on the single terminal `EventDone`. `RunTurn` is unchanged (one turn per call, already correct).

**Tech Stack:** Go 1.25, harness `runtime` + `llm` packages, `llmtest.Base`, testify.

## Global Constraints

- Module `github.com/sausheong/harness`. Work on branch `run-total-usage` — do NOT switch branches.
- `EventDone` stays the SINGLE terminal event (subagent layer keys `Done` on it). No new events, no new fields.
- Behavior preserved: usage-less providers → terminal `EventDone.Usage` stays nil; single-turn runs → identical to today; the calibrator block keeps using the per-turn `event.Usage`, NOT the accumulator.
- Only `runtime/runtime.go`'s `Run` changes. Do NOT modify `RunTurn` (`runtime/runturn.go`).
- Sum ALL FOUR `llm.Usage` fields: `InputTokens`, `OutputTokens`, `CacheCreationInputTokens`, `CacheReadInputTokens`.
- Gates per task: `go build ./...`, `go vet ./...` clean; named tests pass.

---

### Task 1: addUsage helper (pure)

**Files:**
- Modify: `runtime/runtime.go` (add the helper; or create `runtime/usage.go` — implementer's choice, but keep it in package `runtime`)
- Test: `runtime/usage_test.go`

**Interfaces:**
- Consumes: `llm.Usage` (fields `InputTokens`, `OutputTokens`, `CacheCreationInputTokens`, `CacheReadInputTokens`, all `int`).
- Produces: `func addUsage(acc, turn *llm.Usage) *llm.Usage` — field-wise sum; nil treated as zero; returns a fresh pointer; returns nil only when both args are nil; never mutates either argument.

- [ ] **Step 1: Write the failing test**

Create `runtime/usage_test.go`:

```go
package runtime

import (
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAddUsage_BothNil(t *testing.T) {
	assert.Nil(t, addUsage(nil, nil))
}

func TestAddUsage_AccNil(t *testing.T) {
	turn := &llm.Usage{InputTokens: 7, OutputTokens: 3, CacheCreationInputTokens: 2, CacheReadInputTokens: 1}
	got := addUsage(nil, turn)
	require.NotNil(t, got)
	assert.Equal(t, llm.Usage{InputTokens: 7, OutputTokens: 3, CacheCreationInputTokens: 2, CacheReadInputTokens: 1}, *got)
	// input not mutated
	assert.Equal(t, 7, turn.InputTokens)
}

func TestAddUsage_TurnNil(t *testing.T) {
	acc := &llm.Usage{InputTokens: 5}
	got := addUsage(acc, nil)
	require.NotNil(t, got)
	assert.Equal(t, 5, got.InputTokens)
}

func TestAddUsage_SumsAllFourFields(t *testing.T) {
	acc := &llm.Usage{InputTokens: 100, OutputTokens: 40, CacheCreationInputTokens: 10, CacheReadInputTokens: 6}
	turn := &llm.Usage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 1, CacheReadInputTokens: 2}
	got := addUsage(acc, turn)
	require.NotNil(t, got)
	assert.Equal(t, 110, got.InputTokens)
	assert.Equal(t, 45, got.OutputTokens)
	assert.Equal(t, 11, got.CacheCreationInputTokens)
	assert.Equal(t, 8, got.CacheReadInputTokens)
	// neither input mutated
	assert.Equal(t, 100, acc.InputTokens)
	assert.Equal(t, 10, turn.InputTokens)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./runtime/ -run TestAddUsage -v`
Expected: FAIL — `addUsage` undefined.

- [ ] **Step 3: Implement the helper**

Add to `runtime/runtime.go` (near other unexported helpers) or a new `runtime/usage.go`:

```go
// addUsage returns the field-wise sum of two usage snapshots. A nil operand is
// treated as zero. It returns a fresh pointer and never mutates its arguments,
// so callers may safely pass a pointer that aliases a provider stream event.
// Returns nil only when BOTH operands are nil, preserving the contract that
// EventDone.Usage stays nil for providers that never report usage.
func addUsage(acc, turn *llm.Usage) *llm.Usage {
	if acc == nil && turn == nil {
		return nil
	}
	var out llm.Usage
	if acc != nil {
		out = *acc
	}
	if turn != nil {
		out.InputTokens += turn.InputTokens
		out.OutputTokens += turn.OutputTokens
		out.CacheCreationInputTokens += turn.CacheCreationInputTokens
		out.CacheReadInputTokens += turn.CacheReadInputTokens
	}
	return &out
}
```

If creating `runtime/usage.go`, start it with `package runtime` and `import "github.com/sausheong/harness/llm"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./runtime/ -run TestAddUsage -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/usage.go runtime/usage_test.go runtime/runtime.go
git commit -m "feat(runtime): add pure addUsage helper for field-wise usage summation"
```

(Adjust `git add` to whichever files you actually touched.)

---

### Task 2: Accumulate run-total usage in Run

**Files:**
- Modify: `runtime/runtime.go` (the `Run` method — accumulator decl, the `llm.EventDone` case ~line 586-591, the terminal emit ~line 716)
- Test: `runtime/run_usage_test.go`

**Interfaces:**
- Consumes: `addUsage` (Task 1); harness `llmtest.Base`, `llm.ChatEvent`, `llm.EventDone`, `llm.Usage`; `runtime.AgentEvent`, `runtime.EventDone`.
- Produces: corrected terminal `EventDone.Usage` (run total). No new exported symbols.

- [ ] **Step 1: Write the failing test**

Create `runtime/run_usage_test.go`. This uses a scripted provider that emits a tool call on turn 1 (forcing a second turn) with usage, then text + usage on turn 2, mirroring the `threeToolCallLLM` pattern already in `agent_test.go`.

```go
package runtime

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// twoTurnUsageLLM: turn 1 emits a noop tool call + EventDone{Usage:u1};
// turn 2 (after the tool result) emits text + EventDone{Usage:u2} and stops.
type twoTurnUsageLLM struct {
	llmtest.Base
	u1, u2 *llm.Usage
	calls  int
}

func (f *twoTurnUsageLLM) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 8)
	first := f.calls == 0
	f.calls++
	go func() {
		defer close(ch)
		if first {
			tc := llm.ToolCall{ID: "tc_0", Name: "noop", Input: json.RawMessage(`{}`)}
			ch <- llm.ChatEvent{Type: llm.EventToolCallStart, ToolCall: &tc}
			ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &tc}
			ch <- llm.ChatEvent{Type: llm.EventDone, Usage: f.u1}
			return
		}
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "done"}
		ch <- llm.ChatEvent{Type: llm.EventDone, Usage: f.u2}
	}()
	return ch, nil
}

// noopExecutor returns "ok" for a "noop" tool.
type noopExecutor struct{}

func (noopExecutor) Execute(_ context.Context, _ string, _ json.RawMessage) (tool.ToolResult, error) {
	return tool.ToolResult{Output: "ok"}, nil
}
func (noopExecutor) ToolDefs() []llm.ToolDef      { return []llm.ToolDef{{Name: "noop"}} }
func (noopExecutor) Names() []string              { return []string{"noop"} }
func (noopExecutor) Get(string) (tool.Tool, bool) { return nil, false }

func drainUsage(t *testing.T, events <-chan AgentEvent) *llm.Usage {
	t.Helper()
	var done *llm.Usage
	var sawDone bool
	for ev := range events {
		if ev.Type == EventDone {
			done = ev.Usage
			sawDone = true
		}
	}
	require.True(t, sawDone, "expected a terminal EventDone")
	return done
}

func TestRun_UsageAccumulatesAcrossTurns(t *testing.T) {
	r := &Runtime{
		LLM: &twoTurnUsageLLM{
			u1: &llm.Usage{InputTokens: 100, OutputTokens: 40, CacheCreationInputTokens: 10, CacheReadInputTokens: 6},
			u2: &llm.Usage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 1, CacheReadInputTokens: 2},
		},
		Tools:    noopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	require.NotNil(t, got)
	assert.Equal(t, 110, got.InputTokens)
	assert.Equal(t, 45, got.OutputTokens)
	assert.Equal(t, 11, got.CacheCreationInputTokens)
	assert.Equal(t, 8, got.CacheReadInputTokens)
}

func TestRun_UsageSingleTurn(t *testing.T) {
	r := &Runtime{
		LLM: &twoTurnUsageLLM{
			u2: &llm.Usage{InputTokens: 7, OutputTokens: 3},
		},
		Tools:    noopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	// calls=0 path emits a tool call; to get a pure single turn, mark calls so
	// the first ChatStream takes the text+stop branch.
	r.LLM.(*twoTurnUsageLLM).calls = 1
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	require.NotNil(t, got)
	assert.Equal(t, 7, got.InputTokens)
	assert.Equal(t, 3, got.OutputTokens)
}

func TestRun_UsageNilWhenProviderSilent(t *testing.T) {
	r := &Runtime{
		LLM:      &twoTurnUsageLLM{calls: 1}, // single turn, u2 nil → EventDone with nil usage
		Tools:    noopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	assert.Nil(t, got, "usage must stay nil when the provider never reports it")
}
```

Note: confirm the exact `tool.Executor` interface method set by reading `tool/registry.go` or an existing fake in `agent_test.go` (e.g. `cancelOnNthExecutor`). Match its method signatures exactly — `Execute`, `ToolDefs`, `Names`, `Get` are what the existing fake implements; adjust if the interface differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./runtime/ -run TestRun_Usage -v`
Expected: FAIL — `TestRun_UsageAccumulatesAcrossTurns` gets 10/5 (last turn only), not 110/45, because Run still emits `lastUsage`.

- [ ] **Step 3: Implement the accumulation**

In `runtime/runtime.go`, in the `Run` method:

(a) Declare the accumulator just BEFORE the turn loop `for turn := 0; turn < maxTurns; turn++ {` (~line 397):

```go
		var runUsage *llm.Usage // accumulated across all turns; nil until first reported
```

(b) In the `case llm.EventDone:` block (~line 586-591), after `lastUsage = event.Usage`, fold into the accumulator:

```go
				case llm.EventDone:
					refused = event.StopReason == llm.StopReasonRefusal
					refusalCategory = event.StopCategory
					if event.Usage != nil {
						lastUsage = event.Usage
						runUsage = addUsage(runUsage, event.Usage)
					}
					// (existing calibrator block below stays exactly as-is —
					//  it uses event.Usage, the per-turn figure, not runUsage)
```

(c) At the terminal completed emit (~line 716), change `lastUsage` to `runUsage`:

```go
				stopReason = "completed"
				r.emit(AgentEvent{Type: EventDone, Usage: runUsage})
```

Do NOT change the calibrator block, `lastUsage`'s other uses, or any other emission. Confirm via `grep -n "EventDone" runtime/runtime.go` that line 716 is the only `AgentEvent{Type: EventDone}` emit in `Run` (it is, per the spec); if there are others on terminal paths, switch them to `runUsage` too and note it in the report.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./runtime/ -run 'TestRun_Usage|TestAddUsage' -v`
Expected: PASS (accumulate=110/45/11/8; single-turn=7/3; nil-when-silent=nil).

- [ ] **Step 5: Full runtime suite (no regression)**

Run: `go test ./runtime/... && go vet ./... && go build ./...`
Expected: all PASS / clean. If any pre-existing test asserted the old last-turn-only usage, investigate and report before "fixing" it — that would be a behavior the change intentionally alters; flag it rather than silently editing.

- [ ] **Step 6: Commit**

```bash
git add runtime/runtime.go runtime/run_usage_test.go
git commit -m "fix(runtime): emit run-total token usage on terminal EventDone

Run accumulated only the last turn's usage into the terminal EventDone, so
multi-turn runs under-reported tokens to consumers metering AgentEvent.Usage.
Sum all four usage fields across turns; RunTurn is unchanged (one turn/call)."
```

---

### Task 3: Document the field semantics

**Files:**
- Modify: `runtime/runtime.go` (the `AgentEvent.Usage` field doc comment, line ~48)
- Modify: `developer.md` (the usage example region, ~line 471 — only if it characterizes the figure; otherwise leave)

**Interfaces:**
- Consumes: nothing.
- Produces: accurate doc comment.

- [ ] **Step 1: Update the field doc comment**

In `runtime/runtime.go`, change the `AgentEvent.Usage` comment (~line 48) from:

```go
	Usage      *llm.Usage         // populated for EventDone when the provider reported it
```

to:

```go
	Usage      *llm.Usage         // on the terminal EventDone, the token total accumulated across all turns of the run (nil if the provider never reported usage); per-turn figures are available via RunTurn's TurnResult.Usage
```

- [ ] **Step 2: Check developer.md**

Read `developer.md` around line 471. If the surrounding prose describes `ev.Usage` as per-call/per-turn in a way the change now contradicts, update it to say the terminal `EventDone` carries the run total. If it only prints the numbers without characterizing them, leave it unchanged.

- [ ] **Step 3: Verify build (docs/comment only)**

Run: `go build ./... && go vet ./...`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add runtime/runtime.go developer.md
git commit -m "docs(runtime): clarify EventDone.Usage is the run total"
```

(Drop `developer.md` from the add if you didn't change it.)

---

## Self-Review Notes

- **Spec coverage:** addUsage helper (T1), accumulator + terminal emit + preserved calibrator/nil/single-turn behavior (T2), field doc (T3). All spec items mapped.
- **Scope guard:** plan explicitly forbids touching `RunTurn` and the calibrator block; T2 step 3 calls this out.
- **Type consistency:** `addUsage(acc, turn *llm.Usage) *llm.Usage` used identically in T1 and T2; the four `llm.Usage` field names match across tests and helper.
- **Behavior preservation** is directly tested: single-turn (T2 test 2) and nil-when-silent (T2 test 3), not just asserted in prose.
- **Test-fixture risk flagged:** T2 step 1 tells the implementer to verify the `tool.Executor` interface method set against the real interface / existing fakes rather than trust the snippet.
