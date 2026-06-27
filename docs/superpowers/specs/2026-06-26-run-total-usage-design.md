# Design: Run-Total Token Usage on Terminal EventDone

**Date:** 2026-06-26
**Project:** harness (`github.com/sausheong/harness`)
**Status:** Approved (design)

## Motivation

`Runtime.Run` executes a multi-turn agent loop. Each turn captures the provider's
per-turn token usage into a local `lastUsage`, but the terminal
`AgentEvent{Type: EventDone}` is emitted once carrying only the **last turn's**
usage (`runtime/runtime.go:716`). For a multi-turn run, all prior turns' tokens
are silently dropped.

Downstream consumers that meter spend from `AgentEvent.Usage` (e.g. sidecar's
daily token budget cap) therefore undercount multi-turn runs — the cap reads one
turn instead of the whole run. The field's doc comment ("populated for EventDone
when the provider reported it") implies a complete figure; today it is partial.

## Goal

Make the terminal `EventDone.Usage` carry the **accumulated total across all
turns of the run**, summing all four `llm.Usage` fields (InputTokens,
OutputTokens, CacheCreationInputTokens, CacheReadInputTokens).

## Non-goals

- No new events and no new fields: `EventDone` stays the single terminal event
  (the subagent layer keys `Done` on it — `runtime/subagent.go:134` — so adding
  per-turn `EventDone`s would break that contract). This is a value-correctness
  fix to the existing field, not a protocol change.
- No per-turn usage telemetry surface. (Out of scope; `RunTurn` already exposes
  per-turn usage via `TurnResult.Usage` for callers that want it.)

## Scope: `Run` only, not `RunTurn`

`RunTurn` executes **exactly one turn per call** by contract; its caller drives
the loop and accumulates `TurnResult.Usage` itself. Its `EventDone.Usage` (one
turn) is therefore already correct. **Do not change `RunTurn`.** The bug is
solely in `Run`'s internal `for turn := 0; turn < maxTurns; turn++` loop.

## Design

In `Runtime.Run` (`runtime/runtime.go`), introduce a run-scoped accumulator
declared **outside** the turn loop (the loop starts at line 397):

```go
var runUsage *llm.Usage // accumulated across all turns; nil until first reported
```

Inside the turn loop, at the point where the provider's `llm.EventDone` is
handled and `lastUsage = event.Usage` is set (currently ~line 589–591), fold the
turn's usage into the accumulator:

```go
case llm.EventDone:
    refused = event.StopReason == llm.StopReasonRefusal
    refusalCategory = event.StopCategory
    if event.Usage != nil {
        lastUsage = event.Usage
        runUsage = addUsage(runUsage, event.Usage)
    }
    // ... existing calibrator block unchanged (it still uses event.Usage,
    //     the per-turn figure — calibration is per-request, not per-run) ...
```

The terminal completed emission (line 716) changes from `lastUsage` to the
accumulator:

```go
stopReason = "completed"
r.emit(AgentEvent{Type: EventDone, Usage: runUsage})
```

### Helper

Add an unexported helper (in `runtime/runtime.go` or a small `usage.go` in
package runtime):

```go
// addUsage returns the field-wise sum of two usage snapshots. A nil operand is
// treated as zero. Returns a new pointer; never mutates its arguments. Returns
// nil only when both are nil (so EventDone.Usage stays nil when the provider
// never reported usage — preserving today's behavior for usage-less providers).
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

### Why a non-mutating helper

`event.Usage` points into the provider's stream event; mutating it could corrupt
provider-side state or other observers. `addUsage` always returns a fresh value.

## Behavior preservation

- **Usage-less providers** (never send `llm.EventDone.Usage`): `runUsage` stays
  nil → terminal `EventDone.Usage` is nil, exactly as today.
- **Single-turn runs:** `runUsage` == that turn's usage → identical to today.
- **Calibrator:** unchanged — it consumes the per-turn `event.Usage`
  (per-request calibration), not the run total.
- **`lastUsage`:** retained; still used by the non-streaming/other terminal
  paths if any reference it. (Verify during implementation that every terminal
  `EventDone` emission in `Run` uses `runUsage`; if there is more than one, all
  must switch. The known one is line 716.)

## Error handling

No new failure modes. `addUsage` is pure and total. nil-safe on both operands.

## Testing

Mirror the existing scripted-provider pattern in `runtime/agent_test.go` (a fake
`llm.LLMProvider` whose `ChatStream` emits canned events including
`llm.EventDone` with a `*llm.Usage`).

| Test | Setup | Assert |
|------|-------|--------|
| `TestRun_UsageAccumulatesAcrossTurns` | scripted provider: turn 1 emits a tool call + `EventDone{Usage: 100/40/…}`, turn 2 emits text + `EventDone{Usage: 10/5/…}` and stops | terminal `AgentEvent{EventDone}.Usage` == field-wise sum (110 in / 45 out / summed cache) |
| `TestRun_UsageSingleTurn` | one turn, `EventDone{Usage: 7/3}` | terminal usage == 7/3 (no regression) |
| `TestRun_UsageNilWhenProviderSilent` | provider never sets `EventDone.Usage` | terminal `EventDone.Usage` == nil |
| `TestAddUsage` (pure) | nil+nil, nil+x, x+nil, x+y across all four fields | correct sums; inputs not mutated |

Gate: `go test ./runtime/...`, `go vet ./...`, `go build ./...` clean. Run the
broader `go test ./...` to confirm no other runtime test asserted the old
last-turn-only behavior.

## Downstream note (sidecar)

This fixes the root cause of sidecar's documented budget undercount for the
**coding** runtime (multi-turn) and the **evaluator** (up to 8 turns). After
harness ships this, sidecar's CLAUDE.md / sidecar.yaml caveat about multi-turn
undercount can be softened to note only provider-reported-usage caveats. (Triage
is single-turn and already exact.) That doc update is a separate sidecar change,
not part of this harness spec.

## Files

- Modified: `runtime/runtime.go` (accumulator + helper + terminal emission).
- New test cases: `runtime/runtime_usage_test.go` (or appended to
  `runtime/agent_test.go`).
- New: this design doc.
