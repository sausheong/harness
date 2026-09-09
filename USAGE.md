# Usage accounting

`llm.Usage.InputTokens` is total input, including cached input. Cache-read and
cache-creation counters are subsets used for pricing and analysis. Do not add
them to InputTokens when calculating context occupancy or total input. The
Anthropic adapter normalizes its separate wire counters to this contract;
OpenAI-style prompt-token totals already include all input.

This changes Anthropic InputTokens from its old uncached-only value. Consumers
upgrading from v0.3.9 must remove their extra addition of cache counters. For
example, 10 uncached + 42 cache creation + 17 cache read becomes InputTokens=69,
not 10 and not 128 after adding cache twice.

`runtime.EventRequestUsage` carries one `llm.RequestUsage` per provider attempt:
request ID, model, category (`generation`, `retry`, `compaction`), status,
source (`reported` or `unavailable`) and optional usage. A nil usage value is
unknown, including on potentially billable failures. It must never be priced
as zero or replaced by the preceding request's usage. No estimated figure is
labelled reported by this layer.

Use the newest applicable generation/retry record for the active model when
showing context occupancy. Use the sum of records for consumption. Ten
20,000-input requests consume 200,000 reported input tokens while the newest
request occupies approximately 20,000 input tokens. An unavailable newest
record makes current occupancy unknown; model/session changes also require
resetting a consumer's gauge.

`EventDone.Usage` is a reported-only aggregate for the run, including observed
retries and run-owned background and synchronous compaction. It is incomplete whenever any request
record has unavailable usage. RunTurn returns the aggregate for that call,
including refusal retries, rather than only the last attempt. Request records
are available on error paths even when no EventDone is emitted.

Compaction returns its attempt records in `compaction.Result.Requests`.
`Manager.OnUsage` optionally observes all compaction attempts, including
background compaction. The observer must be concurrency-safe. Background work
started with `MaybeCompactAsyncContext` retains the caller's cancellation and
usage observer. Runtime joins its background producer and flushes the session
before emitting EventDone or closing the event stream. Cancellation during
that join produces EventAborted instead of a successful EventDone. This may
add summarisation latency at the end of a run; consume the stream until it
closes, including after cancellation.

The legacy `MaybeCompactAsync` API still creates detached work. Applications
using it must own that producer separately and collect its session accounting.
`JoinInFlight` cancels on caller cancellation and still waits for the producer
to exit; prevent new launches while joining or releasing session ownership.
Providers must honour cancellation for bounded shutdown. When combining event and
callback streams, deduplicate by request ID.

`llm.WithUsageObserver` can be used by other adapters with `llm.ObserveChat`.
Observers are invoked before the terminal event is forwarded; providers must
honour cancellation and end their streams after EventDone/EventError.
`RunTurn` preserves serial delivery of its TurnEmit callbacks. UsageLedger is
intended for a bounded operation, not an unbounded global history. Persistence,
price schedules and budget admission remain application responsibilities.

Context-owned compaction retains its completed result until `JoinInFlight` (or
`ForgetSession`) consumes it. Always join, even when `HasInFlight` is false:
that method reports active execution, not an unread result. Repeated launches
return the existing handle until consumption, preventing an unread failure
from being overwritten. `WaitForInFlight` observes without consuming. This
retains at most one result per owned session; release it when ownership ends.
