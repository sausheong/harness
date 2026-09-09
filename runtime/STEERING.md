# Steering boundary contract

`WithSteering` installs a prompt source for `Runtime.Run`/`RunSync`. It is polled
after a model response and between joined tools. Without a source, existing tool
concurrency and streaming kickoff behaviour is unchanged. With a source, tools
run serially and streaming kickoff is disabled, providing a boundary before
each remaining proposed call.

A source returns nil when no correction is pending. Otherwise it returns a
stable ID, nonempty UTF-8 text bounded to 8 MiB after JSON encoding, up to 16 images
with at most 32 MiB of image bytes in total, and an optional acknowledgement
callback. The source must retain that correction until acknowledgement, honour
cancellation, and return promptly. IDs are at most 128 bytes. A delivered ID
must never be reused with different text.

On delivery, remaining calls receive explicit skipped results, then the user
correction is appended under its stable ID and flushed. Only then does Harness
acknowledge the source. The next model turn sees that history. Source or
persistence errors stop execution. An acknowledgement failure also stops the
run; redelivery of the same ID/text acknowledges without appending a duplicate.

This API does not cancel an already executing tool or take over its approval.
The host must manage queue ownership, freeze text while delivery is claimed,
reconcile acknowledgement failures, and render pending/delivered state. A
correction arriving during a pending approval waits for that tool's execution
boundary; separate cancellation remains available. `RunTurn` does not currently
use this API. Native platform and integrated Hand qualification remain pending.

Steering images use the session attachment store and are verified/hydrated for
identity comparison on redelivery. `MatchesSteering` exposes that semantic
comparison to hosts reconciling interrupted acknowledgements. Text and image
bytes must remain immutable while a delivery is unresolved. These limits bound
the data; hosts must still enforce their model image capabilities and file
attachment policies before supplying a correction.
