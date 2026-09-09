package runtime

import (
	"context"
	"errors"
	"time"

	"github.com/sausheong/harness/budget"
)

func (r *Runtime) DecideTimeBudget(ctx context.Context, allowance time.Duration) error {
	if !r.runMu.TryLock() {
		return errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if r.Compaction != nil && r.Compaction.HasInFlight(r.Session) {
		return errors.New("compaction is running")
	}
	return budget.DecideDeadline(ctx, r.Session, allowance)
}
func (r *Runtime) TimeBudget(ctx context.Context) (budget.DeadlineState, error) {
	if !r.runMu.TryLock() {
		return budget.DeadlineState{}, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return budget.DeadlineState{}, err
	}
	return budget.ReadDeadline(r.Session)
}

// SessionDeadlineContext is for owned operations such as manual compaction.
// The caller must join child work before invoking the returned cancel function.
func (r *Runtime) SessionDeadlineContext(ctx context.Context) (context.Context, context.CancelFunc, error) {
	if !r.runMu.TryLock() {
		return ctx, func() {}, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	return budget.WithDeadline(ctx, r.Session)
}
func abortedEvent(ctx context.Context) AgentEvent {
	if cause := context.Cause(ctx); cause != nil && cause != context.Canceled {
		return AgentEvent{Type: EventError, Error: cause}
	}
	return AgentEvent{Type: EventAborted}
}

// Preserve explicit cancellation causes without attributing deadlines to users.
func toolAbortReason(ctx context.Context) string {
	if cause := context.Cause(ctx); cause != nil && cause != context.Canceled {
		return cause.Error()
	}
	return "aborted by user"
}
