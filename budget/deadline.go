package budget

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/sausheong/harness/session"
)

var ErrTimeExhausted = fmt.Errorf("session time budget: %w", ErrExhausted)

const deadlineKind = "harness.budget.deadline.v1"

type DeadlineState struct {
	Version   int       `json:"version"`
	DecidedAt time.Time `json:"decided_at"`
	Deadline  time.Time `json:"deadline"`
}

// DecideDeadline starts a new explicitly approved wall-clock allowance. Idle
// time counts, and restarting does not reset the deadline. No prior cost/token
// charges are changed. There is deliberately no implicit disable or renewal.
func DecideDeadline(ctx context.Context, s *session.Session, allowance time.Duration) error {
	return decideDeadline(ctx, s, deadlineKind, allowance)
}
func decideDeadline(ctx context.Context, s *session.Session, kind string, allowance time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s == nil || allowance <= 0 || allowance > 30*24*time.Hour {
		return errors.New("time allowance must be positive and at most 30 days")
	}
	now := time.Now().UTC()
	raw, err := json.Marshal(DeadlineState{Version: 1, DecidedAt: now, Deadline: now.Add(allowance)})
	if err != nil {
		return err
	}
	return s.Annotate(kind, raw)
}
func ReadDeadline(s *session.Session) (DeadlineState, error) {
	return readDeadline(s, deadlineKind)
}
func readDeadline(s *session.Session, kind string) (DeadlineState, error) {
	var state DeadlineState
	if s == nil {
		return state, nil
	}
	records := s.Annotations(kind)
	if len(records) == 0 {
		return state, nil
	}
	if err := decodeBudgetJSON(records[len(records)-1].Payload, &state); err != nil {
		return state, err
	}
	if state.Version != 1 || state.DecidedAt.IsZero() || !state.Deadline.After(state.DecidedAt) || state.Deadline.Sub(state.DecidedAt) > 30*24*time.Hour {
		return state, errors.New("invalid session time budget record")
	}
	return state, nil
}

// WithDeadline binds owned work to the stored session deadline. The caller must
// cancel after all child work joins. Expired sessions fail before dispatch;
// active work is cancelled with ErrTimeExhausted as its context cause.
func WithDeadline(ctx context.Context, s *session.Session) (context.Context, context.CancelFunc, error) {
	return withDeadline(ctx, s, deadlineKind, ErrTimeExhausted)
}
func withDeadline(ctx context.Context, s *session.Session, kind string, cause error) (context.Context, context.CancelFunc, error) {
	noop := func() {}
	state, err := readDeadline(s, kind)
	if err != nil {
		return ctx, noop, err
	}
	if state.Version == 0 {
		return ctx, noop, nil
	}
	if err := ctx.Err(); err != nil {
		return ctx, noop, err
	}
	if !time.Now().Before(state.Deadline) {
		return ctx, noop, cause
	}
	child, cancel := context.WithDeadlineCause(ctx, state.Deadline, cause)
	return child, cancel, nil
}
