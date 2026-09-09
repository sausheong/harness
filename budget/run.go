package budget

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sausheong/harness/session"
)

var ErrRunTimeExhausted = fmt.Errorf("run time budget: %w", ErrExhausted)

// Run IDs are stable logical-operation identities, not process-local counters.
// Reusing an ID reopens its existing charges and deadline. A caller must bind
// the same ID to retries, summaries, continuation and recovery of that run.
func runKind(base, id string) (string, error) {
	if len(id) == 0 || len(id) > 64 || strings.IndexFunc(id, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-')
	}) >= 0 {
		return "", errors.New("run budget ID must contain 1-64 ASCII letters, digits, underscores or hyphens")
	}
	return base + ".run." + id, nil
}

// OpenRunTokenLedger opens a durable per-run ledger. Attach its admission guard
// alongside the session guard; opening this ledger does not replace either
// scope or grant a new allowance.
func OpenRunTokenLedger(s *session.Session, id string) (*TokenLedger, error) {
	kind, err := runKind(tokenKind, id)
	if err != nil {
		return nil, err
	}
	return openTokenLedger(s, kind)
}
func OpenRunCostLedger(s *session.Session, id string) (*CostLedger, error) {
	kind, err := runKind(costKind, id)
	if err != nil {
		return nil, err
	}
	return openCostLedger(s, kind)
}
func DecideRunDeadline(ctx context.Context, s *session.Session, id string, allowance time.Duration) error {
	kind, err := runKind(deadlineKind, id)
	if err != nil {
		return err
	}
	return decideDeadline(ctx, s, kind, allowance)
}
func ReadRunDeadline(s *session.Session, id string) (DeadlineState, error) {
	kind, err := runKind(deadlineKind, id)
	if err != nil {
		return DeadlineState{}, err
	}
	return readDeadline(s, kind)
}
func WithRunDeadline(ctx context.Context, s *session.Session, id string) (context.Context, context.CancelFunc, error) {
	kind, err := runKind(deadlineKind, id)
	if err != nil {
		return ctx, func() {}, err
	}
	return withDeadline(ctx, s, kind, ErrRunTimeExhausted)
}
