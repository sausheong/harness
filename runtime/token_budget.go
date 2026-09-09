package runtime

import (
	"context"
	"errors"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
)

type tokenBudgetContextKey struct{ session *session.Session }

// DecideTokenBudget records an explicit absolute session token ceiling. It may
// resume exhaustion, but never erases prior charges or uncertain reservations.
func (r *Runtime) DecideTokenBudget(ctx context.Context, limit int64) error {
	if !r.runMu.TryLock() {
		return errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if r.Compaction != nil && r.Compaction.HasInFlight(r.Session) {
		return errors.New("compaction is running")
	}
	ledger, err := budget.OpenTokenLedger(r.Session)
	if err != nil {
		return err
	}
	return ledger.Decide(ctx, limit)
}
func (r *Runtime) TokenBudget(ctx context.Context) (budget.TokenState, error) {
	if !r.runMu.TryLock() {
		return budget.TokenState{}, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return budget.TokenState{}, err
	}
	ledger, err := budget.OpenTokenLedger(r.Session)
	if err != nil {
		return budget.TokenState{}, err
	}
	return ledger.State()
}

// tokenBudgetContext is called under runMu. A session without a configured
// ceiling retains existing behaviour. A configured ceiling is reloaded on every
// admission, including after restart. Inherited parent guards remain attached.
func (r *Runtime) tokenBudgetContext(ctx context.Context) (context.Context, error) {
	if r.Session == nil {
		return ctx, nil
	}
	key := tokenBudgetContextKey{r.Session}
	if ctx.Value(key) != nil {
		return ctx, nil
	}
	ledger, err := budget.OpenTokenLedger(r.Session)
	if err != nil {
		return ctx, err
	}
	state, err := ledger.State()
	if err != nil {
		return ctx, err
	}
	if state.Limit == 0 {
		return ctx, nil
	}
	guard := ledger.Admission(func(req llm.ChatRequest) (int64, error) {
		prompt := req.SystemPrompt
		if len(req.SystemPromptParts) > 0 {
			prompt = llm.JoinSystemPromptParts(req.SystemPromptParts)
		}
		return int64(tokens.Estimate(req.Messages, prompt, req.Tools)), nil
	})
	return context.WithValue(llm.WithCallAdmission(ctx, guard), key, true), nil
}
