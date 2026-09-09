package runtime

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
)

var ErrRunTokenExhausted = fmt.Errorf("run token budget: %w", budget.ErrExhausted)
var ErrRunCostExhausted = fmt.Errorf("run cost budget: %w", budget.ErrExhausted)

type RunBudgetState struct {
	ID       string               `json:"id"`
	Tokens   budget.TokenState    `json:"tokens"`
	Cost     budget.CostState     `json:"cost"`
	Deadline budget.DeadlineState `json:"deadline"`
}

type runBudgetKey struct{ session *session.Session }
type runBudgetBinding struct {
	id           string
	tokens, cost bool
	deadline     time.Time
}

func (r *Runtime) withIdleRunBudget(ctx context.Context, fn func() error) error {
	if !r.runMu.TryLock() {
		return errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.Session == nil {
		return errors.New("session unavailable")
	}
	if r.Compaction != nil && r.Compaction.HasInFlight(r.Session) {
		return errors.New("compaction is running")
	}
	return fn()
}
func (r *Runtime) DecideRunTokenBudget(ctx context.Context, id string, limit int64) error {
	return r.withIdleRunBudget(ctx, func() error {
		l, err := budget.OpenRunTokenLedger(r.Session, id)
		if err != nil {
			return err
		}
		return l.Decide(ctx, limit)
	})
}
func (r *Runtime) DecideRunCostBudget(ctx context.Context, id, currency string, limitNano int64, strict bool) error {
	return r.withIdleRunBudget(ctx, func() error {
		l, err := budget.OpenRunCostLedger(r.Session, id)
		if err != nil {
			return err
		}
		return l.Decide(ctx, currency, limitNano, strict)
	})
}
func (r *Runtime) DecideRunTimeBudget(ctx context.Context, id string, allowance time.Duration) error {
	return r.withIdleRunBudget(ctx, func() error { return budget.DecideRunDeadline(ctx, r.Session, id, allowance) })
}
func (r *Runtime) RunBudget(ctx context.Context, id string) (state RunBudgetState, err error) {
	err = r.withIdleRunBudget(ctx, func() error { var e error; state, e = r.runBudgetState(id); return e })
	return
}
func (r *Runtime) runBudgetState(id string) (RunBudgetState, error) {
	state := RunBudgetState{ID: id}
	token, err := budget.OpenRunTokenLedger(r.Session, id)
	if err != nil {
		return state, err
	}
	cost, err := budget.OpenRunCostLedger(r.Session, id)
	if err != nil {
		return state, err
	}
	state.Tokens, err = token.State()
	if err != nil {
		return state, err
	}
	state.Cost, err = cost.State()
	if err != nil {
		return state, err
	}
	state.Deadline, err = budget.ReadRunDeadline(r.Session, id)
	return state, err
}

// PrepareRunBudget binds an explicitly configured logical run to all provider
// attempts, including retries and compaction, while retaining inherited guards.
// Use the returned context for the whole logical run (including continuations).
// Cancel only after every owned operation joins. Preparing/repreparing never
// changes limits or renews the deadline; unconfigured IDs fail closed.
func (r *Runtime) PrepareRunBudget(ctx context.Context, id string) (context.Context, context.CancelFunc, error) {
	bound := ctx
	cancel := func() {}
	err := r.withIdleRunBudget(ctx, func() error {
		key := runBudgetKey{r.Session}
		if prior, ok := ctx.Value(key).(runBudgetBinding); ok {
			if prior.id != id {
				return errors.New("cannot replace an inherited run budget")
			}
			return r.validateRunBinding(ctx)
		}
		state, err := r.runBudgetState(id)
		if err != nil {
			return err
		}
		if state.Tokens.Limit == 0 && state.Cost.LimitNano == 0 && state.Deadline.Version == 0 {
			return errors.New("run budget has no explicit decision")
		}
		bound, err = r.budgetContext(ctx)
		if err != nil {
			return err
		}
		if state.Tokens.Limit > 0 {
			l, e := budget.OpenRunTokenLedger(r.Session, id)
			if e != nil {
				return e
			}
			bound = llm.WithCallAdmission(bound, runAdmission(l.Admission(runInputEstimate), ErrRunTokenExhausted))
		}
		if state.Cost.LimitNano > 0 {
			l, e := budget.OpenRunCostLedger(r.Session, id)
			if e != nil {
				return e
			}
			prices, e := r.costPrices()
			if e != nil {
				return e
			}
			resolver := func(req llm.ChatRequest, _ llm.CallCategory) (*budget.PriceSnapshot, int64, error) {
				input, e := runInputEstimate(req)
				if e != nil {
					return nil, 0, e
				}
				for _, p := range prices {
					if p.Provider == req.Route.Provider && p.Destination == req.Route.Destination && p.Model == req.Model {
						return &p, input, nil
					}
				}
				return nil, input, nil
			}
			bound = llm.WithCallAdmission(bound, runAdmission(l.Admission(resolver), ErrRunCostExhausted))
		}
		withSession, sessionCancel, e := budget.WithDeadline(bound, r.Session)
		if e != nil {
			return e
		}
		withRun, runCancel, e := budget.WithRunDeadline(withSession, r.Session, id)
		if e != nil {
			sessionCancel()
			return e
		}
		cancel = func() { runCancel(); sessionCancel() }
		bound = context.WithValue(withRun, key, runBudgetBinding{id: id, tokens: state.Tokens.Limit > 0, cost: state.Cost.LimitNano > 0, deadline: state.Deadline.Deadline})
		return nil
	})
	if err != nil {
		cancel()
		return ctx, func() {}, err
	}
	return bound, cancel, nil
}
func runInputEstimate(req llm.ChatRequest) (int64, error) {
	prompt := req.SystemPrompt
	if len(req.SystemPromptParts) > 0 {
		prompt = llm.JoinSystemPromptParts(req.SystemPromptParts)
	}
	return int64(tokens.Estimate(req.Messages, prompt, req.Tools)), nil
}
func runAdmission(guard llm.CallAdmission, cause error) llm.CallAdmission {
	classify := func(err error) error {
		if errors.Is(err, budget.ErrExhausted) {
			return errors.Join(cause, err)
		}
		return err
	}
	return func(ctx context.Context, req llm.ChatRequest, category llm.CallCategory, id string) (func(llm.RequestUsage) error, error) {
		settle, err := guard(ctx, req, category, id)
		if err != nil {
			return nil, classify(err)
		}
		return func(usage llm.RequestUsage) error { return classify(settle(usage)) }, nil
	}
}

// A newly enabled dimension or changed deadline needs a fresh binding from
// the owner context. Existing ledger guards already reload changed ceilings.
func (r *Runtime) validateRunBinding(ctx context.Context) error {
	binding, ok := ctx.Value(runBudgetKey{r.Session}).(runBudgetBinding)
	if !ok {
		return nil
	}
	state, err := r.runBudgetState(binding.id)
	if err != nil {
		return err
	}
	if binding.tokens != (state.Tokens.Limit > 0) || binding.cost != (state.Cost.LimitNano > 0) || !binding.deadline.Equal(state.Deadline.Deadline) {
		return errors.New("run budget decision changed; prepare a fresh context from its owner")
	}
	return nil
}
