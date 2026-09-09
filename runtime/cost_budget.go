package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
)

const costPricesKind = "harness.budget.prices.v1"

type costPricesRecord struct {
	Version int                    `json:"version"`
	Prices  []budget.PriceSnapshot `json:"prices"`
}
type costBudgetContextKey struct{ session *session.Session }

func validateCostPrices(prices []budget.PriceSnapshot) error {
	if len(prices) > 16 {
		return errors.New("at most 16 route prices are allowed")
	}
	type key struct{ provider, model, destination string }
	seen := map[key]bool{}
	for _, p := range prices {
		if err := p.Validate(); err != nil {
			return err
		}
		k := key{p.Provider, p.Model, p.Destination}
		if seen[k] {
			return errors.New("duplicate route price")
		}
		seen[k] = true
	}
	return nil
}

// SetCostPrices durably replaces explicitly reviewed session route tariffs.
// It does not change the budget ceiling or erase historical request snapshots.
func (r *Runtime) SetCostPrices(ctx context.Context, prices []budget.PriceSnapshot) error {
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
	if err := validateCostPrices(prices); err != nil {
		return err
	}
	raw, err := json.Marshal(costPricesRecord{Version: 1, Prices: prices})
	if err != nil {
		return err
	}
	return r.Session.Annotate(costPricesKind, raw)
}
func (r *Runtime) costPrices() ([]budget.PriceSnapshot, error) {
	if r.Session == nil {
		return nil, errors.New("session unavailable")
	}
	records := r.Session.Annotations(costPricesKind)
	if len(records) == 0 {
		return nil, nil
	}
	var record costPricesRecord
	d := json.NewDecoder(bytes.NewReader(records[len(records)-1].Payload))
	d.DisallowUnknownFields()
	if err := d.Decode(&record); err != nil {
		return nil, err
	}
	if d.Decode(new(any)) != io.EOF || record.Version != 1 {
		return nil, errors.New("invalid session price record")
	}
	return record.Prices, validateCostPrices(record.Prices)
}
func (r *Runtime) CostPrices(ctx context.Context) ([]budget.PriceSnapshot, error) {
	if !r.runMu.TryLock() {
		return nil, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return r.costPrices()
}
func (r *Runtime) DecideCostBudget(ctx context.Context, currency string, limitNano int64, strict bool) error {
	if !r.runMu.TryLock() {
		return errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if r.Compaction != nil && r.Compaction.HasInFlight(r.Session) {
		return errors.New("compaction is running")
	}
	l, err := budget.OpenCostLedger(r.Session)
	if err != nil {
		return err
	}
	return l.Decide(ctx, currency, limitNano, strict)
}
func (r *Runtime) CostBudget(ctx context.Context) (budget.CostState, error) {
	if !r.runMu.TryLock() {
		return budget.CostState{}, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return budget.CostState{}, err
	}
	l, err := budget.OpenCostLedger(r.Session)
	if err != nil {
		return budget.CostState{}, err
	}
	return l.State()
}
func (r *Runtime) costBudgetContext(ctx context.Context) (context.Context, error) {
	if r.Session == nil {
		return ctx, nil
	}
	key := costBudgetContextKey{r.Session}
	if ctx.Value(key) != nil {
		return ctx, nil
	}
	l, err := budget.OpenCostLedger(r.Session)
	if err != nil {
		return ctx, err
	}
	state, err := l.State()
	if err != nil {
		return ctx, err
	}
	if state.LimitNano == 0 {
		return ctx, nil
	}
	prices, err := r.costPrices()
	if err != nil {
		return ctx, err
	}
	guard := l.Admission(func(req llm.ChatRequest, _ llm.CallCategory) (*budget.PriceSnapshot, int64, error) {
		prompt := req.SystemPrompt
		if len(req.SystemPromptParts) > 0 {
			prompt = llm.JoinSystemPromptParts(req.SystemPromptParts)
		}
		input := int64(tokens.Estimate(req.Messages, prompt, req.Tools))
		for _, p := range prices {
			if p.Provider == req.Route.Provider && p.Destination == req.Route.Destination && p.Model == req.Model {
				return &p, input, nil
			}
		}
		return nil, input, nil
	})
	return context.WithValue(llm.WithCallAdmission(ctx, guard), key, true), nil
}
func (r *Runtime) budgetContext(ctx context.Context) (context.Context, error) {
	if err := r.validateRunBinding(ctx); err != nil {
		return ctx, err
	}
	ctx, err := r.tokenBudgetContext(ctx)
	if err != nil {
		return ctx, err
	}
	return r.costBudgetContext(ctx)
}
