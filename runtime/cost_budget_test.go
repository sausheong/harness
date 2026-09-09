package runtime

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func runtimeFixturePrice(route llm.CallRoute, model string) budget.PriceSnapshot {
	return budget.PriceSnapshot{Provider: route.Provider, Destination: route.Destination, Model: model, Currency: "USD", Source: "local test tariff", Version: "1", EffectiveAt: time.Now().Add(-time.Hour), ExpiresAt: time.Now().Add(time.Hour), InputNanoPerMillion: 1000000, OutputNanoPerMillion: 1000000, AllChargesBounded: true}
}
func TestCostBudgetRuntimeAndIndependentSummary(t *testing.T) {
	ctx := context.Background()
	provider := &recordingProvider{reply: "<summary>done</summary>"}
	s := session.NewSession("a", "cost")
	main := llm.CallRoute{Provider: "local", Destination: "http://main.example/v1"}
	summary := llm.CallRoute{Provider: "local", Destination: "http://summary.example/v1"}
	r := &Runtime{Session: s, Route: main, Provider: "local", Model: "fixture", LLM: provider, Tools: tool.NewRegistry(), MaxTurns: 1}
	if err := r.DecideCostBudget(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	events, err := r.Run(ctx, "unpriced", nil)
	if err != nil {
		t.Fatal(err)
	}
	unknown := false
	for event := range events {
		if errors.Is(event.Error, budget.ErrUnknownPrice) {
			unknown = true
		}
	}
	if !unknown || len(provider.requests) != 0 {
		t.Fatal("unpriced runtime reached provider")
	}
	prices := []budget.PriceSnapshot{runtimeFixturePrice(main, "fixture"), runtimeFixturePrice(summary, "fixture")}
	if err = r.SetCostPrices(ctx, prices); err != nil {
		t.Fatal(err)
	}
	prices[0].Source = "caller mutation"
	events, err = r.Run(ctx, "priced", nil)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	state, err := r.CostBudget(ctx)
	if err != nil || len(state.Attempts) != 1 || state.Attempts[0].Quote.Snapshot.Destination != main.Destination {
		t.Fatalf("state %+v err %v", state, err)
	}
	for i := 0; i < 10; i++ {
		s.Append(session.UserMessageEntry("task"))
		s.Append(session.AssistantMessageEntry("evidence"))
	}
	manager := &compaction.Manager{Summarizer: &compaction.Summarizer{Route: summary, Provider: provider, Model: "fixture", MaxOutputTokens: 100}, PreserveTurns: 2}
	manual, err := r.CompactionContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
	result, err := manager.MaybeCompact(manual, s, compaction.ReasonManual, "")
	if err != nil || !result.Compacted {
		t.Fatalf("compaction %+v %v", result, err)
	}
	state, err = r.CostBudget(ctx)
	if err != nil || len(state.Attempts) != 2 || state.Attempts[1].Category != llm.CallCompaction || state.Attempts[1].Quote.Snapshot.Destination != summary.Destination {
		t.Fatalf("state %+v err %v", state, err)
	}
	installed, err := r.CostPrices(ctx)
	if err != nil || installed[0].Source == "caller mutation" {
		t.Fatal("caller mutation changed durable prices")
	}
}
func TestRunTurnCannotBypassSessionBudgets(t *testing.T) {
	ctx := context.Background()
	provider := &recordingProvider{reply: "done"}
	r := &Runtime{Session: session.NewSession("a", "turn"), LLM: provider, Tools: tool.NewRegistry(), Model: "fixture"}
	if err := r.DecideTokenBudget(ctx, 1); err != nil {
		t.Fatal(err)
	}
	result, err := r.RunTurn(ctx, "blocked", nil, nil)
	if !errors.Is(err, budget.ErrExhausted) && !errors.Is(result.Err, budget.ErrExhausted) {
		t.Fatalf("result %+v err %v", result, err)
	}
	if len(provider.requests) != 0 {
		t.Fatal("RunTurn bypassed token budget")
	}
}

func TestCostPricesRestartAndInvalidReplacement(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "prices"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "prices")
	if err != nil {
		t.Fatal(err)
	}
	r := &Runtime{Session: s}
	price := runtimeFixturePrice(llm.CallRoute{Provider: "local", Destination: "http://fixture.example/v1"}, "fixture")
	if err = r.SetCostPrices(ctx, []budget.PriceSnapshot{price}); err != nil {
		t.Fatal(err)
	}
	if err = r.SetCostPrices(ctx, []budget.PriceSnapshot{price, price}); err == nil {
		t.Fatal("duplicate route accepted")
	}
	if err = r.DecideCostBudget(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = store.LoadExclusive("a", "prices")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	r.Session = s
	prices, err := r.CostPrices(ctx)
	if err != nil || len(prices) != 1 || prices[0].Source != price.Source || prices[0].Destination != price.Destination {
		t.Fatalf("prices %+v err %v", prices, err)
	}
	state, err := r.CostBudget(ctx)
	if err != nil || state.LimitNano != 1000000 || !state.Strict {
		t.Fatalf("state %+v err %v", state, err)
	}
	r.runMu.Lock()
	err = r.SetCostPrices(ctx, nil)
	r.runMu.Unlock()
	if err == nil {
		t.Fatal("busy price replacement accepted")
	}
}
