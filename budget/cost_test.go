package budget

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func currentFixturePrice() PriceSnapshot {
	p := fixturePrice()
	p.EffectiveAt = time.Now().Add(-time.Hour)
	p.ExpiresAt = time.Now().Add(time.Hour)
	return p
}
func priceResolver(p *PriceSnapshot) CostResolver {
	return func(llm.ChatRequest, llm.CallCategory) (*PriceSnapshot, int64, error) { return p, 10, nil }
}
func TestCostLedgerRestartAndPricingProvenance(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "cost"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "cost")
	if err != nil {
		t.Fatal(err)
	}
	l, _ := OpenCostLedger(s)
	if err = l.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	p := currentFixturePrice()
	_, err = l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallRetry, "interrupted")
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = store.LoadExclusive("a", "cost")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	l, err = OpenCostLedger(s)
	if err != nil {
		t.Fatal(err)
	}
	state, err := l.State()
	if err != nil || state.CommittedNano != 337507 || len(state.Attempts) != 1 || state.Attempts[0].Status != "reserved" {
		t.Fatalf("state %+v err %v", state, err)
	}
	if state.Attempts[0].Quote.Snapshot.Source != p.Source || state.Attempts[0].Category != llm.CallRetry {
		t.Fatal("provenance lost")
	}
	state.Attempts[0].Quote.Snapshot.Source = "caller mutation"
	state, _ = l.State()
	if state.Attempts[0].Quote.Snapshot.Source != p.Source {
		t.Fatal("mutable state shared")
	}
	done, err := l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallCompaction, "summary")
	if err != nil {
		t.Fatal(err)
	}
	if err = done(llm.RequestUsage{ID: "summary", Status: "completed", Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5}}); err != nil {
		t.Fatal(err)
	}
	state, _ = l.State()
	if state.CommittedNano != 442514 || state.Attempts[1].Status != "reported" {
		t.Fatalf("state %+v", state)
	}
}
func TestStrictCostBlocksUnknownAndAdvisoryPersistsUncertainty(t *testing.T) {
	ctx := context.Background()
	l, _ := OpenCostLedger(session.NewSession("a", "cost"))
	if err := l.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	observed := llm.WithCallAdmission(ctx, l.Admission(priceResolver(nil)))
	_, err := llm.ObserveChat(observed, llm.ChatRequest{Model: "m", MaxTokens: 20}, llm.CallGeneration, func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
		t.Fatal("strict unknown price dispatched")
		return nil, nil
	})
	if !errors.Is(err, ErrUnknownPrice) {
		t.Fatal(err)
	}
	if err = l.Decide(ctx, "USD", 1000000, false); err != nil {
		t.Fatal(err)
	}
	done, err := l.Admission(priceResolver(nil))(ctx, llm.ChatRequest{Model: "m", MaxTokens: 20}, llm.CallGeneration, "unknown")
	if err != nil {
		t.Fatal(err)
	}
	if err = done(llm.RequestUsage{ID: "unknown", Status: "completed", Usage: &llm.Usage{InputTokens: 10}}); err != nil {
		t.Fatal(err)
	}
	state, _ := l.State()
	if state.Unknown != 1 || state.Attempts[0].Quote.Known || state.Attempts[0].Status != "uncertain" {
		t.Fatalf("state %+v", state)
	}
	if err = l.Decide(ctx, "USD", 2000000, true); err == nil {
		t.Fatal("unknown historic charge disappeared in strict mode")
	}
	if err = l.Decide(ctx, "EUR", 2000000, false); err == nil {
		t.Fatal("implicit currency conversion accepted")
	}
}
func TestCostConcurrentOwnersAndExplicitResume(t *testing.T) {
	ctx := context.Background()
	s := session.NewSession("a", "cost")
	a, _ := OpenCostLedger(s)
	b, _ := OpenCostLedger(s)
	if err := a.Decide(ctx, "USD", 500000, true); err != nil {
		t.Fatal(err)
	}
	p := currentFixturePrice()
	var wg sync.WaitGroup
	results := make(chan error, 2)
	for i, l := range []*CostLedger{a, b} {
		wg.Add(1)
		go func(i int, l *CostLedger) {
			defer wg.Done()
			_, err := l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallRetry, string(rune('a'+i)))
			results <- err
		}(i, l)
	}
	wg.Wait()
	close(results)
	success := 0
	for err := range results {
		if err == nil {
			success++
		} else if !errors.Is(err, ErrExhausted) {
			t.Fatal(err)
		}
	}
	state, _ := a.State()
	if success != 1 || state.CommittedNano != 337507 || !state.Exhausted {
		t.Fatalf("state %+v success %d", state, success)
	}
	if err := a.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	state, _ = a.State()
	if state.Exhausted || state.CommittedNano != 337507 {
		t.Fatal("resume erased charges")
	}
}
func TestCostFailedUsageRetainsReservationAndOverrunExhausts(t *testing.T) {
	ctx := context.Background()
	l, _ := OpenCostLedger(session.NewSession("a", "cost"))
	p := currentFixturePrice()
	if err := l.Decide(ctx, "USD", 500000, true); err != nil {
		t.Fatal(err)
	}
	done, err := l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallGeneration, "partial")
	if err != nil {
		t.Fatal(err)
	}
	if err = done(llm.RequestUsage{ID: "partial", Status: "cancelled", Usage: &llm.Usage{InputTokens: 1}}); err != nil {
		t.Fatal(err)
	}
	state, _ := l.State()
	if state.CommittedNano != 337507 {
		t.Fatal("partial usage released uncertain cost")
	}
	if err = l.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	done, err = l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallGeneration, "overrun")
	if err != nil {
		t.Fatal(err)
	}
	err = done(llm.RequestUsage{ID: "overrun", Status: "completed", Usage: &llm.Usage{OutputTokens: 100}})
	if !errors.Is(err, ErrExhausted) {
		t.Fatal("overrun did not stop admission")
	}
	state, _ = l.State()
	if state.CommittedNano != 1837514 || !state.Exhausted {
		t.Fatalf("state %+v", state)
	}
}

func TestCostOversizedSettlementPreservesReadableJournal(t *testing.T) {
	ctx := context.Background()
	l, _ := OpenCostLedger(session.NewSession("a", "cost"))
	p := currentFixturePrice()
	p.OutputNanoPerMillion = 1000000000000000
	if err := l.Decide(ctx, "USD", 1000000000000, true); err != nil {
		t.Fatal(err)
	}
	done, err := l.Admission(priceResolver(&p))(ctx, llm.ChatRequest{Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, Model: p.Model, MaxTokens: 20}, llm.CallGeneration, "oversized")
	if err != nil {
		t.Fatal(err)
	}
	before, _ := l.State()
	if err = done(llm.RequestUsage{ID: "oversized", Status: "completed", Usage: &llm.Usage{OutputTokens: 2000000}}); err == nil {
		t.Fatal("oversized settlement accepted")
	}
	after, err := l.State()
	if err != nil || after.CommittedNano != before.CommittedNano || after.Attempts[0].Status != "reserved" {
		t.Fatalf("journal corrupted: %+v %v", after, err)
	}
}

func TestCostAdmissionRejectsSameModelAtDifferentDestination(t *testing.T) {
	ctx := context.Background()
	l, _ := OpenCostLedger(session.NewSession("a", "route"))
	p := currentFixturePrice()
	if err := l.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	for _, route := range []llm.CallRoute{{}, {Provider: p.Provider, Destination: "http://other.example/v1"}, {Provider: "other", Destination: p.Destination}} {
		request := llm.ChatRequest{Route: route, Model: p.Model, MaxTokens: 20}
		_, err := llm.ObserveChat(llm.WithCallAdmission(ctx, l.Admission(priceResolver(&p))), request, llm.CallGeneration, func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
			t.Fatal("mismatched tariff dispatched")
			return nil, nil
		})
		if err == nil {
			t.Fatal("route mismatch accepted")
		}
	}
	state, err := l.State()
	if err != nil || len(state.Attempts) != 0 {
		t.Fatalf("mismatch reserved funds: %+v %v", state, err)
	}
}
