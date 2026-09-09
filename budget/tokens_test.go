package budget

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func estimate(llm.ChatRequest) (int64, error) { return 10, nil }
func reserve(t *testing.T, l *TokenLedger, id string) func(llm.RequestUsage) error {
	t.Helper()
	done, err := l.Admission(estimate)(context.Background(), llm.ChatRequest{Model: "m", MaxTokens: 20}, llm.CallCompaction, id)
	if err != nil {
		t.Fatal(err)
	}
	return done
}
func TestTokenLedgerRestartRetainsUncertainAndRequiresDecision(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "s"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "s")
	if err != nil {
		t.Fatal(err)
	}
	l, err := OpenTokenLedger(s)
	if err != nil {
		t.Fatal(err)
	}
	if err = l.Decide(ctx, 50); err != nil {
		t.Fatal(err)
	}
	reserve(t, l, "interrupted")
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = store.LoadExclusive("a", "s")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	l, err = OpenTokenLedger(s)
	if err != nil {
		t.Fatal(err)
	}
	state, err := l.State()
	if err != nil || state.Committed != 30 || state.Attempts[0].Status != "reserved" {
		t.Fatalf("restart %+v %v", state, err)
	}
	_, err = l.Admission(estimate)(ctx, llm.ChatRequest{MaxTokens: 20}, llm.CallRetry, "next")
	if !errors.Is(err, ErrExhausted) {
		t.Fatal(err)
	}
	// Even a smaller request cannot implicitly resume an exhausted decision.
	_, err = l.Admission(estimate)(ctx, llm.ChatRequest{MaxTokens: 1}, llm.CallRetry, "small")
	if !errors.Is(err, ErrExhausted) {
		t.Fatal("exhaustion not latched")
	}
	if err = l.Decide(ctx, 80); err != nil {
		t.Fatal(err)
	}
	done := reserve(t, l, "resumed")
	if err = done(llm.RequestUsage{ID: "resumed", Status: "completed", Usage: &llm.Usage{InputTokens: 8, OutputTokens: 2}}); err != nil {
		t.Fatal(err)
	}
	state, _ = l.State()
	if state.Committed != 40 || state.Limit != 80 || state.Exhausted {
		t.Fatalf("state %+v", state)
	}
	s.Compact("summary", 0)
	if err = s.Flush(); err != nil {
		t.Fatal(err)
	}
	l, err = OpenTokenLedger(s)
	if err != nil {
		t.Fatal(err)
	}
	state, _ = l.State()
	if state.Committed != 40 {
		t.Fatal("compaction erased ledger")
	}
}
func TestTokenLedgerOwnersCannotDoubleReserve(t *testing.T) {
	s := session.NewSession("a", "s")
	first, _ := OpenTokenLedger(s)
	second, _ := OpenTokenLedger(s)
	if err := first.Decide(context.Background(), 30); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	results := make(chan error, 2)
	for i, l := range []*TokenLedger{first, second} {
		wg.Add(1)
		go func(i int, l *TokenLedger) {
			defer wg.Done()
			_, err := l.Admission(estimate)(context.Background(), llm.ChatRequest{MaxTokens: 20}, llm.CallGeneration, string(rune('a'+i)))
			results <- err
		}(i, l)
	}
	wg.Wait()
	close(results)
	ok := 0
	for err := range results {
		if err == nil {
			ok++
		} else if !errors.Is(err, ErrExhausted) {
			t.Fatal(err)
		}
	}
	state, err := first.State()
	if err != nil || ok != 1 || state.Committed != 30 || len(state.Attempts) != 1 {
		t.Fatalf("state %+v success %d err %v", state, ok, err)
	}
}
func TestTokenLedgerSettlementUncertaintyAndOverrun(t *testing.T) {
	for _, tc := range []struct {
		name, status string
		usage        *llm.Usage
		want         int64
		exhausted    bool
	}{
		{"missing", "completed", nil, 30, false},
		{"partial", "cancelled", &llm.Usage{InputTokens: 1}, 30, false},
		{"not_sent", "not_dispatched", nil, 0, false},
		{"overrun", "completed", &llm.Usage{InputTokens: 80, OutputTokens: 40}, 120, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, _ := OpenTokenLedger(session.NewSession("a", "s"))
			if err := l.Decide(context.Background(), 100); err != nil {
				t.Fatal(err)
			}
			done := reserve(t, l, "attempt")
			r := llm.RequestUsage{ID: "attempt", Status: tc.status, Usage: tc.usage}
			if err := done(r); (err != nil) != tc.exhausted || (err != nil && !errors.Is(err, ErrExhausted)) {
				t.Fatal(err)
			}
			if err := done(r); (err != nil) != tc.exhausted || (err != nil && !errors.Is(err, ErrExhausted)) {
				t.Fatal(err)
			}
			state, err := l.State()
			if err != nil || state.Committed != tc.want || state.Exhausted != tc.exhausted {
				t.Fatalf("state %+v err %v", state, err)
			}
		})
	}
}
func TestTokenLedgerPersistenceFailurePreventsDispatch(t *testing.T) {
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "s"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "s")
	if err != nil {
		t.Fatal(err)
	}
	l, _ := OpenTokenLedger(s)
	if err = l.Decide(context.Background(), 100); err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	ctx := llm.WithCallAdmission(context.Background(), l.Admission(estimate))
	_, err = llm.ObserveChat(ctx, llm.ChatRequest{MaxTokens: 20}, llm.CallGeneration, func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
		t.Fatal("failed reservation dispatched")
		return nil, nil
	})
	if err == nil {
		t.Fatal("closed journal accepted reservation")
	}
}
