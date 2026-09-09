package budget

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestRunLedgersComposeWithSessionAndSurviveRestart(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "runs"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "runs")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if s != nil {
			s.Close()
		}
	}()
	total, _ := OpenTokenLedger(s)
	run, _ := OpenRunTokenLedger(s, "job-1")
	if err = total.Decide(ctx, 100); err != nil {
		t.Fatal(err)
	}
	if err = run.Decide(ctx, 1); err != nil {
		t.Fatal(err)
	}
	guarded := llm.WithCallAdmission(llm.WithCallAdmission(ctx, total.Admission(estimate)), run.Admission(estimate))
	calls := 0
	provider := func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
		calls++
		out := make(chan llm.ChatEvent, 1)
		out <- llm.ChatEvent{Type: llm.EventDone, Usage: &llm.Usage{InputTokens: 8, OutputTokens: 2}}
		close(out)
		return out, nil
	}
	req := llm.ChatRequest{Model: "m", MaxTokens: 20}
	if _, err = llm.ObserveChat(guarded, req, llm.CallGeneration, provider); !errors.Is(err, ErrExhausted) {
		t.Fatal(err)
	}
	state, err := total.State()
	if err != nil || state.Committed != 0 || calls != 0 {
		t.Fatalf("denied run charged session: %+v %v calls=%d", state, err, calls)
	}
	if err = run.Decide(ctx, 80); err != nil {
		t.Fatal(err)
	}
	for _, category := range []llm.CallCategory{llm.CallGeneration, llm.CallRetry, llm.CallCompaction} {
		events, e := llm.ObserveChat(guarded, req, category, provider)
		if e != nil {
			t.Fatal(e)
		}
		for ev := range events {
			if ev.Error != nil {
				t.Fatal(ev.Error)
			}
		}
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s = nil
	s, err = store.LoadExclusive("a", "runs")
	if err != nil {
		t.Fatal(err)
	}
	total, _ = OpenTokenLedger(s)
	run, _ = OpenRunTokenLedger(s, "job-1")
	other, _ := OpenRunTokenLedger(s, "job-2")
	for _, l := range []*TokenLedger{total, run} {
		state, e := l.State()
		if e != nil || state.Committed != 30 {
			t.Fatalf("lost charge: %+v %v", state, e)
		}
	}
	untouched, e := other.State()
	if e != nil || untouched.Limit != 0 || untouched.Committed != 0 {
		t.Fatalf("scope collision: %+v %v", untouched, e)
	}
}
func TestRunCostAndDeadlinePersistWithoutSessionReset(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "run-cost"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "run-cost")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if s != nil {
			s.Close()
		}
	}()
	run, _ := OpenRunCostLedger(s, "operation")
	if err = run.Decide(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	p := currentFixturePrice()
	req := llm.ChatRequest{Model: p.Model, Route: llm.CallRoute{Provider: p.Provider, Destination: p.Destination}, MaxTokens: 20}
	// An interrupted attempt remains reserved under the same run after reopen.
	if _, err = run.Admission(priceResolver(&p))(ctx, req, llm.CallRetry, "attempt"); err != nil {
		t.Fatal(err)
	}
	if err = DecideRunDeadline(ctx, s, "operation", time.Hour); err != nil {
		t.Fatal(err)
	}
	before, err := ReadRunDeadline(s, "operation")
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s = nil
	s, err = store.LoadExclusive("a", "run-cost")
	if err != nil {
		t.Fatal(err)
	}
	run, _ = OpenRunCostLedger(s, "operation")
	state, err := run.State()
	if err != nil || state.CommittedNano != 337507 || state.Attempts[0].Status != "reserved" {
		t.Fatalf("lost run cost: %+v %v", state, err)
	}
	after, err := ReadRunDeadline(s, "operation")
	if err != nil || !after.Deadline.Equal(before.Deadline) {
		t.Fatalf("deadline renewed: %+v %v", after, err)
	}
	sessionCost, _ := OpenCostLedger(s)
	total, err := sessionCost.State()
	if err != nil || total.CommittedNano != 0 || total.LimitNano != 0 {
		t.Fatalf("scope contaminated session: %+v %v", total, err)
	}
	sessionDeadline, err := ReadDeadline(s)
	if err != nil || sessionDeadline.Version != 0 {
		t.Fatal("run set session deadline")
	}
	parent, cancel := context.WithCancel(ctx)
	child, stop, err := WithRunDeadline(parent, s, "operation")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	cancel()
	<-child.Done()
	if !errors.Is(context.Cause(child), context.Canceled) {
		t.Fatal("parent cancellation lost")
	}
}
func TestRunBudgetIDsRejectAmbiguousNamespaces(t *testing.T) {
	s := session.NewSession("a", "ids")
	for _, id := range []string{"", "x.y", "../session", "with space", "a/b", strings.Repeat("a", 65), "日本"} {
		if _, err := OpenRunTokenLedger(s, id); err == nil {
			t.Fatalf("accepted %q", id)
		}
		if _, err := OpenRunCostLedger(s, id); err == nil {
			t.Fatalf("accepted %q", id)
		}
		if err := DecideRunDeadline(context.Background(), s, id, time.Hour); err == nil {
			t.Fatalf("accepted %q", id)
		}
	}
}

func TestRunDeadlineExpiryDoesNotRenewOnRebind(t *testing.T) {
	s := session.NewSession("a", "expiry")
	ctx := context.Background()
	if err := DecideRunDeadline(ctx, s, "run", 20*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	child, cancel, err := WithRunDeadline(ctx, s, "run")
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	select {
	case <-child.Done():
	case <-time.After(time.Second):
		t.Fatal("run deadline did not expire")
	}
	if !errors.Is(context.Cause(child), ErrRunTimeExhausted) {
		t.Fatal(context.Cause(child))
	}
	if _, stop, err := WithRunDeadline(ctx, s, "run"); !errors.Is(err, ErrRunTimeExhausted) {
		stop()
		t.Fatal("expired run implicitly renewed", err)
	}
	if err := DecideRunDeadline(ctx, s, "run", time.Hour); err != nil {
		t.Fatal(err)
	}
	_, stop, err := WithRunDeadline(ctx, s, "run")
	defer stop()
	if err != nil {
		t.Fatal(err)
	}
}
