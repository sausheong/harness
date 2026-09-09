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

func TestRunBudgetBindingChargesRunTurnAndCompactionOnce(t *testing.T) {
	ctx := context.Background()
	p := &recordingProvider{reply: "<summary>done</summary>"}
	r := &Runtime{Session: session.NewSession("a", "run"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture", MaxTurns: 1}
	if err := r.DecideTokenBudget(ctx, 1000000); err != nil {
		t.Fatal(err)
	}
	if err := r.DecideRunTokenBudget(ctx, "logical-run", 1000000); err != nil {
		t.Fatal(err)
	}
	bound, cancel, err := r.PrepareRunBudget(ctx, "logical-run")
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	// Repreparing must not attach another reservation for the same scope.
	same, stop, err := r.PrepareRunBudget(bound, "logical-run")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	if _, stop, e := r.PrepareRunBudget(bound, "replacement"); e == nil {
		stop()
		t.Fatal("inherited scope replaced")
	}
	events, err := r.Run(same, "first", nil)
	if err != nil {
		t.Fatal(err)
	}
	for ev := range events {
		if ev.Error != nil {
			t.Fatal(ev.Error)
		}
	}
	result, err := r.RunTurn(same, "second", nil, nil)
	if err != nil || result.Err != nil {
		t.Fatalf("turn %+v %v", result, err)
	}
	for i := 0; i < 10; i++ {
		r.Session.Append(session.UserMessageEntry("task"))
		r.Session.Append(session.AssistantMessageEntry("evidence"))
	}
	manager := &compaction.Manager{Summarizer: &compaction.Summarizer{Provider: p, Model: "fixture", MaxOutputTokens: 100}, PreserveTurns: 2}
	manual, err := r.CompactionContext(same)
	if err != nil {
		t.Fatal(err)
	}
	compacted, err := manager.MaybeCompact(manual, r.Session, compaction.ReasonManual, "")
	if err != nil || !compacted.Compacted {
		t.Fatalf("summary %+v %v", compacted, err)
	}
	run, err := r.RunBudget(ctx, "logical-run")
	if err != nil {
		t.Fatal(err)
	}
	total, err := r.TokenBudget(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.Tokens.Attempts) != 3 || len(total.Attempts) != 3 || run.Tokens.Committed != total.Committed || run.Tokens.Attempts[2].Category != llm.CallCompaction {
		t.Fatalf("run=%+v total=%+v", run, total)
	}
}
func TestRunBudgetBindingCannotBypassEitherScope(t *testing.T) {
	for _, small := range []string{"run", "session"} {
		t.Run(small, func(t *testing.T) {
			ctx := context.Background()
			p := &recordingProvider{reply: "done"}
			r := &Runtime{Session: session.NewSession("a", "blocked"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture"}
			runLimit, sessionLimit := int64(100000), int64(100000)
			if small == "run" {
				runLimit = 1
			} else {
				sessionLimit = 1
			}
			if err := r.DecideTokenBudget(ctx, sessionLimit); err != nil {
				t.Fatal(err)
			}
			if err := r.DecideRunTokenBudget(ctx, "job", runLimit); err != nil {
				t.Fatal(err)
			}
			bound, cancel, err := r.PrepareRunBudget(ctx, "job")
			if err != nil {
				t.Fatal(err)
			}
			defer cancel()
			events, err := r.Run(bound, "blocked", nil)
			if err != nil {
				t.Fatal(err)
			}
			var terminal error
			for ev := range events {
				if ev.Error != nil {
					terminal = ev.Error
				}
			}
			if !errors.Is(terminal, budget.ErrExhausted) || len(p.requests) != 0 {
				t.Fatalf("admission failed: %v requests=%d", terminal, len(p.requests))
			}
			if small == "run" && !errors.Is(terminal, ErrRunTokenExhausted) {
				t.Fatal("run reason lost", terminal)
			}
			run, _ := r.RunBudget(ctx, "job")
			total, _ := r.TokenBudget(ctx)
			if run.Tokens.Committed != 0 || total.Committed != 0 {
				t.Fatal("undispatched work charged")
			}
		})
	}
}
func TestRunBudgetDeadlineCancelsOwnedProvider(t *testing.T) {
	ctx := context.Background()
	p := &deadlineProvider{started: make(chan struct{}), stopped: make(chan struct{})}
	r := &Runtime{Session: session.NewSession("a", "time"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture"}
	if _, stop, err := r.PrepareRunBudget(ctx, "missing"); err == nil {
		stop()
		t.Fatal("unconfigured run allowed")
	}
	if err := r.DecideRunTimeBudget(ctx, "timed", time.Second); err != nil {
		t.Fatal(err)
	}
	bound, cancel, err := r.PrepareRunBudget(ctx, "timed")
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	events, err := r.Run(bound, "wait", nil)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for ev := range events {
		if errors.Is(ev.Error, budget.ErrRunTimeExhausted) {
			found = true
		}
	}
	if !found {
		t.Fatal("run deadline cause lost")
	}
	select {
	case <-p.stopped:
	case <-time.After(time.Second):
		t.Fatal("provider not stopped")
	}
	if _, stop, e := r.PrepareRunBudget(ctx, "timed"); !errors.Is(e, budget.ErrRunTimeExhausted) {
		stop()
		t.Fatal("expired deadline renewed", e)
	}
}

func TestRunBudgetStaleBindingCannotOmitNewDimension(t *testing.T) {
	ctx := context.Background()
	p := &recordingProvider{reply: "done"}
	r := &Runtime{Session: session.NewSession("a", "stale"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture"}
	if err := r.DecideRunTimeBudget(ctx, "job", time.Hour); err != nil {
		t.Fatal(err)
	}
	bound, cancel, err := r.PrepareRunBudget(ctx, "job")
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	if err = r.DecideRunTokenBudget(ctx, "job", 1); err != nil {
		t.Fatal(err)
	}
	if _, err = r.Run(bound, "must not bypass", nil); err == nil {
		t.Fatal("stale binding bypassed new token limit")
	}
	if len(p.requests) != 0 {
		t.Fatal("stale binding dispatched")
	}
}
func TestRunCostBindingStrictAdmissionAndSharedCharges(t *testing.T) {
	ctx := context.Background()
	p := &recordingProvider{reply: "done"}
	r := &Runtime{Session: session.NewSession("a", "money"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture", MaxTurns: 1, Route: llm.CallRoute{Provider: "local", Destination: "fixture"}}
	if err := r.DecideCostBudget(ctx, "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	if err := r.SetCostPrices(ctx, []budget.PriceSnapshot{runtimeFixturePrice(r.Route, r.Model)}); err != nil {
		t.Fatal(err)
	}
	if err := r.DecideRunCostBudget(ctx, "job", "USD", 1, true); err != nil {
		t.Fatal(err)
	}
	bound, cancel, err := r.PrepareRunBudget(ctx, "job")
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	events, err := r.Run(bound, "blocked", nil)
	if err != nil {
		t.Fatal(err)
	}
	var terminal error
	for ev := range events {
		if ev.Error != nil {
			terminal = ev.Error
		}
	}
	if !errors.Is(terminal, ErrRunCostExhausted) || len(p.requests) != 0 {
		t.Fatalf("run cost bypass: %v", terminal)
	}
	total, err := r.CostBudget(ctx)
	if err != nil || total.CommittedNano != 0 {
		t.Fatalf("denied attempt charged session: %+v %v", total, err)
	}
	if err = r.DecideRunCostBudget(ctx, "job", "USD", 1000000, true); err != nil {
		t.Fatal(err)
	}
	events, err = r.Run(bound, "resumed", nil)
	if err != nil {
		t.Fatal(err)
	}
	for ev := range events {
		if ev.Error != nil {
			t.Fatal(ev.Error)
		}
	}
	state, err := r.RunBudget(ctx, "job")
	if err != nil {
		t.Fatal(err)
	}
	total, err = r.CostBudget(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if state.Cost.CommittedNano <= 0 || state.Cost.CommittedNano != total.CommittedNano || len(p.requests) != 1 {
		t.Fatalf("charges run=%+v total=%+v", state.Cost, total)
	}
}
