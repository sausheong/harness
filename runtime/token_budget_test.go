package runtime

import (
	"context"
	"errors"
	"testing"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func TestTokenBudgetGovernsRuntimeAndManualCompaction(t *testing.T) {
	ctx := context.Background()
	provider := &recordingProvider{reply: "<summary>done</summary>"}
	s := session.NewSession("a", "budget")
	r := &Runtime{Session: s, LLM: provider, Model: "fixture", Tools: tool.NewRegistry(), MaxTurns: 1}
	if err := r.DecideTokenBudget(ctx, 1); err != nil {
		t.Fatal(err)
	}
	events, err := r.Run(ctx, "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	exhausted := false
	for event := range events {
		if errors.Is(event.Error, budget.ErrExhausted) {
			exhausted = true
		}
	}
	if !exhausted || len(provider.requests) != 0 {
		t.Fatal("exhausted runtime dispatched provider or lost cause")
	}
	state, err := r.TokenBudget(ctx)
	if err != nil || !state.Exhausted {
		t.Fatalf("state %+v %v", state, err)
	}
	if err = r.DecideTokenBudget(ctx, 100000); err != nil {
		t.Fatal(err)
	}
	events, err = r.Run(ctx, "continue", nil)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	state, err = r.TokenBudget(ctx)
	if err != nil || len(state.Attempts) != 1 || state.Attempts[0].Category != llm.CallGeneration {
		t.Fatalf("state %+v %v", state, err)
	}
	for i := 0; i < 10; i++ {
		s.Append(session.UserMessageEntry("task"))
		s.Append(session.AssistantMessageEntry("evidence"))
	}
	manager := &compaction.Manager{Summarizer: &compaction.Summarizer{Provider: provider, Model: "fixture", MaxOutputTokens: 100}, PreserveTurns: 2}
	manual, err := r.CompactionContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// Reusing a prepared context must not reserve twice against this session.
	manual, err = r.CompactionContext(manual)
	if err != nil {
		t.Fatal(err)
	}
	result, err := manager.MaybeCompact(manual, s, compaction.ReasonManual, "")
	if err != nil || !result.Compacted {
		t.Fatalf("result %+v err %v", result, err)
	}
	state, err = r.TokenBudget(ctx)
	if err != nil || len(state.Attempts) != 2 || state.Attempts[1].Category != llm.CallCompaction || state.Committed <= 0 {
		t.Fatalf("state %+v err %v", state, err)
	}
}
