package runtime

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

type deadlineProvider struct {
	recordingProvider
	started, stopped chan struct{}
}

func (p *deadlineProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	close(p.started)
	out := make(chan llm.ChatEvent)
	go func() { defer close(out); <-ctx.Done(); close(p.stopped) }()
	return out, nil
}
func TestTimeBudgetCancelsRunningProviderAndBlocksNextRun(t *testing.T) {
	ctx := context.Background()
	p := &deadlineProvider{started: make(chan struct{}), stopped: make(chan struct{})}
	r := &Runtime{Session: session.NewSession("a", "deadline"), LLM: p, Tools: tool.NewRegistry(), Model: "fixture", MaxTurns: 1}
	if err := r.DecideTimeBudget(ctx, time.Second); err != nil {
		t.Fatal(err)
	}
	events, err := r.Run(ctx, "wait", nil)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.started:
	case <-time.After(3 * time.Second):
		t.Fatal("provider not started")
	}
	finished := make(chan bool, 1)
	go func() {
		exhausted := false
		for event := range events {
			if errors.Is(event.Error, budget.ErrTimeExhausted) {
				exhausted = true
			}
		}
		finished <- exhausted
	}()
	select {
	case exhausted := <-finished:
		if !exhausted {
			t.Fatal("deadline lost budget outcome")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("runtime failed to join after deadline")
	}
	select {
	case <-p.stopped:
	case <-time.After(time.Second):
		t.Fatal("provider did not observe cancellation")
	}
	if _, err = r.Run(ctx, "cannot renew", nil); !errors.Is(err, budget.ErrTimeExhausted) {
		t.Fatalf("expired new run: %v", err)
	}
}

func TestToolAbortReason(t *testing.T) {
	user, cancel := context.WithCancel(context.Background())
	cancel()
	if got := toolAbortReason(user); got != "aborted by user" {
		t.Fatal(got)
	}
	deadline, stop := context.WithCancelCause(context.Background())
	stop(budget.ErrTimeExhausted)
	if got := toolAbortReason(deadline); got != budget.ErrTimeExhausted.Error() {
		t.Fatal(got)
	}
}
