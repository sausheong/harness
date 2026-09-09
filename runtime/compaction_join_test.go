package runtime

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
)

type joinedSummaryProvider struct {
	llmtest.Base
	entered, cancelled, release chan struct{}
	beforeReturn                func()
}

func (p *joinedSummaryProvider) ChatStream(ctx context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	close(p.entered)
	select {
	case <-p.release:
	case <-ctx.Done():
		close(p.cancelled)
		<-p.release
		return nil, ctx.Err()
	}
	if p.beforeReturn != nil {
		p.beforeReturn()
	}
	ch := make(chan llm.ChatEvent, 2)
	ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Summary of original work and constraints."}
	ch <- llm.ChatEvent{Type: llm.EventDone, Usage: &llm.Usage{InputTokens: 70, OutputTokens: 7}}
	close(ch)
	return ch, nil
}

func TestRuntimeJoinsDeferredBackgroundCompactionAndUsage(t *testing.T) {
	for _, cancelRun := range []bool{false, true} {
		name := "complete"
		if cancelRun {
			name = "cancel"
		}
		t.Run(name, func(t *testing.T) {
			sess := session.NewSession("agent", "key")
			for i := 0; i < 8; i++ {
				sess.Append(session.UserMessageEntry("question"))
				sess.Append(session.AssistantMessageEntry("answer"))
			}
			summary := &joinedSummaryProvider{entered: make(chan struct{}), cancelled: make(chan struct{}), release: make(chan struct{})}
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(summary.release) }) }
			defer release()
			manager := &compaction.Manager{PreserveTurns: 1, Summarizer: &compaction.Summarizer{Provider: summary, Model: "summary", Timeout: time.Second}}
			rt := &Runtime{Tools: usageNoopExecutor{}, LLM: &requestUsageProvider{total: 1}, Session: sess, Model: "generation", ContextWindow: 1000000, MaxTurns: 1, Compaction: manager}
			rt.AgentLoop.Hooks.OnStop = func(ctx context.Context, _ string) {
				manager.MaybeCompactAsyncContext(ctx, sess, compaction.ReasonManual)
			}
			ledger := &llm.UsageLedger{}
			ctx, cancel := context.WithCancel(llm.WithUsageObserver(context.Background(), ledger.Record))
			defer cancel()
			events, err := rt.Run(ctx, "go", nil)
			if err != nil {
				t.Fatal(err)
			}
			joined := make(chan struct{})
			var observed []AgentEvent
			go func() {
				for event := range events {
					observed = append(observed, event)
				}
				close(joined)
			}()
			select {
			case <-summary.entered:
			case <-time.After(3 * time.Second):
				t.Fatal("background summary not started")
			}
			if cancelRun {
				cancel()
				select {
				case <-summary.cancelled:
				case <-time.After(time.Second):
					t.Fatal("parent cancellation lost")
				}
			}
			select {
			case <-joined:
				t.Fatal("runtime stream closed before background joined")
			case <-time.After(20 * time.Millisecond):
			}
			release()
			select {
			case <-joined:
			case <-time.After(3 * time.Second):
				t.Fatal("runtime did not join background")
			}
			if manager.HasInFlight(sess) {
				t.Fatal("producer remains after runtime close")
			}
			var doneCount, abortedCount int
			for i, event := range observed {
				if event.Type == EventDone {
					doneCount++
					if i != len(observed)-1 || event.Usage == nil || event.Usage.InputTokens != 20070 {
						t.Fatalf("done must be last with final accounting: %+v", event)
					}
				}
				if event.Type == EventAborted {
					abortedCount++
				}
				if event.Type == EventError && !cancelRun {
					t.Fatalf("unexpected error: %v", event.Error)
				}
			}
			if cancelRun && (doneCount != 0 || abortedCount != 1) {
				t.Fatalf("cancel: done=%d aborted=%d", doneCount, abortedCount)
			}
			if !cancelRun && (doneCount != 1 || abortedCount != 0) {
				t.Fatalf("success: done=%d aborted=%d", doneCount, abortedCount)
			}
			records := ledger.Records()
			if len(records) != 2 || records[1].Category != llm.CallCompaction {
				t.Fatal("background observer context lost", records)
			}
			if cancelRun {
				if records[1].Status != "cancelled" {
					t.Fatal(records[1])
				}
			} else if records[1].Usage == nil || records[1].Usage.InputTokens != 70 {
				t.Fatal(records[1])
			}
		})
	}
}

func TestRuntimeReportsAlreadyFinishedCompactionError(t *testing.T) {
	sess := session.NewSession("agent", "key")
	for i := 0; i < 8; i++ {
		sess.Append(session.UserMessageEntry("question"))
		sess.Append(session.AssistantMessageEntry("answer"))
	}
	summary := &joinedSummaryProvider{entered: make(chan struct{}), cancelled: make(chan struct{}), release: make(chan struct{})}
	// A concurrent branch change makes the summary's source view stale. This is
	// a real CommitCompaction error, without making the session writer fail.
	summary.beforeReturn = func() { sess.Append(session.UserMessageEntry("new branch work")) }
	close(summary.release)
	manager := &compaction.Manager{PreserveTurns: 1, Summarizer: &compaction.Summarizer{Provider: summary, Model: "summary", Timeout: time.Second}}
	rt := &Runtime{Tools: usageNoopExecutor{}, LLM: &requestUsageProvider{total: 1}, Session: sess, Model: "generation", ContextWindow: 1000000, MaxTurns: 1, Compaction: manager}
	rt.AgentLoop.Hooks.OnStop = func(ctx context.Context, _ string) {
		<-manager.MaybeCompactAsyncContext(ctx, sess, compaction.ReasonManual)
	}
	events, err := rt.Run(context.Background(), "go", nil)
	if err != nil {
		t.Fatal(err)
	}
	var errorsSeen, doneSeen int
	for event := range events {
		if event.Type == EventError {
			if !errors.Is(event.Error, session.ErrSessionChanged) {
				t.Fatalf("expected stale-session error, got %v", event.Error)
			}
			errorsSeen++
		}
		if event.Type == EventDone {
			doneSeen++
		}
	}
	if errorsSeen != 1 || doneSeen != 0 {
		t.Fatalf("completed failed producer: errors=%d done=%d", errorsSeen, doneSeen)
	}
	if err := sess.Flush(); err != nil {
		t.Fatalf("error must originate in compaction, not persistence: %v", err)
	}
}
