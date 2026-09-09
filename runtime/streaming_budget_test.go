package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

type interruptedToolProvider struct {
	recordingProvider
	started      <-chan struct{}
	cause        error
	usage        *llm.Usage
	waitDeadline bool
}

func (p *interruptedToolProvider) ChatStream(ctx context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	out := make(chan llm.ChatEvent)
	go func() {
		defer close(out)
		tc := llm.ToolCall{ID: "running-read", Name: "read", Input: json.RawMessage(`{}`)}
		for _, typ := range []llm.EventType{llm.EventToolCallStart, llm.EventToolCallDone} {
			select {
			case out <- llm.ChatEvent{Type: typ, ToolCall: &tc}:
			case <-ctx.Done():
				return
			}
		}
		select {
		case <-p.started:
		case <-ctx.Done():
			return
		}
		// The terminal error is deliberately emitted only AFTER the concurrent
		// tool has started, so a sequential implementation cannot pass this test.
		if p.waitDeadline {
			<-ctx.Done()
			return
		}
		terminal := llm.ChatEvent{Type: llm.EventError, Error: p.cause}
		if p.usage != nil {
			terminal = llm.ChatEvent{Type: llm.EventDone, Usage: p.usage}
		}
		select {
		case out <- terminal:
		case <-ctx.Done():
		}
	}()
	return out, nil
}
func TestStreamingBudgetExhaustionJoinsAndPersistsStartedTool(t *testing.T) {
	for _, kind := range []string{"deadline", "token_settlement", "cost_settlement", "provider_error"} {
		cause := budget.ErrTimeExhausted
		switch kind {
		case "token_settlement":
			cause = budget.ErrExhausted
		case "cost_settlement":
			cause = budget.ErrCostExhausted
		case "provider_error":
			cause = errors.New("provider failed after tool start")
		}
		t.Run(kind, func(t *testing.T) {
			t.Setenv("HARNESS_STREAMING_TOOLS", "1")
			store := session.NewStore(t.TempDir())
			if err := store.Create("a", "stream-budget"); err != nil {
				t.Fatal(err)
			}
			sess, err := store.LoadExclusive("a", "stream-budget")
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if sess != nil {
					sess.Close()
				}
			}()
			started := make(chan struct{})
			exec := newTimedExecutor()
			exec.addTool("read")
			exec.markSafe("read")
			exec.blockUntil = make(chan struct{})
			exec.onExecuteStart = func() { close(started) }
			p := &interruptedToolProvider{started: started, cause: cause}
			rt := &Runtime{Session: sess, LLM: p, Tools: exec, Model: "fixture", MaxTurns: 1}
			switch kind {
			case "deadline":
				p.waitDeadline = true
				if err = rt.DecideTimeBudget(context.Background(), time.Second); err != nil {
					t.Fatal(err)
				}
			case "token_settlement":
				p.usage = &llm.Usage{InputTokens: 2000000, OutputTokens: 1}
				if err = rt.DecideTokenBudget(context.Background(), 100000); err != nil {
					t.Fatal(err)
				}
			case "cost_settlement":
				p.usage = &llm.Usage{InputTokens: 2000000, OutputTokens: 1}
				rt.Route = llm.CallRoute{Provider: "local", Destination: "fixture"}
				if err = rt.SetCostPrices(context.Background(), []budget.PriceSnapshot{runtimeFixturePrice(rt.Route, rt.Model)}); err != nil {
					t.Fatal(err)
				}
				if err = rt.DecideCostBudget(context.Background(), "USD", 100000, true); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events, err := rt.Run(ctx, "read", nil)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan bool, 1)
			go func() {
				found := false
				for ev := range events {
					if errors.Is(ev.Error, cause) {
						found = true
					}
				}
				done <- found
			}()
			select {
			case found := <-done:
				if !found {
					t.Fatal("budget cause lost")
				}
			case <-time.After(2 * time.Second):
				cancel()
				<-done
				t.Fatal("stream failure did not stop and join running tool")
			}
			if err = sess.Close(); err != nil {
				t.Fatal(err)
			}
			sess = nil
			sess, err = store.LoadExclusive("a", "stream-budget")
			if err != nil {
				t.Fatal(err)
			}
			if kind == "token_settlement" {
				ledger, e := budget.OpenTokenLedger(sess)
				if e != nil {
					t.Fatal(e)
				}
				state, e := ledger.State()
				if e != nil || !state.Exhausted || state.Committed != 2000001 {
					t.Fatalf("token ledger %+v %v", state, e)
				}
			}
			if kind == "cost_settlement" {
				ledger, e := budget.OpenCostLedger(sess)
				if e != nil {
					t.Fatal(e)
				}
				state, e := ledger.State()
				if e != nil || !state.Exhausted || state.CommittedNano != 2000001 {
					t.Fatalf("cost ledger %+v %v", state, e)
				}
			}
			calls, results := 0, 0
			for _, e := range sess.View() {
				if e.Type == session.EntryTypeToolCall {
					calls++
				}
				if e.Type == session.EntryTypeToolResult {
					results++
					var d session.ToolResultData
					if err = json.Unmarshal(e.Data, &d); err != nil {
						t.Fatal(err)
					}
					if d.ToolCallID != "running-read" || !d.Aborted || !d.IsError || !strings.Contains(d.Error, cause.Error()) {
						t.Fatalf("bad result: %+v", d)
					}
				}
			}
			if calls != 1 || results != 1 {
				t.Fatalf("lost pair: calls=%d results=%d", calls, results)
			}
		})
	}
}
