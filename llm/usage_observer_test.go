package llm

import (
	"context"
	"errors"
	"testing"
)

func TestUsageObserverFailedRequestCanCarryReportedUsage(t *testing.T) {
	ledger := &UsageLedger{}
	ctx := WithUsageObserver(context.Background(), ledger.Record)
	ch, err := ObserveChat(ctx, ChatRequest{Model: "m"}, CallRetry, func(context.Context, ChatRequest) (<-chan ChatEvent, error) {
		ch := make(chan ChatEvent, 1)
		ch <- ChatEvent{Type: EventError, Error: errors.New("failed"), Usage: &Usage{InputTokens: 42}}
		close(ch)
		return ch, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	records := ledger.Records()
	if len(records) != 1 || records[0].Status != "failed" || records[0].Source != "reported" || ledger.Total().InputTokens != 42 {
		t.Fatalf("records %+v total %+v", records, ledger.Total())
	}
	records[0].Usage.InputTokens = 100
	if ledger.Total().InputTokens != 42 {
		t.Fatal("records alias total")
	}
}
func TestUsageObserverReportsMissingTerminal(t *testing.T) {
	ledger := &UsageLedger{}
	ctx := WithUsageObserver(context.Background(), ledger.Record)
	ch, err := ObserveChat(ctx, ChatRequest{Model: "m"}, CallGeneration, func(context.Context, ChatRequest) (<-chan ChatEvent, error) {
		ch := make(chan ChatEvent)
		close(ch)
		return ch, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var event ChatEvent
	for e := range ch {
		event = e
	}
	if event.Type != EventError || len(ledger.Records()) != 1 || ledger.Total() != nil {
		t.Fatal("missing terminal treated as success")
	}
}
