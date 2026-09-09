package llm

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestAdmissionDenialUnwindsWithoutDispatch(t *testing.T) {
	denied := errors.New("budget exhausted")
	var records []RequestUsage
	ctx := WithCallAdmission(context.Background(), func(_ context.Context, _ ChatRequest, category CallCategory, id string) (func(RequestUsage) error, error) {
		if category != CallRetry || id == "" {
			t.Fatal("missing attempt identity")
		}
		return func(r RequestUsage) error { records = append(records, r); return nil }, nil
	})
	ctx = WithCallAdmission(ctx, func(context.Context, ChatRequest, CallCategory, string) (func(RequestUsage) error, error) {
		return nil, denied
	})
	_, err := ObserveChat(ctx, ChatRequest{Model: "m"}, CallRetry, func(context.Context, ChatRequest) (<-chan ChatEvent, error) {
		t.Fatal("denied provider dispatched")
		return nil, nil
	})
	if !errors.Is(err, denied) || len(records) != 1 || records[0].Status != "not_dispatched" || records[0].Usage != nil {
		t.Fatalf("err %v records %+v", err, records)
	}
}

func TestAdmissionSettlementPrecedesTerminalAndFailureIsVisible(t *testing.T) {
	failure := errors.New("ledger persistence failed")
	for _, category := range []CallCategory{CallGeneration, CallRetry, CallCompaction} {
		t.Run(string(category), func(t *testing.T) {
			settled := false
			ctx := WithCallAdmission(context.Background(), func(_ context.Context, _ ChatRequest, actual CallCategory, id string) (func(RequestUsage) error, error) {
				if actual != category {
					t.Fatal("wrong category")
				}
				return func(r RequestUsage) error {
					if r.ID != id || r.Usage == nil || r.Usage.OutputTokens != 7 || r.Status != "completed" {
						t.Errorf("record %+v", r)
					}
					settled = true
					return failure
				}, nil
			})
			ch, err := ObserveChat(ctx, ChatRequest{Model: "m"}, category, func(context.Context, ChatRequest) (<-chan ChatEvent, error) {
				ch := make(chan ChatEvent, 1)
				ch <- ChatEvent{Type: EventDone, Usage: &Usage{OutputTokens: 7}}
				close(ch)
				return ch, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for event := range ch {
				count++
				if !settled || event.Type != EventError || !errors.Is(event.Error, failure) {
					t.Fatalf("event %+v settled %v", event, settled)
				}
			}
			if count != 1 {
				t.Fatal("wrong terminal count")
			}
		})
	}
}

func TestAdmissionConcurrentReservations(t *testing.T) {
	var mu sync.Mutex
	reserved, settlements := 0, 0
	guard := func(context.Context, ChatRequest, CallCategory, string) (func(RequestUsage) error, error) {
		mu.Lock()
		defer mu.Unlock()
		if reserved == 1 {
			return nil, errors.New("budget exhausted")
		}
		reserved++
		return func(r RequestUsage) error { mu.Lock(); defer mu.Unlock(); reserved--; settlements++; return nil }, nil
	}
	ctx := WithCallAdmission(context.Background(), guard)
	source := make(chan ChatEvent)
	first, err := ObserveChat(ctx, ChatRequest{}, CallGeneration, func(context.Context, ChatRequest) (<-chan ChatEvent, error) { return source, nil })
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := ObserveChat(ctx, ChatRequest{}, CallCompaction, func(context.Context, ChatRequest) (<-chan ChatEvent, error) {
				t.Error("oversubscribed provider dispatched")
				return nil, nil
			})
			if err == nil {
				t.Error("oversubscription accepted")
			}
		}()
	}
	wg.Wait()
	source <- ChatEvent{Type: EventDone}
	close(source)
	for range first {
	}
	mu.Lock()
	defer mu.Unlock()
	if reserved != 0 || settlements != 1 {
		t.Fatalf("reserved %d settlements %d", reserved, settlements)
	}
}

func TestAdmissionCancellationSettlesUnknownOnce(t *testing.T) {
	settled := make(chan RequestUsage, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx = WithCallAdmission(ctx, func(context.Context, ChatRequest, CallCategory, string) (func(RequestUsage) error, error) {
		return func(r RequestUsage) error { settled <- r; return nil }, nil
	})
	ch, err := ObserveChat(ctx, ChatRequest{}, CallGeneration, func(context.Context, ChatRequest) (<-chan ChatEvent, error) { return make(chan ChatEvent), nil })
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	for range ch {
	}
	r := <-settled
	if r.Status != "cancelled" || r.Usage != nil {
		t.Fatalf("record %+v", r)
	}
	select {
	case <-settled:
		t.Fatal("double settlement")
	default:
	}
}
