package llm

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"sync"
)

type CallCategory string

const (
	CallGeneration CallCategory = "generation"
	CallRetry      CallCategory = "retry"
	CallCompaction CallCategory = "compaction"
)

// RequestUsage represents one attempted provider call. Nil Usage explicitly
// means unavailable, including potentially billable failures; it is not zero.
// InputTokens includes cached input. Cache fields are subsets, not additions.
type RequestUsage struct {
	ID       string       `json:"request_id"`
	Model    string       `json:"model"`
	Category CallCategory `json:"category"`
	Status   string       `json:"status"`
	Source   string       `json:"source"`
	Usage    *Usage       `json:"usage"`
}

type usageObserverKey struct{}

// WithUsageObserver chains observers. Callbacks can run concurrently and must
// not mutate their record. Each callback completes before its terminal event
// is forwarded to the consumer.
func WithUsageObserver(ctx context.Context, observer func(RequestUsage)) context.Context {
	previous, _ := ctx.Value(usageObserverKey{}).(func(RequestUsage))
	return context.WithValue(ctx, usageObserverKey{}, func(record RequestUsage) {
		if previous != nil {
			previous(record)
		}
		if observer != nil {
			observer(record)
		}
	})
}

// ObserveChat observes streaming and non-streaming adapters using one contract.
// Providers must terminate after EventDone/EventError and honour cancellation.
func ObserveChat(ctx context.Context, req ChatRequest, category CallCategory, call func(context.Context, ChatRequest) (<-chan ChatEvent, error)) (<-chan ChatEvent, error) {
	observer, _ := ctx.Value(usageObserverKey{}).(func(RequestUsage))
	guards, _ := ctx.Value(callAdmissionKey{}).([]CallAdmission)
	if observer == nil && len(guards) == 0 {
		return call(ctx, req)
	}
	record := RequestUsage{ID: rand.Text(), Model: req.Model, Category: category, Source: "unavailable"}
	settle, err := admitCall(ctx, req, record, guards)
	if err != nil {
		return nil, fmt.Errorf("provider admission: %w", err)
	}
	publish := func(status string, usage *Usage) error {
		record.Status = status
		if usage != nil {
			copy := *usage
			record.Usage = &copy
			record.Source = "reported"
		}
		err := settle(record)
		if observer != nil {
			observer(record)
		}
		return err
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, context.Cause(ctx), publish("not_dispatched", nil))
	}
	source, err := call(ctx, req)
	if err != nil {
		status := "failed"
		if ctx.Err() != nil {
			status = "cancelled"
		}
		return nil, errors.Join(err, context.Cause(ctx), publish(status, nil))
	}
	if source == nil {
		return nil, errors.Join(fmt.Errorf("provider returned a nil event stream"), publish("failed", nil))
	}
	out := make(chan ChatEvent)
	go func() {
		defer close(out)
		var usage *Usage
		for {
			select {
			case <-ctx.Done():
				publish("cancelled", usage)
				return
			case event, ok := <-source:
				if !ok {
					settlementErr := publish("failed", usage)
					select {
					case out <- ChatEvent{Type: EventError, Error: errors.Join(fmt.Errorf("provider stream closed without terminal event"), settlementErr, context.Cause(ctx))}:
					case <-ctx.Done():
					}
					return
				}
				if event.Usage != nil {
					copy := *event.Usage
					usage = &copy
				}
				if event.Type == EventError && ctx.Err() != nil {
					event.Error = errors.Join(event.Error, context.Cause(ctx))
				}
				terminal := event.Type == EventDone || event.Type == EventError
				if terminal {
					status := "completed"
					if event.Type == EventError {
						status = "failed"
					}
					if err := publish(status, usage); err != nil {
						event.Type = EventError
						event.Error = errors.Join(event.Error, fmt.Errorf("provider settlement: %w", err))
					}
				}
				select {
				case out <- event:
				case <-ctx.Done():
					if !terminal {
						publish("cancelled", usage)
					}
					return
				}
				if terminal {
					return
				}
			}
		}
	}()
	return out, nil
}

// UsageLedger keeps immutable request records and a reported-only total.
// Unknown requests remain visible through Records rather than being priced as
// free. Create a ledger per bounded operation, not a process-global history.
type UsageLedger struct {
	mu      sync.Mutex
	records []RequestUsage
	total   *Usage
}

func (l *UsageLedger) Record(record RequestUsage) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if record.Usage != nil {
		u := *record.Usage
		record.Usage = &u
		if l.total == nil {
			l.total = &Usage{}
		}
		l.total.InputTokens += u.InputTokens
		l.total.OutputTokens += u.OutputTokens
		l.total.CacheCreationInputTokens += u.CacheCreationInputTokens
		l.total.CacheReadInputTokens += u.CacheReadInputTokens
	}
	l.records = append(l.records, record)
}
func (l *UsageLedger) Total() *Usage {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.total == nil {
		return nil
	}
	u := *l.total
	return &u
}
func (l *UsageLedger) Records() []RequestUsage {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := append([]RequestUsage(nil), l.records...)
	for i := range out {
		if out[i].Usage != nil {
			u := *out[i].Usage
			out[i].Usage = &u
		}
	}
	return out
}
