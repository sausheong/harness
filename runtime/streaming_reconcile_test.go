package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

type mixedToolExecutor struct {
	*timedExecutor
	running        chan struct{}
	pendingInvoked atomic.Bool
}

func (e *mixedToolExecutor) Execute(ctx context.Context, name string, in json.RawMessage) (tool.ToolResult, error) {
	if name == "pending" {
		e.pendingInvoked.Store(true)
	}
	if name == "running" {
		close(e.running)
		<-ctx.Done()
		return tool.ToolResult{}, ctx.Err()
	}
	return tool.ToolResult{Output: "completed evidence"}, nil
}

type mixedToolProvider struct {
	recordingProvider
	order              []string
	running, completed <-chan struct{}
	cause              error
}

func (p *mixedToolProvider) ChatStream(ctx context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	out := make(chan llm.ChatEvent)
	go func() {
		defer close(out)
		for _, name := range p.order {
			tc := llm.ToolCall{ID: name, Name: name, Input: json.RawMessage(`{}`)}
			for _, typ := range []llm.EventType{llm.EventToolCallStart, llm.EventToolCallDone} {
				select {
				case out <- llm.ChatEvent{Type: typ, ToolCall: &tc}:
				case <-ctx.Done():
					return
				}
			}
		}
		select {
		case <-p.running:
		case <-ctx.Done():
			return
		}
		select {
		case <-p.completed:
		case <-ctx.Done():
			return
		}
		select {
		case out <- llm.ChatEvent{Type: llm.EventError, Error: p.cause}:
		case <-ctx.Done():
		}
	}()
	return out, nil
}
func TestStreamingReconcileMixedResults(t *testing.T) {
	for _, order := range [][]string{{"running", "completed", "pending"}, {"completed", "running", "pending"}, {"completed"}} {
		t.Run(order[0], func(t *testing.T) {
			t.Setenv("HARNESS_STREAMING_TOOLS", "1")
			store := session.NewStore(t.TempDir())
			if err := store.Create("a", "mixed"); err != nil {
				t.Fatal(err)
			}
			sess, err := store.LoadExclusive("a", "mixed")
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if sess != nil {
					sess.Close()
				}
			}()
			base := newTimedExecutor()
			for _, name := range order {
				base.addTool(name)
				if name != "pending" {
					base.markSafe(name)
				}
			}
			exec := &mixedToolExecutor{timedExecutor: base, running: make(chan struct{})}
			if len(order) == 1 {
				close(exec.running)
			}
			completed := make(chan struct{})
			cause := errors.New("interrupted mixed stream")
			p := &mixedToolProvider{order: order, running: exec.running, completed: completed, cause: cause}
			rt := &Runtime{Session: sess, LLM: p, Tools: exec, Model: "fixture", MaxTurns: 1}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events, err := rt.Run(ctx, "mixed", nil)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan bool, 1)
			go func() {
				found := false
				for ev := range events {
					if ev.Type == EventToolResult && ev.ToolCall.ID == "completed" {
						close(completed)
					}
					if errors.Is(ev.Error, cause) {
						found = true
					}
				}
				done <- found
			}()
			select {
			case found := <-done:
				if !found {
					t.Error("original error lost")
				}
			case <-time.After(3 * time.Second):
				t.Fatal("result reconciliation failed to join")
			}
			if exec.pendingInvoked.Load() {
				t.Error("pending tool was executed after cancellation")
			}
			if err = sess.Close(); err != nil {
				t.Fatal(err)
			}
			sess = nil
			sess, err = store.LoadExclusive("a", "mixed")
			if err != nil {
				t.Fatal(err)
			}
			calls := map[string]int{}
			results := map[string][]session.ToolResultData{}
			for _, entry := range sess.View() {
				if entry.Type == session.EntryTypeToolCall {
					var d session.ToolCallData
					if err = json.Unmarshal(entry.Data, &d); err != nil {
						t.Fatal(err)
					}
					calls[d.ID]++
				}
				if entry.Type == session.EntryTypeToolResult {
					var d session.ToolResultData
					if err = json.Unmarshal(entry.Data, &d); err != nil {
						t.Fatal(err)
					}
					results[d.ToolCallID] = append(results[d.ToolCallID], d)
				}
			}
			for _, name := range order {
				if calls[name] != 1 || len(results[name]) != 1 {
					t.Errorf("%s pairs: %d/%d", name, calls[name], len(results[name]))
					continue
				}
				d := results[name][0]
				if name == "completed" {
					if d.Aborted || d.IsError || d.Output != "completed evidence" {
						t.Errorf("completed result changed: %+v", d)
					}
				} else if !d.Aborted || !d.IsError || d.Error != cause.Error() {
					t.Errorf("bad aborted result: %+v", d)
				}
			}
		})
	}
}
