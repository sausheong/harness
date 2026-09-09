package runtime

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestMalformedToolCallIDsCannotExecuteTwice(t *testing.T) {
	for _, mode := range []string{"streaming", "sequential", "runturn"} {
		for _, kind := range []string{"duplicate", "empty"} {
			t.Run(mode+"_"+kind, func(t *testing.T) {
				t.Setenv("HARNESS_STREAMING_TOOLS", "0")
				if mode == "streaming" {
					t.Setenv("HARNESS_STREAMING_TOOLS", "1")
				}
				tc := llm.ToolCall{ID: "same", Name: "read", Input: json.RawMessage(`{}`)}
				if kind == "empty" {
					tc.ID = ""
				}
				scripted := []llm.ChatEvent{{Type: llm.EventToolCallStart, ToolCall: &tc}, {Type: llm.EventToolCallDone, ToolCall: &tc}}
				if kind == "duplicate" {
					scripted = append(scripted, llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &tc})
				}
				scripted = append(scripted, llm.ChatEvent{Type: llm.EventDone})
				exec := newTimedExecutor()
				exec.addTool("read")
				exec.markSafe("read")
				rt := &Runtime{Session: session.NewSession("a", "ids"), LLM: &mockLLMProvider{events: scripted}, Tools: exec, Model: "fixture", MaxTurns: 1}
				var terminal error
				if mode == "runturn" {
					result, err := rt.RunTurn(context.Background(), "read", nil, nil)
					terminal = err
					if terminal == nil {
						terminal = result.Err
					}
				} else {
					events, err := rt.Run(context.Background(), "read", nil)
					if err != nil {
						t.Fatal(err)
					}
					for ev := range events {
						if ev.Type == EventError {
							terminal = ev.Error
						}
					}
				}
				if terminal == nil || !strings.Contains(terminal.Error(), "tool call ID") {
					t.Fatalf("missing identity error: %v", terminal)
				}
				allowed := 0
				if mode == "streaming" && kind == "duplicate" {
					allowed = 1
				}
				if exec.callCount() > allowed {
					t.Fatalf("executed %d calls, maximum %d", exec.callCount(), allowed)
				}
				seen := map[string]bool{}
				for _, e := range rt.Session.View() {
					if e.Type == session.EntryTypeToolCall {
						var d session.ToolCallData
						if err := json.Unmarshal(e.Data, &d); err != nil {
							t.Fatal(err)
						}
						if d.ID == "" || seen[d.ID] {
							t.Fatalf("invalid persisted identity %q", d.ID)
						}
						seen[d.ID] = true
					}
				}
			})
		}
	}
}
