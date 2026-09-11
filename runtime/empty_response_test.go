package runtime

import (
	"context"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"strings"
	"testing"
)

type emptyResponseProvider struct {
	llmtest.Base
	responses [][]llm.ChatEvent
	calls     int
	limits    []int
}

func (p *emptyResponseProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.limits = append(p.limits, req.MaxTokens)
	i := min(p.calls, len(p.responses)-1)
	p.calls++
	ch := make(chan llm.ChatEvent, len(p.responses[i]))
	for _, e := range p.responses[i] {
		ch <- e
	}
	close(ch)
	return ch, nil
}
func TestEmptyAndTruncatedResponses(t *testing.T) {
	done := llm.ChatEvent{Type: llm.EventDone, StopReason: "stop"}
	text := llm.ChatEvent{Type: llm.EventTextDelta, Text: "Answer"}
	for _, tc := range []struct {
		name      string
		responses [][]llm.ChatEvent
		calls     int
		want      string
	}{
		{"empty", [][]llm.ChatEvent{{done}}, 2, "no answer"},
		{"recovery", [][]llm.ChatEvent{{done}, {text, done}}, 2, ""},
		{"normal", [][]llm.ChatEvent{{text, done}}, 1, ""},
		{"length", [][]llm.ChatEvent{{{Type: llm.EventDone, StopReason: "length", Usage: &llm.Usage{OutputTokens: 2048}}}}, 1, "output limit"},
		{"partial", [][]llm.ChatEvent{{text, {Type: llm.EventDone, StopReason: "max_tokens"}}}, 1, "output limit"},
		{"whitespace", [][]llm.ChatEvent{{{Type: llm.EventTextDelta, Text: " \n"}, done}}, 1, "no answer"},
		{"thinking", [][]llm.ChatEvent{{{Type: llm.EventThinkingBlock, ThinkingBlock: &llm.ThinkingBlock{Thinking: "opaque"}}, done}}, 2, "no answer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &emptyResponseProvider{responses: tc.responses}
			rt := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("test", "test"), Model: "fixture", MaxOutputTokens: 2048, MaxTurns: 2}
			reason := ""
			rt.AgentLoop.Hooks.OnStop = func(_ context.Context, r string) { reason = r }
			events, err := rt.Run(context.Background(), "question", nil)
			if err != nil {
				t.Fatal(err)
			}
			var failure error
			for e := range events {
				if e.Type == EventError {
					failure = e.Error
				}
			}
			if tc.want == "" {
				if failure != nil || reason != "completed" {
					t.Fatalf("%v %s", failure, reason)
				}
			} else if failure == nil || !strings.Contains(failure.Error(), tc.want) || reason == "completed" {
				t.Fatalf("%v %s", failure, reason)
			}
			if p.calls != tc.calls {
				t.Fatalf("calls %d", p.calls)
			}
			for _, limit := range p.limits {
				if limit != 2048 {
					t.Fatalf("changed limit %d", limit)
				}
			}
		})
	}
}

func TestSingleTurnRejectsEmptyAndTruncatedResponses(t *testing.T) {
	for _, reason := range []string{"stop", "length", "max_tokens"} {
		t.Run(reason, func(t *testing.T) {
			p := &emptyResponseProvider{responses: [][]llm.ChatEvent{{{Type: llm.EventDone, StopReason: reason}}}}
			rt := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("test", "single"), Model: "fixture", MaxOutputTokens: 2048}
			result, err := rt.RunTurn(context.Background(), "question", nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if result.Err == nil || result.StopReason == "completed" || p.calls != 1 {
				t.Fatalf("%+v calls=%d", result, p.calls)
			}
		})
	}
}
