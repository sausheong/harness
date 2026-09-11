package runtime

import (
	"context"
	"errors"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"testing"
)

type outputLimitProvider struct {
	llmtest.Base
	limits   []int
	fallback bool
}

func (p *outputLimitProvider) ChatStream(_ context.Context, r llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.limits = append(p.limits, r.MaxTokens)
	if p.fallback && len(p.limits) == 1 {
		return nil, errors.New("529 overloaded")
	}
	ch := make(chan llm.ChatEvent, 2)
	ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Answer"}
	ch <- llm.ChatEvent{Type: llm.EventDone}
	close(ch)
	return ch, nil
}
func TestGenerationOutputLimitReachesRequestsAndFallback(t *testing.T) {
	for _, single := range []bool{false, true} {
		for _, limit := range []int{0, 123, 32768, -1} {
			p := &outputLimitProvider{fallback: true}
			r := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("a", "b"), Model: "primary", FallbackModel: "fallback", MaxTurns: 1, MaxOutputTokens: limit}
			var err error
			if single {
				var result TurnResult
				result, err = r.RunTurn(context.Background(), "hello", nil, nil)
				if err == nil {
					err = result.Err
				}
			} else {
				var events <-chan AgentEvent
				events, err = r.Run(context.Background(), "hello", nil)
				if err == nil {
					for e := range events {
						if e.Error != nil {
							err = e.Error
						}
					}
				}
			}
			if limit < 0 {
				if err == nil || len(p.limits) != 0 {
					t.Fatal("negative limit dispatched", err, p.limits)
				}
				continue
			}
			if err != nil {
				t.Fatal(err)
			}
			want := limit
			if want == 0 {
				want = 8192
			}
			if len(p.limits) != 2 {
				t.Fatal("missing fallback", p.limits)
			}
			for _, got := range p.limits {
				if got != want {
					t.Fatalf("single=%v limit=%d got=%d", single, limit, got)
				}
			}
		}
	}
}
