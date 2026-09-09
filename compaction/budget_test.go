package compaction

import (
	"context"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"strings"
	"testing"
)

type budgetProvider struct {
	fakeProvider
	requests []llm.ChatRequest
}

func (p *budgetProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.requests = append(p.requests, req)
	return p.fakeProvider.ChatStream(ctx, req)
}
func TestSummariserOutputBudgetAndResponseBound(t *testing.T) {
	entries := []session.SessionEntry{session.UserMessageEntry("task")}
	for _, limit := range []int{0, 1, 2048, 32768, -1, 32769} {
		provider := &budgetProvider{fakeProvider: fakeProvider{text: "summary"}}
		s := &Summarizer{Provider: provider, MaxOutputTokens: limit}
		_, err := s.Summarize(context.Background(), entries, "")
		if limit < 0 || limit > 32768 {
			if err == nil || len(provider.requests) != 0 {
				t.Fatal("invalid budget dispatched")
			}
			continue
		}
		want := limit
		if want == 0 {
			want = 4096
		}
		if err != nil || len(provider.requests) != 1 || provider.requests[0].MaxTokens != want {
			t.Fatalf("budget %d: %v %+v", limit, err, provider.requests)
		}
	}
	provider := &budgetProvider{fakeProvider: fakeProvider{text: strings.Repeat("x", (256<<10)+1)}}
	s := &Summarizer{Provider: provider}
	out, err := s.Summarize(context.Background(), entries, "")
	if err == nil || out != "" || len(provider.requests) != 1 {
		t.Fatal("oversized summary accepted or retried")
	}
}
