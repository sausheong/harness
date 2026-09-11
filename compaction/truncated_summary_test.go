package compaction

import (
	"context"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"strings"
	"testing"
)

type truncatedSummaryProvider struct{ llmtest.Base }

func (truncatedSummaryProvider) ChatStream(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 2)
	ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Incomplete summary"}
	ch <- llm.ChatEvent{Type: llm.EventDone, StopReason: "length"}
	close(ch)
	return ch, nil
}
func TestRejectTruncatedSummary(t *testing.T) {
	s := Summarizer{Provider: truncatedSummaryProvider{}, Model: "fixture"}
	out, err := s.callOnce(context.Background(), "history", "")
	if err == nil || out != "" || !strings.Contains(err.Error(), "output limit") {
		t.Fatalf("%q %v", out, err)
	}
}
