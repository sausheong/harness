package compaction

import (
	"context"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/stretchr/testify/require"
)

type usageSummaryProvider struct{ llmtest.Base }

func (usageSummaryProvider) ChatStream(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 2)
	ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Summary of prior work."}
	ch <- llm.ChatEvent{Type: llm.EventDone, Usage: &llm.Usage{InputTokens: 1200, OutputTokens: 20}}
	close(ch)
	return ch, nil
}
func TestCompactionUsageRecordsAndBackgroundObserver(t *testing.T) {
	ledger := &llm.UsageLedger{}
	manager := &Manager{Summarizer: &Summarizer{Provider: usageSummaryProvider{}, Model: "summary-model"}, OnUsage: ledger.Record, PreserveTurns: 4}
	result, err := manager.MaybeCompact(context.Background(), longSession(), ReasonManual, "")
	require.NoError(t, err)
	require.True(t, result.Compacted)
	require.Len(t, result.Requests, 1)
	require.Equal(t, llm.CallCompaction, result.Requests[0].Category)
	require.Equal(t, "summary-model", result.Requests[0].Model)
	<-manager.MaybeCompactAsync(longSession(), ReasonPreventive)
	require.Len(t, ledger.Records(), 2)
	require.Equal(t, 2400, ledger.Total().InputTokens)
}
