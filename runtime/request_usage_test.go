package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/stretchr/testify/require"
)

type requestUsageProvider struct {
	llmtest.Base
	calls       int
	total       int
	failFirst   bool
	missingLast bool
}

func (p *requestUsageProvider) ChatStream(_ context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.calls++
	if p.failFirst && p.calls == 1 {
		return nil, errors.New("529 overloaded")
	}
	ch := make(chan llm.ChatEvent, 2)
	if p.calls < p.total {
		ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: req.Model, Name: "noop", Input: json.RawMessage(`{}`)}}
	}
	usage := &llm.Usage{InputTokens: 20000, OutputTokens: 100, CacheReadInputTokens: 15000}
	if p.missingLast && p.calls == p.total {
		usage = nil
	}
	ch <- llm.ChatEvent{Type: llm.EventDone, Usage: usage}
	close(ch)
	return ch, nil
}
func TestRequestUsageSeparatesLatestFromCumulative(t *testing.T) {
	p := &requestUsageProvider{total: 10}
	r := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("a", "k"), Model: "primary", MaxTurns: 10}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	var records []llm.RequestUsage
	var total *llm.Usage
	for e := range events {
		if e.Type == EventRequestUsage {
			records = append(records, *e.RequestUsage)
		}
		if e.Type == EventDone {
			total = e.Usage
		}
	}
	require.Len(t, records, 10)
	require.Equal(t, 20000, records[9].Usage.InputTokens)
	require.Equal(t, 200000, total.InputTokens)
	for _, r := range records {
		require.Equal(t, llm.CallGeneration, r.Category)
		require.Equal(t, "reported", r.Source)
	}
}
func TestRequestUsageRetainsUnknownFailedAttemptAndModelChange(t *testing.T) {
	p := &requestUsageProvider{total: 2, failFirst: true, missingLast: true}
	r := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("a", "k"), Model: "primary", FallbackModel: "fallback", MaxTurns: 2}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	var records []llm.RequestUsage
	for e := range events {
		if e.Type == EventRequestUsage {
			records = append(records, *e.RequestUsage)
		}
	}
	require.Len(t, records, 2)
	require.Equal(t, "failed", records[0].Status)
	require.Nil(t, records[0].Usage)
	require.Equal(t, "primary", records[0].Model)
	require.Equal(t, "fallback", records[1].Model)
	require.Equal(t, llm.CallRetry, records[1].Category)
	require.Equal(t, "unavailable", records[1].Source)
}

type refusalUsageProvider struct {
	llmtest.Base
	calls int
}

func (p *refusalUsageProvider) ChatStream(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.calls++
	ch := make(chan llm.ChatEvent, 1)
	e := llm.ChatEvent{Type: llm.EventDone, Usage: &llm.Usage{InputTokens: 10000}}
	if p.calls == 1 {
		e.StopReason = llm.StopReasonRefusal
		e.Usage.InputTokens = 20000
	}
	ch <- e
	close(ch)
	return ch, nil
}
func TestRunTurnUsageIncludesRefusedAttempt(t *testing.T) {
	p := &refusalUsageProvider{}
	r := &Runtime{LLM: p, Tools: usageNoopExecutor{}, Session: session.NewSession("a", "k"), Model: "primary", FallbackModel: "fallback"}
	var records []llm.RequestUsage
	result, err := r.RunTurn(context.Background(), "go", nil, func(e AgentEvent) {
		if e.RequestUsage != nil {
			records = append(records, *e.RequestUsage)
		}
	})
	require.NoError(t, err)
	require.NoError(t, result.Err)
	require.Equal(t, 30000, result.Usage.InputTokens)
	require.Len(t, records, 2)
	require.Equal(t, llm.CallRetry, records[1].Category)
}
