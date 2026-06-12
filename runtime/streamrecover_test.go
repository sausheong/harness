package runtime

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// streamErrThenOKProvider emits EventError (no token) on the first stream
// call, then clean text on the second.
type streamErrThenOKProvider struct {
	llmtest.Base
	calls     atomic.Int64
	streamErr error
	finalText string
}

func (p *streamErrThenOKProvider) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	n := p.calls.Add(1)
	ch := make(chan llm.ChatEvent, 4)
	if n == 1 {
		ch <- llm.ChatEvent{Type: llm.EventError, Error: p.streamErr}
	} else {
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: p.finalText}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}
	close(ch)
	return ch, nil
}

func TestStreamErrorRetryableEngagesFallback(t *testing.T) {
	prov := &streamErrThenOKProvider{
		streamErr: errors.New("429 rate limit exceeded"),
		finalText: "RECOVERED_VIA_FALLBACK",
	}
	rt := &Runtime{
		LLM:           prov,
		Tools:         tool.NewRegistry(),
		Session:       session.NewSession("a", "k"),
		AgentID:       "a",
		Model:         "claude-opus-4-8",
		FallbackModel: "claude-sonnet-4-5",
		Provider:      "anthropic",
		MaxTurns:      2,
		Workspace:     t.TempDir(),
	}
	out, err := rt.RunSync(context.Background(), "do it", nil)
	require.NoError(t, err)
	assert.Contains(t, out, "RECOVERED_VIA_FALLBACK")
	assert.EqualValues(t, 2, prov.calls.Load(), "must retry once after the pre-first-token 429")
}

func TestStreamErrorNonRetryableStillAborts(t *testing.T) {
	prov := &streamErrThenOKProvider{
		streamErr: errors.New("400 invalid input"),
		finalText: "SHOULD_NOT_BE_REACHED",
	}
	rt := &Runtime{
		LLM:       prov,
		Tools:     tool.NewRegistry(),
		Session:   session.NewSession("a", "k"),
		AgentID:   "a",
		Model:     "claude-opus-4-8",
		Provider:  "anthropic",
		MaxTurns:  2,
		Workspace: t.TempDir(),
	}
	_, err := rt.RunSync(context.Background(), "do it", nil)
	require.Error(t, err)
	assert.EqualValues(t, 1, prov.calls.Load(), "non-retryable pre-first-token error must abort, not retry")
}
