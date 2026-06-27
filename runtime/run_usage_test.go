package runtime

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// twoTurnUsageLLM: turn 1 emits a noop tool call + EventDone{Usage:u1};
// turn 2 (after the tool result) emits text + EventDone{Usage:u2} and stops.
type twoTurnUsageLLM struct {
	llmtest.Base
	u1, u2 *llm.Usage
	calls  int
}

func (f *twoTurnUsageLLM) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 8)
	first := f.calls == 0
	f.calls++
	go func() {
		defer close(ch)
		if first {
			tc := llm.ToolCall{ID: "tc_0", Name: "noop", Input: json.RawMessage(`{}`)}
			ch <- llm.ChatEvent{Type: llm.EventToolCallStart, ToolCall: &tc}
			ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &tc}
			ch <- llm.ChatEvent{Type: llm.EventDone, Usage: f.u1}
			return
		}
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "done"}
		ch <- llm.ChatEvent{Type: llm.EventDone, Usage: f.u2}
	}()
	return ch, nil
}

// usageNoopExecutor returns "ok" for a "noop" tool. (Distinct from the
// error-returning noopExecutor in agent_test.go, which has no tool defs.)
type usageNoopExecutor struct{}

func (usageNoopExecutor) Execute(_ context.Context, _ string, _ json.RawMessage) (tool.ToolResult, error) {
	return tool.ToolResult{Output: "ok"}, nil
}
func (usageNoopExecutor) ToolDefs() []llm.ToolDef      { return []llm.ToolDef{{Name: "noop"}} }
func (usageNoopExecutor) Names() []string              { return []string{"noop"} }
func (usageNoopExecutor) Get(string) (tool.Tool, bool) { return nil, false }

func drainUsage(t *testing.T, events <-chan AgentEvent) *llm.Usage {
	t.Helper()
	var done *llm.Usage
	var sawDone bool
	for ev := range events {
		if ev.Type == EventDone {
			done = ev.Usage
			sawDone = true
		}
	}
	require.True(t, sawDone, "expected a terminal EventDone")
	return done
}

func TestRun_UsageAccumulatesAcrossTurns(t *testing.T) {
	r := &Runtime{
		LLM: &twoTurnUsageLLM{
			u1: &llm.Usage{InputTokens: 100, OutputTokens: 40, CacheCreationInputTokens: 10, CacheReadInputTokens: 6},
			u2: &llm.Usage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 1, CacheReadInputTokens: 2},
		},
		Tools:    usageNoopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	require.NotNil(t, got)
	assert.Equal(t, 110, got.InputTokens)
	assert.Equal(t, 45, got.OutputTokens)
	assert.Equal(t, 11, got.CacheCreationInputTokens)
	assert.Equal(t, 8, got.CacheReadInputTokens)
}

func TestRun_UsageSingleTurn(t *testing.T) {
	r := &Runtime{
		LLM: &twoTurnUsageLLM{
			u2: &llm.Usage{InputTokens: 7, OutputTokens: 3},
		},
		Tools:    usageNoopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	// calls=0 path emits a tool call; to get a pure single turn, mark calls so
	// the first ChatStream takes the text+stop branch.
	r.LLM.(*twoTurnUsageLLM).calls = 1
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	require.NotNil(t, got)
	assert.Equal(t, 7, got.InputTokens)
	assert.Equal(t, 3, got.OutputTokens)
}

func TestRun_UsageNilWhenProviderSilent(t *testing.T) {
	r := &Runtime{
		LLM:      &twoTurnUsageLLM{calls: 1}, // single turn, u2 nil → EventDone with nil usage
		Tools:    usageNoopExecutor{},
		Session:  session.NewSession("a", "k"),
		AgentID:  "a",
		Model:    "test-model",
		MaxTurns: 5,
	}
	events, err := r.Run(context.Background(), "go", nil)
	require.NoError(t, err)
	got := drainUsage(t, events)
	assert.Nil(t, got, "usage must stay nil when the provider never reports it")
}
