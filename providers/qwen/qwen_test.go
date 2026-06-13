package qwen

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

func TestQwenResolveSystemPromptPrefersParts(t *testing.T) {
	got := qwenResolveSystemPrompt(llm.ChatRequest{
		SystemPrompt:      "legacy",
		SystemPromptParts: []llm.SystemPromptPart{{Text: "new-a"}, {Text: "new-b"}},
	})
	require.Equal(t, "new-a\nnew-b", got)
}

func TestQwenResolveSystemPromptFallback(t *testing.T) {
	got := qwenResolveSystemPrompt(llm.ChatRequest{SystemPrompt: "legacy"})
	require.Equal(t, "legacy", got)
}

func TestQwenResolveSystemPromptEmpty(t *testing.T) {
	got := qwenResolveSystemPrompt(llm.ChatRequest{})
	require.Equal(t, "", got)
}

func TestEmitToolCalls_OrderedByIndex(t *testing.T) {
	mk := func(id, name string) *pendingTC {
		p := &pendingTC{id: id, name: name}
		p.args.WriteString("{}")
		return p
	}
	toolCalls := map[int]*pendingTC{1: mk("b", "second"), 0: mk("a", "first")}
	ch := make(chan llm.ChatEvent, 10)
	emitToolCalls(ch, toolCalls)
	close(ch)
	var ids []string
	for ev := range ch {
		if ev.Type == llm.EventToolCallDone {
			ids = append(ids, ev.ToolCall.ID)
		}
	}
	require.Equal(t, []string{"a", "b"}, ids)
}
