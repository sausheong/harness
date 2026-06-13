package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

func TestGeminiResolveSystemPromptPrefersParts(t *testing.T) {
	got := geminiResolveSystemPrompt(llm.ChatRequest{
		SystemPrompt:      "legacy",
		SystemPromptParts: []llm.SystemPromptPart{{Text: "alpha"}, {Text: "beta"}},
	})
	require.Equal(t, "alpha\nbeta", got)
}

func TestGeminiResolveSystemPromptFallsBackToString(t *testing.T) {
	got := geminiResolveSystemPrompt(llm.ChatRequest{SystemPrompt: "only-string"})
	require.Equal(t, "only-string", got)
}

func TestGeminiResolveSystemPromptEmpty(t *testing.T) {
	got := geminiResolveSystemPrompt(llm.ChatRequest{})
	require.Equal(t, "", got)
}

func TestGeminiUsageBuffer_KeepsLast(t *testing.T) {
	var last *llm.Usage
	last = updateUsage(last, 10, 5)
	last = updateUsage(last, 20, 9) // final cumulative
	require.NotNil(t, last)
	require.Equal(t, 20, last.InputTokens)
	require.Equal(t, 9, last.OutputTokens)
}
