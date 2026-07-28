package gemini

import (
	"strings"
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

// TestSyntheticToolCallID_UniqueAndAnthropicShaped guards against the bug
// where an empty Gemini FunctionCall.ID fell back to the function name.
// That fallback isn't a valid provider-neutral tool-call ID shape, and
// worse, it isn't unique: two calls to the same tool (parallel calls, or
// the same tool called again later in a conversation — routine for a
// coding agent re-reading files) collided on one ID. When that history
// was later replayed to Anthropic on a routing switch, the duplicate
// tool_use ids were rejected with a 400.
func TestSyntheticToolCallID_UniqueAndAnthropicShaped(t *testing.T) {
	seen := make(map[string]bool)
	for range 100 {
		id := syntheticToolCallID()
		require.True(t, strings.HasPrefix(id, "toolu_"),
			"synthetic ID must be shaped like an Anthropic tool_use id so a later replay to Anthropic is accepted")
		require.False(t, seen[id], "synthetic IDs must be unique across calls in the same process")
		seen[id] = true
	}
}
