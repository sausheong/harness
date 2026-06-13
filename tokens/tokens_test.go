package tokens

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/sausheong/harness/llm"
)

func TestEstimateBasic(t *testing.T) {
	msgs := []llm.Message{
		{Role: "user", Content: "hello world"},   // 11 chars
		{Role: "assistant", Content: "hi there"}, // 8 chars
	}
	got := Estimate(msgs, "", nil)
	// Total = 11 + 8 + len("user")*2 + len("assistant") + len("user") = 11 + 8 + 4 + 9 + 4 = 36
	// /4 = 9
	assert.GreaterOrEqual(t, got, 9)
	assert.LessOrEqual(t, got, 12)
}

func TestEstimateWithSystemPromptAndTools(t *testing.T) {
	msgs := []llm.Message{{Role: "user", Content: "hi"}}
	tools := []llm.ToolDef{
		{Name: "read_file", Description: "read a file", Parameters: []byte(`{"type":"object"}`)},
	}
	withoutSys := Estimate(msgs, "", nil)
	withSys := Estimate(msgs, "you are a helpful assistant", tools)
	assert.Greater(t, withSys, withoutSys, "system prompt + tools should bump estimate")
}

func TestContextWindowKnown(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"anthropic/claude-3-5-sonnet-20241022", 200000},
		{"anthropic/claude-3-opus-20240229", 200000},
		{"anthropic/claude-3-haiku-20240307", 200000},
		{"openai/gpt-4o", 128000},
		{"openai/gpt-4o-mini", 128000},
		{"openai/gpt-4-turbo", 128000},
		{"google/gemini-1.5-pro", 2000000},
		{"google/gemini-1.5-flash", 1000000},
	}
	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			assert.Equal(t, tc.want, ContextWindow(tc.model))
		})
	}
}

func TestContextWindowUnknownRemoteReturnsRemoteFallback(t *testing.T) {
	// Non-local unknown models default to 128k — frontier proxies are
	// overwhelmingly 128k+, and reactive compaction handles overflow.
	assert.Equal(t, 128000, ContextWindow("weird/unknown-model"))
	assert.Equal(t, 128000, ContextWindow(""))
}

func TestContextWindowUnknownLocalReturnsLocalFallback(t *testing.T) {
	// Local/ollama unknown models default to 32k — conservative for
	// genuinely small models without a probed registration.
	assert.Equal(t, 32000, ContextWindow("local/some-unprobed-model"))
	assert.Equal(t, 32000, ContextWindow("ollama/some-unprobed-model"))
}

func TestContextWindowForOverrideWins(t *testing.T) {
	// Per-agent override is honoured over auto-detection (e.g. when an
	// auto-detected 200k Claude is too aggressive for a slow proxy).
	assert.Equal(t, 64000, ContextWindowFor("anthropic/claude-3-opus", 64000))
	// Zero override falls through to auto-detection.
	assert.Equal(t, 200000, ContextWindowFor("anthropic/claude-3-opus", 0))
	// Negative override is ignored (treated as "no override").
	assert.Equal(t, 200000, ContextWindowFor("anthropic/claude-3-opus", -1))
}

// TestContextWindowProxyProviderDetectedByModelFamily covers the case
// where the provider prefix is a relay/proxy (platformai, openrouter,
// bedrock, vertex) — the model family in the suffix should still drive
// the window, not the provider label.
func TestContextWindowProxyProviderDetectedByModelFamily(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"platformai/claude-sonnet-4-6-asia-southeast1", 1000000},
		{"openrouter/anthropic/claude-3-opus", 200000},
		{"bedrock/anthropic.claude-3-haiku", 200000},
		{"vertex/gemini-1.5-pro-001", 2000000},
		{"openrouter/openai/gpt-4o-2024-08-06", 128000},
	}
	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			assert.Equal(t, tc.want, ContextWindow(tc.model))
		})
	}
}

func TestEstimate_CountsImages(t *testing.T) {
	textOnly := []llm.Message{{Role: "user", Content: "hello"}}
	withImg := []llm.Message{{Role: "user", Content: "hello", Images: []llm.ImageContent{
		{MimeType: "image/png", Data: []byte("x")},
		{MimeType: "image/png", Data: []byte("y")},
	}}}
	base := Estimate(textOnly, "", nil)
	withImages := Estimate(withImg, "", nil)
	require.GreaterOrEqual(t, withImages-base, 2*perImageTokens)
}

func TestContextWindowOllamaDefault(t *testing.T) {
	// Without RegisterOllamaContext call, ollama models fall back to a sane default
	assert.Equal(t, 32000, ContextWindow("local/qwen2.5:3b-instruct"))
}

func TestContextWindowOllamaRegistered(t *testing.T) {
	RegisterOllamaContext("qwen2.5:3b-instruct", 32768)
	defer ResetOllamaContexts()
	assert.Equal(t, 32768, ContextWindow("local/qwen2.5:3b-instruct"))
	assert.Equal(t, 32768, ContextWindow("ollama/qwen2.5:3b-instruct"))
}

func TestCalibratorStartsAtOne(t *testing.T) {
	c := NewCalibrator()
	assert.Equal(t, 100, c.Adjust(100))
}

func TestCalibratorConvergesTowardActual(t *testing.T) {
	c := NewCalibrator()
	// After 5 identical samples of (actual=150, estimated=100), the running mean
	// should reach exactly 1.5, so Adjust(100) returns 150.
	c.Update(150, 100)
	c.Update(150, 100)
	c.Update(150, 100)
	c.Update(150, 100)
	c.Update(150, 100)
	got := c.Adjust(100)
	assert.GreaterOrEqual(t, got, 148, "calibrator should learn ratio≈1.5")
	assert.LessOrEqual(t, got, 150)
}

func TestCalibratorIgnoresZeroOrNegative(t *testing.T) {
	c := NewCalibrator()
	c.Update(0, 100)
	c.Update(100, 0)
	c.Update(-5, 100)
	assert.Equal(t, 100, c.Adjust(100), "bad samples must be ignored")
}

// TestContextWindow1MClaudeFamily pins the 1M-window Claude generation:
// Fable 5, Mythos 5, Opus 4.6+, and Sonnet 4.6 all ship a 1M context
// window. The old blanket claude→200k rule made compaction fire ~5x too
// early on these models. Older Claude (3.x, 4.0-4.5 era) stays 200k.
func TestContextWindow1MClaudeFamily(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"anthropic/claude-fable-5", 1000000},
		{"platformai/claude-fable-5-global", 1000000},
		{"anthropic/claude-mythos-5", 1000000},
		{"anthropic/claude-opus-4-6", 1000000},
		{"anthropic/claude-opus-4-7", 1000000},
		{"anthropic/claude-opus-4-8", 1000000},
		{"platformai/claude-sonnet-4-6-asia-southeast1", 1000000},
		// Older generations keep 200k.
		{"anthropic/claude-haiku-4-5-20251001", 200000},
		{"anthropic/claude-sonnet-4-5", 200000},
		{"anthropic/claude-3-5-sonnet-20241022", 200000},
		{"anthropic/claude-opus-4-0-20250514", 200000},
	}
	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			assert.Equal(t, tc.want, ContextWindow(tc.model))
		})
	}
}
