// Package local provides an llm.LLMProvider for a locally-served model —
// Ollama, LM Studio, llama.cpp's server, vLLM, or anything else exposing
// an OpenAI-compatible endpoint on the caller's own machine or network.
// Like providers/litellm and providers/openrouter, this adds no new
// request/stream-translation code of its own — it's a thin constructor
// around harness's existing OpenAI-compatible support.
package local

import "github.com/sausheong/harness/providers/openai"

// DefaultBaseURL is used when NewLocalProvider is called with an empty
// baseURL. It's Ollama's endpoint specifically — the most common local
// server — but any other one works the same way via an explicit baseURL.
const DefaultBaseURL = "http://localhost:11434/v1"

// NewLocalProvider returns an OpenAI-compatible provider pointed at
// baseURL, or DefaultBaseURL when baseURL is empty. No API key is
// needed — local servers generally don't authenticate requests.
// kind="local" suppresses reasoning_effort, since locally-served models
// don't uniformly support it, and keeps retryable-error classification
// from treating a down local server the way it would a hosted API
// outage (see llm/retry.go).
func NewLocalProvider(baseURL string) *openai.OpenAIProvider {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return openai.NewOpenAIProviderWithKind("", baseURL, "local")
}
