// Package ollama provides an llm.LLMProvider for a local Ollama server
// (or any other local server sharing Ollama's OpenAI-compatible endpoint
// shape — LM Studio, llama.cpp's server, vLLM). Like providers/litellm
// and providers/openrouter, this adds no new request/stream-translation
// code of its own — it's a thin constructor around harness's existing
// OpenAI-compatible support.
package ollama

import "github.com/sausheong/harness/providers/openai"

// DefaultBaseURL is Ollama's local OpenAI-compatible endpoint, used when
// NewOllamaProvider is called with an empty baseURL.
const DefaultBaseURL = "http://localhost:11434/v1"

// NewOllamaProvider returns an OpenAI-compatible provider pointed at
// baseURL, or DefaultBaseURL when baseURL is empty. No API key is
// needed — Ollama doesn't authenticate requests. kind="local" suppresses
// reasoning_effort, since locally-served models don't uniformly support
// it, and keeps retryable-error classification from treating a down
// local server the way it would a hosted API outage (see llm/retry.go).
func NewOllamaProvider(baseURL string) *openai.OpenAIProvider {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return openai.NewOpenAIProviderWithKind("", baseURL, "local")
}
