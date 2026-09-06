// Package openrouter provides an llm.LLMProvider for OpenRouter, an
// aggregator that fronts hundreds of underlying models behind one
// OpenAI-compatible endpoint. Like providers/litellm, this adds no new
// request/stream-translation code of its own — it's a thin constructor
// around harness's existing OpenAI-compatible support.
package openrouter

import "github.com/sausheong/harness/providers/openai"

// DefaultBaseURL is OpenRouter's public API endpoint, used when
// NewOpenRouterProvider is called with an empty baseURL.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// NewOpenRouterProvider returns an OpenAI-compatible provider pointed at
// baseURL, or DefaultBaseURL when baseURL is empty. kind="openai-compatible"
// suppresses reasoning_effort, since OpenRouter's underlying models don't
// uniformly support it.
func NewOpenRouterProvider(apiKey, baseURL string) *openai.OpenAIProvider {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return openai.NewOpenAIProviderWithKind(apiKey, baseURL, "openai-compatible")
}
