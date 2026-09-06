// Package litellm provides an llm.LLMProvider for a self-hosted LiteLLM
// proxy. LiteLLM speaks the OpenAI chat-completions wire format and
// itself routes to dozens of underlying vendors, so this package adds no
// new request/stream-translation code of its own — it's a thin
// constructor around harness's existing OpenAI-compatible support.
package litellm

import "github.com/sausheong/harness/providers/openai"

// NewLiteLLMProvider returns an OpenAI-compatible provider pointed at
// baseURL, a self-hosted LiteLLM proxy endpoint. Unlike OpenRouter,
// LiteLLM has no public default endpoint — callers must supply a
// non-empty baseURL (cmd/hand's buildProvider, for example, rejects an
// empty one with a clear error before ever reaching here).
// kind="openai-compatible" suppresses reasoning_effort, since not every
// model LiteLLM routes a request to is guaranteed to support it.
func NewLiteLLMProvider(apiKey, baseURL string) *openai.OpenAIProvider {
	return openai.NewOpenAIProviderWithKind(apiKey, baseURL, "openai-compatible")
}
