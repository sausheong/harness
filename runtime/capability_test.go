package runtime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	anthropicprovider "github.com/sausheong/harness/providers/anthropic"
)

// cachingCapableFake implements llm.LLMProvider plus the
// llm.PromptCachingProvider capability.
type cachingCapableFake struct {
	llmtest.Stub
}

func (*cachingCapableFake) SupportsPromptCaching() bool { return true }

// TestProviderSupportsCaching_CapabilityInterfaceWins: an Anthropic-shaped
// provider registered under a custom name (platformai, bedrock, any relay)
// must still get prompt caching. Gating on the config block's NAME meant
// every turn on a renamed provider re-processed the full history uncached.
func TestProviderSupportsCaching_CapabilityInterfaceWins(t *testing.T) {
	r := &Runtime{Provider: "platformai", LLM: &cachingCapableFake{}}
	assert.True(t, r.providerSupportsCaching(),
		"capability interface must win over the provider name")
}

func TestProviderSupportsCaching_NameFallbackStillWorks(t *testing.T) {
	r := &Runtime{Provider: "anthropic", LLM: &llmtest.Stub{}}
	assert.True(t, r.providerSupportsCaching(),
		"legacy name-based gate kept for providers without the capability interface")
}

func TestProviderSupportsCaching_NonCachingProviderFalse(t *testing.T) {
	r := &Runtime{Provider: "local", LLM: &llmtest.Stub{}}
	assert.False(t, r.providerSupportsCaching())
}

// TestAnthropicProviderDeclaresPromptCaching pins the wiring end-to-end:
// the real Anthropic provider must implement the capability interface.
func TestAnthropicProviderDeclaresPromptCaching(t *testing.T) {
	p := anthropicprovider.NewAnthropicProvider("k", "")
	pc, ok := any(p).(llm.PromptCachingProvider)
	assert.True(t, ok, "AnthropicProvider must implement llm.PromptCachingProvider")
	if ok {
		assert.True(t, pc.SupportsPromptCaching())
	}
}

// Mid-loop compaction was gated on the same name list; an Anthropic-shaped
// provider under a custom name must support it too.
func TestProviderSupportsMidLoopCompaction_CapabilityInterfaceWins(t *testing.T) {
	r := &Runtime{Provider: "platformai", LLM: &cachingCapableFake{}}
	assert.True(t, r.providerSupportsMidLoopCompaction())
}
