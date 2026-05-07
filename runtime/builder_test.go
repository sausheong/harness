package runtime

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBuildRuntimeSetsProviderAndStaticPrompt(t *testing.T) {
	spec := AgentSpec{ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5"}
	deps := RuntimeDeps{ConfigSummary: "Configured channels: cli"}
	inputs := RuntimeInputs{}

	rt, err := BuildRuntime(deps, inputs, spec)
	require.NoError(t, err)
	require.Equal(t, "anthropic", rt.Provider)
	require.Equal(t, "claude-sonnet-4-5", rt.Model)
	require.NotEmpty(t, rt.StaticSystemPrompt)
	require.Contains(t, rt.StaticSystemPrompt, `"A" agent (id: a)`)
	require.Contains(t, rt.StaticSystemPrompt, "Configured channels: cli")
}

func TestBuildRuntimeLocalProvider(t *testing.T) {
	spec := AgentSpec{ID: "x", Name: "X", Model: "local/qwen2.5:3b"}
	rt, err := BuildRuntime(RuntimeDeps{}, RuntimeInputs{}, spec)
	require.NoError(t, err)
	require.Equal(t, "local", rt.Provider)
}

func TestBuildRuntimeMinimalSpecSafe(t *testing.T) {
	spec := AgentSpec{ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5"}
	rt, err := BuildRuntime(RuntimeDeps{}, RuntimeInputs{}, spec)
	require.NoError(t, err)
	require.Equal(t, "anthropic", rt.Provider)
	require.NotEmpty(t, rt.StaticSystemPrompt)
}

func TestBuildRuntimeUsesCallerProvidedMemoryFiles(t *testing.T) {
	// Caller composes memoryFiles however they want — typical patterns
	// include walking AGENTS.md from workspace + $HOME, building from a
	// wiki dump, or pulling from a doc index. Harness only concatenates
	// it into the static prompt.
	const sentinel = "MEMFILE_END_TO_END_SENTINEL"
	deps := RuntimeDeps{
		MemoryFiles: "\n\n## Project memory: AGENTS.md\n\n" + sentinel,
	}
	rt, err := BuildRuntime(deps, RuntimeInputs{}, AgentSpec{
		ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5",
	})
	require.NoError(t, err)
	require.Contains(t, rt.StaticSystemPrompt, sentinel)
	require.Contains(t, rt.StaticSystemPrompt, "## Project memory:")
}
