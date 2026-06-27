package runtime

import (
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAddUsage_BothNil(t *testing.T) {
	assert.Nil(t, addUsage(nil, nil))
}

func TestAddUsage_AccNil(t *testing.T) {
	turn := &llm.Usage{InputTokens: 7, OutputTokens: 3, CacheCreationInputTokens: 2, CacheReadInputTokens: 1}
	got := addUsage(nil, turn)
	require.NotNil(t, got)
	assert.Equal(t, llm.Usage{InputTokens: 7, OutputTokens: 3, CacheCreationInputTokens: 2, CacheReadInputTokens: 1}, *got)
	// input not mutated, and the result must be a distinct pointer (the
	// non-mutating/aliasing-safety contract the accumulator relies on).
	assert.NotSame(t, turn, got)
	assert.Equal(t, 7, turn.InputTokens)
}

func TestAddUsage_TurnNil(t *testing.T) {
	acc := &llm.Usage{InputTokens: 5}
	got := addUsage(acc, nil)
	require.NotNil(t, got)
	assert.Equal(t, 5, got.InputTokens)
}

func TestAddUsage_SumsAllFourFields(t *testing.T) {
	acc := &llm.Usage{InputTokens: 100, OutputTokens: 40, CacheCreationInputTokens: 10, CacheReadInputTokens: 6}
	turn := &llm.Usage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 1, CacheReadInputTokens: 2}
	got := addUsage(acc, turn)
	require.NotNil(t, got)
	assert.Equal(t, 110, got.InputTokens)
	assert.Equal(t, 45, got.OutputTokens)
	assert.Equal(t, 11, got.CacheCreationInputTokens)
	assert.Equal(t, 8, got.CacheReadInputTokens)
	// neither input mutated
	assert.Equal(t, 100, acc.InputTokens)
	assert.Equal(t, 10, turn.InputTokens)
}
