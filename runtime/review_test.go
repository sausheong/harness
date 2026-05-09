package runtime

import (
	"errors"
	"testing"
	"time"

	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/assert"
)

func TestReviewSpec_Defaults(t *testing.T) {
	// ReviewSpec with only required fields should compile.
	spec := ReviewSpec{
		Prompt: "review the conversation",
		Tools:  tool.NewRegistry(),
	}
	assert.NotNil(t, spec.Tools)
	assert.Equal(t, "review the conversation", spec.Prompt)
	// Optional fields default to zero.
	assert.Equal(t, 0, spec.MaxTurns)
	assert.Equal(t, time.Duration(0), spec.Timeout)
	assert.Nil(t, spec.OnEvent)
}

func TestReviewResult_Zero(t *testing.T) {
	res := ReviewResult{}
	assert.Empty(t, res.Actions)
	assert.Equal(t, 0, res.ToolCalls)
	assert.Equal(t, 0, res.Turns)
	assert.NoError(t, res.Err)
}

func TestErrReviewRecursion_IsExported(t *testing.T) {
	assert.True(t, errors.Is(ErrReviewRecursion, ErrReviewRecursion))
	assert.NotEmpty(t, ErrReviewRecursion.Error())
}

func TestReviewPrompts_Exported(t *testing.T) {
	assert.NotEmpty(t, ReviewPromptDefault)
	assert.NotEmpty(t, ReviewPromptVerbose)
	// Verbose should be at least as long as default (more guidance).
	assert.GreaterOrEqual(t, len(ReviewPromptVerbose), len(ReviewPromptDefault))
	// Both should reference the canonical action verbs.
	for _, p := range []string{ReviewPromptDefault, ReviewPromptVerbose} {
		assert.Contains(t, p, "memory")
		assert.Contains(t, p, "skill")
	}
}
