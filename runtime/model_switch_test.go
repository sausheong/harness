package runtime

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

// TestModelSwitchMidSession_ActuallyChangesRequestModelAndHint reproduces,
// at the harness level, exactly what a caller like hand's tui.Controller
// does on a live model switch: mutate Model and DynamicIdentityHint on an
// already-running Runtime between two Run calls in the same session. This
// is the mechanism a caller-side fix for "the model still thinks it's the
// old one" ultimately rests on — if the second Run's actual ChatRequest
// didn't target the new model, no amount of system-prompt wording could
// have fixed the symptom, since the model actually being queried would
// still be the old one.
func TestModelSwitchMidSession_ActuallyChangesRequestModelAndHint(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	reg := tool.NewRegistry()

	rt := &Runtime{
		LLM:                 rec,
		Tools:               reg,
		Session:             sess,
		Model:               "meta/muse-spark-1.3",
		Provider:            "openrouter",
		Workspace:           t.TempDir(),
		MaxTurns:            5,
		DynamicIdentityHint: "You are running as the model \"openrouter/meta/muse-spark-1.3\".",
	}

	events, err := rt.Run(context.Background(), "which model am I using", nil)
	require.NoError(t, err)
	for range events {
	}

	// Simulate exactly what Controller.SwitchModel does.
	rt.Model = "anthropic/claude-sonnet-5"
	rt.DynamicIdentityHint = "You are running as the model \"openrouter/anthropic/claude-sonnet-5\"."

	events, err = rt.Run(context.Background(), "which model am I using now", nil)
	require.NoError(t, err)
	for range events {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.Len(t, rec.requests, 2)
	require.Equal(t, "meta/muse-spark-1.3", rec.requests[0].Model)
	require.Equal(t, "anthropic/claude-sonnet-5", rec.requests[1].Model, "second request must target the switched model")

	part2 := rec.requests[1].SystemPromptParts
	require.Len(t, part2, 2, "expected static + dynamic system prompt parts")
	require.Contains(t, part2[1].Text, "openrouter/anthropic/claude-sonnet-5")
	require.NotContains(t, part2[1].Text, "muse-spark-1.3", "dynamic hint must not still name the old model")
}
