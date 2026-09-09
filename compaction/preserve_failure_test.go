package compaction

import (
	"context"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestFailedSummarisationPreservesEffectiveAndRawHistory(t *testing.T) {
	sess := longSession()
	before := sess.View()
	raw := sess.Entries()
	leaf := sess.LeafID()
	manager := &Manager{Summarizer: &Summarizer{Provider: &fakeProvider{text: ""}, Model: "test"}, PreserveTurns: 4}
	result, err := manager.MaybeCompact(context.Background(), sess, ReasonManual, "")
	require.NoError(t, err)
	require.False(t, result.Compacted)
	require.NotEmpty(t, result.Skipped)
	require.Equal(t, before, sess.View())
	require.Equal(t, raw, sess.Entries())
	require.Equal(t, leaf, sess.LeafID())
}
