package runtime

import (
	"context"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
	"github.com/sausheong/harness/tool"
	"github.com/stretchr/testify/require"
)

// usageDoneProvider emits a usage-bearing EventDone so the agent loop can
// populate (and persist) the per-session token calibrator.
type usageDoneProvider struct {
	llmtest.Base
}

func (p *usageDoneProvider) ChatStream(_ context.Context, _ llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	ch := make(chan llm.ChatEvent, 4)
	ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "answer"}
	ch <- llm.ChatEvent{Type: llm.EventDone, Usage: &llm.Usage{InputTokens: 1000, OutputTokens: 10}}
	close(ch)
	return ch, nil
}

// TestCalibratorPersistsByEndOfRun guards that the per-session calibrator is
// persisted to the CalibratorStore by the time Run completes. After the P8
// debounce change the Save moves from per-round to a single deferred flush at
// Run end; this test asserts the observable end state (persisted) is unchanged.
func TestCalibratorPersistsByEndOfRun(t *testing.T) {
	dir := t.TempDir()
	store := tokens.NewCalibratorStore(dir)
	rt := &Runtime{
		LLM:     &usageDoneProvider{},
		Tools:   tool.NewRegistry(),
		Session: session.NewSession("a", "k"),
		AgentID: "a",
		Model:   "claude-sonnet-4-5",
		// Compaction must be non-nil for the runtime to lazily create the
		// per-session calibrator (the calibrator-creation path is gated on
		// it). MessageCap/threshold left at defaults so no compaction fires
		// during this short run.
		Compaction:      newCompactionMgr(),
		Provider:        "anthropic",
		MaxTurns:        2,
		Workspace:       t.TempDir(),
		CalibratorStore: store,
	}
	_, err := rt.RunSync(context.Background(), "go", nil)
	require.NoError(t, err)

	ratio, count := tokens.NewCalibratorStore(dir).Load("a", "k")
	require.Greater(t, count, 0, "calibrator must be persisted by end of Run")
	require.Greater(t, ratio, 0.0)
}
