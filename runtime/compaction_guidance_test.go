package runtime

import (
	"context"
	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/session"
	"strings"
	"testing"
	"time"
)

func TestPinnedGuidanceReachesManualAndAsyncSummariser(t *testing.T) {
	sess := session.NewSession("hand", "focus")
	provider := &recordingProvider{reply: "<summary>usable summary</summary>"}
	r := &Runtime{Session: sess}
	pin := ContextPin{ID: "constraint", Kind: "constraint", Text: "PRESERVE EXACT CONSTRAINT"}
	if err := r.SetContextPins(context.Background(), []ContextPin{pin}); err != nil {
		t.Fatal(err)
	}
	if err := r.SetContextState(context.Background(), 0, []ContextStateItem{{ID: "decision", Kind: "decision", Text: "PRESERVE EXACT DECISION"}}); err != nil {
		t.Fatal(err)
	}
	ctx, err := r.CompactionContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	manager := &compaction.Manager{Summarizer: &compaction.Summarizer{Provider: provider, Model: "fixture"}, PreserveTurns: 2}
	add := func() {
		for i := 0; i < 10; i++ {
			sess.Append(session.UserMessageEntry("task detail"))
			sess.Append(session.AssistantMessageEntry("decision and evidence"))
		}
	}
	add()
	result, err := manager.MaybeCompact(ctx, sess, compaction.ReasonManual, "FOCUS ON UNRESOLVED WORK")
	if err != nil || !result.Compacted {
		t.Fatalf("manual: %+v %v", result, err)
	}
	add()
	done := manager.MaybeCompactAsyncContext(ctx, sess, compaction.ReasonPreventive)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("async compaction did not join")
	}
	if len(provider.requests) != 2 {
		t.Fatalf("requests %d", len(provider.requests))
	}
	for _, request := range provider.requests {
		if !strings.Contains(request.Messages[0].Content, pin.Text) || !strings.Contains(request.Messages[0].Content, "PRESERVE EXACT DECISION") {
			t.Fatal("pin omitted from summariser request")
		}
	}
	if !strings.Contains(provider.requests[0].Messages[0].Content, "FOCUS ON UNRESOLVED WORK") {
		t.Fatal("manual focus omitted")
	}
	if strings.Contains(provider.requests[1].Messages[0].Content, "FOCUS ON UNRESOLVED WORK") {
		t.Fatal("one-off focus leaked into automatic compaction")
	}
}
