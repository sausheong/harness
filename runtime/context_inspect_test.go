package runtime

import (
	"context"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
	"github.com/sausheong/harness/tool"
	"strings"
	"testing"
)

func TestInspectContextEstimatesWithoutChangingHistory(t *testing.T) {
	s := session.NewSession("test", "inspect")
	s.Append(session.UserMessageEntry("private content must not be returned"))
	reg := tool.NewRegistry()
	reg.Register(echoTool{})
	r := &Runtime{Session: s, Tools: reg, StaticSystemPrompt: strings.Repeat("a", 100)}
	got, err := r.InspectContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := tokens.Estimate([]llm.Message{{Role: "user", Content: "private content must not be returned"}}, r.StaticSystemPrompt, reg.ToolDefs())
	if got.EstimatedTokens != want || len(got.Contributions) != 4 {
		t.Fatalf("inspection: %+v want %d", got, want)
	}
	if got.Contributions[1].Count != 1 || got.Contributions[3].Count != 1 || !strings.Contains(got.EstimateMethod, "not provider-reported") {
		t.Fatalf("metadata: %+v", got)
	}
	if len(s.View()) != 1 {
		t.Fatal("inspection changed history")
	}
	r.runMu.Lock()
	_, err = r.InspectContext(context.Background())
	r.runMu.Unlock()
	if err == nil {
		t.Fatal("busy inspection accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = r.InspectContext(ctx); err == nil {
		t.Fatal("cancelled inspection accepted")
	}
}
