package runtime

import (
	"context"
	"encoding/json"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"strings"
	"testing"
)

func TestContextPinsSurviveCompactionAndRestart(t *testing.T) {
	store := session.NewStore(t.TempDir())
	if err := store.Create("agent", "pins"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "pins")
	if err != nil {
		t.Fatal(err)
	}
	provider := &recordingProvider{reply: "done"}
	r := &Runtime{Session: sess, LLM: provider, Tools: tool.NewRegistry(), StaticSystemPrompt: "identity", MaxTurns: 1}
	pins := []ContextPin{{ID: "constraint", Kind: "constraint", Text: "KEEP THIS EXACT CONSTRAINT"}, {ID: "objective", Kind: "objective", Text: "Deliver working implementation"}}
	if err = r.SetContextPins(context.Background(), pins); err != nil {
		t.Fatal(err)
	}
	pins[0].Text = "caller mutation"
	for i := 0; i < 3; i++ {
		sess.Compact("summary deliberately omits pins", 0)
		if err = sess.Flush(); err != nil {
			t.Fatal(err)
		}
		if i == 1 {
			if err = sess.Close(); err != nil {
				t.Fatal(err)
			}
			sess, err = store.LoadExclusive("agent", "pins")
			if err != nil {
				t.Fatal(err)
			}
			r.Session = sess
		}
		events, err := r.Run(context.Background(), "continue", nil)
		if err != nil {
			t.Fatal(err)
		}
		done := false
		for event := range events {
			if event.Type == EventError {
				t.Fatal(event.Error)
			}
			if event.Type == EventDone {
				done = true
			}
		}
		if !done {
			t.Fatal("run did not complete")
		}
	}
	defer sess.Close()
	if len(provider.requests) != 3 {
		t.Fatal("missing requests")
	}
	for _, request := range provider.requests {
		prompt := llm.JoinSystemPromptParts(request.SystemPromptParts)
		if !strings.Contains(prompt, "KEEP THIS EXACT CONSTRAINT") || !strings.Contains(prompt, "Deliver working implementation") {
			t.Fatal("pins omitted after compaction")
		}
	}
	inspected, err := r.InspectContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if inspected.Contributions[len(inspected.Contributions)-1].Kind != "pins" {
		t.Fatal("pins not counted")
	}
	if err = r.SetContextPins(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	got, err := r.ContextPins(context.Background())
	if err != nil || len(got) != 0 {
		t.Fatal("clear failed", err)
	}
}
func TestContextPinsValidationAndCorruptRecord(t *testing.T) {
	r := &Runtime{Session: session.NewSession("agent", "pins")}
	for _, pins := range [][]ContextPin{{{ID: "bad id", Kind: "constraint", Text: "x"}}, {{ID: "ok", Kind: "other", Text: "x"}}, {{ID: "ok", Kind: "objective", Text: strings.Repeat("x", 2049)}}} {
		if err := r.SetContextPins(context.Background(), pins); err == nil {
			t.Fatal("invalid pins accepted")
		}
	}
	if len(r.Session.Annotations(contextPinsKind)) != 0 {
		t.Fatal("invalid write persisted")
	}
	r.runMu.Lock()
	err := r.SetContextPins(context.Background(), nil)
	r.runMu.Unlock()
	if err == nil {
		t.Fatal("busy update accepted")
	}
	if err = r.Session.Annotate(contextPinsKind, json.RawMessage(`{"version":99,"pins":[]}`)); err != nil {
		t.Fatal(err)
	}
	if _, err = r.ContextPins(context.Background()); err == nil {
		t.Fatal("unsupported record ignored")
	}
}
