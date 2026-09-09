package runtime

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func TestRunTurnPinsAfterCompactionRestartAndCorruption(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("agent", "turn-pins"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "turn-pins")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { sess.Close() }()
	provider := &recordingProvider{reply: "done"}
	r := &Runtime{Session: sess, LLM: provider, Tools: tool.NewRegistry(), StaticSystemPrompt: "identity"}
	if err = r.SetContextPins(ctx, []ContextPin{{ID: "objective", Kind: "objective", Text: "EXACT OBJECTIVE"}, {ID: "constraint", Kind: "constraint", Text: "EXACT CONSTRAINT"}}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		sess.Compact("summary omits all pins", 0)
		if err = sess.Flush(); err != nil {
			t.Fatal(err)
		}
		if err = sess.Close(); err != nil {
			t.Fatal(err)
		}
		sess, err = store.LoadExclusive("agent", "turn-pins")
		if err != nil {
			t.Fatal(err)
		}
		r.Session = sess
		result, err := r.RunTurn(ctx, "continue", nil, nil)
		if err != nil || result.Err != nil || result.StopReason != "completed" {
			t.Fatalf("%+v %v", result, err)
		}
		req := provider.requests[len(provider.requests)-1]
		prompt := llm.JoinSystemPromptParts(req.SystemPromptParts)
		for _, text := range []string{"EXACT OBJECTIVE", "EXACT CONSTRAINT"} {
			if strings.Count(prompt, text) != 1 {
				t.Fatalf("missing or duplicated pin: %s", prompt)
			}
		}
	}
	if err = sess.Annotate(contextPinsKind, json.RawMessage(`{"version":99,"pins":[]}`)); err != nil {
		t.Fatal(err)
	}
	before := len(provider.requests)
	result, err := r.RunTurn(ctx, "must not dispatch", nil, nil)
	if err == nil && result.Err == nil {
		t.Fatal("corrupt pins accepted")
	}
	if len(provider.requests) != before {
		t.Fatal("provider dispatched with corrupt pins")
	}
}
