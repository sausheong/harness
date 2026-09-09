package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func TestStructuredContextSurvivesSummaryAndRestart(t *testing.T) {
	for _, turn := range []bool{false, true} {
		t.Run(map[bool]string{false: "Run", true: "RunTurn"}[turn], func(t *testing.T) {
			ctx := context.Background()
			store := session.NewStore(t.TempDir())
			if err := store.Create("agent", "state"); err != nil {
				t.Fatal(err)
			}
			sess, err := store.LoadExclusive("agent", "state")
			if err != nil {
				t.Fatal(err)
			}
			defer func() { sess.Close() }()
			provider := &recordingProvider{reply: "done"}
			r := &Runtime{Session: sess, LLM: provider, Tools: tool.NewRegistry(), MaxTurns: 1}
			items := []ContextStateItem{{ID: "goal", Kind: "objective", Text: "EXACT GOAL"}, {ID: "design", Kind: "decision", Text: "EXACT DECISION"}, {ID: "todo", Kind: "unresolved_work", Text: "EXACT TODO"}, {ID: "tests", Kind: "verification_reference", Text: "EXACT TEST SCOPE", Reference: "evidence/run.json#snapshot-abc"}}
			if err = r.SetContextState(ctx, 0, items); err != nil {
				t.Fatal(err)
			}
			if err = r.SetContextState(ctx, 0, nil); !errors.Is(err, session.ErrAnnotationConflict) {
				t.Fatalf("stale replacement: %v", err)
			}
			for i := 0; i < 3; i++ {
				sess.Compact("summary deliberately omits all task facts", 0)
				if err = sess.Flush(); err != nil {
					t.Fatal(err)
				}
				if err = sess.Close(); err != nil {
					t.Fatal(err)
				}
				sess, err = store.LoadExclusive("agent", "state")
				if err != nil {
					t.Fatal(err)
				}
				r.Session = sess
				if turn {
					result, err := r.RunTurn(ctx, "continue", nil, nil)
					if err != nil || result.Err != nil {
						t.Fatalf("%+v %v", result, err)
					}
				} else {
					events, err := r.Run(ctx, "continue", nil)
					if err != nil {
						t.Fatal(err)
					}
					for e := range events {
						if e.Type == EventError {
							t.Fatal(e.Error)
						}
					}
				}
				prompt := llm.JoinSystemPromptParts(provider.requests[len(provider.requests)-1].SystemPromptParts)
				for _, item := range items {
					if !strings.Contains(prompt, item.Text) || !strings.Contains(prompt, item.Reference) {
						t.Fatal("structured state lost", prompt)
					}
				}
				state, err := r.ContextState(ctx)
				if err != nil || state.Revision != 1 || !reflect.DeepEqual(state.Items, items) {
					t.Fatalf("%+v %v", state, err)
				}
			}
			view, err := r.InspectContext(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if view.Contributions[len(view.Contributions)-1].Kind != "task_state" {
				t.Fatal("context estimate omits task state")
			}
			if err = r.SetContextState(ctx, 1, nil); err != nil {
				t.Fatal(err)
			}
			state, err := r.ContextState(ctx)
			if err != nil || state.Revision != 2 || len(state.Items) != 0 {
				t.Fatal(state, err)
			}
		})
	}
}
func TestStructuredContextRejectsInvalidAndCorruptState(t *testing.T) {
	ctx := context.Background()
	p := &recordingProvider{reply: "done"}
	r := &Runtime{Session: session.NewSession("a", "s"), LLM: p, Tools: tool.NewRegistry()}
	invalid := [][]ContextStateItem{
		{{ID: "a", Kind: "verification_reference", Text: "test"}},
		{{ID: "a", Kind: "unknown", Text: "test"}},
		{{ID: "a", Kind: "objective", Text: "x"}, {ID: "a", Kind: "decision", Text: "y"}},
		{{ID: "a", Kind: "objective", Text: strings.Repeat("x", 2049)}},
	}
	large := []ContextStateItem{}
	for _, id := range []string{"a", "b", "c", "d"} {
		large = append(large, ContextStateItem{ID: id, Kind: "objective", Text: strings.Repeat("x", 2048)})
	}
	invalid = append(invalid, large)
	for _, items := range invalid {
		if err := r.SetContextState(ctx, 0, items); err == nil {
			t.Fatal("invalid state accepted")
		}
	}
	if len(r.Session.Annotations(contextStateKind)) != 0 {
		t.Fatal("invalid records persisted")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := r.SetContextState(cancelled, 0, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := r.Session.Annotate(contextStateKind, json.RawMessage(`{"version":2,"items":[]}`)); err != nil {
		t.Fatal(err)
	}
	result, err := r.RunTurn(ctx, "do not dispatch", nil, nil)
	if err == nil && result.Err == nil {
		t.Fatal("corrupt state accepted")
	}
	if len(p.requests) != 0 {
		t.Fatal("corrupt state dispatched provider")
	}
	if _, err = r.CompactionContext(ctx); err == nil {
		t.Fatal("corrupt state compacted")
	}
}
