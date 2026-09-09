package runtime

import (
	"context"
	"errors"
	"github.com/sausheong/harness/llm"
	"testing"
)

func TestRunLifecycleStartFailureAndFinishCannotChangeOutcome(t *testing.T) {
	for _, mode := range []string{"run", "runturn"} {
		for _, deny := range []bool{false, true} {
			t.Run(mode+map[bool]string{true: "-deny", false: "-allow"}[deny], func(t *testing.T) {
				starts, finishes := 0, 0
				reason := ""
				blocked := errors.New("start denied")
				rt := hookFixture(t, LifecycleHooks{OnRunStart: func(context.Context) error {
					starts++
					if deny {
						return blocked
					}
					return nil
				}, OnRunFinish: func(ctx context.Context, outcome string) {
					finishes++
					reason = outcome
					if ctx.Err() != nil {
						t.Error("finish context already cancelled")
					}
					if _, ok := ctx.Deadline(); !ok {
						t.Error("finish unbounded")
					}
					panic("observer cannot rewrite outcome")
				}}, []llm.ChatEvent{{Type: llm.EventDone}}, nil)
				defer rt.Close()
				defer rt.Session.Close()
				failed := false
				if mode == "run" {
					events, err := rt.Run(context.Background(), "hello", nil)
					if err != nil {
						t.Fatal(err)
					}
					for event := range events {
						if event.Error != nil {
							failed = true
						}
					}
				} else {
					result, err := rt.RunTurn(context.Background(), "hello", nil, nil)
					failed = err != nil || result.Err != nil
				}
				if starts != 1 || finishes != 1 || failed != deny {
					t.Fatal(starts, finishes, failed, deny)
				}
				if deny {
					if reason != "error" || len(rt.Session.View()) != 0 {
						t.Fatal("denied start mutated history", reason, rt.Session.View())
					}
				} else if reason != "completed" {
					t.Fatal(reason)
				}
			})
		}
	}
}
