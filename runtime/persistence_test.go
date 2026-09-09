package runtime

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

type persistenceProvider struct {
	usageDoneProvider
	before func()
	calls  int
}

func (p *persistenceProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.calls++
	if p.before != nil {
		p.before()
	}
	return p.usageDoneProvider.ChatStream(ctx, req)
}

func TestPersistenceFailurePreventsSuccessfulRuntimeCompletion(t *testing.T) {
	for _, stage := range []string{"prompt", "answer"} {
		t.Run(stage, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "store")
			sess := session.NewSession("agent", "key")
			sess.SetStore(session.NewStore(root))
			provider := &persistenceProvider{}
			if stage == "prompt" {
				if err := os.WriteFile(root, []byte("obstruction"), 0600); err != nil {
					t.Fatal(err)
				}
			} else {
				provider.before = func() {
					path := filepath.Join(root, "agent", "key.jsonl")
					if err := os.Remove(path); err != nil {
						t.Error(err)
					}
					if err := os.Mkdir(path, 0700); err != nil {
						t.Error(err)
					}
				}
			}
			rt := &Runtime{AgentID: "agent", Provider: "local", Model: "fixture", Session: sess, Tools: tool.NewRegistry(), LLM: provider, MaxTurns: 1}
			events, err := rt.Run(context.Background(), "work", nil)
			if err != nil {
				t.Fatal(err)
			}
			failed, done := false, false
			for event := range events {
				if event.Type == EventError && event.Error != nil {
					failed = true
				}
				if event.Type == EventDone {
					done = true
				}
			}
			if !failed || done || sess.PersistenceError() == nil {
				t.Fatal("persistence failure became success", failed, done)
			}
			if stage == "prompt" && provider.calls != 0 {
				t.Fatal("unsaved prompt sent to provider")
			}
		})
	}
}

func TestRunSyncJoinsAfterError(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	rt := &Runtime{AgentID: "agent", Provider: "local", Model: "fixture", Session: session.NewSession("agent", "key"), Tools: tool.NewRegistry(), LLM: &mockLLMProvider{events: []llm.ChatEvent{{Type: llm.EventError, Error: errors.New("provider failed")}}}, MaxTurns: 1}
	rt.AgentLoop.Hooks.OnStop = func(context.Context, string) { close(entered); <-release }
	result := make(chan error, 1)
	go func() { _, err := rt.RunSync(context.Background(), "work", nil); result <- err }()
	<-entered
	select {
	case err := <-result:
		t.Fatal("RunSync returned before cleanup", err)
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-result; err == nil {
		t.Fatal("failure lost while joining")
	}
}
