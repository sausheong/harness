//go:build darwin || linux

package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sausheong/harness/budget"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/bash"
)

func TestTimeBudgetStopsShellDescendantAndPersistsPair(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("streaming_"+mode, func(t *testing.T) {
			t.Setenv("HARNESS_STREAMING_TOOLS", mode)
			checkTimeBudgetShell(t)
		})
	}
}

func checkTimeBudgetShell(t *testing.T) {
	work := t.TempDir()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "shell-deadline"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("a", "shell-deadline")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if sess != nil {
			sess.Close()
		}
	}()
	tc := llm.ToolCall{ID: "deadline-shell", Name: "bash", Input: json.RawMessage(`{"command":"printf started > started; (sleep 3; printf survived > survived) & wait"}`)}
	provider := &mockLLMProvider{events: []llm.ChatEvent{{Type: llm.EventToolCallStart, ToolCall: &tc}, {Type: llm.EventToolCallDone, ToolCall: &tc}, {Type: llm.EventDone}}}
	registry := tool.NewRegistry()
	registry.Register(&bash.BashTool{WorkDir: work})
	rt := &Runtime{Session: sess, LLM: provider, Tools: registry, Model: "fixture", AgentID: "a", MaxTurns: 2}
	if err = rt.DecideTimeBudget(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	begin := time.Now()
	events, err := rt.Run(context.Background(), "exercise deadline", nil)
	if err != nil {
		t.Fatal(err)
	}
	ended := make(chan bool, 1)
	go func() {
		exhausted := false
		for event := range events {
			if errors.Is(event.Error, budget.ErrTimeExhausted) {
				exhausted = true
			}
		}
		ended <- exhausted
	}()
	select {
	case exhausted := <-ended:
		if !exhausted {
			t.Fatal("time exhaustion cause lost")
		}
	case <-time.After(6 * time.Second):
		t.Fatal("shell operation failed to join")
	}
	if _, err = os.Stat(filepath.Join(work, "started")); err != nil {
		t.Fatal("shell never started", err)
	}
	// Wait beyond the descendant's scheduled write: returning alone does not
	// establish that the owned process group stopped.
	if remaining := 3500*time.Millisecond - time.Since(begin); remaining > 0 {
		time.Sleep(remaining)
	}
	if _, err = os.Stat(filepath.Join(work, "survived")); !os.IsNotExist(err) {
		t.Fatalf("descendant survived expiry: %v", err)
	}
	if err = sess.Close(); err != nil {
		t.Fatal(err)
	}
	sess = nil
	sess, err = store.LoadExclusive("a", "shell-deadline")
	if err != nil {
		t.Fatal(err)
	}
	calls, results := 0, 0
	for _, entry := range sess.View() {
		if entry.Type == session.EntryTypeToolCall {
			var data session.ToolCallData
			if err = json.Unmarshal(entry.Data, &data); err != nil {
				t.Fatal(err)
			}
			if data.ID == tc.ID {
				calls++
			}
		}
		if entry.Type == session.EntryTypeToolResult {
			var data session.ToolResultData
			if err = json.Unmarshal(entry.Data, &data); err != nil {
				t.Fatal(err)
			}
			if data.ToolCallID == tc.ID {
				results++
				if !data.Aborted || !data.IsError || data.Error != budget.ErrTimeExhausted.Error() {
					t.Fatalf("cancelled tool recorded success: %+v", data)
				}
			}
		}
	}
	if calls != 1 || results != 1 {
		t.Fatalf("persisted calls=%d results=%d", calls, results)
	}
	t.Logf("deadline process/descendant check and reopen completed in %s", time.Since(begin))
}
