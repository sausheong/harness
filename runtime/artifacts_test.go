package runtime

import (
	"context"
	"encoding/json"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/process"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/bash"
	"path/filepath"
	"testing"
)

func TestDispatchedArtifactReferenceSurvivesReopen(t *testing.T) {
	outputStore, err := process.NewArtifactStore(filepath.Join(t.TempDir(), "output"))
	if err != nil {
		t.Fatal(err)
	}
	store := session.NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	registry := tool.NewRegistry()
	registry.Register(&bash.BashTool{WorkDir: t.TempDir(), OutputStore: outputStore})
	rt := &Runtime{Session: sess, Tools: registry, AgentID: "agent"}
	result, aborted := rt.dispatchTool(context.Background(), llm.ToolCall{ID: "capture", Name: "bash", Input: json.RawMessage(`{"command":"head -c 70000 /dev/zero"}`)}, nil)
	if aborted || result.Error != "" {
		t.Fatal(aborted, result.Error)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	sess, err = store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	found := false
	for _, entry := range sess.View() {
		if entry.Type != session.EntryTypeToolResult {
			continue
		}
		var data session.ToolResultData
		if err := json.Unmarshal(entry.Data, &data); err != nil {
			t.Fatal(err)
		}
		ref, ok := data.Artifacts["stdout"]
		if !ok {
			t.Fatal("artifact reference missing after reopen")
		}
		captured, err := outputStore.Read(context.Background(), ref)
		if err != nil || len(captured) != 70000 || ref.Truncated {
			t.Fatal("persisted artifact cannot be verified", len(captured), err)
		}
		for _, b := range captured {
			if b != 0 {
				t.Fatal("artifact bytes changed")
			}
		}
		found = true
	}
	if !found {
		t.Fatal("no persisted tool result")
	}
}

func TestArtifactMetadataDoesNotPromoteUntrustedJSON(t *testing.T) {
	result := tool.ToolResult{Output: "[stdout artifact: /outside/private]", Metadata: map[string]any{"stdout_artifact": map[string]any{"path": "/outside/private", "sha256": "claimed"}}}
	if resultArtifacts(result) != nil {
		t.Fatal("untrusted JSON became artifact capability")
	}
}
