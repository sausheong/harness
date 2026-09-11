package runtime

import (
	"context"
	"encoding/json"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tool/skills"
	"github.com/sausheong/harness/tool/skills/disk"
	"strings"
	"testing"
	"time"
)

type authoringProvider struct {
	llmtest.Base
	prompts []string
}

func (p *authoringProvider) ChatStream(_ context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.prompts = append(p.prompts, llm.JoinSystemPromptParts(req.SystemPromptParts))
	ch := make(chan llm.ChatEvent, 2)
	switch len(p.prompts) {
	case 1:
		ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "author", Name: "skill_manage", Input: json.RawMessage(`{"action":"create","name":"authored-skill","body":"---\ndescription: newly authored workflow\n---\nunique procedure body"}`)}}
	case 2:
		ch <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "load", Name: "load_skill", Input: json.RawMessage(`{"name":"authored-skill"}`)}}
	default:
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Skill created and loaded."}
	}
	ch <- llm.ChatEvent{Type: llm.EventDone}
	close(ch)
	return ch, nil
}

func TestSkillAuthoringRefreshesNextModelRequest(t *testing.T) {
	store := disk.NewStore(t.TempDir())
	reg := tool.NewRegistry()
	reg.Register(&skills.SkillTool{Store: store})
	provider := &authoringProvider{}
	rt, err := BuildRuntime(RuntimeDeps{Skills: store.AsSkillProvider()}, RuntimeInputs{Tools: reg, Session: session.NewSession("test", "skills")}, AgentSpec{ID: "test", SystemPrompt: "fixed identity", MaxTurns: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	rt.LLM = provider
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	events, err := rt.Run(ctx, "create then load the workflow", nil)
	if err != nil {
		t.Fatal(err)
	}
	done := false
	loaded := false
	for event := range events {
		if event.Type == EventError {
			t.Errorf("runtime error: %v", event.Error)
		}
		if event.Type == EventDone {
			done = true
		}
		if event.Type == EventToolResult && event.ToolCall != nil && event.ToolCall.Name == "load_skill" {
			encoded, _ := json.Marshal(event)
			loaded = strings.Contains(string(encoded), "unique procedure body")
		}
	}
	if !done || !loaded || len(provider.prompts) != 3 {
		t.Fatalf("done=%v loaded=%v requests=%d", done, loaded, len(provider.prompts))
	}
	if strings.Contains(provider.prompts[0], "newly authored workflow") {
		t.Fatal("skill existed before authoring")
	}
	for _, prompt := range provider.prompts[1:] {
		if !strings.Contains(prompt, "newly authored workflow") || !strings.Contains(prompt, "fixed identity") {
			t.Fatalf("next request omitted refreshed skill: %s", prompt)
		}
	}
	if _, ok, err := store.Get(ctx, "authored-skill"); err != nil || !ok {
		t.Fatalf("skill not persisted: %v", err)
	}
}
