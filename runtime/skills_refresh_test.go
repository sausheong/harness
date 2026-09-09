package runtime

import (
	"context"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tool/skills"
	"github.com/sausheong/harness/tool/skills/disk"
	"strings"
	"testing"
)

func TestRefreshSkillsDiscoversAuthoredAndRemovedSkills(t *testing.T) {
	store := disk.NewStore(t.TempDir())
	rt, err := BuildRuntime(RuntimeDeps{Skills: store.AsSkillProvider(), ConfigSummary: "configuration retained", MemoryFiles: "memory retained"}, RuntimeInputs{Tools: tool.NewRegistry()}, AgentSpec{ID: "test", SystemPrompt: "identity retained"})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	initial := rt.StaticSystemPrompt
	if _, err = store.Create(context.Background(), skills.Skill{Name: "new-skill", Description: "new description", Body: "procedure"}); err != nil {
		t.Fatal(err)
	}
	rt.runMu.Lock()
	err = rt.RefreshSkills()
	rt.runMu.Unlock()
	if err == nil || rt.StaticSystemPrompt != initial {
		t.Fatal("busy refresh mutated prompt")
	}
	if err = rt.RefreshSkills(); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"new-skill", "new description", "identity retained", "configuration retained", "memory retained"} {
		if !strings.Contains(rt.StaticSystemPrompt, want) {
			t.Fatalf("missing %s", want)
		}
	}
	if err = store.Remove(context.Background(), "new-skill"); err != nil {
		t.Fatal(err)
	}
	if err = rt.RefreshSkills(); err != nil {
		t.Fatal(err)
	}
	if rt.StaticSystemPrompt != initial {
		t.Fatal("removed skill remains or unrelated context changed")
	}
}

func TestRefreshSkillsUsesCurrentProviderForToolAndIndex(t *testing.T) {
	ctx := context.Background()
	first := disk.NewStore(t.TempDir())
	second := disk.NewStore(t.TempDir())
	for _, entry := range []struct {
		store *disk.Store
		body  string
	}{{first, "original"}, {second, "replacement"}} {
		if _, err := entry.store.Create(ctx, skills.Skill{Name: "selected", Description: entry.body, Body: entry.body}); err != nil {
			t.Fatal(err)
		}
	}
	rt, err := BuildRuntime(RuntimeDeps{Skills: first.AsSkillProvider()}, RuntimeInputs{Tools: tool.NewRegistry()}, AgentSpec{ID: "test"})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	rt.Skills = second.AsSkillProvider()
	if err = rt.RefreshSkills(); err != nil {
		t.Fatal(err)
	}
	result, err := rt.Tools.Execute(ctx, "load_skill", []byte(`{"name":"selected"}`))
	if err != nil || result.Error != "" || !strings.Contains(result.Output, "replacement") || strings.Contains(result.Output, "original") || !strings.Contains(rt.StaticSystemPrompt, "replacement") {
		t.Fatal(result, err, rt.StaticSystemPrompt)
	}
	rt.Skills = nil
	if err = rt.RefreshSkills(); err != nil {
		t.Fatal(err)
	}
	result, err = rt.Tools.Execute(ctx, "load_skill", []byte(`{"name":"selected"}`))
	if err != nil || result.Error == "" || strings.Contains(rt.StaticSystemPrompt, "replacement") {
		t.Fatal("removed provider still used", result, err)
	}
}
