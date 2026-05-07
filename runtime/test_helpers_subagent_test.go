package runtime

import (
	"testing"

	"github.com/sausheong/harness/llm"
)

// testCfg is a tiny stand-in for whatever live-config object a real
// consumer would hold. It bundles a SubagentResolver with the small
// accessor helpers (EligibleSubagents, GetAgent) that subagent tests
// reach for, so subagent_test.go and inheritance_test.go can keep
// their `cfg.X()` call shape.
type testCfg struct {
	specs    map[string]AgentSpec
	subagent map[string]bool   // id → registered as subagent
	desc     map[string]string // id → description (for EligibleSubagents)
}

func (c *testCfg) Resolve(id string) (SubagentSpec, bool) {
	spec, ok := c.specs[id]
	if !ok {
		return SubagentSpec{}, false
	}
	return SubagentSpec{Spec: spec, Registered: c.subagent[id]}, true
}

func (c *testCfg) EligibleSubagents() map[string]string {
	out := map[string]string{}
	for id, ok := range c.subagent {
		if ok {
			out[id] = c.desc[id]
		}
	}
	return out
}

func (c *testCfg) GetAgent(id string) (AgentSpec, bool) {
	s, ok := c.specs[id]
	return s, ok
}

// newTwoAgentCfg builds a fixture with a parent ("default") and one
// subagent ("researcher") — the minimal shape needed to exercise the
// task-tool dispatch path.
func newTwoAgentCfg(t *testing.T) *testCfg {
	t.Helper()
	parent := AgentSpec{
		ID:        "default",
		Name:      "Parent",
		Workspace: t.TempDir(),
		Model:     "test/parent",
		MaxTurns:  5,
	}
	sub := AgentSpec{
		ID:        "researcher",
		Name:      "Researcher",
		Workspace: t.TempDir(),
		Model:     "test/researcher",
		MaxTurns:  5,
	}
	return &testCfg{
		specs:    map[string]AgentSpec{"default": parent, "researcher": sub},
		subagent: map[string]bool{"researcher": true},
		desc:     map[string]string{"researcher": "Investigates topics and reports findings."},
	}
}

// subagentBuilderForLLM returns a SubagentBuildFn that hands every subagent
// the supplied LLM, a fresh in-memory Session, and a noop executor.
func subagentBuilderForLLM(subLLM llm.LLMProvider) SubagentBuildFn {
	return func(spec AgentSpec) (RuntimeInputs, error) {
		return RuntimeInputs{
			Provider:     subLLM,
			Tools:        noopExecutor{},
			Session:      NewSubagentSession(spec.ID),
			Compaction:   nil,
			IngestSource: "",
		}, nil
	}
}
