package runtime

import (
	"context"
	"fmt"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tokens"
)

// ContextContribution describes size, never the potentially sensitive body.
type ContextContribution struct {
	Kind            string `json:"kind"`
	Name            string `json:"name"`
	Count           int    `json:"count"`
	EstimatedTokens int    `json:"estimated_tokens"`
}

// ContextSource describes a component already included in the installed prompt.
// These estimates are subsets of the static category, not additional tokens.
type ContextSource struct {
	Kind            string `json:"kind"`
	Path            string `json:"path"`
	EstimatedTokens int    `json:"estimated_tokens"`
}

type SkillIndexSnapshotter interface {
	IndexSnapshot() (string, []ContextSource)
}

func skillIndexSnapshot(provider SkillProvider) (string, []ContextSource) {
	if provider == nil {
		return "", nil
	}
	if snapshotter, ok := provider.(SkillIndexSnapshotter); ok {
		return snapshotter.IndexSnapshot()
	}
	return provider.FormatIndex(), nil
}

type ContextInspection struct {
	Sources         []ContextSource       `json:"sources,omitempty"`
	OmittedSources  int                   `json:"omitted_sources,omitempty"`
	EstimateMethod  string                `json:"estimate_method"`
	EstimatedTokens int                   `json:"estimated_tokens"`
	Contributions   []ContextContribution `json:"contributions"`
	Limitations     []string              `json:"limitations"`
}

// InspectContext measures the installed static prompt, current effective history
// and registered schemas while idle. It does not make provider calls or refresh
// discovery. Dynamic retrieval and a future user message are not included.
func (r *Runtime) InspectContext(ctx context.Context) (ContextInspection, error) {
	var out ContextInspection
	if !r.runMu.TryLock() {
		return out, fmt.Errorf("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return out, err
	}
	var messages []llm.Message
	if r.Session != nil {
		history, err := r.Session.ResolveImages(ctx, r.Session.View())
		if err != nil {
			return out, err
		}
		messages = assembleMessages(history)
	}
	var defs []llm.ToolDef
	if r.Tools != nil {
		defs = r.Tools.ToolDefs()
	}
	if r.contextSources != nil {
		sources := r.contextSources()
		if len(sources) > 128 {
			out.OmittedSources = len(sources) - 128
			sources = sources[:128]
		}
		out.Sources = append([]ContextSource(nil), sources...)
	}
	out.EstimateMethod = "UTF-8 bytes / 4 with message framing and fixed image allowance; not provider-reported usage"
	pinned, err := r.contextPinsPrompt()
	if err != nil {
		return out, err
	}
	statePrompt, err := r.contextStatePrompt()
	if err != nil {
		return out, err
	}
	out.EstimatedTokens = tokens.Estimate(messages, r.StaticSystemPrompt+pinned+statePrompt, defs)
	out.Contributions = append(out.Contributions, ContextContribution{Kind: "static_prompt", Name: "Installed identity, guidance, skills and memory", Count: 1, EstimatedTokens: tokens.Estimate(nil, r.StaticSystemPrompt, nil)})
	var ordinary, results []llm.Message
	for _, m := range messages {
		if m.Role == "tool" {
			results = append(results, m)
		} else {
			ordinary = append(ordinary, m)
		}
	}
	out.Contributions = append(out.Contributions,
		ContextContribution{Kind: "messages", Name: "Effective conversation", Count: len(ordinary), EstimatedTokens: tokens.Estimate(ordinary, "", nil)},
		ContextContribution{Kind: "tool_results", Name: "Effective tool results", Count: len(results), EstimatedTokens: tokens.Estimate(results, "", nil)},
		ContextContribution{Kind: "tool_schemas", Name: "Registered tool schemas", Count: len(defs), EstimatedTokens: tokens.Estimate(nil, "", defs)})
	if pinned != "" {
		pins, _ := r.contextPins()
		out.Contributions = append(out.Contributions, ContextContribution{Kind: "pins", Name: "Pinned objectives and constraints", Count: len(pins), EstimatedTokens: tokens.Estimate(nil, pinned, nil)})
	}
	if statePrompt != "" {
		state, _ := r.contextState()
		out.Contributions = append(out.Contributions, ContextContribution{Kind: "task_state", Name: "Structured objectives, decisions, unresolved work and evidence references", Count: len(state.Items), EstimatedTokens: tokens.Estimate(nil, statePrompt, nil)})
	}
	out.Limitations = []string{"Source estimates are subsets of the static prompt category, not additional tokens.", "Excludes the next user message, per-request date/identity suffix and dynamic retrieval.", "Estimates include only effective history after compaction; raw archived history is not counted.", "Image estimates use a fixed allowance, not provider-specific tiles or resolution.", "Components round down individually; their sum can differ slightly from the total."}
	return out, nil
}
