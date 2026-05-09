// Package runtime — Review primitive.
//
// runtime.Review runs a one-shot reviewer Runtime against a finished
// parent session, designed to be called from LifecycleHooks.OnStop in
// a goroutine. The reviewer sees a snapshot of the parent's
// conversation but writes durable memory/skills into the SHARED stores
// the parent reads from — that's how an agent built on harness curates
// its own memory and skill library across sessions.
//
// Typical wiring (10 lines from OnStop):
//
//	spec.Loop.Hooks.OnStop = func(ctx context.Context, reason string) {
//	    if reason != "completed" { return }
//	    go func() {
//	        runtime.Review(context.Background(), rt, runtime.ReviewSpec{
//	            Prompt: runtime.ReviewPromptDefault,
//	            Tools:  reviewTools,
//	        })
//	    }()
//	}
//
// Note: pass context.Background() — the parent's ctx is canceled by
// the time OnStop fires.
package runtime

import (
	"context"
	"errors"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
)

// ReviewSpec describes one reviewer pass over the parent Runtime's
// just-finished session. Only Prompt and Tools are required.
type ReviewSpec struct {
	// Prompt is the user-message-shaped review instruction handed to
	// the reviewer LLM. Use ReviewPromptDefault or ReviewPromptVerbose
	// from review_prompts.go, or write your own.
	Prompt string
	// Tools is the restricted toolset the reviewer can call. Typically
	// just write-only memory + skill tools so the reviewer can't read
	// files or execute shell commands. Required.
	Tools tool.Executor

	// MaxTurns caps the reviewer's tool-use loop, counted independently
	// of the parent's MaxTurns. 0 → default 16.
	MaxTurns int
	// Timeout is the hard envelope for the reviewer's Run. The reviewer
	// is canceled when exceeded; partial Actions are returned. 0 →
	// default 60 seconds.
	Timeout time.Duration

	// Provider, when non-nil, routes the reviewer to a (typically
	// cheaper) auxiliary LLM. nil → inherit parent.LLM.
	Provider llm.LLMProvider
	// Model overrides the parent's model id. "" → inherit parent.Model.
	Model string
	// ContextWindow overrides the parent's context window. 0 → inherit.
	ContextWindow int
	// SystemPrompt overrides the reviewer's static system prompt. "" →
	// the reviewer gets a minimal default identity composed from its
	// tool names.
	SystemPrompt string

	// Permission, when non-nil, gates the reviewer's tool calls.
	// nil → allow-all (the reviewer's tool registry is already
	// restricted by spec.Tools).
	Permission tool.PermissionChecker
	// Memory, when non-nil, overrides the parent's MemoryProvider.
	// nil → inherit (writes share the parent's memory store).
	Memory MemoryProvider
	// Skills, when non-nil, overrides the parent's SkillProvider.
	// nil → inherit (writes share the parent's skill store).
	Skills SkillProvider

	// OnEvent, when non-nil, receives a copy of every reviewer event
	// as it arrives. nil → events are drained silently. Run on the
	// reviewer goroutine; keep callbacks fast.
	OnEvent func(AgentEvent)
}

// ReviewResult summarizes one Review pass.
type ReviewResult struct {
	// Actions is the list of human-readable one-liners extracted from
	// the reviewer's successful tool calls (the "message" field of the
	// canonical {"success": true, "message": "...", "target": "..."}
	// JSON envelope).
	Actions []string
	// ToolCalls is the count of tool_call entries the reviewer made
	// (regardless of success).
	ToolCalls int
	// Turns is the count of LLM turns the reviewer used.
	Turns int
	// Duration is the wall time from the start of Review to its return
	// (including snapshot + Run + extraction).
	Duration time.Duration
	// Err is non-nil for setup failures (recursion guard, nil Tools,
	// build errors). Mid-Run failures (LLM errors, tool errors) are
	// counted in Turns/ToolCalls but do not populate Err — partial
	// Actions are returned.
	Err error
}

// ErrReviewRecursion is returned (via ReviewResult.Err) when Review is
// called on a Runtime whose AgentID is the reserved "__review__"
// sentinel — i.e., the reviewer would itself spawn a reviewer.
var ErrReviewRecursion = errors.New("runtime: review recursion (parent is itself a reviewer)")

// reviewerAgentID is the reserved AgentID assigned to every reviewer
// Runtime. Review checks parent.AgentID against this value and refuses
// to recurse.
const reviewerAgentID = "__review__"

// Review is a stub that later tasks fill in.
func Review(ctx context.Context, parent *Runtime, spec ReviewSpec) ReviewResult {
	return ReviewResult{Err: errors.New("not implemented")}
}
