// Package runtime — subagent factory + adapter.
//
// This file wires the per-Runtime task tool: it constructs a tool.SubagentFactory
// that builds a fresh Runtime for the named subagent, sets its Parent
// pointer (for event forwarding) and Depth (for the recursion cap), and adapts
// Runtime.Run to tool.SubagentRunner so the task tool can drain it.
//
// The factory enforces the depth cap (maxAgentDepth()) before constructing
// anything — a subagent that would exceed the limit returns an error that
// TaskTool surfaces as a tool result error, which the parent LLM then sees.
package runtime

import (
	"context"
	"fmt"

	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

// SubagentBuildFn constructs the per-subagent RuntimeInputs given the resolved
// AgentSpec. Each call site provides this closure because tool-registry
// construction (file/bash/web tools, MCP, send_message) is environment-specific.
//
// Implementations MUST:
//   - Build a fresh tool.Executor for the subagent (workspace = spec.Workspace)
//   - Resolve the LLM provider from spec.Model
//   - Create a fresh in-memory Session via NewSubagentSession
//   - Set IngestSource to "" (subagents are short-lived; no KG ingest)
//   - Set Compaction to whatever the call site's chat path uses for this agent
type SubagentBuildFn func(spec AgentSpec) (RuntimeInputs, error)

// SubagentSpec carries the per-call-site policy (whether to inherit parent
// history, registration check) the framework needs to honour without
// importing consumer config types.
type SubagentSpec struct {
	Spec AgentSpec
	// Registered reports whether the consumer considers this subagent ID
	// valid for parent-driven invocation. False prevents a parent agent
	// from spawning this child via the task tool.
	Registered bool
	// InheritContext, when true, copies the parent session's view into
	// the child session before BuildRuntime fires.
	InheritContext bool
}

// MakeSubagentFactory returns a tool.SubagentFactory that builds a Runtime
// for the named subagent and adapts it to tool.SubagentRunner. Enforces
// the recursion depth cap before constructing anything.
//
// resolve looks up the subagent's spec at call-time so consumers can
// hot-reload registrations.
func MakeSubagentFactory(
	resolve func(agentID string) (SubagentSpec, bool),
	deps RuntimeDeps,
	buildInputs SubagentBuildFn,
	parent *Runtime,
) tool.SubagentFactory {
	return func(ctx context.Context, agentID string, parentDepth int) (tool.SubagentRunner, error) {
		// Depth-cap enforcement.
		if parentDepth+1 > parent.maxAgentDepth() {
			return nil, fmt.Errorf("subagent depth limit %d reached", parent.maxAgentDepth())
		}
		ss, ok := resolve(agentID)
		if !ok {
			return nil, fmt.Errorf("subagent %q not found", agentID)
		}
		if !ss.Registered {
			return nil, fmt.Errorf("agent %q is not registered as a subagent", agentID)
		}
		inputs, err := buildInputs(ss.Spec)
		if err != nil {
			return nil, fmt.Errorf("subagent %q: build inputs: %w", agentID, err)
		}
		// InheritContext: pre-populate the subagent's fresh session with
		// copies of parent's session entries. The subagent's first LLM
		// call then sees the parent's history; the existing
		// CacheLastMessage path naturally caches the inherited prefix on
		// the subagent's subsequent turns.
		if ss.InheritContext && parent != nil && parent.Session != nil && inputs.Session != nil {
			inheritParentHistory(inputs.Session, parent.Session)
		}
		rt, err := BuildRuntime(deps, inputs, ss.Spec)
		if err != nil {
			return nil, fmt.Errorf("subagent %q: build runtime: %w", agentID, err)
		}
		rt.Parent = parent
		rt.Depth = parentDepth + 1
		return &subagentRunnerAdapter{rt: rt}, nil
	}
}

// inheritParentHistory copies the parent session's current view (post any
// compaction) into the destination subagent session via Append.
//
// Operates outside Append's normal "stitch into leaf" path: the FIRST
// inherited entry gets its ParentID cleared so the subagent's empty leaf
// doesn't cause Append to leave a dangling pointer to a parent entry.
func inheritParentHistory(dst, src *session.Session) {
	view := src.View()
	for i, e := range view {
		if i == 0 {
			e.ParentID = ""
		}
		dst.Append(e)
	}
}

// subagentRunnerAdapter satisfies tool.SubagentRunner by adapting Runtime.Run.
type subagentRunnerAdapter struct{ rt *Runtime }

func (s *subagentRunnerAdapter) Run(ctx context.Context, prompt string) (<-chan tool.AgentEventLike, error) {
	raw, err := s.rt.Run(ctx, prompt, nil)
	if err != nil {
		return nil, err
	}
	out := make(chan tool.AgentEventLike, 16)
	go func() {
		defer close(out)
		for ev := range raw {
			out <- adaptEvent(ev)
		}
	}()
	return out, nil
}

// adaptEvent translates an AgentEvent into the tool.AgentEventLike shape
// that TaskTool understands. Only the fields TaskTool actually inspects
// are filled.
func adaptEvent(ev AgentEvent) tool.AgentEventLike {
	return tool.AgentEventLike{
		Type:    int(ev.Type),
		Text:    ev.Text,
		Done:    ev.Type == EventDone,
		Aborted: ev.Type == EventAborted,
		Err:     ev.Error,
	}
}

// NewSubagentSession returns a fresh in-memory Session for a subagent run.
// SetStore is NOT called — subagent sessions are ephemeral and do not
// write JSONL to disk. The parent's session is the durable record.
func NewSubagentSession(agentID string) *session.Session {
	return session.NewSession(agentID, "subagent")
}
