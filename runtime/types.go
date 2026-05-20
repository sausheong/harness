package runtime

import (
	"context"
	"encoding/json"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

// AgentSpec is the framework's view of an agent. Consumers convert their
// own agent-config types into this struct before invoking BuildRuntime.
//
// Holds only the fields Runtime actually consumes. Anything richer that
// a consumer's config carries (per-agent tool policies, MCP server
// bindings, channel routing, inheritContext flags, subagent
// registration, identity files) is resolved by the consumer before this
// struct is filled in.
type AgentSpec struct {
	// ID identifies the agent for logging, session paths, and subagent
	// resolution. Used as session.Session.AgentID.
	ID string
	// Name is the human-readable display name (shown in chat UIs).
	Name string
	// Model is the "provider/model" string parsed by ParseProviderModel
	// (e.g., "anthropic/claude-haiku-4-5"). The provider portion is exposed
	// on Runtime.Provider for caching capability checks.
	Model string
	// FallbackModel is the bare model name used on transient model errors
	// (Anthropic 429/529, OpenAI 429/5xx). Same provider as Model.
	FallbackModel string
	// Workspace is the agent's working directory — the root for the
	// post-compact file-restore path and the spill directory.
	Workspace string
	// SystemPrompt overrides the built-in default identity. Empty ⇒
	// BuildStaticSystemPrompt composes a default identity string from
	// the registered tool names.
	SystemPrompt string
	// MaxTurns caps the tool-use loop (default 25 when 0).
	MaxTurns int
	// ContextWindow overrides the auto-detected window from
	// tokens.ContextWindow. 0 ⇒ auto-detect.
	ContextWindow int
	// Reasoning maps to each provider's native reasoning knob. Use
	// llm.ReasoningOff (zero value), Low, Medium, or High.
	Reasoning string
	// Loop controls the runtime's tool-execution behavior (concurrency
	// cap, depth cap, streaming-tools toggle). Zero value ⇒ env-var
	// fallback then compiled-in defaults.
	Loop LoopConfig
	// MCPServers, when non-empty, are connected by BuildRuntime: each
	// server's tools are namespaced ("mcp__<Name>__<tool>") and added
	// to the agent's tool registry. Runtime.Close releases the
	// underlying sessions. Connection failures abort BuildRuntime.
	MCPServers []mcp.ServerConfig
}

// LoopConfig tunes the agent runtime's tool-execution behavior. Each
// field has a built-in env-var fallback (HARNESS_MAX_TOOL_CONCURRENCY,
// HARNESS_MAX_AGENT_DEPTH, HARNESS_STREAMING_TOOLS); set the field
// explicitly from your own config to bypass the env vars.
type LoopConfig struct {
	// MaxToolConcurrency caps parallel tool dispatch within a safe batch.
	// 0 ⇒ env fallback then default 10.
	MaxToolConcurrency int
	// MaxAgentDepth caps subagent recursion depth. 0 ⇒ env fallback
	// then default 3.
	MaxAgentDepth int
	// StreamingTools enables mid-stream concurrency-safe tool kickoff.
	// false ⇒ env fallback (HARNESS_STREAMING_TOOLS=1) then off.
	StreamingTools bool
	// MaxToolResultLen caps the in-context length of any single tool
	// result; longer results are truncated to their first MaxToolResultLen
	// chars (cut at a newline boundary) with the remainder either spilled
	// to disk via spillConfig or simply dropped. 0 ⇒ env fallback
	// (HARNESS_MAX_TOOL_RESULT_LEN) then default 4000. Common pick for
	// engineering-style agents (reading source files, running tests) is
	// 16000-25000.
	MaxToolResultLen int
	// Hooks are optional callbacks the runtime fires at well-known
	// points in the loop. Zero value (all nil fields) disables every
	// hook with no overhead.
	Hooks LifecycleHooks
}

// HookDecision is the outcome of a BeforeToolUse hook. Mirrors
// tool.Decision so callers familiar with PermissionChecker get a
// consistent shape: Allow=true permits the call to continue (still
// subject to PermissionChecker); Allow=false denies it with Reason
// surfaced as the tool result error.
type HookDecision struct {
	Allow  bool
	Reason string
}

// LifecycleHooks bundles optional callbacks the runtime fires at
// well-known points. All fields are nil-safe — leave a field nil and
// that hook is skipped. Hooks run synchronously on the runtime
// goroutine; expensive work should be deferred by the implementation.
type LifecycleHooks struct {
	// OnUserPromptSubmit fires once at the top of Run, BEFORE the
	// user message is appended to the session. The hook may rewrite
	// the prompt and/or images; the rewritten values are what the
	// session and the LLM see. Returning err != nil aborts Run with
	// EventError.
	OnUserPromptSubmit func(ctx context.Context, prompt string, images []llm.ImageContent) (string, []llm.ImageContent, error)

	// OnSessionStart fires once at the top of Run, AFTER the user
	// message is appended to the session. Observe-only.
	OnSessionStart func(ctx context.Context, sess *session.Session)

	// BeforeToolUse fires inside dispatchTool / executeToolKickoff
	// BEFORE PermissionChecker. Returning Allow=false denies the call
	// with the given Reason (same shape as a PermissionChecker
	// denial). Returning err != nil treats the call as a denial with
	// err.Error() as the reason. nil → no-op (call proceeds).
	BeforeToolUse func(ctx context.Context, toolName string, input json.RawMessage) (HookDecision, error)

	// AfterToolUse fires after the tool's Execute returns (including
	// on error or denial). Observe-only — no result rewrite.
	AfterToolUse func(ctx context.Context, toolName string, input json.RawMessage, result tool.ToolResult)

	// OnStop fires exactly once at the end of Run, regardless of
	// outcome. reason is one of: "completed", "max_turns", "error",
	// "aborted". Observe-only.
	OnStop func(ctx context.Context, reason string)
}

// SkillProvider lets the runtime advertise the available skills in the
// static system prompt and resolve a skill body on demand via the
// load_skill tool. Optional — pass nil on Deps.Skills to disable.
//
// FormatIndex is called once at BuildRuntime time; the returned string
// is concatenated into the cacheable static system prompt. Get is wired
// through tool.LoadSkillTool's Lookup closure; load_skill.Execute
// calls it at agent-loop time.
type SkillProvider interface {
	FormatIndex() string
	Get(name string) (body string, ok bool)
}

// MemoryProvider mirrors SkillProvider for an on-demand memory-entries
// store. FormatIndex contributes to the static prompt; Get backs the
// load_memory tool. Optional.
type MemoryProvider interface {
	FormatIndex() string
	Get(id string) (body string, ok bool)
}

// Message is a minimal conversation tuple the runtime hands to the
// KnowledgeGraph. The runtime accumulates these inside Run as the LLM
// produces text, calls tools, and consumes results. Implementations may
// translate to their own domain types.
type Message struct {
	Role    string // "user" | "assistant"
	Content string
}

// KnowledgeGraph is the optional plug point for a long-term memory /
// knowledge-graph backend. Wire your own implementation if you need
// recall + ingest hooks; leave deps.KGFn nil to disable the entire
// pathway.
//
//   - ShouldRecall is called synchronously at the start of every Run
//     before scheduling the recall goroutine. Cheap; should return
//     false for trivial messages ("ok", "thanks", greetings) where
//     recall would not help.
//
//   - Recall runs in a background goroutine. The runtime caps the wait
//     at 800ms; implementations should respect ctx cancellation. The
//     returned string is concatenated verbatim into the dynamic system
//     prompt suffix — return "" for no hint, or a pre-formatted block
//     ready for the prompt.
//
//   - Ingest fires deferred-async at the end of Run with the full
//     conversation thread. The runtime calls it with a fresh
//     context.Background — the request ctx may already be cancelled.
//     Implementations decide sync/async batching internally.
//
// Pass nil on Deps.KG to disable the entire pathway.
type KnowledgeGraph interface {
	ShouldRecall(query string) bool
	Recall(ctx context.Context, query string) string
	Ingest(ctx context.Context, thread []Message)
}

// SubagentResolver resolves a subagent ID to its AgentSpec. The runtime
// calls this inside MakeSubagentFactory before constructing a
// child Runtime. Returning ok=false makes TaskTool surface a "subagent
// %q not found" error to the parent LLM.
//
// Implementations typically read from a live config object (often via
// an atomic.Value) so subagent definitions hot-reload without
// restarting the runtime.
type SubagentResolver func(agentID string) (spec AgentSpec, registered bool, ok bool)
