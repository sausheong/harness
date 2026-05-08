package runtime

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

// RuntimeDeps holds the long-lived dependencies that every Runtime in this
// process shares. Built once at startup and reused for every Runtime
// construction (including subagent runtimes built by the task tool factory).
type RuntimeDeps struct {
	// Skills, when non-nil, contributes a Skills Index to the static
	// system prompt and backs the load_skill tool. Optional.
	Skills SkillProvider
	// Memory, when non-nil, contributes a Memory Index to the static
	// system prompt and backs the load_memory tool. Optional.
	Memory MemoryProvider
	// Permission gates tool execution. nil → allow-all.
	Permission tool.PermissionChecker
	// KGFn returns the per-agent KnowledgeGraph for the given model
	// string. Nil at the call site means KG is disabled for this Runtime.
	// The function shape (rather than a single KG) lets consumers route
	// the chatting agent's own model to the KG's underlying LLM — useful
	// when the KG runs an extractor on the same provider/model as the
	// agent itself.
	KGFn func(model string) KnowledgeGraph
	// AgentLoop carries the loop tunables (concurrency cap, depth cap,
	// streaming-tools toggle). Copied verbatim into every Runtime built
	// here. Zero value → readers fall back to env vars then defaults.
	AgentLoop LoopConfig
	// CalibratorStore is the per-(agentID, sessionKey) persistence layer
	// for the token Calibrator. nil disables persistence; in-memory
	// learning still happens.
	CalibratorStore *tokens.CalibratorStore
	// ConfigSummary, when non-empty, is prepended to the static system
	// prompt's context block. Consumers can use this to advertise the
	// list of configured agents, channel bindings, etc. — anything that
	// changes only on hot-reload.
	ConfigSummary string
	// MemoryFiles, when non-empty, is appended to the static system
	// prompt. Consumers compose this themselves — typical pattern is to
	// concatenate AGENTS.md (and any project-specific identity files)
	// from the workspace and the user's $HOME.
	MemoryFiles string
}

// RuntimeInputs holds the per-Runtime-instance inputs that genuinely vary
// per call site: the resolved LLM provider for this agent's model, the tool
// executor (different per cron/chat/subagent path), the session, the
// per-agent compaction manager, and the IngestSource flag.
type RuntimeInputs struct {
	Provider     llm.LLMProvider
	Tools        tool.Executor
	Session      *session.Session
	Compaction   *compaction.Manager
	IngestSource string // "" | "chat" | "cron"
}

// BuildRuntime constructs a Runtime for the given AgentSpec using the
// supplied deps + inputs. Centralises three patterns:
//  1. Parsing the model identifier (provider/model)
//  2. Parsing the reasoning mode (default-to-off + warn on invalid)
//  3. Resolving the per-agent KnowledgeGraph via deps.KGFn (nil-safe)
//
// Returns the constructed Runtime and a nil error today, but callers MUST
// check the error: the return is reserved for future validation.
func BuildRuntime(deps RuntimeDeps, inputs RuntimeInputs, spec AgentSpec) (*Runtime, error) {
	provider, modelName := llm.ParseProviderModel(spec.Model)
	reasoning, err := llm.ParseReasoningMode(spec.Reasoning)
	if err != nil {
		slog.Error("invalid reasoning mode in agent spec; defaulting to off",
			"agent", spec.ID, "value", spec.Reasoning, "err", err)
		reasoning = llm.ReasoningOff
	}
	var kg KnowledgeGraph
	if deps.KGFn != nil {
		kg = deps.KGFn(spec.Model)
	}

	// Register the on-demand load tools onto the agent's tool registry
	// when the corresponding manager is non-nil. Type-assert to
	// *tool.Registry rather than widening tool.Executor — production
	// callers always pass *tool.Registry; test paths that don't won't
	// get load tools registered (which is fine for them).
	var mcpClients []*mcp.Client
	if reg, ok := inputs.Tools.(*tool.Registry); ok {
		if deps.Skills != nil {
			skills := deps.Skills
			reg.Register(&tool.LoadSkillTool{
				Lookup: func(name string) (string, bool) {
					return skills.Get(name)
				},
			})
		}
		if deps.Memory != nil {
			mem := deps.Memory
			reg.Register(&tool.LoadMemoryTool{
				Lookup: func(id string) (string, bool) {
					return mem.Get(id)
				},
			})
		}
		// Connect declared MCP servers and register their adapter tools.
		// On any failure, close everything connected so far before
		// returning — a partially-built Runtime would leak processes.
		for _, srv := range spec.MCPServers {
			cli, err := mcp.Connect(context.Background(), srv)
			if err != nil {
				for _, c := range mcpClients {
					_ = c.Close()
				}
				return nil, fmt.Errorf("mcp server %q: %w", srv.Name, err)
			}
			for _, t := range cli.Tools() {
				reg.Register(t)
			}
			mcpClients = append(mcpClients, cli)
		}
	} else if len(spec.MCPServers) > 0 {
		return nil, fmt.Errorf("AgentSpec.MCPServers requires inputs.Tools to be *tool.Registry")
	}

	// Pre-compute the static portion of the system prompt so the per-turn
	// hot loop never reads config or rebuilds the indices. The load tools
	// are registered above so toolNames includes them in the default
	// identity tool hints.
	skillsIndex := ""
	if deps.Skills != nil {
		skillsIndex = deps.Skills.FormatIndex()
	}
	memoryIndex := ""
	if deps.Memory != nil {
		memoryIndex = deps.Memory.FormatIndex()
	}
	var toolNames []string
	if inputs.Tools != nil {
		toolNames = inputs.Tools.Names()
	}
	staticPrompt := BuildStaticSystemPrompt(
		spec.Workspace, spec.SystemPrompt, spec.ID, spec.Name,
		toolNames, deps.ConfigSummary, skillsIndex,
		memoryIndex, deps.MemoryFiles,
	)

	// Strip the provider prefix off FallbackModel so the runtime hands
	// the same provider client a bare model id on retry. Cross-provider
	// fallback isn't supported here.
	fallbackModel := ""
	if spec.FallbackModel != "" {
		fbProvider, fbModel := llm.ParseProviderModel(spec.FallbackModel)
		if fbProvider != "" && fbProvider != provider {
			slog.Warn("fallbackModel ignored: cross-provider fallback not supported",
				"agent", spec.ID,
				"primary_provider", provider,
				"fallback", spec.FallbackModel)
		} else {
			fallbackModel = fbModel
		}
	}

	rt := &Runtime{
		LLM:                inputs.Provider,
		Tools:              inputs.Tools,
		Session:            inputs.Session,
		AgentID:            spec.ID,
		AgentName:          spec.Name,
		Model:              modelName,
		FallbackModel:      fallbackModel,
		ContextWindow:      spec.ContextWindow,
		Provider:           provider,
		Reasoning:          reasoning,
		Workspace:          spec.Workspace,
		MaxTurns:           spec.MaxTurns,
		SystemPrompt:       spec.SystemPrompt,
		Skills:             deps.Skills,
		Memory:             deps.Memory,
		KG:                 kg,
		Permission:         deps.Permission,
		Compaction:         inputs.Compaction,
		IngestSource:       inputs.IngestSource,
		AgentLoop:          deps.AgentLoop,
		StaticSystemPrompt: staticPrompt,
		CalibratorStore:    deps.CalibratorStore,
		mcpClients:         mcpClients,
	}

	// Seed the calibrator from prior (ratio, count) for this session so a
	// long session that's been split across many rebuilds retains its
	// learned chars→tokens ratio. Skipped for subagent sessions and when
	// no store is configured.
	if deps.CalibratorStore != nil && inputs.Session != nil && inputs.Session.Key != "" && inputs.Session.Key != "subagent" {
		ratio, count := deps.CalibratorStore.Load(spec.ID, inputs.Session.Key)
		if count > 0 {
			rt.calibrator = tokens.NewCalibrator()
			rt.calibrator.Restore(ratio, count)
		}
	}

	return rt, nil
}
