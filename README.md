# harness

A reusable Go agentic platform for building LLM agents. Implements the
streaming agent loop, tool registry, session storage, compaction, and
token budgeting needed to run a multi-provider agent in production. BYO
concrete tools, BYO provider clients, BYO memory/knowledge-graph plugins.

## Packages

```
github.com/sausheong/harness/
├── llm/                # LLMProvider interface, Message/ToolDef/ChatRequest types
├── session/            # Append-only session DAG (Session, SessionEntry, Store)
├── tokens/             # char/4 estimator + Calibrator + CalibratorStore
├── compaction/         # Three-stage summarize-and-splice manager
├── tool/               # Tool interface, Executor, Registry, PermissionChecker,
│                       # SubagentFactory/Runner, JobScheduler, LoadSkillTool,
│                       # LoadMemoryTool, TaskTool, CronTool
├── runtime/            # Runtime, Run loop, partition, subagent factory,
│                       # AgentSpec, RuntimeDeps, RuntimeInputs, LoopConfig,
│                       # MemoryProvider/SkillProvider/KnowledgeGraph interfaces
├── providers/
│   ├── anthropic/      # Anthropic LLMProvider (with prompt caching)
│   ├── openai/         # OpenAI / OpenAI-compatible / local Ollama
│   ├── gemini/         # Google Gemini via google.golang.org/genai
│   └── qwen/           # Alibaba Qwen (OpenAI-compatible endpoint)
└── tools/              # Batteries-included concrete tools (each importable separately)
    ├── file/           # read_file (with vision), write_file, edit_file
    ├── bash/           # bash (with ExecPolicy: deny | allowlist | full)
    ├── web/            # web_fetch, web_search, ssrf guard
    ├── browser/        # chromedp wrapper with per-session reuse
    └── todo/           # todo_write
```

## Quick start

```go
import (
    "context"
    "github.com/sausheong/harness/runtime"
    "github.com/sausheong/harness/session"
    "github.com/sausheong/harness/tool"
    "github.com/sausheong/harness/providers/anthropic"
    "github.com/sausheong/harness/tools/file"
    "github.com/sausheong/harness/tools/bash"
)

func main() {
    reg := tool.NewRegistry()
    reg.Register(&file.ReadFileTool{WorkDir: "/tmp/work"})
    reg.Register(&bash.BashTool{WorkDir: "/tmp/work"})

    rt, _ := runtime.New(
        runtime.AgentSpec{
            ID: "demo", Name: "Demo",
            Model: "claude-haiku-4-5-20251001",
            Workspace: "/tmp/work",
            SystemPrompt: "You are a helpful coding assistant.",
            MaxTurns: 25,
        },
        runtime.Deps{
            LLM:     anthropic.New("sk-ant-..."),
            Tools:   reg,
            Session: session.NewSession("demo", "main"),
        },
    )

    events, _ := rt.Run(context.Background(), "list files in workspace", nil)
    for ev := range events {
        if ev.Type == runtime.EventTextDelta {
            print(ev.Text)
        }
    }
}
```

## Components

A general agent harness has ~13 conceptual components: the loop, the
model invocation layer, the tool registry, permission gating,
context/compaction, the system prompt, sub-agents, hooks, MCP, skills,
session persistence, UI, and entrypoints. harness implements the
in-process, library-shaped subset of those. The rest is left to the
caller — you own the binary, you own the UI, you own the channel.

| # | Component | Where it lives in harness |
|---|---|---|
| 1 | **Agent loop** | `runtime.Runtime.Run()` — single goroutine, streaming-first; `<-chan ChatEvent` |
| 2 | **Model invocation** | `llm.Provider` interface + `ChatStream`; non-streaming fallback retries the same `ChatRequest` byte-for-byte to preserve the prompt-cache prefix |
| 3 | **Tool registry & schemas** | `tool.Registry`; `ToolDefs()` is sorted by name so the request prefix is stable across turns (cache hit) |
| 4 | **Tool execution & permissions** | `tool.PermissionChecker` (`Check` + `FilterToolDefs`); concurrency-safe partitioning via `Tool.IsConcurrencySafe` + `LoopConfig.MaxToolConcurrency` (default 10) |
| 5 | **Context & compaction** | `compaction.Manager` — summarize-and-splice at a clean user-message boundary; tool-result pruning at request time |
| 6 | **System prompt assembly** | `llm.SystemPromptPart` — splits the static cacheable prefix from the per-turn dynamic suffix; provider-side `cache_control` placement is automatic |
| 7 | **Sub-agents / delegation** | `runtime.SubagentResolver` interface + `MaxAgentDepth=3` cap; subagents run as in-process goroutines with their own `Runtime` |
| 10 | **Skills** | `runtime.SkillProvider` (`FormatIndex` + `Get`); auto-registers `load_skill` so the model can pull a skill on demand. See `examples/support-agent/` for a `//go:embed kb/*.md` example |
| 11 | **Session persistence** | `session.Session` (in-memory, append-only) + optional `session.Store` for JSONL persistence and cross-process resume |

Deliberately not in scope (numbers from the same conceptual list):

- **8 — Hooks**: no PreToolUse/PostToolUse/Stop hooks. If you need
  them, wrap the event channel.
- **9 — MCP**: not built in. Register MCP-backed tools manually as
  `tool.Tool` implementations.
- **12 — UI**: harness emits events; the caller renders.
- **13 — Entrypoints**: harness is a library. There is no `harness`
  binary. The example agents in `examples/` show how a binary is
  assembled.

## Design notes

- **Streaming-first.** The loop yields `EventTextDelta`,
  `EventToolCallStart`, `EventToolResult`, `EventDone`, and friends
  through a `<-chan ChatEvent`. There is no buffered "give me the
  whole response" path — even the non-streaming fallback re-emits
  through the same channel.
- **Prompt-cache discipline.** Tool definitions are sorted, system
  prompt parts are split into cached/dynamic blocks, and
  `buildMessageParams` is a pure function (proven by
  `TestBuildMessageParamsIsPure` in
  `providers/anthropic/anthropic_test.go`). The non-streaming retry
  path depends on byte-identical params to keep the cache prefix
  valid.
- **Compaction at a clean boundary.** `compaction.Split` walks
  backward counting user messages and cuts at one — never inside a
  tool-call/tool-result pair. The prompt itself (`compaction/prompt.go`)
  is adapted from Claude Code's `BASE_COMPACT_PROMPT` and emits a
  9-section structured summary inside an `<analysis>` + `<summary>`
  envelope; the `<analysis>` block is stripped before re-injection.
- **Plug-points are nil-friendly.** `MemoryProvider`,
  `KnowledgeGraph`, `SkillProvider`, `SubagentResolver`,
  `PermissionChecker`, and `session.Store` are all optional. Pass
  `nil` and the corresponding feature disappears with no degraded
  behaviour.
- **Four providers, one interface.** Anthropic, OpenAI (and
  OpenAI-compatible / local Ollama), Gemini, and Qwen ship built-in.
  Adding a fifth is implementing `llm.Provider` (~300 LOC).

See [`developer.md`](./developer.md) for a step-by-step guide to
building agents on top of harness.

## Status

Pre-1.0. The `runtime` API surface is still being shaped; expect
breaking changes until a v0.1.0 tag.

## License

MIT.
