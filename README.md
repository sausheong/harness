# harness

A reusable Go agentic platform for building LLM agents. Implements the
streaming agent loop, tool registry, session storage, compaction, and
token budgeting needed to run a multi-provider agent in production. BYO
concrete tools, BYO provider clients, BYO memory/knowledge-graph plugins.

> **Status: pre-1.0.** The `runtime` API surface is still being shaped
> — expect breaking changes until a `v0.1.0` tag.

## Why harness

If you're building an LLM agent in Go and you find yourself writing
the same things over and over — a streaming think-act loop, parallel
tool dispatch, prompt-cache-stable request building, summarize-and-
splice context compaction, JSONL session storage, MCP-server glue —
that's what harness is. It's the substrate. You bring the tools, the
prompts, and the binary; it brings the loop.

It is **not** a CLI, a UI, an agent product, or a framework with
opinions about how your agents should be deployed. There is no
`harness` binary. You import packages, compose a `Runtime`, and call
`rt.Run(ctx, msg, nil)`.

## Requirements

- **Go 1.25.1 or newer.** The module declares `go 1.25.1` and uses
  `iter.Seq` from `iter` (Go 1.23+) plus a few stdlib calls that
  hardened in 1.25. Older toolchains will fail at `go build`.
- **An LLM API key**, depending on which provider(s) you wire in:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, Google ADC for Gemini, or a
  Qwen DashScope key. For local models, point `openai.NewOpenAIProviderWithKind`
  at an Ollama endpoint — no key required.
- **Optional: an MCP server** for `tools/mcp` integrations (any
  binary that speaks the Model Context Protocol — `npx
  @modelcontextprotocol/server-filesystem`, etc.).

## Install

```bash
go get github.com/sausheong/harness@latest
```

The library has no submodules — every package is importable directly.
The `tools/` subpackages are individually importable so you can pull
in only what you need (e.g. `tools/file` without `tools/browser`).

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
    ├── mcp/            # Connect to external MCP servers; adapt their tools
    └── todo/           # todo_write
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/sausheong/harness/providers/anthropic"
    "github.com/sausheong/harness/runtime"
    "github.com/sausheong/harness/session"
    "github.com/sausheong/harness/tool"
    "github.com/sausheong/harness/tools/bash"
    "github.com/sausheong/harness/tools/file"
)

func main() {
    reg := tool.NewRegistry()
    reg.Register(&file.ReadFileTool{WorkDir: "/tmp/work"})
    reg.Register(&bash.BashTool{WorkDir: "/tmp/work"})

    rt, err := runtime.BuildRuntime(
        runtime.RuntimeDeps{},
        runtime.RuntimeInputs{
            Provider: anthropic.NewAnthropicProvider(os.Getenv("ANTHROPIC_API_KEY"), ""),
            Tools:    reg,
            Session:  session.NewSession("demo", "main"),
        },
        runtime.AgentSpec{
            ID:           "demo",
            Name:         "Demo",
            Model:        "anthropic/claude-haiku-4-5-20251001",
            Workspace:    "/tmp/work",
            SystemPrompt: "You are a helpful coding assistant.",
            MaxTurns:     25,
        },
    )
    if err != nil {
        panic(err)
    }
    defer rt.Close() // releases MCP sessions if any were declared

    events, _ := rt.Run(context.Background(), "list files in workspace", nil)
    for ev := range events {
        if ev.Type == runtime.EventTextDelta {
            fmt.Print(ev.Text)
        }
    }
}
```

See [`examples/minimal/`](./examples/minimal) for the runnable
end-to-end version, and [`developer.md`](./developer.md) for
how to plug in hooks, MCP servers, skills, permissions, and more.

## Examples

Four end-to-end agents ship in [`examples/`](./examples). Each is a
self-contained `main.go` you can `go run`:

| Example | What it shows | API surface |
|---|---|---|
| [`minimal/`](./examples/minimal) | Smallest useful agent — file + bash + web tools, REPL, streaming output. ~80 lines. | `tool.Registry`, `session.Session`, `anthropic.NewAnthropicProvider`, `runtime.BuildRuntime` |
| [`data-agent/`](./examples/data-agent) | BYO custom tools (no `tools/*` deps) hitting Singapore's open-data APIs (`data.gov.sg`). Includes a `//go:build live` smoke test. | `tool.Tool` interface for 5 custom HTTP-backed tools |
| [`lta-agent/`](./examples/lta-agent) | Same shape as data-agent, talks to LTA Datamall. Demonstrates `IsConcurrencySafe=true` for parallel HTTP fanout. | Custom tools + parallel dispatch via `MaxToolConcurrency` |
| [`support-agent/`](./examples/support-agent) | Mock customer-support agent showing **`SkillProvider` + `PermissionChecker` + `LifecycleHooks`** wired together. KB articles via `//go:embed`; supervisor-mode gating; JSONL audit log. Closest to a real product. | All of the above, plus `runtime.SkillProvider`, `tool.PermissionChecker`, `runtime.LifecycleHooks` (`AfterToolUse` + `OnStop`) |

To run any of them:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run ./examples/minimal
go run ./examples/support-agent
# etc.
```

The `lta-agent` additionally needs `LTA_DATAMALL_KEY`. The
`support-agent` honours `--supervisor` (or
`SUPPORT_AGENT_SUPERVISOR=1`) to unlock write tools and
`SUPPORT_AGENT_AUDIT_LOG=path/to/audit.jsonl` to redirect the audit
trail.

## Configuration

Three environment variables tune the runtime when you don't override
them via `LoopConfig`:

| Variable | Default | What it controls |
|---|---|---|
| `HARNESS_MAX_TOOL_CONCURRENCY` | `10` | Max parallel tool calls within a concurrency-safe batch |
| `HARNESS_MAX_AGENT_DEPTH` | `3` | Max subagent recursion depth (a child at depth N can spawn at N+1) |
| `HARNESS_STREAMING_TOOLS` | unset (off) | When `"1"`, concurrency-safe tools start executing while the model is still streaming text — latency win for I/O-bound tools |

`LoopConfig` field values win over env vars when set. Set the field
to its zero value (or omit it) to fall back to env, then to defaults.

Everything else (provider API keys, model selection, workspace path,
skill paths, etc.) is your config's responsibility — harness reads
none of it.

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
| 8 | **Hooks** | `runtime.LifecycleHooks` (on `LoopConfig.Hooks`): `OnUserPromptSubmit`, `OnSessionStart`, `BeforeToolUse`, `AfterToolUse`, `OnStop`. Go callbacks — type-safe, in-process, nil-friendly |
| 9 | **MCP** | `tools/mcp` package: `mcp.Connect(ctx, ServerConfig)` adapts an external MCP server's tools as `tool.Tool`s. Declare `AgentSpec.MCPServers` for auto-wiring; `Runtime.Close()` releases sessions |
| 10 | **Skills** | `runtime.SkillProvider` (`FormatIndex` + `Get`); auto-registers `load_skill` so the model can pull a skill on demand. See `examples/support-agent/` for a `//go:embed kb/*.md` example |
| 11 | **Session persistence** | `session.Session` (in-memory, append-only) + optional `session.Store` for JSONL persistence and cross-process resume |

Deliberately not in scope (numbers from the same conceptual list):

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
  `PermissionChecker`, `LifecycleHooks`, and `session.Store` are all
  optional. Pass `nil` (or leave the field zero) and the corresponding
  feature disappears with no degraded behaviour.
- **Hooks are Go callbacks, not shell scripts.** `LifecycleHooks`
  fires `OnUserPromptSubmit` (may rewrite the prompt),
  `OnSessionStart`, `BeforeToolUse` (may deny like a
  `PermissionChecker`), `AfterToolUse`, and `OnStop` (with reason:
  `completed | max_turns | error | aborted`). Hooks run synchronously
  on the runtime goroutine — keep them quick.
- **MCP is just another tool source.** `mcp.Connect` returns a
  `Client` whose `Tools()` are regular `tool.Tool`s namespaced as
  `mcp__<server>__<tool>`. Either register them yourself or declare
  `AgentSpec.MCPServers` and let `BuildRuntime` wire them in. Stdio
  (`Command + Args`) and Streamable HTTP (`URL` + optional `Headers`
  / `HTTPClient`) are both supported via the official
  `github.com/modelcontextprotocol/go-sdk`. For OAuth, pass an
  `oauth2.NewClient(ctx, tokenSource)` (or
  `clientcredentials.Config.Client(ctx)`) as `HTTPClient` —
  auto-refresh comes for free. Call `Runtime.Close()` to release
  sessions.
- **Four providers, one interface.** Anthropic, OpenAI (and
  OpenAI-compatible / local Ollama), Gemini, and Qwen ship built-in.
  Adding a fifth is implementing `llm.Provider` (~300 LOC).

See [`developer.md`](./developer.md) for a step-by-step guide to
building agents on top of harness, including diagrams of the
composition surface and the per-Run loop.

## Non-goals

The list of things harness deliberately does **not** do is as much
part of its design as the list of things it does. If you need any of
these, you'll need to bring them yourself or layer them on top:

- **A binary or CLI.** harness is a library. The `examples/` agents
  show how a binary is assembled — pick the one closest to your
  shape and copy it.
- **A UI or TUI.** The runtime emits `<-chan AgentEvent`; rendering
  is the caller's job. Wire it to stdout, a TUI, a websocket, an
  HTTP SSE endpoint, a Slack adapter — whatever you need.
- **Channel adapters.** No WhatsApp / Telegram / Slack / Discord
  bridges. Harness doesn't know what a "channel" is — it speaks
  `(userMsg, images) → events`. Build the bridge in your own
  binary.
- **Hot-reload of MCP servers mid-Run.** MCP servers are connected
  at `BuildRuntime` time. To add or replace one, build a new
  Runtime.
- **Server-side MCP.** The `tools/mcp` package is client-side only —
  it consumes other servers' tools. Exposing harness's tools as an
  MCP server for other agents is a future addition, not a current
  feature.
- **A skill / plugin marketplace.** Skills are `//go:embed` markdown
  in your binary, or whatever your `SkillProvider` resolves at
  runtime. There is no central registry.
- **Auth / billing / multi-tenant infrastructure.** Per-agent
  policy lives in `PermissionChecker`; everything else is your
  problem.
- **Shell-command hooks (Claude Code style).** Hooks are typed Go
  callbacks on `LifecycleHooks`, not subprocess invocations.

## Testing

```bash
go test ./...        # unit tests for every package
go vet ./...         # static checks
go build ./...       # all packages + examples compile
```

Live integration tests (the ones that hit real APIs) are gated
behind the `live` build tag so the default `go test ./...` stays
hermetic:

```bash
LTA_DATAMALL_KEY=... go test -tags live ./examples/lta-agent/...
```

The runtime test suite includes some rare timing-sensitive cases —
if a streaming-tools test flakes once, re-run before treating it as
a real failure.

## Status

Pre-1.0. The `runtime` API surface is still being shaped; expect
breaking changes until a `v0.1.0` tag. Specifically:

- `RuntimeDeps` / `RuntimeInputs` / `AgentSpec` field names may
  shift as new optional plug-points land.
- New `LifecycleHooks` callbacks may be added (additive, but watch
  for new fields if you construct the struct positionally — use
  named fields).
- The `tools/mcp` package's `ServerConfig` may grow new fields for
  the MCP-spec OAuth handshake.

Tagged releases will follow the usual Go module semver discipline
once the v0.x line stabilizes.

## License

MIT — see [`LICENSE`](./LICENSE).
