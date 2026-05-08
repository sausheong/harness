# Writing agents with harness

This guide walks you through building agents on top of `harness`. It assumes
you've read [`README.md`](./README.md) and have a Go 1.25+ project that depends
on `github.com/sausheong/harness`.

> Two complete examples ship in [`examples/`](./examples) — `minimal/` (file +
> bash + web tools) and `lta-agent/` and `data-agent/` (BYO tools talking to
> live Singapore government APIs). Read them after this guide.

---

## 1. Mental model

There are two views worth holding in your head: **how a Runtime is
composed** (what you assemble at boot) and **what one Run does** (the
think-act loop that fires per user message).

### Composition: what you wire to build a Runtime

`runtime.BuildRuntime(deps, inputs, spec)` takes three arguments. The
required pieces are inside `inputs`; everything in `deps` is optional;
`spec` is per-agent config.

```
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │ llm.LLMProvider│  │ *tool.Registry │  │ *session.Session│
   │ (anthropic /   │  │ (built-in tools│  │ (in-memory or   │
   │  openai / …)   │  │  + your own)   │  │  JSONL-backed)  │
   └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
            │                   │                   │
            └────────┬──────────┴───────────────────┘
                     ▼
              RuntimeInputs                  ◄── required
                     │
                     │
   RuntimeDeps  ────►├◄──── AgentSpec
   (optional)        │      (id, name, model,
   • Skills          │       workspace, MaxTurns,
   • Memory          │       SystemPrompt,
   • Permission      │       LoopConfig{Hooks…},
   • KGFn            │       MCPServers, …)
   • LifecycleHooks  │
   • Compaction      │
   • CalibratorStore │
                     ▼
            runtime.BuildRuntime
                     │
                     ▼
              runtime.Runtime
                     │
                     │ events, _ := rt.Run(ctx, userMsg, images)
                     ▼
            <-chan runtime.AgentEvent
        (EventTextDelta, EventToolCallStart,
         EventToolResult, EventDone, EventError, EventAborted)
```

The four mandatory pieces:

* **`llm.LLMProvider`** (in `RuntimeInputs.Provider`) — a streaming
  chat client (Anthropic / OpenAI / Gemini / Qwen, or your own).
* **`tool.Registry`** (in `RuntimeInputs.Tools`) — the tools the LLM
  may invoke. Each tool implements `tool.Tool`.
* **`session.Session`** (in `RuntimeInputs.Session`) — append-only
  conversation history. In-memory by default; attach a
  `*session.Store` to persist as JSONL on disk.
* **`runtime.AgentSpec`** — per-agent identity (id, model, workspace,
  system prompt, MaxTurns, LoopConfig, MCPServers).

Everything in `RuntimeDeps` is optional. Leave a field zero (or pass
`nil`) and the corresponding subsystem disappears.

### Per-Run: what `rt.Run(ctx, msg, images)` actually does

```
   userMsg, images
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Hooks.OnUserPromptSubmit  (may rewrite prompt/images)       │
   │  Session.Append(user message)                                │
   │  Hooks.OnSessionStart                                        │
   │  KG.Recall (optional, ≤800ms cap)                            │
   └──────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌── for turn := 0; turn < MaxTurns; turn++ ────────────────────┐
   │                                                              │
   │   Provider.ChatStream(messages, system, tools)               │
   │              │                                               │
   │              │  streams text + tool_use blocks               │
   │              ▼                                               │
   │   ┌─────────────────────────────────────────────┐            │
   │   │  emit EventTextDelta as text arrives        │            │
   │   │  collect tool_use blocks                    │            │
   │   │  (StreamingTools? execute concurrency-safe  │            │
   │   │   tools BEFORE stream ends)                 │            │
   │   └─────────────────────────────────────────────┘            │
   │              │                                               │
   │              ▼                                               │
   │   no tool calls? ───► emit EventDone, return                 │
   │              │                                               │
   │              ▼ tool calls present                            │
   │                                                              │
   │   partition into concurrency-safe batches                    │
   │   for each batch (parallel up to MaxToolConcurrency):        │
   │     • Hooks.BeforeToolUse  (may deny)                        │
   │     • Permission.Check     (may deny)                        │
   │     • tool.Execute                                           │
   │     • Hooks.AfterToolUse   (observe)                         │
   │     • Session.Append(tool_call + tool_result)                │
   │     • emit EventToolResult                                   │
   │                                                              │
   │   loop: feed tool_results back to Provider for next turn     │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Hooks.OnStop(reason)                                        │
   │      reason ∈ {completed, max_turns, error, aborted}         │
   │  KG.Ingest (deferred-async, optional)                        │
   │  Compaction.MaybeCompactAsync (if near threshold)            │
   └──────────────────────────────────────────────────────────────┘
```

This is what every section in the rest of this guide is hooking into
or extending.

---

## 2. Minimum viable agent

The smallest useful agent: REPL + Anthropic + the bash tool.

```go
package main

import (
    "bufio"
    "context"
    "fmt"
    "os"
    "path/filepath"

    "github.com/sausheong/harness/providers/anthropic"
    "github.com/sausheong/harness/runtime"
    "github.com/sausheong/harness/session"
    "github.com/sausheong/harness/tool"
    "github.com/sausheong/harness/tools/bash"
)

func main() {
    workspace, _ := filepath.Abs("./_scratch")
    _ = os.MkdirAll(workspace, 0o755)

    reg := tool.NewRegistry()
    reg.Register(&bash.BashTool{WorkDir: workspace})

    rt, err := runtime.BuildRuntime(
        runtime.RuntimeDeps{},                        // no Skills/Memory/KG/Compaction
        runtime.RuntimeInputs{
            Provider: anthropic.NewAnthropicProvider(os.Getenv("ANTHROPIC_API_KEY"), ""),
            Tools:    reg,
            Session:  session.NewSession("demo", "main"),
        },
        runtime.AgentSpec{
            ID:           "demo",
            Name:         "Demo",
            Model:        "anthropic/claude-haiku-4-5-20251001",
            Workspace:    workspace,
            SystemPrompt: "You are a concise shell assistant. Use bash to inspect the workspace.",
            MaxTurns:     10,
        },
    )
    if err != nil {
        panic(err)
    }

    in := bufio.NewScanner(os.Stdin)
    for {
        fmt.Print("> ")
        if !in.Scan() || in.Text() == "" {
            return
        }
        events, _ := rt.Run(context.Background(), in.Text(), nil)
        for ev := range events {
            switch ev.Type {
            case runtime.EventTextDelta:
                fmt.Print(ev.Text)
            case runtime.EventDone:
                fmt.Println()
            }
        }
    }
}
```

That's it. Three required dependencies (`Provider`, `Tools`, `Session`), a
spec, and a streaming loop.

---

## 3. The four core types

### `runtime.AgentSpec`

Static configuration for one agent. Built once per Runtime construction.

| Field           | Required | Notes                                              |
|-----------------|----------|----------------------------------------------------|
| `ID`            | yes      | Used for logging, session paths, subagent lookup   |
| `Name`          | yes      | Display name                                       |
| `Model`         | yes      | `"provider/model"` — e.g. `"anthropic/claude-haiku-4-5"` |
| `Workspace`     | rec.     | Working directory for file/bash tools              |
| `SystemPrompt`  | rec.     | Empty → harness composes a default identity        |
| `MaxTurns`      | no       | Tool-loop cap. 0 → 25                              |
| `ContextWindow` | no       | 0 → auto-detect from model id                      |
| `Reasoning`     | no       | `""`/`"low"`/`"medium"`/`"high"` (see `llm.ReasoningMode`) |
| `FallbackModel` | no       | Bare model name for transient-error retries (same provider) |

### `runtime.RuntimeDeps`

Long-lived deps shared across every Runtime in a process. Can be reused.

```go
deps := runtime.RuntimeDeps{
    AgentLoop: runtime.LoopConfig{
        MaxToolConcurrency: 3,    // parallel tool calls per safe batch (default 10)
        MaxAgentDepth:      1,    // subagent recursion cap (default 3)
        StreamingTools:     false,// kick off tools mid-stream (default off)
    },
    Skills:          nil,         // optional SkillProvider
    Memory:          nil,         // optional MemoryProvider
    KGFn:            nil,         // optional func(model) KnowledgeGraph
    Permission:      nil,         // nil → allow-all
    CalibratorStore: nil,         // optional token-calibrator persistence
    ConfigSummary:   "",          // prepended to system prompt context block
    MemoryFiles:     "",          // appended to system prompt
}
```

### `runtime.RuntimeInputs`

Per-Runtime inputs that vary per construction.

```go
inputs := runtime.RuntimeInputs{
    Provider:     anthropic.NewAnthropicProvider(key, ""),
    Tools:        reg,                                // tool.Executor
    Session:      session.NewSession("agent-id", "main"),
    Compaction:   nil,                                // optional *compaction.Manager
    IngestSource: "",                                 // "", "chat", or "cron"
}
```

### `runtime.Runtime`

What you actually call. One method matters:

```go
rt.Run(ctx context.Context, userMsg string, images []llm.ImageContent) (<-chan runtime.AgentEvent, error)
```

Returns a channel of events; closes when the turn (and any internal tool
loop) completes.

---

## 4. Writing your own tool

Tools implement `tool.Tool`:

```go
type Tool interface {
    Name() string
    Description() string
    Parameters() json.RawMessage          // JSON Schema
    Execute(ctx context.Context, input json.RawMessage) (ToolResult, error)
    IsConcurrencySafe(input json.RawMessage) bool
}
```

A worked example — a tool that fetches a single environment variable from
the host and returns it:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "os"

    "github.com/sausheong/harness/tool"
)

type EnvTool struct{}

func (*EnvTool) Name() string { return "get_env" }

func (*EnvTool) Description() string {
    return "Read a single environment variable from the host. Returns the " +
        "value or an empty string if the variable is unset."
}

func (*EnvTool) Parameters() json.RawMessage {
    return json.RawMessage(`{
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Variable name, e.g. 'PATH'."}
        },
        "required": ["name"]
    }`)
}

// Pure read, no shared state mutation → safe to run alongside other safe tools.
func (*EnvTool) IsConcurrencySafe(json.RawMessage) bool { return true }

func (*EnvTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
    var in struct{ Name string `json:"name"` }
    if err := json.Unmarshal(input, &in); err != nil {
        return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
    }
    if in.Name == "" {
        return tool.ToolResult{Error: "name is required"}, nil
    }
    return tool.ToolResult{Output: os.Getenv(in.Name)}, nil
}
```

Register it:

```go
reg := tool.NewRegistry()
reg.Register(&EnvTool{})
```

### Tool-design rules of thumb

* **Return errors as `ToolResult.Error`, not Go errors**, unless something
  is genuinely unrecoverable (network down, disk full). The LLM reads
  `Error` and can retry or pivot. Returning a Go error from `Execute` aborts
  the run.
* **`IsConcurrencySafe`** is consulted by the partitioner. Return `true`
  for pure reads (HTTP GETs, file reads, lookups). Return `false` for
  anything that mutates state or has ordering sensitivity (file writes,
  shell, write-back caches). Tools with the same name across a batch are
  always serialized regardless.
* **`Description`** is the LLM's only documentation. Spell out *when* to
  use the tool, what its inputs mean, and what the output looks like.
  Reference other tools by name where flow matters
  (e.g. `data-agent/tools.go` chains `datagov_search_datasets` →
  `datagov_query_dataset`).
* **`Parameters`** must be valid JSON Schema. Constrain types, mark
  required fields, use `enum` for closed sets — every constraint here
  catches a class of LLM mistakes before `Execute` runs.
* **Don't panic.** The partitioner wraps `IsConcurrencySafe` in
  `recover()`, but a panic inside `Execute` will tear down the run.

### Returning images

Vision-capable models can see images returned from a tool. Set
`ToolResult.Images`:

```go
return tool.ToolResult{
    Output: "screenshot captured",
    Images: []llm.ImageContent{{
        MediaType: "image/png",
        Data:      base64Bytes,
    }},
}, nil
```

The runtime forwards them to the model on the next turn.

---

## 5. Picking a provider

Each provider has its own constructor. All return an `llm.LLMProvider`.

| Provider     | Constructor                                            | Env / args                          |
|--------------|--------------------------------------------------------|-------------------------------------|
| Anthropic    | `anthropic.NewAnthropicProvider(apiKey, baseURL)`      | `baseURL == ""` → official endpoint |
| OpenAI       | `openai.NewOpenAIProvider(apiKey, baseURL)`            | `baseURL == ""` → official endpoint |
| OpenAI-compat| `openai.NewOpenAIProviderWithKind(apiKey, baseURL, "ollama"\|"compat")` | for local Ollama, etc.   |
| Gemini       | `gemini.NewGeminiProvider(ctx, apiKey)`                | takes `context.Context`             |
| Qwen         | `qwen.NewQwenProvider(apiKey, baseURL)`                | OpenAI-compatible endpoint          |

The provider you pass on `RuntimeInputs.Provider` only needs to match the
provider prefix in `AgentSpec.Model` — `runtime.BuildRuntime` parses
`"anthropic/claude-haiku-4-5"` into `provider="anthropic"` and
`model="claude-haiku-4-5"` and hands the bare model id to the provider on
each call. Cross-provider fallback is *not* supported via `FallbackModel`
— it must use the same provider as `Model`.

### Bringing your own provider

`llm.LLMProvider` is small:

```go
type LLMProvider interface {
    ChatStream(ctx context.Context, req ChatRequest) (<-chan ChatEvent, error)
    Models() []ModelInfo
    NormalizeToolSchema(tools []ToolDef) ([]ToolDef, []Diagnostic)
}
```

`llm/llmtest` has helpers (`MockProvider`, scripted-event sources) you can
use to drive tests without hitting a live API.

---

## 6. Sessions: in-memory vs persistent

By default `session.NewSession(agentID, key)` is in-memory only. Attach a
store to persist as JSONL on disk:

```go
store := session.NewStore("/var/lib/myapp/sessions")
sess, err := store.Load("demo", "main")     // creates if absent
if err != nil {
    panic(err)
}
sess.SetStore(store)                         // every Append flushes a JSONL line
```

Load semantics:

* **`store.Load(agentID, key)`** — returns the session, creating it on
  disk if missing.
* **`sess.SetStore(store)`** — wires append-time persistence. Without this
  the session lives in RAM only.
* The on-disk file is a strict append-log of `SessionEntry` records — no
  schema migrations required. Backups are `cp`.

For subagents you almost always want
`runtime.NewSubagentSession(agentID)` — a fresh in-memory session that
deliberately does not call `SetStore` (the parent's session is the durable
record).

---

## 7. The streaming event loop

`Run` returns a channel of `runtime.AgentEvent`. Drain it; the channel
closes when the turn completes.

```go
events, err := rt.Run(ctx, "find files larger than 1MB in the workspace", nil)
if err != nil {
    return err
}
for ev := range events {
    switch ev.Type {
    case runtime.EventTextDelta:        // streamed assistant text
        fmt.Print(ev.Text)
    case runtime.EventToolCallStart:    // model decided to call a tool
        fmt.Printf("\n  [%s] ", ev.ToolCall.Name)
    case runtime.EventToolResult:       // tool returned
        if ev.Result != nil && ev.Result.Error != "" {
            fmt.Printf("✗ %s", ev.Result.Error)
        } else {
            fmt.Print("✓")
        }
    case runtime.EventDone:             // turn complete
        if ev.Usage != nil {
            log.Printf("[%d in / %d out tokens]", ev.Usage.InputTokens, ev.Usage.OutputTokens)
        }
    case runtime.EventError:            // fatal — turn ended early
        log.Printf("error: %v", ev.Error)
    case runtime.EventAborted:          // ctx cancelled mid-turn
        log.Print("aborted")
    case runtime.EventCompactionStart, runtime.EventCompactionDone, runtime.EventCompactionSkipped:
        // only fire if you wired a compaction.Manager into RuntimeInputs
    }
}
```

You **must** drain the channel to completion, even on early exit — the
runtime relies on that to write paired session entries (especially when
`StreamingTools` is on).

---

## 8. The LoopConfig knobs

```go
type LoopConfig struct {
    MaxToolConcurrency int            // 0 → env HARNESS_MAX_TOOL_CONCURRENCY → 10
    MaxAgentDepth      int            // 0 → env HARNESS_MAX_AGENT_DEPTH      → 3
    StreamingTools     bool           // false → env HARNESS_STREAMING_TOOLS=="1" → off
    Hooks              LifecycleHooks // zero value = all hooks disabled
}
```

* **`MaxToolConcurrency`** — When the model emits multiple tool calls in
  one turn, the partitioner groups them into "safe batches" of up to
  `MaxToolConcurrency` calls (each batch contains only tools whose
  `IsConcurrencySafe` returned `true`, with no within-batch name
  duplicates). Batches run in parallel inside their group, then move to
  the next batch sequentially.
* **`MaxAgentDepth`** — How deeply the `task` tool can spawn child
  agents. A child at depth `n` can spawn children at depth `n+1`; the
  cap rejects anything beyond.
* **`StreamingTools`** — When true, concurrency-safe tools start
  executing as soon as the model finishes emitting their call, *while*
  the model is still streaming text. Latency win for I/O-bound tools;
  harmless to keep off until you measure that you need it.
* **`Hooks`** — Lifecycle callbacks fired on key events
  (user prompt submitted, tool about to run, tool result, run
  finished). Use these for audit logging, dynamic gating, prompt
  rewriting. See section 10 (`LifecycleHooks`) for the full surface.

---

## 9. Subagents

`harness` ships a `task` tool (`tool.TaskTool`) that lets the parent agent
spawn child agents to handle subtasks. It's wired through a factory:

```go
import "github.com/sausheong/harness/tool"

scheduler := myJobScheduler{}                      // implements tool.JobScheduler
factory := runtime.MakeSubagentFactory(
    func(id string) (runtime.SubagentSpec, bool) {
        spec, ok := myConfig.Subagents[id]
        return runtime.SubagentSpec{
            Spec:           spec,                  // resolved AgentSpec
            Registered:     true,                  // gate parent-driven invocation
            InheritContext: false,                 // copy parent history into child?
        }, ok
    },
    deps,
    func(spec runtime.AgentSpec) (runtime.RuntimeInputs, error) {
        // Build a fresh tool registry, provider, and session for this child.
        return runtime.RuntimeInputs{
            Provider: providerFor(spec.Model),
            Tools:    childToolRegistry(spec),
            Session:  runtime.NewSubagentSession(spec.ID),
        }, nil
    },
    parentRuntime,
)
reg.Register(&tool.TaskTool{Factory: factory})
```

A few things worth knowing:

* The factory **builds child inputs lazily**, on every spawn. You decide
  per-spec which tools the child gets — restrict aggressively.
* Child events (`EventTextDelta`, etc.) are forwarded up to the parent's
  channel with `AgentEvent.AgentID` set to the child's id, so a single
  consumer loop sees everything.
* Set `InheritContext: false` unless you specifically need the child to
  see parent history. Inheritance copies the parent's view into the
  child's session and increases token cost on every child turn.

---

## 10. Optional plug points

All four are off if their field is `nil`. Wire them only when you need
them.

### Permission (`tool.PermissionChecker`)

Per-agent allow/deny rules:

```go
deps.Permission = tool.NewStaticChecker(map[string]tool.Policy{
    "demo": {Allowed: []string{"read_file", "bash"}}, // others denied
    // unmapped agents get allow-all
})
```

The checker is consulted **before** every tool call and also at
tool-list-assembly time, so denied tools never appear in the model's tool
list — preventing wasted tool-calls.

### Compaction (`*compaction.Manager`)

When sessions get long, hand the manager into `RuntimeInputs.Compaction`
and the runtime triggers a summarize-and-splice pass when token usage
crosses `Threshold` (fraction of context window).

```go
mgr := &compaction.Manager{
    Summarizer: &compaction.Summarizer{
        Provider: anthropic.NewAnthropicProvider(key, ""),
        Model:    "claude-haiku-4-5",
        Timeout:  60 * time.Second,
    },
    PreserveTurns: 4,    // keep K most recent user turns verbatim
    Threshold:     0.6,  // compact when context usage > 60%
    MessageCap:    0,    // hard backstop on message count; 0 disables
}
inputs.Compaction = mgr
```

### Skills + Memory (`SkillProvider`, `MemoryProvider`)

When set, the runtime registers two more tools (`load_skill`, `load_memory`)
and inlines a *short* index of available items into the system prompt;
the body is loaded on demand only when the model asks for it.

```go
type fileSkills struct{ root string }

func (f *fileSkills) FormatIndex() string { /* ~50-line summary */ }
func (f *fileSkills) Get(name string) (string, bool) { /* read file */ }

deps.Skills = &fileSkills{root: "/var/lib/myapp/skills"}
```

This is the right shape for prompt-cacheable, lazy-loaded knowledge —
the index goes in the static (cacheable) prompt prefix; bodies don't.

### KnowledgeGraph

Synchronous recall + async ingest hooks for an external long-term memory:

```go
type myKG struct{ /* … */ }
func (k *myKG) ShouldRecall(query string) bool                       { /* cheap gate */ }
func (k *myKG) Recall(ctx context.Context, query string) string      { /* ≤800ms */ }
func (k *myKG) Ingest(ctx context.Context, thread []runtime.Message) { /* async */ }

deps.KGFn = func(model string) runtime.KnowledgeGraph { return &myKG{} }
```

`Recall` is racy by design — the runtime caps the wait at 800ms and
moves on; respect `ctx`. `Ingest` runs deferred-async with
`context.Background()`.

### Lifecycle hooks (`runtime.LifecycleHooks`)

Five callback fields on `LoopConfig.Hooks`. Every field is optional;
zero value disables the hook entirely. Hooks run **synchronously on
the runtime goroutine** — keep them quick, defer heavy work yourself.

```go
deps.AgentLoop.Hooks = runtime.LifecycleHooks{
    // Rewrite (or reject) the user's message before the loop sees it.
    OnUserPromptSubmit: func(ctx context.Context, prompt string, imgs []llm.ImageContent) (string, []llm.ImageContent, error) {
        return strings.TrimSpace(prompt), imgs, nil
    },

    // Fires once at Run start, after the user message is appended.
    OnSessionStart: func(ctx context.Context, sess *session.Session) {
        slog.Info("agent run started", "session", sess.Key)
    },

    // Fires BEFORE PermissionChecker. Returning Allow=false is a
    // denial — same shape as a PermissionChecker denial.
    BeforeToolUse: func(ctx context.Context, name string, input json.RawMessage) (runtime.HookDecision, error) {
        if name == "bash" && businessHoursOnly() && !isBusinessHours() {
            return runtime.HookDecision{Allow: false, Reason: "bash disabled outside business hours"}, nil
        }
        return runtime.HookDecision{Allow: true}, nil
    },

    // Fires AFTER Execute. Observe-only — perfect for audit trails.
    AfterToolUse: func(ctx context.Context, name string, input json.RawMessage, res tool.ToolResult) {
        auditLog.Record(name, input, res.Error == "")
    },

    // Fires once at Run end. reason ∈ {completed, max_turns, error, aborted}.
    OnStop: func(ctx context.Context, reason string) {
        metrics.AgentRuns.WithLabelValues(reason).Inc()
    },
}
```

Common uses:

* **Audit logging** — `AfterToolUse` is the right place to write a
  durable record of what the agent did. It fires on every tool call,
  including denials and errors.
* **Dynamic policy** — `BeforeToolUse` can deny based on runtime
  state (time of day, user role, rate limits) that a static
  `PermissionChecker` can't express.
* **Prompt sanitization** — `OnUserPromptSubmit` can strip secrets,
  expand short-codes, or redact PII before the message reaches the
  LLM. The rewritten prompt is what gets persisted to the session.
* **Metrics & tracing** — `OnStop`'s reason argument distinguishes
  clean completions from aborts and max-turn timeouts.

### MCP servers (`AgentSpec.MCPServers` / `tools/mcp`)

Two ways to wire in an external MCP server. Both produce regular
`tool.Tool`s namespaced as `mcp__<server>__<tool>`.

**Declarative — let `BuildRuntime` do it:**

```go
import "github.com/sausheong/harness/tools/mcp"

spec := runtime.AgentSpec{
    ID: "demo", Name: "Demo", Model: "anthropic/claude-haiku-4-5-20251001",
    MCPServers: []mcp.ServerConfig{
        {
            Name:    "fs",
            Command: "npx",
            Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
        },
        {
            Name: "github",
            URL:  "https://mcp.example.com/github",
        },
    },
}

rt, err := runtime.BuildRuntime(deps, inputs, spec)
if err != nil { /* connection failure */ }
defer rt.Close() // releases all MCP sessions
```

`BuildRuntime` connects each server, lists its tools, registers each
one into the agent's `*tool.Registry`, and stores the client handles
on the Runtime so `Close` can release them. A connection failure on
any server aborts `BuildRuntime` and tears down whatever was already
connected — no half-built state.

**Imperative — wire it yourself:**

```go
cli, err := mcp.Connect(ctx, mcp.ServerConfig{
    Name: "fs", Command: "npx",
    Args: []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
})
if err != nil { /* … */ }
defer cli.Close()

for _, t := range cli.Tools() {
    reg.Register(t)
}
```

Use this when you want full control over registration order, want to
filter the server's tools, or want to keep one MCP client alive
across many `Runtime`s.

**Streamable HTTP with auth:**

Most remote MCP servers need a bearer token or API key. `Headers` is
attached to every outgoing request via a RoundTripper wrapper around
`HTTPClient.Transport` — your custom client (mTLS, retries, etc.)
still runs.

```go
mcp.ServerConfig{
    Name: "github",
    URL:  "https://mcp.githubmcp.com/v1",
    Headers: map[string]string{
        "Authorization": "Bearer " + os.Getenv("GITHUB_MCP_TOKEN"),
        "X-Tenant":      "acme",
    },
    // Optional: full client control. Default is &http.Client{}.
    HTTPClient: &http.Client{Timeout: 30 * time.Second},
}
```

The wrapper clones each request before adding headers, so SDK retries
see a clean request and your `Headers` map is read-only at runtime.

**OAuth with auto-refresh (recommended for production):**

For OAuth-protected MCP servers, the cleanest path is to use
`golang.org/x/oauth2` to build an `*http.Client` whose transport
handles token refresh automatically, then pass it via `HTTPClient`.
The MCP transport sees a normal `*http.Client`; the oauth2 layer
adds `Authorization: Bearer <token>` and refreshes when the token
nears expiry.

Client-credentials (M2M agent — no user in the loop):

```go
import "golang.org/x/oauth2/clientcredentials"

cfg := clientcredentials.Config{
    ClientID:     os.Getenv("CLIENT_ID"),
    ClientSecret: os.Getenv("CLIENT_SECRET"),
    TokenURL:     "https://auth.example.com/oauth/token",
    Scopes:       []string{"mcp:read", "mcp:write"},
}

mcp.ServerConfig{
    Name:       "internal",
    URL:        "https://mcp.example.com",
    HTTPClient: cfg.Client(ctx), // auto-fetches + refreshes
}
```

Authorization code with refresh token (you've already done the user
handshake elsewhere and held onto the refresh token):

```go
import "golang.org/x/oauth2"

ts := myProvider.TokenSource(ctx, &oauth2.Token{
    AccessToken:  accessToken,
    RefreshToken: refreshToken,
    Expiry:       expiry,
})
mcp.ServerConfig{
    Name:       "remote",
    URL:        "https://mcp.example.com",
    HTTPClient: oauth2.NewClient(ctx, ts),
}
```

Both keep the OAuth lifecycle out of `tools/mcp` and let the
canonical `golang.org/x/oauth2` package own it. The MCP-spec OAuth
handshake (server-driven discovery + dynamic client registration +
PKCE) is **not** wired through `ServerConfig` today — if you hit a
server that requires it, drop down to the SDK's `auth` / `oauthex`
packages and register tools imperatively.

A few notes:

* The official `github.com/modelcontextprotocol/go-sdk` v1.5.0+ is
  the underlying transport. Stdio (`Command + Args + Env`) and
  Streamable HTTP (`URL` + optional `Headers` / `HTTPClient`) are
  both supported; the package picks the right transport from the
  fields you set.
* MCP tools default to `IsConcurrencySafe=false` because MCP doesn't
  expose a per-tool safety hint. Wrap the adapter if you know a
  specific tool is read-only and want it in the parallel batch.
* MCP `IsError` results surface as `tool.ToolResult.Error` so the
  agent sees them the same way it sees a harness-native tool error.
* `Runtime.Close()` is idempotent and safe to call when no MCP
  servers were declared — useful as a default `defer`.

---

## 11. Common patterns

### Pattern: separate provider per agent

`RuntimeInputs.Provider` is per-Runtime, so you can stand up different
providers for different agents while sharing one `RuntimeDeps`:

```go
deps := runtime.RuntimeDeps{ /* … */ }

chatRT, _ := runtime.BuildRuntime(deps,
    runtime.RuntimeInputs{Provider: anthropicProv, Tools: chatTools, Session: chatSess},
    runtime.AgentSpec{ID: "chat", Model: "anthropic/claude-haiku-4-5", /* … */})

cronRT, _ := runtime.BuildRuntime(deps,
    runtime.RuntimeInputs{Provider: openaiProv, Tools: cronTools, Session: cronSess, IngestSource: "cron"},
    runtime.AgentSpec{ID: "cron", Model: "openai/gpt-4o-mini", /* … */})
```

### Pattern: parallel HTTP tools

Five tools that all do `http.Get` against different endpoints? Mark them
all `IsConcurrencySafe == true` and set `MaxToolConcurrency: 5`. The
partitioner schedules them into a single batch and the model gets all
five results in one turn — the LTA agent does this for bus arrival,
traffic incidents, and carpark availability.

### Pattern: stateful tools

Caching, connection pools, rate-limiters all belong on the tool struct.
Tools are constructed once and live for the process lifetime:

```go
type CachedAPI struct {
    apiKey string

    mu        sync.Mutex
    cached    []byte
    cachedAt  time.Time
}

reg.Register(&CachedAPI{apiKey: os.Getenv("API_KEY")})
```

See `examples/lta-agent/tools.go` for two real cases — the carpark tool
caches the upstream response for 30 seconds; the bus-stops tool lazy-loads
~6,000 records on first use.

### Pattern: hot-reloading subagent specs

`MakeSubagentFactory` takes a `resolve` closure, not a static map. Read
through your live config object on every call so subagent registrations
hot-reload:

```go
func(id string) (runtime.SubagentSpec, bool) {
    cfg := configRef.Load().(*Config)        // atomic.Value
    s, ok := cfg.Subagents[id]
    return runtime.SubagentSpec{Spec: s, Registered: ok}, ok
}
```

---

## 12. Where to look next

* **`examples/minimal/`** — file + bash + web tools, ~80 lines. The
  smallest end-to-end you can poke at.
* **`examples/lta-agent/`** — five custom tools, no harness/tools/* deps.
  Pattern to copy when you're integrating against your own external API.
* **`examples/data-agent/`** — same shape, talking to Singapore's
  open-data APIs (`data.gov.sg`). Includes a `//go:build live` smoke test
  showing how to test custom tools against real endpoints without
  blocking `go test ./...`.
* **`examples/support-agent/`** — `SkillProvider` + `PermissionChecker`
  + `LifecycleHooks` (`AfterToolUse` audit trail) all wired into one
  agent. Closest pattern to copy if you're building a real product.
* **`runtime/types.go`** — every interface a consumer can implement
  (`LifecycleHooks`, `HookDecision`, `SkillProvider`, `MemoryProvider`,
  `KnowledgeGraph`, `SubagentResolver`) is defined here with full doc
  comments.
* **`tool/tool.go`** — the `Tool`, `Executor`, `Registry`, `ToolResult`
  surface. Read it before writing anything non-trivial.
* **`tools/mcp/`** — the MCP adapter package. `mcp.go` shows the
  Connect / Close lifecycle, `adapter.go` shows the
  `tool.Tool` ↔ MCP CallTool translation, `mcp_test.go` shows how to
  drive it with an in-process MCP server for tests.
* **`compaction/compaction.go`** — if you want to understand exactly what
  triggers a compaction and what survives one.

For bug reports or API friction, open an issue at
<https://github.com/sausheong/harness>.
