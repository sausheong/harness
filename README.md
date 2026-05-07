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

## Design

The harness is intentionally minimal. Every agent gets:

- A streaming think-act loop with parallel tool dispatch
- A pluggable LLM provider interface (Anthropic, OpenAI, Gemini, Qwen
  built in; bring your own to add more)
- Append-only session storage with optional JSONL persistence
- Three-stage summarize-and-splice compaction with prompt-cache-stable
  prefixes
- A token budget calibrator that learns the chars→tokens ratio per
  session

Everything else — skills, long-term memory, knowledge graphs,
permission policies, subagent dispatch, on-disk session stores — is an
optional plug-point. Pass `nil` and it disappears.

See [`developer.md`](./developer.md) for a step-by-step guide to
building agents on top of harness.

## Status

Pre-1.0. The `runtime` API surface is still being shaped; expect
breaking changes until a v0.1.0 tag.

## License

MIT.
