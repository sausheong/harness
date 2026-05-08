// Minimal harness example: a single agent with file + bash tools, talking
// to Anthropic, streaming output to stdout. Demonstrates the full
// composition path consumers go through.
//
// Run with:
//
//	ANTHROPIC_API_KEY=sk-ant-... go run ./examples/minimal
//
// The agent has access to a scratch directory (./_scratch) it can read,
// write, and execute bash in.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/providers/anthropic"
	"github.com/sausheong/harness/runtime"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/bash"
	"github.com/sausheong/harness/tools/file"
	"github.com/sausheong/harness/tools/web"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY required")
		os.Exit(1)
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})))

	workspace, _ := filepath.Abs("./_scratch")
	_ = os.MkdirAll(workspace, 0o755)

	// 1. Tool registry — register only what this agent needs.
	reg := tool.NewRegistry()
	reg.Register(&file.ReadFileTool{WorkDir: workspace})
	reg.Register(&file.WriteFileTool{WorkDir: workspace})
	reg.Register(&file.EditFileTool{WorkDir: workspace})
	reg.Register(&bash.BashTool{WorkDir: workspace})
	reg.Register(&web.WebFetchTool{})

	// 2. Session — in-memory only here (no SetStore call → no JSONL).
	sess := session.NewSession("demo", "main")

	// 3. LLM provider — direct construction, no factory.
	prov := anthropic.NewAnthropicProvider(apiKey, "")

	// 4. Build the runtime.
	rt, err := runtime.BuildRuntime(
		runtime.RuntimeDeps{
			AgentLoop: runtime.LoopConfig{
				MaxToolConcurrency: 4,
				MaxAgentDepth:      1,
			},
		},
		runtime.RuntimeInputs{
			Provider: prov,
			Tools:    reg,
			Session:  sess,
		},
		runtime.AgentSpec{
			ID:           "demo",
			Name:         "Demo",
			Model:        "anthropic/claude-haiku-4-5-20251001",
			Workspace:    workspace,
			SystemPrompt: "You are a terse coding assistant. Use tools to inspect or modify files in the workspace.",
			MaxTurns:     10,
		},
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build runtime:", err)
		os.Exit(1)
	}
	defer rt.Close() // no-op here (no MCP), but idiomatic

	// 5. REPL loop.
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("harness demo — type a message, blank line to exit")
	for {
		fmt.Print("\n> ")
		if !in.Scan() || in.Text() == "" {
			return
		}
		events, err := rt.Run(context.Background(), in.Text(), nil)
		if err != nil {
			fmt.Fprintln(os.Stderr, "run:", err)
			continue
		}
		for ev := range events {
			switch ev.Type {
			case runtime.EventTextDelta:
				fmt.Print(ev.Text)
			case runtime.EventToolCallStart:
				if ev.ToolCall != nil {
					fmt.Printf("\n[tool: %s] ", ev.ToolCall.Name)
				}
			case runtime.EventToolResult:
				fmt.Print(" ✓")
			case runtime.EventError:
				fmt.Fprintln(os.Stderr, "\nerror:", ev.Error)
			case runtime.EventDone:
				if ev.Usage != nil {
					fmt.Printf("\n[%d in / %d out tokens]\n",
						ev.Usage.InputTokens, ev.Usage.OutputTokens)
				}
			}
		}
	}
	_ = llm.ReasoningOff // keep llm import live for clarity in the demo
}
