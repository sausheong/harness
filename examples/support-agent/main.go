// Mock customer-support agent — third example demonstrating three harness
// surfaces the previous agents skip:
//
//   - SkillProvider: the knowledge base is exposed as harness "skills".
//     The runtime auto-registers a load_skill tool and the static system
//     prompt advertises the available articles by slug. The full body
//     loads on demand.
//
//   - PermissionChecker: write-side tools (tickets_update, escalate_to_human)
//     are gated on a "supervisor mode" flag. By default the agent can
//     read tickets and the KB but cannot mutate state — flip
//     SUPPORT_AGENT_SUPERVISOR=1 (or pass --supervisor) to unlock.
//
//   - LifecycleHooks: every tool call (including denials and errors) is
//     written to an audit log via AfterToolUse. The OnStop hook records
//     the run outcome. This is the pattern any compliance-sensitive
//     support deployment will want.
//
// Backend is fully in-memory: a few seed tickets are pre-populated. Run:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/support-agent
//	# or, to unlock write tools:
//	SUPPORT_AGENT_SUPERVISOR=1 go run ./examples/support-agent
//
// Try asking:
//   - "What's our refund policy for usage-based charges?"
//   - "Look up ticket T-1002 — what's it about?"
//   - "A customer's account was hacked. File a ticket and walk me through next steps."
//   - "Mark ticket T-1001 as resolved." (supervisor only)
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/sausheong/harness/providers/anthropic"
	"github.com/sausheong/harness/runtime"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func main() {
	supervisorFlag := flag.Bool("supervisor", false,
		"unlock write tools (tickets_update, escalate_to_human). "+
			"Equivalent to SUPPORT_AGENT_SUPERVISOR=1.")
	flag.Parse()

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY required")
		os.Exit(1)
	}
	supervisor := *supervisorFlag || os.Getenv("SUPPORT_AGENT_SUPERVISOR") == "1"

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr,
		&slog.HandlerOptions{Level: slog.LevelWarn})))

	// Single shared store: seeded with a handful of in-flight tickets so
	// the demo is interactive on first launch.
	store := NewTicketStore()
	store.Seed()

	// Five concrete tools. The runtime auto-registers a sixth (load_skill)
	// because we pass deps.Skills below.
	reg := tool.NewRegistry()
	reg.Register(&KBSearchTool{KB: NewKBSkills()})
	reg.Register(&TicketGetTool{Store: store})
	reg.Register(&TicketCreateTool{Store: store})
	reg.Register(&TicketUpdateTool{Store: store})
	reg.Register(&EscalateTool{Webhook: os.Getenv("SUPPORT_AGENT_WEBHOOK")})

	sess := session.NewSession("support-agent", "main")
	prov := anthropic.NewAnthropicProvider(apiKey, "")

	// LifecycleHooks: write a JSONL audit log of every tool call (with
	// outcome) and a record of every Run's stop reason. Compliance-
	// sensitive support deployments will want this kind of trail.
	auditPath := "support-agent-audit.jsonl"
	if v := os.Getenv("SUPPORT_AGENT_AUDIT_LOG"); v != "" {
		auditPath = v
	}
	audit, err := newAuditLog(auditPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open audit log:", err)
		os.Exit(1)
	}
	defer audit.Close()
	fmt.Fprintf(os.Stderr, "Auditing tool calls to %s\n", auditPath)

	rt, err := runtime.BuildRuntime(
		runtime.RuntimeDeps{
			AgentLoop: runtime.LoopConfig{
				MaxToolConcurrency: 3,
				MaxAgentDepth:      1,
				Hooks: runtime.LifecycleHooks{
					AfterToolUse: func(_ context.Context, name string, input json.RawMessage, res tool.ToolResult) {
						audit.Log(map[string]any{
							"kind":    "tool",
							"name":    name,
							"input":   json.RawMessage(input),
							"ok":      res.Error == "",
							"error":   res.Error,
							"outchars": len(res.Output),
						})
					},
					OnStop: func(_ context.Context, reason string) {
						audit.Log(map[string]any{
							"kind":   "run_end",
							"reason": reason,
						})
					},
				},
			},
			// SkillProvider: contributes a KB index to the static system
			// prompt and backs the auto-registered load_skill tool.
			Skills: NewKBSkills(),
			// PermissionChecker: gates the two write-side tools on
			// supervisor mode. nil-safe (would default to allow-all),
			// but we want the gating behaviour even in default mode.
			Permission: NewSupportChecker(supervisor),
		},
		runtime.RuntimeInputs{
			Provider: prov,
			Tools:    reg,
			Session:  sess,
		},
		runtime.AgentSpec{
			ID:    "support",
			Name:  "Acme Support",
			Model: "anthropic/claude-haiku-4-5-20251001",
			SystemPrompt: `You are a customer support agent for Acme, a SaaS company. ` +
				`Be empathetic, concise, and accurate.

Workflow:
  1. When the customer describes their issue, FIRST use kb_search to find
     relevant policy articles, then load_skill to read the full body of any
     article you'll cite. Do not invent policy from memory — quote the KB.
  2. If the customer references an existing ticket, call tickets_get to
     pull its current state before responding.
  3. If the issue is new and warrants tracking (refund request, account
     compromise, billing dispute, downtime claim), call tickets_create.
  4. Status mutations (mark resolved, change priority) and escalation to
     a human go through tickets_update / escalate_to_human. These tools
     are restricted in non-supervisor mode — if you receive a permission
     denial, tell the customer their request will be handed to a senior
     agent and stop.

Tone:
  - Acknowledge the customer's situation in one sentence before quoting policy.
  - Cite KB article slugs in brackets, e.g. "[refund-policy]".
  - Never promise a specific resolution timeline you can't verify from
     a ticket or KB article.`,
			MaxTurns: 12,
		},
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build runtime:", err)
		os.Exit(1)
	}
	defer rt.Close() // no-op here (no MCP), but idiomatic

	mode := "agent mode"
	if supervisor {
		mode = "supervisor mode (write tools unlocked)"
	}
	fmt.Printf("Acme Support — %s\n", mode)
	fmt.Println("Type a customer message; blank line to exit.")
	fmt.Println("Examples:")
	fmt.Println("  • What's your refund policy for usage charges?")
	fmt.Println("  • Look up ticket T-1002.")
	fmt.Println("  • My account was just compromised — what do I do?")
	if !supervisor {
		fmt.Println("  (Re-run with --supervisor to test write tools.)")
	}
	fmt.Println()

	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !in.Scan() || in.Text() == "" {
			return
		}
		runOne(rt, in.Text())
		fmt.Println()
	}
}

// runOne drains one Run, formatting events for terminal display.
func runOne(rt *runtime.Runtime, prompt string) {
	events, err := rt.Run(context.Background(), prompt, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nrun error:", err)
		return
	}
	for ev := range events {
		switch ev.Type {
		case runtime.EventTextDelta:
			fmt.Print(ev.Text)
		case runtime.EventToolCallStart:
			if ev.ToolCall != nil {
				fmt.Printf("\n  [%s] ", ev.ToolCall.Name)
			}
		case runtime.EventToolResult:
			if ev.Result != nil && ev.Result.Error != "" {
				fmt.Printf("✗ %s", ev.Result.Error)
			} else {
				fmt.Print("✓")
			}
		case runtime.EventError:
			fmt.Fprintln(os.Stderr, "\nerror:", ev.Error)
		case runtime.EventDone:
			fmt.Println()
			if ev.Usage != nil {
				fmt.Fprintf(os.Stderr, "[%d in / %d out tokens]\n",
					ev.Usage.InputTokens, ev.Usage.OutputTokens)
			}
		}
	}
}
