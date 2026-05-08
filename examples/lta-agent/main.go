// LTA Datamall agent — example of building a domain-specific agent on
// the harness framework with custom tools (no harness/tools/* deps).
//
// Composition surface used:
//   - tool.Registry (BYO concrete tools — none from harness/tools/*)
//   - session.NewSession (in-memory, no SetStore so nothing persists)
//   - anthropic.NewAnthropicProvider
//   - runtime.BuildRuntime + runtime.AgentSpec + runtime.RuntimeDeps
//   - runtime.Run streaming loop
//
// Deliberately NOT used:
//   - Skills, Memory, KnowledgeGraph (this agent doesn't need them)
//   - Subagents, compaction, calibrator persistence
//
// Run:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	export LTA_DATAMALL_KEY=<your account key>
//	go run ./examples/lta-agent
//
// Try asking:
//   - "When's the next bus 7 at the stop opposite Bugis MRT?"
//   - "Are there any traffic incidents on the AYE?"
//   - "How many lots are free at Marina Bay Sands carpark?"
//   - "Any train disruptions right now?"
package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/sausheong/harness/providers/anthropic"
	"github.com/sausheong/harness/runtime"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY required")
		os.Exit(1)
	}
	ltaKey := os.Getenv("LTA_DATAMALL_KEY")
	if ltaKey == "" {
		fmt.Fprintln(os.Stderr, "LTA_DATAMALL_KEY required — register at https://datamall.lta.gov.sg/")
		os.Exit(1)
	}

	// Quiet the default slog so REPL output stays clean. Switch to
	// LevelInfo if you want to see tool-result trace + API call timing.
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr,
		&slog.HandlerOptions{Level: slog.LevelWarn})))

	// Tool registry — five concrete tools, all defined in this example.
	// None are imported from harness/tools/*, proving the framework
	// genuinely supports BYO tools.
	reg := tool.NewRegistry()
	reg.Register(&BusArrivalTool{AccountKey: ltaKey})
	reg.Register(&TrafficIncidentsTool{AccountKey: ltaKey})
	reg.Register(&CarparkAvailabilityTool{AccountKey: ltaKey})
	reg.Register(&TrainAlertsTool{AccountKey: ltaKey})
	reg.Register(&BusStopsSearchTool{AccountKey: ltaKey})

	sess := session.NewSession("lta-agent", "main")
	prov := anthropic.NewAnthropicProvider(apiKey, "")

	rt, err := runtime.BuildRuntime(
		runtime.RuntimeDeps{
			AgentLoop: runtime.LoopConfig{
				MaxToolConcurrency: 3,
				MaxAgentDepth:      1,
			},
			// No Skills, no Memory, no KGFn, no CalibratorStore — all
			// optional. The harness wires nil-safe defaults for each.
		},
		runtime.RuntimeInputs{
			Provider: prov,
			Tools:    reg,
			Session:  sess,
		},
		runtime.AgentSpec{
			ID:    "lta",
			Name:  "Singapore Transit Assistant",
			Model: "anthropic/claude-haiku-4-5-20251001",
			SystemPrompt: `You are a Singapore transit assistant. You help users find bus arrival times, ` +
				`current traffic conditions, public carpark availability, and MRT/LRT alerts. ` +
				`Use the lta_* tools to fetch real-time data from LTA Datamall. ` +
				`When the user asks about a location (e.g. "Bugis", "Orchard"), ` +
				`first call lta_bus_stops to find the matching BusStopCode, ` +
				`then call lta_bus_arrival with that code. ` +
				`Format responses concisely — Singaporeans value brevity. ` +
				`Times are in minutes from now; "Arr" means arriving in <1 minute.`,
			MaxTurns: 12,
		},
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build runtime:", err)
		os.Exit(1)
	}
	defer rt.Close() // no-op here (no MCP), but idiomatic

	fmt.Println("LTA Singapore Transit Assistant — type a question, blank line to exit")
	fmt.Println("Examples:")
	fmt.Println("  • When's the next bus 7 near Bugis?")
	fmt.Println("  • Any traffic incidents on the AYE right now?")
	fmt.Println("  • How many lots free at Marina Bay carparks?")
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
