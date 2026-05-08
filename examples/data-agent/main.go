// data.gov.sg agent — example agent built on harness that talks to
// Singapore's open-data APIs (real-time environment + CKAN dataset
// catalog).
//
// Composition surface used (identical to lta-agent):
//   - tool.Registry (BYO concrete tools — none from harness/tools/*)
//   - session.NewSession (in-memory only)
//   - anthropic.NewAnthropicProvider
//   - runtime.BuildRuntime + runtime.AgentSpec + runtime.RuntimeInputs
//   - runtime.Run streaming loop
//
// Unlike the LTA Datamall, data.gov.sg's open APIs need no API key, so
// the only env var required is ANTHROPIC_API_KEY.
//
// Run:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/data-agent
//
// Try asking:
//   - "What's the weather in Bedok right now?"
//   - "How's the air quality in the west?"
//   - "What's the current temperature across Singapore?"
//   - "Find datasets about HDB resale prices."
//   - "Pull a few records from the latest HDB resale price dataset."
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

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr,
		&slog.HandlerOptions{Level: slog.LevelWarn})))

	// Five concrete tools, all defined in this example. None are imported
	// from harness/tools/* — same BYO pattern as lta-agent.
	reg := tool.NewRegistry()
	reg.Register(&WeatherForecastTool{})
	reg.Register(&AirQualityTool{})
	reg.Register(&EnvironmentReadingTool{})
	reg.Register(&DatasetSearchTool{})
	reg.Register(&DatasetQueryTool{})

	sess := session.NewSession("data-agent", "main")
	prov := anthropic.NewAnthropicProvider(apiKey, "")

	rt, err := runtime.BuildRuntime(
		runtime.RuntimeDeps{
			AgentLoop: runtime.LoopConfig{
				MaxToolConcurrency: 3,
				MaxAgentDepth:      1,
			},
		},
		runtime.RuntimeInputs{
			Provider: prov,
			Tools:    reg,
			Session:  sess,
		},
		runtime.AgentSpec{
			ID:    "datagov",
			Name:  "Singapore Open-Data Assistant",
			Model: "anthropic/claude-haiku-4-5-20251001",
			SystemPrompt: `You answer questions about Singapore using data.gov.sg open data. ` +
				`Use the datagov_* tools to fetch live readings and search the dataset catalog.

Tool selection guide:
  - Weather forecast (rain, "Partly Cloudy", etc.) by area → datagov_weather
  - PSI / PM2.5 / haze by region → datagov_air_quality
  - Numeric environment readings (temperature, humidity, rainfall, UV, wind) → datagov_environment
  - Find a dataset by topic (HDB, COE, population, etc.) → datagov_search_datasets
  - Pull rows from a known dataset → datagov_query_dataset (needs the resource_id from search)

When the user asks about a Singapore neighbourhood (e.g. "Bedok", "Tampines"), pass it as the area
to datagov_weather. For air quality, use one of: north, south, east, west, central, national.

Format responses concisely. Round temperatures to 1 decimal. Quote PSI / PM2.5 with their band
(0-50 Good, 51-100 Moderate, 101-200 Unhealthy, 201-300 Very Unhealthy, 301+ Hazardous).`,
			MaxTurns: 12,
		},
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build runtime:", err)
		os.Exit(1)
	}
	defer rt.Close() // no-op here (no MCP), but idiomatic

	fmt.Println("Singapore Open-Data Assistant — type a question, blank line to exit")
	fmt.Println("Examples:")
	fmt.Println("  • What's the weather in Bedok right now?")
	fmt.Println("  • How's the air quality in the west?")
	fmt.Println("  • Find datasets about HDB resale prices.")
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
