//go:build live

package stagehand

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sausheong/harness/providers/openai"
)

// TestLiveStagehandLocalChrome drives a local Chrome through a full
// navigate → observe → extract → act → screenshot flow, with Stagehand's
// inference routed through the harness OpenAI provider.
//
//	OPENAI_API_KEY=... go test -tags live -run TestLiveStagehand ./tools/stagehand/
func TestLiveStagehandLocalChrome(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	model := os.Getenv("STAGEHAND_TEST_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}
	st, err := NewStagehandTool(Config{
		Provider:          openai.NewOpenAIProvider(key, ""),
		Model:             model,
		BrowserbaseAPIKey: os.Getenv("BROWSERBASE_API_KEY"),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	run := func(input string) string {
		t.Helper()
		res, err := st.Execute(ctx, json.RawMessage(input))
		if err != nil {
			t.Fatalf("%s: %v", input, err)
		}
		if res.Error != "" {
			t.Fatalf("%s: tool error: %s", input, res.Error)
		}
		t.Logf("%s\n→ %s", input, res.Output)
		return res.Output
	}

	if out := run(`{"action":"navigate","url":"https://example.com","session":"live"}`); !strings.Contains(out, "Example Domain") {
		t.Errorf("navigate output missing title: %s", out)
	}
	run(`{"action":"observe","instruction":"find the links on the page","session":"live"}`)
	if out := run(`{"action":"extract","instruction":"the main heading of the page","schema":{"type":"object","properties":{"heading":{"type":"string"}}},"session":"live"}`); !strings.Contains(out, "Example Domain") {
		t.Errorf("extract output missing heading: %s", out)
	}
	run(`{"action":"act","instruction":"click the link on the page","session":"live"}`)

	res, err := st.Execute(ctx, json.RawMessage(`{"action":"screenshot","session":"live"}`))
	if err != nil || res.Error != "" {
		t.Fatalf("screenshot: %v %s", err, res.Error)
	}
	if len(res.Images) != 1 || len(res.Images[0].Data) == 0 {
		t.Error("screenshot returned no image")
	}
	run(`{"action":"close","session":"live"}`)
}
