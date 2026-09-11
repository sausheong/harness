package openai

import (
	"context"
	"fmt"
	"github.com/sausheong/harness/llm"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPreservesOutputLimitAndTrailingUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"partial","type":"function","function":{"name":"bash","arguments":"{\"command\":"}}]},"finish_reason":"length"}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":2048,"total_tokens":2098}}`+"\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	p := NewOpenAIProviderWithKind("fixture", srv.URL, "openai-compatible")
	events, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for e := range events {
		if e.Type == llm.EventToolCallDone {
			t.Fatal("truncated tool accepted")
		}
		if e.Type == llm.EventError {
			t.Fatal(e.Error)
		}
		if e.Type == llm.EventDone {
			found = true
			if e.StopReason != "length" || e.Usage == nil || e.Usage.OutputTokens != 2048 {
				t.Fatalf("%+v", e)
			}
		}
	}
	if !found {
		t.Fatal("missing terminal event")
	}
}
