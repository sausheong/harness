package openai

import (
	"context"
	"fmt"
	"github.com/sausheong/harness/llm"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRepeatedFinishEmitsToolOnlyOnce(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_one\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"}}]}}]}\n\n")
		for i := 0; i < 2; i++ {
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := NewOpenAIProviderWithKind("fixture", srv.URL, "openai-compatible")
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	calls, done := 0, 0
	for e := range stream {
		if e.Type == llm.EventToolCallDone {
			calls++
		}
		if e.Type == llm.EventDone {
			done++
		}
		if e.Type == llm.EventError {
			t.Fatal(e.Error)
		}
	}
	if calls != 1 || done != 1 {
		t.Fatalf("calls=%d done=%d", calls, done)
	}
}
