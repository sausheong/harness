package runtime

import (
	"context"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"testing"
)

func TestToolReadyCarriesArgumentsBeforeResult(t *testing.T) {
	provider := &emptyResponseProvider{responses: [][]llm.ChatEvent{
		{{Type: llm.EventToolCallStart, ToolCall: &llm.ToolCall{ID: "one", Name: "noop"}}, {Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "one", Name: "noop", Input: []byte(`{"command":"echo hi"}`)}}, {Type: llm.EventDone}},
		{{Type: llm.EventTextDelta, Text: "Done"}, {Type: llm.EventDone}},
	}}
	rt := &Runtime{LLM: provider, Tools: usageNoopExecutor{}, Session: session.NewSession("test", "ready"), Model: "fixture", MaxTurns: 2}
	stream, err := rt.Run(context.Background(), "go", nil)
	if err != nil {
		t.Fatal(err)
	}
	ready, results := 0, 0
	for e := range stream {
		switch e.Type {
		case EventToolCallReady:
			ready++
			if e.ToolCall == nil || string(e.ToolCall.Input) != `{"command":"echo hi"}` {
				t.Fatalf("%+v", e)
			}
		case EventToolResult:
			results++
			if ready != 1 {
				t.Fatal("result before arguments")
			}
		case EventError:
			t.Fatal(e.Error)
		}
	}
	if ready != 1 || results != 1 {
		t.Fatalf("ready=%d results=%d", ready, results)
	}
}
