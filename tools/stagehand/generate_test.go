package stagehand

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	sh "github.com/browserbase/stagehand/packages/sdk-go/v4"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
)

// scripted emits a fixed event sequence and records the request.
type scripted struct {
	llmtest.Base
	events []llm.ChatEvent
	got    llm.ChatRequest
}

func (s *scripted) ChatStream(_ context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	s.got = req
	ch := make(chan llm.ChatEvent, len(s.events))
	for _, ev := range s.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func userText(text string) sh.LLMMessage {
	return sh.LLMMessage{Role: sh.LLMRoleUser, Content: sh.LLMMessageContent{sh.TextContentBlock(sh.LLMTextContent{Text: text})}}
}

func TestGenerateStructuredUsesForcedToolCall(t *testing.T) {
	p := &scripted{events: []llm.ChatEvent{
		{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "t1", Name: structuredToolName, Input: json.RawMessage(`{"title":"Hello"}`)}},
		{Type: llm.EventDone, StopReason: "tool_use", Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5}},
	}}
	sys := "extract things"
	params := sh.StructuredGenerateParams(sh.LLMStructuredGenerateParams{
		Messages:     []sh.LLMMessage{userText("page content")},
		SystemPrompt: &sys,
		ResponseFormat: sh.LLMJSONSchemaResponseFormat{
			Name:   "extraction",
			Schema: json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"title":{"type":"string"}}}`),
		},
	})

	res, err := NewGenerateFunc(p, "m1")(context.Background(), params)
	if err != nil {
		t.Fatal(err)
	}
	structured, ok := res.AsStructured()
	if !ok {
		t.Fatalf("want structured result, got %#v", res)
	}
	if string(structured.StructuredContent) != `{"title":"Hello"}` {
		t.Errorf("structured content = %s", structured.StructuredContent)
	}
	if structured.Usage == nil || structured.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", structured.Usage)
	}

	if p.got.Model != "m1" {
		t.Errorf("model = %q", p.got.Model)
	}
	if len(p.got.Tools) != 1 || p.got.Tools[0].Name != structuredToolName {
		t.Fatalf("tools = %+v", p.got.Tools)
	}
	if strings.Contains(string(p.got.Tools[0].Parameters), "$schema") {
		t.Errorf("$schema not stripped: %s", p.got.Tools[0].Parameters)
	}
	if !strings.HasPrefix(p.got.SystemPrompt, "extract things") || !strings.Contains(p.got.SystemPrompt, structuredToolName) {
		t.Errorf("system prompt = %q", p.got.SystemPrompt)
	}
}

func TestGenerateStructuredFallsBackToFencedJSON(t *testing.T) {
	p := &scripted{events: []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "```json\n{\"a\":1}\n```"},
		{Type: llm.EventDone, StopReason: "end_turn"},
	}}
	params := sh.StructuredGenerateParams(sh.LLMStructuredGenerateParams{
		Messages:       []sh.LLMMessage{userText("x")},
		ResponseFormat: sh.LLMJSONSchemaResponseFormat{Name: "r", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	res, err := NewGenerateFunc(p, "m")(context.Background(), params)
	if err != nil {
		t.Fatal(err)
	}
	structured, _ := res.AsStructured()
	if string(structured.StructuredContent) != `{"a":1}` {
		t.Errorf("structured content = %s", structured.StructuredContent)
	}
}

func TestGenerateStructuredErrorsWithoutOutput(t *testing.T) {
	p := &scripted{events: []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "I can't find that."},
		{Type: llm.EventDone, StopReason: "end_turn"},
	}}
	params := sh.StructuredGenerateParams(sh.LLMStructuredGenerateParams{
		Messages:       []sh.LLMMessage{userText("x")},
		ResponseFormat: sh.LLMJSONSchemaResponseFormat{Name: "r", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if _, err := NewGenerateFunc(p, "m")(context.Background(), params); err == nil {
		t.Fatal("want error for missing structured output")
	}
}

func TestGenerateMessageMapsToolsAndHistory(t *testing.T) {
	p := &scripted{events: []llm.ChatEvent{
		{Type: llm.EventTextDelta, Text: "clicking"},
		{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "c2", Name: "click", Input: json.RawMessage(`{"selector":"#go"}`)}},
		{Type: llm.EventDone, StopReason: "tool_use"},
	}}
	desc := "Click an element"
	png := base64.StdEncoding.EncodeToString([]byte("fakepng"))
	isErr := true
	params := sh.MessageGenerateParams(sh.LLMMessageGenerateParams{
		Messages: []sh.LLMMessage{
			{Role: sh.LLMRoleUser, Content: sh.LLMMessageContent{
				sh.TextContentBlock(sh.LLMTextContent{Text: "do it"}),
				sh.ImageContentBlock(sh.LLMImageContent{Data: "data:image/png;base64," + png, MIMEType: "image/png"}),
			}},
			{Role: sh.LLMRoleAssistant, Content: sh.LLMMessageContent{
				sh.ToolUseContentBlock(sh.LLMToolUseContent{ID: "c1", Name: "click", Input: sh.LLMToolUseContentInput{"selector": json.RawMessage(`"#a"`)}}),
			}},
			{Role: sh.LLMRoleUser, Content: sh.LLMMessageContent{
				sh.ToolResultContentBlock(sh.LLMToolResultContent{ToolUseID: "c1", IsError: &isErr, Content: []sh.LLMToolResultContentBlock{
					sh.ToolResultTextBlock(sh.LLMTextContent{Text: "not found"}),
				}}),
			}},
		},
		Tools: []sh.LLMClientTool{{Name: "click", Description: &desc, InputSchema: sh.LLMToolJSON{Type: "object"}}},
	})

	res, err := NewGenerateFunc(p, "m")(context.Background(), params)
	if err != nil {
		t.Fatal(err)
	}

	msgs := p.got.Messages
	if len(msgs) != 3 {
		t.Fatalf("messages = %+v", msgs)
	}
	if msgs[0].Role != "user" || msgs[0].Content != "do it" || len(msgs[0].Images) != 1 || string(msgs[0].Images[0].Data) != "fakepng" {
		t.Errorf("user message = %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || len(msgs[1].ToolCalls) != 1 || msgs[1].ToolCalls[0].ID != "c1" {
		t.Errorf("assistant message = %+v", msgs[1])
	}
	if msgs[2].Role != "tool" || msgs[2].ToolCallID != "c1" || !msgs[2].IsError || msgs[2].Content != "not found" {
		t.Errorf("tool result message = %+v", msgs[2])
	}
	if len(p.got.Tools) != 1 || p.got.Tools[0].Name != "click" || p.got.Tools[0].Description != desc {
		t.Errorf("tools = %+v", p.got.Tools)
	}

	message, ok := res.AsMessage()
	if !ok {
		t.Fatalf("want message result, got %#v", res)
	}
	if message.StopReason == nil || *message.StopReason != "tool_use" {
		t.Errorf("stop reason = %v", message.StopReason)
	}
	if len(message.Content) != 2 {
		t.Fatalf("content = %+v", message.Content)
	}
	if txt, ok := message.Content[0].AsText(); !ok || txt.Text != "clicking" {
		t.Errorf("content[0] = %+v", message.Content[0])
	}
	tu, ok := message.Content[1].AsToolUse()
	if !ok || tu.ID != "c2" || string(tu.Input["selector"]) != `"#go"` {
		t.Errorf("content[1] = %+v", message.Content[1])
	}
	// The result must survive the SDK's own JSON encoding (it crosses an RPC boundary).
	if _, err := json.Marshal(res); err != nil {
		t.Errorf("marshal result: %v", err)
	}
}

func TestGenerateMessageToolChoiceNoneSendsNoTools(t *testing.T) {
	p := &scripted{events: []llm.ChatEvent{{Type: llm.EventTextDelta, Text: "ok"}, {Type: llm.EventDone}}}
	none := sh.LLMToolChoiceModeNone
	params := sh.MessageGenerateParams(sh.LLMMessageGenerateParams{
		Messages:   []sh.LLMMessage{userText("hi")},
		Tools:      []sh.LLMClientTool{{Name: "click", InputSchema: sh.LLMToolJSON{Type: "object"}}},
		ToolChoice: &sh.LLMToolChoice{Mode: &none},
	})
	if _, err := NewGenerateFunc(p, "m")(context.Background(), params); err != nil {
		t.Fatal(err)
	}
	if len(p.got.Tools) != 0 {
		t.Errorf("tools sent despite tool_choice none: %+v", p.got.Tools)
	}
}

func TestGenerateSurfacesRefusalAndStreamErrors(t *testing.T) {
	params := sh.MessageGenerateParams(sh.LLMMessageGenerateParams{Messages: []sh.LLMMessage{userText("hi")}})
	refusal := &scripted{events: []llm.ChatEvent{{Type: llm.EventDone, StopReason: llm.StopReasonRefusal}}}
	if _, err := NewGenerateFunc(refusal, "m")(context.Background(), params); err == nil {
		t.Error("want error on refusal")
	}
	stub := &llmtest.Stub{ChatErr: context.DeadlineExceeded}
	if _, err := NewGenerateFunc(stub, "m")(context.Background(), params); err == nil {
		t.Error("want error on chat failure")
	}
}
