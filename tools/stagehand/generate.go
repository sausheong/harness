package stagehand

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	sh "github.com/browserbase/stagehand/packages/sdk-go/v4"
	"github.com/sausheong/harness/llm"
)

const (
	// generateMaxTokens bounds each inference call Stagehand makes. act/observe
	// return short action lists; extract can return larger tables, so this is
	// generous relative to typical responses.
	generateMaxTokens = 8192
	// structuredToolName is the synthetic tool the model is forced to call for
	// structured output. Harness providers have no portable response_format
	// knob, but every provider supports tool calls whose input is validated
	// against a JSON Schema — the tool input IS the structured result.
	structuredToolName = "respond"
)

// NewGenerateFunc adapts a harness LLMProvider to Stagehand's LLMGenerateFunc
// so Stagehand's internal inference (act, observe, extract) runs through the
// same provider — and the same usage observers and admission guards — as the
// agent itself, instead of a separately configured model and API key.
//
// Structured requests (extract, and act/observe's action selection) are
// served by forcing a single tool call whose parameters are the requested
// schema. Message requests pass Stagehand's tools through unchanged.
func NewGenerateFunc(provider llm.LLMProvider, model string) sh.LLMGenerateFunc {
	return func(ctx context.Context, params sh.LLMGenerateParams) (sh.LLMGenerateResult, error) {
		if provider == nil {
			return sh.LLMGenerateResult{}, errors.New("stagehand: no LLM provider configured")
		}
		if structured, ok := params.AsStructured(); ok {
			return generateStructured(ctx, provider, model, structured)
		}
		if message, ok := params.AsMessage(); ok {
			return generateMessage(ctx, provider, model, message)
		}
		return sh.LLMGenerateResult{}, errors.New("stagehand: unsupported generate params")
	}
}

func generateStructured(ctx context.Context, provider llm.LLMProvider, model string, p sh.LLMStructuredGenerateParams) (sh.LLMGenerateResult, error) {
	messages, err := toHarnessMessages(p.Messages)
	if err != nil {
		return sh.LLMGenerateResult{}, err
	}
	description := "Return the result. The input must match the requested schema exactly."
	if p.ResponseFormat.Description != nil && *p.ResponseFormat.Description != "" {
		description = *p.ResponseFormat.Description
	}
	tools, _ := provider.NormalizeToolSchema([]llm.ToolDef{{
		Name:        structuredToolName,
		Description: description,
		Parameters:  stripSchemaKeyword(p.ResponseFormat.Schema),
	}})
	system := deref(p.SystemPrompt)
	instruction := fmt.Sprintf("Respond ONLY by calling the %q tool exactly once. Do not reply with plain text.", structuredToolName)
	if system != "" {
		system += "\n\n" + instruction
	} else {
		system = instruction
	}
	req := llm.ChatRequest{
		Model:        model,
		Messages:     messages,
		Tools:        tools,
		MaxTokens:    generateMaxTokens,
		SystemPrompt: system,
	}
	if p.Temperature != nil {
		req.Temperature, req.TemperatureSet = *p.Temperature, true
	}
	out, err := collect(ctx, provider, req)
	if err != nil {
		return sh.LLMGenerateResult{}, err
	}

	var structured json.RawMessage
	for _, tc := range out.toolCalls {
		if tc.Name == structuredToolName {
			structured = tc.Input
			break
		}
	}
	if structured == nil {
		// Some models answer in text despite the instruction. Accept a bare
		// (optionally fenced) JSON object rather than failing the whole act.
		structured = jsonFromText(out.text)
	}
	if structured == nil {
		return sh.LLMGenerateResult{}, fmt.Errorf("stagehand: model returned no structured output (stop reason %q)", out.stopReason)
	}
	return sh.StructuredGenerateResult(sh.LLMStructuredGenerateResult{
		Role:              sh.LLMRoleAssistant,
		Content:           sh.LLMMessageContent{sh.TextContentBlock(sh.LLMTextContent{Text: string(structured)})},
		StopReason:        stringPtr(out.stopReason),
		Usage:             out.usage,
		StructuredContent: structured,
	}), nil
}

func generateMessage(ctx context.Context, provider llm.LLMProvider, model string, p sh.LLMMessageGenerateParams) (sh.LLMGenerateResult, error) {
	messages, err := toHarnessMessages(p.Messages)
	if err != nil {
		return sh.LLMGenerateResult{}, err
	}
	req := llm.ChatRequest{
		Model:        model,
		Messages:     messages,
		MaxTokens:    generateMaxTokens,
		SystemPrompt: deref(p.SystemPrompt),
	}
	if p.Temperature != nil {
		req.Temperature, req.TemperatureSet = *p.Temperature, true
	}
	// ChatRequest has no tool_choice field: "none" is honoured by sending no
	// tools; "auto" and "required" both send them and let the model decide.
	if !(p.ToolChoice != nil && p.ToolChoice.Mode != nil && *p.ToolChoice.Mode == sh.LLMToolChoiceModeNone) {
		defs := make([]llm.ToolDef, 0, len(p.Tools))
		for _, t := range p.Tools {
			schema, err := json.Marshal(t.InputSchema)
			if err != nil {
				return sh.LLMGenerateResult{}, fmt.Errorf("stagehand: encode schema for tool %q: %w", t.Name, err)
			}
			defs = append(defs, llm.ToolDef{Name: t.Name, Description: deref(t.Description), Parameters: stripSchemaKeyword(schema)})
		}
		if len(defs) > 0 {
			req.Tools, _ = provider.NormalizeToolSchema(defs)
		}
	}
	out, err := collect(ctx, provider, req)
	if err != nil {
		return sh.LLMGenerateResult{}, err
	}

	content := sh.LLMMessageContent{}
	if out.text != "" {
		content = append(content, sh.TextContentBlock(sh.LLMTextContent{Text: out.text}))
	}
	for _, tc := range out.toolCalls {
		input := sh.LLMToolUseContentInput{}
		if len(tc.Input) > 0 {
			if err := json.Unmarshal(tc.Input, &input); err != nil {
				return sh.LLMGenerateResult{}, fmt.Errorf("stagehand: decode tool call %q input: %w", tc.Name, err)
			}
		}
		content = append(content, sh.ToolUseContentBlock(sh.LLMToolUseContent{ID: tc.ID, Name: tc.Name, Input: input}))
	}
	stop := out.stopReason
	if len(out.toolCalls) > 0 {
		stop = "tool_use"
	}
	return sh.MessageGenerateResult(sh.LLMMessageGenerateResult{
		Role:       sh.LLMRoleAssistant,
		Content:    content,
		StopReason: stringPtr(stop),
		Usage:      out.usage,
	}), nil
}

// toHarnessMessages flattens Stagehand's block-structured messages into
// harness messages. Tool results become separate "tool" role messages, which
// is how the harness providers expect them.
func toHarnessMessages(in []sh.LLMMessage) ([]llm.Message, error) {
	out := make([]llm.Message, 0, len(in))
	for _, m := range in {
		msg := llm.Message{Role: string(m.Role)}
		var text []string
		var toolResults []llm.Message
		for _, block := range m.Content {
			if t, ok := block.AsText(); ok {
				text = append(text, t.Text)
			} else if img, ok := block.AsImage(); ok {
				decoded, err := decodeImage(img)
				if err != nil {
					return nil, err
				}
				msg.Images = append(msg.Images, decoded)
			} else if tu, ok := block.AsToolUse(); ok {
				input, err := json.Marshal(tu.Input)
				if err != nil {
					return nil, fmt.Errorf("stagehand: encode tool use %q: %w", tu.Name, err)
				}
				msg.ToolCalls = append(msg.ToolCalls, llm.ToolCall{ID: tu.ID, Name: tu.Name, Input: input})
			} else if tr, ok := block.AsToolResult(); ok {
				result := llm.Message{Role: "tool", ToolCallID: tr.ToolUseID, IsError: tr.IsError != nil && *tr.IsError}
				var parts []string
				for _, rb := range tr.Content {
					if t, ok := rb.AsText(); ok {
						parts = append(parts, t.Text)
					} else if img, ok := rb.AsImage(); ok {
						decoded, err := decodeImage(img)
						if err != nil {
							return nil, err
						}
						result.Images = append(result.Images, decoded)
					}
				}
				result.Content = strings.Join(parts, "\n")
				toolResults = append(toolResults, result)
			}
		}
		msg.Content = strings.Join(text, "\n")
		if msg.Content != "" || len(msg.Images) > 0 || len(msg.ToolCalls) > 0 {
			out = append(out, msg)
		}
		out = append(out, toolResults...)
	}
	return out, nil
}

func decodeImage(img sh.LLMImageContent) (llm.ImageContent, error) {
	data := img.Data
	// Accept data URLs as well as bare base64.
	if i := strings.Index(data, ";base64,"); strings.HasPrefix(data, "data:") && i >= 0 {
		data = data[i+len(";base64,"):]
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return llm.ImageContent{}, fmt.Errorf("stagehand: decode image: %w", err)
	}
	return llm.ImageContent{MimeType: img.MIMEType, Data: raw}, nil
}

type collected struct {
	text       string
	toolCalls  []llm.ToolCall
	stopReason string
	usage      *sh.LLMUsage
}

// collect drains one provider stream into a single response.
func collect(ctx context.Context, provider llm.LLMProvider, req llm.ChatRequest) (collected, error) {
	stream, err := llm.ObserveChat(ctx, req, llm.CallGeneration, provider.ChatStream)
	if err != nil {
		return collected{}, fmt.Errorf("stagehand: chat stream: %w", err)
	}
	var out collected
	var sb strings.Builder
	for ev := range stream {
		switch ev.Type {
		case llm.EventTextDelta:
			sb.WriteString(ev.Text)
		case llm.EventToolCallDone:
			if ev.ToolCall != nil {
				out.toolCalls = append(out.toolCalls, *ev.ToolCall)
			}
		case llm.EventDone:
			out.stopReason = ev.StopReason
			if ev.Usage != nil {
				out.usage = &sh.LLMUsage{
					InputTokens:  ev.Usage.InputTokens,
					OutputTokens: ev.Usage.OutputTokens,
					TotalTokens:  ev.Usage.InputTokens + ev.Usage.OutputTokens,
				}
				if ev.Usage.CacheReadInputTokens > 0 {
					cached := ev.Usage.CacheReadInputTokens
					out.usage.CachedInputTokens = &cached
				}
			}
		case llm.EventError:
			return collected{}, fmt.Errorf("stagehand: stream error: %w", ev.Error)
		}
	}
	if err := ctx.Err(); err != nil {
		return collected{}, err
	}
	if out.stopReason == llm.StopReasonRefusal {
		return collected{}, errors.New("stagehand: model refused the request")
	}
	out.text = sb.String()
	return out, nil
}

// stripSchemaKeyword removes a top-level "$schema" key. Stagehand's schemas
// are generated by a JSON Schema reflector that emits it, and several
// provider tool-schema validators reject unknown top-level keywords.
func stripSchemaKeyword(schema json.RawMessage) json.RawMessage {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(schema, &m); err != nil {
		return schema
	}
	if _, ok := m["$schema"]; !ok {
		return schema
	}
	delete(m, "$schema")
	out, err := json.Marshal(m)
	if err != nil {
		return schema
	}
	return out
}

// jsonFromText returns the JSON object in s, tolerating a surrounding
// markdown code fence. Returns nil when s holds no valid object.
func jsonFromText(s string) json.RawMessage {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "{") || !json.Valid([]byte(s)) {
		return nil
	}
	return json.RawMessage(s)
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func stringPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
