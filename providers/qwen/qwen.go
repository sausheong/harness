package qwen

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sausheong/harness/llm"
)

// pendingTC accumulates a streamed tool call across deltas. args uses
// strings.Builder so repeated argument fragments append in O(n) total
// (not O(n²) from string concatenation).
type pendingTC struct {
	id   string
	name string
	args strings.Builder
}

// emitToolCalls sends EventToolCallDone for each completed tool call in
// ascending index order so tool_use block order is deterministic across
// turns (prompt-cache stability).
func emitToolCalls(events chan<- llm.ChatEvent, toolCalls map[int]*pendingTC) {
	maxIdx := -1
	for idx := range toolCalls {
		if idx > maxIdx {
			maxIdx = idx
		}
	}
	for idx := 0; idx <= maxIdx; idx++ {
		tc, ok := toolCalls[idx]
		if !ok || tc.name == "" {
			continue
		}
		events <- llm.ChatEvent{
			Type: llm.EventToolCallDone,
			ToolCall: &llm.ToolCall{
				ID:    tc.id,
				Name:  tc.name,
				Input: json.RawMessage(tc.args.String()),
			},
		}
	}
}

// QwenProvider implements llm.LLMProvider using the Qwen Chat Completions API.
// Qwen uses an OpenAI-compatible API, so we reuse the go-openai client.
type QwenProvider struct {
	client *openai.Client
}

// NewQwenProvider creates a new Qwen LLM provider.
// The Qwen API is OpenAI-compatible, so we use the go-openai client
// with Qwen's base URL.
func NewQwenProvider(apiKey, baseURL string) *QwenProvider {
	cfg := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		cfg.BaseURL = baseURL
	} else {
		// Default to Qwen's official API endpoint
		cfg.BaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	}
	client := openai.NewClientWithConfig(cfg)
	return &QwenProvider{client: client}
}

func (p *QwenProvider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{ID: "qwen-plus", Name: "Qwen Plus", Provider: "qwen"},
		{ID: "qwen-turbo", Name: "Qwen Turbo", Provider: "qwen"},
		{ID: "qwen-max", Name: "Qwen Max", Provider: "qwen"},
		{ID: "qwen-coder-plus", Name: "Qwen Coder Plus", Provider: "qwen"},
		{ID: "qwen-vl-plus", Name: "Qwen VL Plus", Provider: "qwen"},
	}
}

func (p *QwenProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	// Build messages
	msgs := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+1)

	if sysPrompt := qwenResolveSystemPrompt(req); sysPrompt != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: sysPrompt,
		})
	}

	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			if m.ToolCallID != "" {
				if len(m.Images) > 0 {
					// Tool result with images: use multi-content parts
					var parts []openai.ChatMessagePart
					for _, img := range m.Images {
						encoded := base64.StdEncoding.EncodeToString(img.Data)
						dataURI := fmt.Sprintf("data:%s;base64,%s", img.MimeType, encoded)
						parts = append(parts, openai.ChatMessagePart{
							Type: openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{
								URL:    dataURI,
								Detail: openai.ImageURLDetailAuto,
							},
						})
					}
					if m.Content != "" {
						parts = append(parts, openai.ChatMessagePart{
							Type: openai.ChatMessagePartTypeText,
							Text: m.Content,
						})
					}
					msgs = append(msgs, openai.ChatCompletionMessage{
						Role:         openai.ChatMessageRoleTool,
						MultiContent: parts,
						ToolCallID:   m.ToolCallID,
					})
				} else {
					msgs = append(msgs, openai.ChatCompletionMessage{
						Role:       openai.ChatMessageRoleTool,
						Content:    m.Content,
						ToolCallID: m.ToolCallID,
					})
				}
			} else if len(m.Images) > 0 {
				var parts []openai.ChatMessagePart
				for _, img := range m.Images {
					encoded := base64.StdEncoding.EncodeToString(img.Data)
					dataURI := fmt.Sprintf("data:%s;base64,%s", img.MimeType, encoded)
					parts = append(parts, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL:    dataURI,
							Detail: openai.ImageURLDetailAuto,
						},
					})
				}
				if m.Content != "" {
					parts = append(parts, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeText,
						Text: m.Content,
					})
				}
				msgs = append(msgs, openai.ChatCompletionMessage{
					Role:         openai.ChatMessageRoleUser,
					MultiContent: parts,
				})
			} else {
				msgs = append(msgs, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: m.Content,
				})
			}
		case "assistant":
			msg := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: m.Content,
			}
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Input),
					},
				})
			}
			msgs = append(msgs, msg)
		}
	}

	// Build tools
	var tools []openai.Tool
	for _, t := range req.Tools {
		var params any
		if err := json.Unmarshal(t.Parameters, &params); err != nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}

		tools = append(tools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}

	model := req.Model
	if model == "" {
		model = "qwen-plus"
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	openaiReq := openai.ChatCompletionRequest{
		Model:               model,
		Messages:            msgs,
		MaxCompletionTokens: maxTokens,
		Stream:              true,
		StreamOptions:       &openai.StreamOptions{IncludeUsage: true},
	}

	if len(tools) > 0 {
		openaiReq.Tools = tools
	}

	if req.Temperature > 0 {
		openaiReq.Temperature = float32(req.Temperature)
	}

	if enabled, diag, ok := p.BuildEnableThinking(model, req.Reasoning); ok {
		// TODO: Qwen DashScope expects a custom top-level "enable_thinking"
		// JSON field that go-openai v1.41 does not expose via any
		// ExtraBody/RawJSON mechanism. Wiring it requires either upgrading
		// to a go-openai version with custom-field support or adding a
		// custom HTTP roundtripper. Tracked as a Phase 2 follow-up.
		_ = enabled
		_ = diag
		slog.Info("qwen thinking requested",
			"model", model,
			"requested", string(req.Reasoning),
			"clamped_to_bool", true,
			"reason", diag.Reason,
			"sdk_limitation", "enable_thinking not yet wired to wire format")
	} else if req.Reasoning != llm.ReasoningOff {
		slog.Info("reasoning ignored",
			"provider", "qwen",
			"model", model,
			"requested", string(req.Reasoning),
			"reason", "model does not support thinking")
	}

	stream, err := p.client.CreateChatCompletionStream(ctx, openaiReq)
	if err != nil {
		return nil, err
	}

	events := make(chan llm.ChatEvent, 100)

	go func() {
		defer close(events)
		defer stream.Close()

		// Track tool calls being built up across deltas
		toolCalls := make(map[int]*pendingTC)

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				events <- llm.ChatEvent{Type: llm.EventError, Error: err}
				return
			}

			for _, choice := range resp.Choices {
				delta := choice.Delta

				// Text content
				if delta.Content != "" {
					events <- llm.ChatEvent{
						Type: llm.EventTextDelta,
						Text: delta.Content,
					}
				}

				// Tool calls
				for _, tc := range delta.ToolCalls {
					idx := 0
					if tc.Index != nil {
						idx = *tc.Index
					}
					pending, exists := toolCalls[idx]
					if !exists {
						pending = &pendingTC{}
						toolCalls[idx] = pending
					}

					if tc.ID != "" {
						pending.id = tc.ID
					}
					if tc.Function.Name != "" {
						pending.name = tc.Function.Name
						events <- llm.ChatEvent{
							Type: llm.EventToolCallStart,
							ToolCall: &llm.ToolCall{
								ID:   pending.id,
								Name: pending.name,
							},
						}
					}
					if tc.Function.Arguments != "" {
						pending.args.WriteString(tc.Function.Arguments)
					}
				}

				// Finish reason
				if choice.FinishReason == openai.FinishReasonToolCalls || choice.FinishReason == openai.FinishReasonStop {
					// Emit completed tool calls in index order.
					emitToolCalls(events, toolCalls)
				}
			}
		}

		events <- llm.ChatEvent{Type: llm.EventDone}
	}()

	return events, nil
}

// NormalizeToolSchema strips $ref and definitions from each tool's
// JSON Schema. Qwen DashScope tracks the OpenAI function-calling
// shape, so the same restricted JSON Schema subset applies (and the
// same openaiUnsupportedFields list is reused).
func (p *QwenProvider) NormalizeToolSchema(tools []llm.ToolDef) ([]llm.ToolDef, []llm.Diagnostic) {
	return llm.ApplyStripList(tools, []string{"$ref", "definitions"})
}

// BuildEnableThinking maps a llm.ReasoningMode to Qwen's enable_thinking
// boolean. Returns (false, empty diag, false) when off or the model
// doesn't support thinking. For any non-off mode on a supported model,
// returns (true, clamped diag, true) — the boolean toggle loses the
// requested low/medium/high granularity, so the diag always fires.
func (p *QwenProvider) BuildEnableThinking(model string, mode llm.ReasoningMode) (bool, llm.Diagnostic, bool) {
	if mode == llm.ReasoningOff {
		return false, llm.Diagnostic{}, false
	}
	if !qwenSupportsThinking(model) {
		return false, llm.Diagnostic{}, false
	}
	return true, llm.Diagnostic{
		Action: "clamped",
		Reason: "qwen reasoning is boolean; granularity ignored",
	}, true
}

// qwenSupportsThinking returns true for Qwen models that support the
// enable_thinking parameter. Conservative — unknown IDs default to
// false (only QwQ and Qwen3 family models accept this field).
func qwenSupportsThinking(model string) bool {
	prefixes := []string{"qwen-qwq", "qwen3"}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// qwenResolveSystemPrompt returns the effective system prompt string,
// preferring SystemPromptParts when present. Mirrors the OpenAI/Gemini
// resolution helpers.
func qwenResolveSystemPrompt(req llm.ChatRequest) string {
	if len(req.SystemPromptParts) > 0 {
		return llm.JoinSystemPromptParts(req.SystemPromptParts)
	}
	return req.SystemPrompt
}
