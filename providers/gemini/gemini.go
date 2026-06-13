package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"
	"github.com/sausheong/harness/llm"
)

// GeminiProvider implements llm.LLMProvider using the Google Gemini API.
type GeminiProvider struct {
	client *genai.Client
}

// NewGeminiProvider creates a new Gemini LLM provider.
func NewGeminiProvider(ctx context.Context, apiKey string) (*GeminiProvider, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("create gemini client: %w", err)
	}
	return &GeminiProvider{client: client}, nil
}

func (p *GeminiProvider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{ID: "gemini-2.5-flash", Name: "Gemini 2.5 Flash", Provider: "gemini"},
		{ID: "gemini-2.5-pro", Name: "Gemini 2.5 Pro", Provider: "gemini"},
		{ID: "gemini-2.0-flash", Name: "Gemini 2.0 Flash", Provider: "gemini"},
	}
}

func (p *GeminiProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	// Build a map from tool call ID to function name for resolving tool results.
	toolIDToName := make(map[string]string)
	for _, m := range req.Messages {
		for _, tc := range m.ToolCalls {
			toolIDToName[tc.ID] = tc.Name
		}
	}

	// Build contents
	var contents []*genai.Content
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			if m.ToolCallID != "" {
				// Tool result — send as function response
				funcName := m.ToolCallID
				if name, ok := toolIDToName[m.ToolCallID]; ok {
					funcName = name
				}
				var response map[string]any
				if m.IsError {
					response = map[string]any{"error": m.Content}
				} else {
					response = map[string]any{"output": m.Content}
				}
				contents = append(contents, &genai.Content{
					Role: "user",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							Name:     funcName,
							ID:       m.ToolCallID,
							Response: response,
						}},
					},
				})
			} else {
				var parts []*genai.Part
				for _, img := range m.Images {
					parts = append(parts, &genai.Part{
						InlineData: &genai.Blob{
							Data:     img.Data,
							MIMEType: img.MimeType,
						},
					})
				}
				if m.Content != "" {
					parts = append(parts, genai.NewPartFromText(m.Content))
				}
				contents = append(contents, &genai.Content{
					Role:  "user",
					Parts: parts,
				})
			}
		case "assistant":
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, genai.NewPartFromText(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				if err := json.Unmarshal(tc.Input, &args); err != nil {
					args = map[string]any{}
				}
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   tc.ID,
						Name: tc.Name,
						Args: args,
					},
				})
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{
					Role:  "model",
					Parts: parts,
				})
			}
		}
	}

	// Build tool declarations
	var tools []*genai.Tool
	if len(req.Tools) > 0 {
		var decls []*genai.FunctionDeclaration
		for _, t := range req.Tools {
			var schema any
			if err := json.Unmarshal(t.Parameters, &schema); err != nil {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			decls = append(decls, &genai.FunctionDeclaration{
				Name:                 t.Name,
				Description:          t.Description,
				ParametersJsonSchema: schema,
			})
		}
		tools = append(tools, &genai.Tool{FunctionDeclarations: decls})
	}

	// Build config
	model := req.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}

	config := &genai.GenerateContentConfig{}

	if sysPrompt := geminiResolveSystemPrompt(req); sysPrompt != "" {
		config.SystemInstruction = genai.NewContentFromText(sysPrompt, "user")
	}

	if req.MaxTokens > 0 {
		config.MaxOutputTokens = int32(req.MaxTokens)
	}

	if req.Temperature > 0 {
		temp := float32(req.Temperature)
		config.Temperature = &temp
	}

	if len(tools) > 0 {
		config.Tools = tools
	}

	if budget, ok := p.BuildThinkingBudget(model, req.Reasoning); ok {
		b := budget // need addressable int32 for *int32 field
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget: &b,
		}
		slog.Info("gemini thinking enabled",
			"model", model,
			"budget_tokens", budget)
	} else if req.Reasoning != llm.ReasoningOff {
		slog.Info("reasoning ignored",
			"provider", "gemini",
			"model", model,
			"requested", string(req.Reasoning),
			"reason", "model does not support thinking")
	}

	// Stream responses
	events := make(chan llm.ChatEvent, 100)

	go func() {
		defer close(events)

		var lastUsage *llm.Usage

		for resp, err := range p.client.Models.GenerateContentStream(ctx, model, contents, config) {
			if err != nil {
				events <- llm.ChatEvent{Type: llm.EventError, Error: err}
				return
			}

			if resp.UsageMetadata != nil {
				lastUsage = updateUsage(lastUsage,
					int(resp.UsageMetadata.PromptTokenCount),
					int(resp.UsageMetadata.CandidatesTokenCount))
			}

			for _, cand := range resp.Candidates {
				if cand.Content == nil {
					continue
				}
				for _, part := range cand.Content.Parts {
					if part.Text != "" {
						events <- llm.ChatEvent{
							Type: llm.EventTextDelta,
							Text: part.Text,
						}
					}
					if part.FunctionCall != nil {
						fc := part.FunctionCall
						argsJSON, err := json.Marshal(fc.Args)
						if err != nil {
							argsJSON = []byte("{}")
						}
						id := fc.ID
						if id == "" {
							id = fc.Name
						}
						events <- llm.ChatEvent{
							Type: llm.EventToolCallStart,
							ToolCall: &llm.ToolCall{
								ID:   id,
								Name: fc.Name,
							},
						}
						events <- llm.ChatEvent{
							Type: llm.EventToolCallDone,
							ToolCall: &llm.ToolCall{
								ID:    id,
								Name:  fc.Name,
								Input: json.RawMessage(argsJSON),
							},
						}
					}
				}
			}
		}

		// Ensure a Done event is always sent
		events <- llm.ChatEvent{Type: llm.EventDone, Usage: lastUsage}
	}()

	return events, nil
}

// updateUsage returns a *llm.Usage carrying the latest cumulative counts.
// Gemini sends cumulative usage on multiple chunks; we keep only the last and
// emit a single EventDone at stream end (mirrors the OpenAI provider's lastUsage).
func updateUsage(_ *llm.Usage, prompt, candidates int) *llm.Usage {
	return &llm.Usage{InputTokens: prompt, OutputTokens: candidates}
}

// geminiUnsupportedFields are JSON Schema fields Gemini's "OpenAPI 3.0
// subset" rejects. This is the broadest strip set across the four
// providers — most of the cross-provider portability gap shows up here.
var geminiUnsupportedFields = []string{"anyOf", "oneOf", "not", "$ref", "format"}

// NormalizeToolSchema strips fields incompatible with Gemini's OpenAPI
// 3.0 subset. Diagnostics list every stripped occurrence with a
// dotted JSON path.
func (p *GeminiProvider) NormalizeToolSchema(tools []llm.ToolDef) ([]llm.ToolDef, []llm.Diagnostic) {
	return llm.ApplyStripList(tools, geminiUnsupportedFields)
}

// BuildThinkingBudget maps a llm.ReasoningMode to Gemini's thinking budget
// (in tokens). Returns (0, false) when reasoning is off or the model
// does not support thinking. Mapping: low=1024, medium=4096, high=16384.
func (p *GeminiProvider) BuildThinkingBudget(model string, mode llm.ReasoningMode) (int32, bool) {
	if mode == llm.ReasoningOff {
		return 0, false
	}
	if !geminiSupportsThinking(model) {
		return 0, false
	}
	switch mode {
	case llm.ReasoningLow:
		return 1024, true
	case llm.ReasoningMedium:
		return 4096, true
	case llm.ReasoningHigh:
		return 16384, true
	default:
		return 0, false
	}
}

// geminiSupportsThinking returns true for Gemini models that accept the
// ThinkingConfig field. Conservative — unknown IDs default to false
// (Gemini's thinking is tightly bound to 2.0-flash-thinking and 2.5
// families; other models reject the field).
func geminiSupportsThinking(model string) bool {
	prefixes := []string{"gemini-2.0-flash-thinking", "gemini-2.5"}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// geminiResolveSystemPrompt returns the effective system prompt string,
// preferring SystemPromptParts when present. Mirrors the OpenAI/Qwen
// resolution path but the per-provider helper avoids importing the
// genai SDK in test code.
func geminiResolveSystemPrompt(req llm.ChatRequest) string {
	if len(req.SystemPromptParts) > 0 {
		return llm.JoinSystemPromptParts(req.SystemPromptParts)
	}
	return req.SystemPrompt
}

