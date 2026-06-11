package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/sausheong/harness/llm"
)

// AnthropicProvider implements llm.LLMProvider using the Anthropic Messages API.
type AnthropicProvider struct {
	client anthropic.Client
}

// NewAnthropicProvider creates a new Anthropic LLM provider.
// If baseURL is non-empty, the client points to that endpoint.
func NewAnthropicProvider(apiKey, baseURL string) *AnthropicProvider {
	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client := anthropic.NewClient(opts...)
	return &AnthropicProvider{client: client}
}

// SupportsPromptCaching implements llm.PromptCachingProvider: the
// Anthropic Messages API honours cache_control breakpoints regardless
// of what the provider block is named in user config.
func (p *AnthropicProvider) SupportsPromptCaching() bool { return true }

func (p *AnthropicProvider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{ID: "claude-sonnet-4-5-20250514", Name: "Claude Sonnet 4.5", Provider: "anthropic"},
		{ID: "claude-opus-4-0-20250514", Name: "Claude Opus 4", Provider: "anthropic"},
		{ID: "claude-haiku-3-5-20241022", Name: "Claude Haiku 3.5", Provider: "anthropic"},
	}
}

// buildMessageParams assembles the anthropic.MessageNewParams shared by
// ChatStream and ChatNonStreaming. Pure: same input → same output (so a
// stream-fallback non-streaming retry sends the same bytes the streaming
// call did, preserving the prompt cache prefix).
func (p *AnthropicProvider) buildMessageParams(req llm.ChatRequest) anthropic.MessageNewParams {
	msgs := buildAnthropicMessages(req.Messages, req.CacheLastMessage)

	tools := make([]anthropic.ToolUnionParam, 0, len(req.Tools))
	for _, t := range req.Tools {
		var props any
		var required []string
		var schema struct {
			Properties any      `json:"properties"`
			Required   []string `json:"required"`
		}
		if err := json.Unmarshal(t.Parameters, &schema); err == nil {
			props = schema.Properties
			required = schema.Required
		}
		tools = append(tools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: props,
					Required:   required,
				},
			},
		})
	}

	maxTokens := int64(req.MaxTokens)
	if maxTokens == 0 {
		maxTokens = 4096
	}
	model := req.Model
	if model == "" {
		model = "claude-sonnet-4-5-20250514"
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  msgs,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	if sys := buildAnthropicSystem(req); len(sys) > 0 {
		params.System = sys
	}
	if req.Temperature > 0 && !anthropicAdaptiveThinkingOnly(model) {
		params.Temperature = anthropic.Float(req.Temperature)
	}
	if anthropicAdaptiveThinkingOnly(model) {
		// Fable 5 / Mythos 5 / Opus 4.7+ reject the legacy
		// enabled+budget_tokens config (400 "thinking.type.enabled is
		// not supported"). They take {type:"adaptive"} plus
		// output_config.effort; sampling params are also removed.
		// Reasoning off → omit thinking entirely (Fable 5 rejects an
		// explicit {type:"disabled"} too).
		if req.Reasoning != llm.ReasoningOff {
			params.Thinking = anthropic.ThinkingConfigParamUnion{
				OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
			}
			params.OutputConfig = anthropic.OutputConfigParam{
				Effort: anthropicEffort(req.Reasoning),
			}
			slog.Info("anthropic adaptive thinking",
				"model", model,
				"effort", string(anthropicEffort(req.Reasoning)))
		}
	} else if cfg, ok := p.BuildThinkingConfig(model, req.Reasoning); ok {
		required := cfg.BudgetTokens + 4096
		if maxTokens < required {
			maxTokens = required
			params.MaxTokens = maxTokens
		}
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: cfg.BudgetTokens,
			},
		}
		slog.Info("anthropic thinking enabled",
			"model", model,
			"budget_tokens", cfg.BudgetTokens,
			"max_tokens", maxTokens)
	} else if req.Reasoning != llm.ReasoningOff {
		slog.Info("reasoning ignored",
			"provider", "anthropic",
			"model", model,
			"requested", string(req.Reasoning),
			"reason", "model does not support thinking")
	}
	return params
}

func (p *AnthropicProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	params := p.buildMessageParams(req)

	// Create stream
	stream := p.client.Messages.NewStreaming(ctx, params)

	events := make(chan llm.ChatEvent, 100)

	go func() {
		defer close(events)
		defer stream.Close()

		// Track tool calls being built up across events.
		//
		// startInput captures any input that arrives on content_block_start.
		// Canonical Anthropic streams send `{}` here and stream the real
		// input via input_json_delta events, but Anthropic-shaped proxies
		// that route non-Anthropic responses (e.g., platformai routing
		// Gemini-Pro/Flash through an Anthropic API) emit the FULL tool
		// input on content_block_start with NO subsequent deltas — Gemini
		// doesn't natively stream tool arguments. Without capturing it
		// here we'd hand the MCP tool an empty input and the server would
		// reject the JSON-RPC envelope as Bad Request.
		type pendingTC struct {
			id         string
			name       string
			inputJSON  string // built from input_json_delta events
			startInput string // captured from content_block_start (proxy fallback)
		}
		var pendingTools []pendingTC
		var currentBlockType string
		var inputTokens, cacheCreationTokens, cacheReadTokens int64

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				u := event.Message.Usage
				inputTokens = u.InputTokens
				cacheCreationTokens = u.CacheCreationInputTokens
				cacheReadTokens = u.CacheReadInputTokens

			case "content_block_start":
				cb := event.ContentBlock
				switch cb.Type {
				case "text":
					currentBlockType = "text"
				case "tool_use":
					currentBlockType = "tool_use"
					pending := pendingTC{
						id:   cb.ID,
						name: cb.Name,
					}
					if cb.Input != nil {
						if data, err := json.Marshal(cb.Input); err == nil {
							s := string(data)
							if s != "null" && s != "{}" {
								pending.startInput = s
							}
						}
					}
					pendingTools = append(pendingTools, pending)
					events <- llm.ChatEvent{
						Type: llm.EventToolCallStart,
						ToolCall: &llm.ToolCall{
							ID:   cb.ID,
							Name: cb.Name,
						},
					}
				}

			case "content_block_delta":
				delta := event.Delta
				switch delta.Type {
				case "text_delta":
					events <- llm.ChatEvent{
						Type: llm.EventTextDelta,
						Text: delta.Text,
					}
				case "input_json_delta":
					if len(pendingTools) > 0 {
						pendingTools[len(pendingTools)-1].inputJSON += delta.PartialJSON
					}
				}

			case "content_block_stop":
				if currentBlockType == "tool_use" && len(pendingTools) > 0 {
					tc := pendingTools[len(pendingTools)-1]
					// Prefer streamed deltas (canonical Anthropic). Fall
					// back to start-time input (proxy emitting full input
					// upfront). Default to "{}" so MCP servers receive a
					// valid empty-args envelope rather than null/empty.
					inp := tc.inputJSON
					if inp == "" {
						inp = tc.startInput
					}
					if inp == "" {
						inp = "{}"
					}
					events <- llm.ChatEvent{
						Type: llm.EventToolCallDone,
						ToolCall: &llm.ToolCall{
							ID:    tc.id,
							Name:  tc.name,
							Input: json.RawMessage(inp),
						},
					}
				}
				currentBlockType = ""

			case "message_delta":
				stopReason := string(event.Delta.StopReason)
				stopCategory := string(event.Delta.StopDetails.Category)
				if stopReason == llm.StopReasonRefusal {
					slog.Warn("anthropic refusal stop reason",
						"category", stopCategory,
						"output_tokens", event.Usage.OutputTokens)
				}
				if event.Usage.OutputTokens > 0 || cacheCreationTokens > 0 || cacheReadTokens > 0 || stopReason != "" {
					slog.Info("anthropic stream usage",
						"input_tokens", inputTokens,
						"output_tokens", event.Usage.OutputTokens,
						"cache_creation_input_tokens", cacheCreationTokens,
						"cache_read_input_tokens", cacheReadTokens,
					)
					events <- llm.ChatEvent{
						Type: llm.EventDone,
						Usage: &llm.Usage{
							InputTokens:              int(inputTokens),
							OutputTokens:             int(event.Usage.OutputTokens),
							CacheCreationInputTokens: int(cacheCreationTokens),
							CacheReadInputTokens:     int(cacheReadTokens),
						},
						StopReason:   stopReason,
						StopCategory: stopCategory,
					}
				}

			case "message_stop":
				// Final event — nothing extra to emit

			case "error":
				slog.Error("anthropic stream error", "event", event.Type)
			}
		}

		if err := stream.Err(); err != nil {
			events <- llm.ChatEvent{
				Type:  llm.EventError,
				Error: err,
			}
		}
	}()

	return events, nil
}

// ChatNonStreaming implements llm.NonStreamingProvider — same request
// shape as ChatStream, but uses the non-streaming Messages.New endpoint
// and synthesises the same event sequence the streaming path would have
// emitted on success. The runtime falls back to this when a stream dies
// mid-flight; the partial output of the failed stream is discarded so
// the prompt cache prefix on the next turn stays byte-identical.
//
// Note: thinking blocks (when reasoning is enabled) are NOT surfaced —
// matches the streaming path which also doesn't emit thinking deltas.
// Tool-use input is converted to JSON via the SDK's RawJSON field.
func (p *AnthropicProvider) ChatNonStreaming(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	params := p.buildMessageParams(req)
	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, err
	}

	events := make(chan llm.ChatEvent, 16)
	go func() {
		defer close(events)
		for _, block := range msg.Content {
			switch block.Type {
			case "text":
				if block.Text != "" {
					events <- llm.ChatEvent{Type: llm.EventTextDelta, Text: block.Text}
				}
			case "tool_use":
				// Stream path emits Start then Done; mirror that so the
				// runtime's collection loop sees the same event ordering
				// either way.
				events <- llm.ChatEvent{
					Type: llm.EventToolCallStart,
					ToolCall: &llm.ToolCall{
						ID:   block.ID,
						Name: block.Name,
					},
				}
				inputJSON, mErr := json.Marshal(block.Input)
				if mErr != nil || len(inputJSON) == 0 || string(inputJSON) == "null" {
					// Default to "{}" so MCP / tool adapters never receive
					// null/empty inputs that would be rejected at the
					// transport layer (Bad Request).
					inputJSON = []byte("{}")
				}
				events <- llm.ChatEvent{
					Type: llm.EventToolCallDone,
					ToolCall: &llm.ToolCall{
						ID:    block.ID,
						Name:  block.Name,
						Input: json.RawMessage(inputJSON),
					},
				}
			}
		}
		events <- llm.ChatEvent{
			Type: llm.EventDone,
			Usage: &llm.Usage{
				InputTokens:              int(msg.Usage.InputTokens),
				OutputTokens:             int(msg.Usage.OutputTokens),
				CacheCreationInputTokens: int(msg.Usage.CacheCreationInputTokens),
				CacheReadInputTokens:     int(msg.Usage.CacheReadInputTokens),
			},
			StopReason:   string(msg.StopReason),
			StopCategory: string(msg.StopDetails.Category),
		}
	}()
	return events, nil
}

// NormalizeToolSchema returns tools unchanged. The Anthropic Messages
// API accepts the full JSON Schema draft-7 dialect (including anyOf,
// oneOf, format, $ref, and definitions), so no fields need stripping
// to be portable. Identity behavior is intentional and durable, not a
// placeholder.
func (p *AnthropicProvider) NormalizeToolSchema(tools []llm.ToolDef) ([]llm.ToolDef, []llm.Diagnostic) {
	return tools, nil
}

// AnthropicThinkingConfig is a provider-internal representation of
// Anthropic's thinking config. Exported for testability — production
// code uses BuildThinkingConfig and wires the result into the SDK
// inside ChatStream.
type AnthropicThinkingConfig struct {
	BudgetTokens int64
}

// BuildThinkingConfig maps a llm.ReasoningMode to Anthropic's thinking
// budget. Returns (nil, false) when reasoning is off or the model
// doesn't support thinking. Mapping: low=1024, medium=4096, high=16384.
//
// Models without thinking support (legacy claude-3.x except 3-7-sonnet)
// always return (nil, false); when the caller sees this for a non-off
// mode it should emit an "ignored" diagnostic.
func (p *AnthropicProvider) BuildThinkingConfig(model string, mode llm.ReasoningMode) (*AnthropicThinkingConfig, bool) {
	if mode == llm.ReasoningOff {
		return nil, false
	}
	if !anthropicSupportsThinking(model) {
		return nil, false
	}
	switch mode {
	case llm.ReasoningLow:
		return &AnthropicThinkingConfig{BudgetTokens: 1024}, true
	case llm.ReasoningMedium:
		return &AnthropicThinkingConfig{BudgetTokens: 4096}, true
	case llm.ReasoningHigh:
		return &AnthropicThinkingConfig{BudgetTokens: 16384}, true
	default:
		return nil, false
	}
}

// anthropicAdaptiveThinkingOnly reports whether the model accepts ONLY
// adaptive thinking: Claude Fable 5, Claude Mythos 5, and Opus 4.7+.
// On these the legacy {type:"enabled", budget_tokens:N} config and
// sampling params (temperature/top_p/top_k) return HTTP 400. Matched by
// substring so gateway model-group names (claude-fable-5-global) and
// Bedrock-style IDs (us.anthropic.claude-opus-4-8-v1) are covered.
func anthropicAdaptiveThinkingOnly(model string) bool {
	adaptiveOnly := []string{
		"claude-fable-5",
		"claude-mythos-5",
		"claude-mythos-preview",
		"claude-opus-4-7",
		"claude-opus-4-8",
		"claude-opus-4-9", // future-proof within the 4.x line
	}
	for _, marker := range adaptiveOnly {
		if strings.Contains(model, marker) {
			return true
		}
	}
	return false
}

// anthropicEffort maps the unified ReasoningMode onto output_config.effort
// for adaptive-thinking-only models.
func anthropicEffort(mode llm.ReasoningMode) anthropic.OutputConfigEffort {
	switch mode {
	case llm.ReasoningLow:
		return anthropic.OutputConfigEffortLow
	case llm.ReasoningMedium:
		return anthropic.OutputConfigEffortMedium
	default:
		return anthropic.OutputConfigEffortHigh
	}
}

// anthropicSupportsThinking returns true for Claude models that accept
// extended thinking. Conservative — unknown IDs default to true so we
// let the API decide rather than silently drop a knob the user set.
func anthropicSupportsThinking(model string) bool {
	noThink := []string{
		"claude-3-haiku", "claude-3-5-haiku",
		"claude-3-sonnet", "claude-3-5-sonnet",
		"claude-3-opus",
	}
	for _, prefix := range noThink {
		if strings.HasPrefix(model, prefix) {
			return false
		}
	}
	return true
}

// buildAnthropicMessages converts the provider-neutral []llm.Message into
// Anthropic's []MessageParam. Consecutive tool-result user messages are
// coalesced into a single user message with multiple tool_result content
// blocks, which is what the Anthropic Messages API requires when an
// assistant turn contained multiple tool_use blocks (e.g. parallel tool
// calls). Without coalescing, the API rejects the second tool_result
// because the immediately preceding message is itself a tool_result, not
// an assistant message containing the matching tool_use.
func buildAnthropicMessages(in []llm.Message, cacheLast bool) []anthropic.MessageParam {
	msgs := make([]anthropic.MessageParam, 0, len(in))
	for i := 0; i < len(in); i++ {
		m := in[i]
		switch m.Role {
		case "user":
			if m.ToolCallID != "" {
				// Collect a run of consecutive tool_result user messages.
				var blocks []anthropic.ContentBlockParamUnion
				for ; i < len(in); i++ {
					cur := in[i]
					if cur.Role != "user" || cur.ToolCallID == "" {
						i-- // un-consume; outer loop will re-process
						break
					}
					blocks = append(blocks, buildToolResultBlock(cur))
				}
				msgs = append(msgs, anthropic.NewUserMessage(blocks...))
			} else if len(m.Images) > 0 {
				var blocks []anthropic.ContentBlockParamUnion
				for _, img := range m.Images {
					encoded := base64.StdEncoding.EncodeToString(img.Data)
					blocks = append(blocks, anthropic.NewImageBlockBase64(img.MimeType, encoded))
				}
				if m.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(m.Content))
				}
				msgs = append(msgs, anthropic.NewUserMessage(blocks...))
			} else {
				msgs = append(msgs, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			}
		case "assistant":
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var input any
				if err := json.Unmarshal(tc.Input, &input); err != nil {
					input = map[string]any{}
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, input, tc.Name))
			}
			if len(blocks) > 0 {
				msgs = append(msgs, anthropic.NewAssistantMessage(blocks...))
			}
		}
	}
	// Cache marker on the last block of the last message. Anthropic accepts
	// up to 4 cache_control markers per request; combined with the static
	// system block this is at most 2.
	if cacheLast && len(msgs) > 0 {
		tail := &msgs[len(msgs)-1]
		if len(tail.Content) > 0 {
			setCacheControlOnBlock(&tail.Content[len(tail.Content)-1])
		}
	}
	return msgs
}

// setCacheControlOnBlock attaches an ephemeral cache_control marker to the
// underlying param of any ContentBlockParamUnion variant that has a
// CacheControl field. The SDK union type does not expose a direct setter,
// so we mutate the variant struct directly. Variants other than text /
// tool_result / image / tool_use are out of scope; the runtime never
// produces them as the tail of a user message.
func setCacheControlOnBlock(block *anthropic.ContentBlockParamUnion) {
	cc := anthropic.NewCacheControlEphemeralParam()
	switch {
	case block.OfText != nil:
		block.OfText.CacheControl = cc
	case block.OfToolResult != nil:
		block.OfToolResult.CacheControl = cc
	case block.OfImage != nil:
		block.OfImage.CacheControl = cc
	case block.OfToolUse != nil:
		block.OfToolUse.CacheControl = cc
	}
}

// buildToolResultBlock converts a single tool_result llm.Message (a user
// message with ToolCallID set) into an Anthropic ContentBlockParamUnion
// suitable for inclusion in a NewUserMessage. Preserves the
// with-images and without-images branches: image-bearing results build
// a structured content slice; plain results use the SDK helper.
func buildToolResultBlock(m llm.Message) anthropic.ContentBlockParamUnion {
	if len(m.Images) > 0 {
		var content []anthropic.ToolResultBlockParamContentUnion
		for _, img := range m.Images {
			encoded := base64.StdEncoding.EncodeToString(img.Data)
			content = append(content, anthropic.ToolResultBlockParamContentUnion{
				OfImage: &anthropic.ImageBlockParam{
					Source: anthropic.ImageBlockParamSourceUnion{
						OfBase64: &anthropic.Base64ImageSourceParam{
							Data:      encoded,
							MediaType: anthropic.Base64ImageSourceMediaType(img.MimeType),
						},
					},
				},
			})
		}
		if m.Content != "" {
			content = append(content, anthropic.ToolResultBlockParamContentUnion{
				OfText: &anthropic.TextBlockParam{Text: m.Content},
			})
		}
		toolBlock := anthropic.ToolResultBlockParam{
			ToolUseID: m.ToolCallID,
			IsError:   anthropic.Bool(m.IsError),
			Content:   content,
		}
		return anthropic.ContentBlockParamUnion{OfToolResult: &toolBlock}
	}
	return anthropic.NewToolResultBlock(m.ToolCallID, m.Content, m.IsError)
}

// buildAnthropicSystem builds the System param array from a llm.ChatRequest.
// Prefers SystemPromptParts when set: each non-empty part becomes one
// TextBlockParam; parts with Cache=true get an ephemeral cache_control marker.
// Falls back to a single un-cached block built from SystemPrompt when parts
// are absent. Returns nil when both inputs are empty.
func buildAnthropicSystem(req llm.ChatRequest) []anthropic.TextBlockParam {
	if len(req.SystemPromptParts) > 0 {
		blocks := make([]anthropic.TextBlockParam, 0, len(req.SystemPromptParts))
		for _, p := range req.SystemPromptParts {
			if p.Text == "" {
				continue
			}
			b := anthropic.TextBlockParam{Text: p.Text}
			if p.Cache {
				b.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			blocks = append(blocks, b)
		}
		if len(blocks) > 0 {
			return blocks
		}
		return nil
	}
	if req.SystemPrompt != "" {
		return []anthropic.TextBlockParam{{Text: req.SystemPrompt}}
	}
	return nil
}
