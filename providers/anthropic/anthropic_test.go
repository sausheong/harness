package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

// helper: extract tool_use IDs (in order) from an assistant MessageParam.
func toolUseIDs(t *testing.T, m anthropic.MessageParam) []string {
	t.Helper()
	require.Equal(t, anthropic.MessageParamRoleAssistant, m.Role)
	var ids []string
	for _, b := range m.Content {
		if b.OfToolUse != nil {
			ids = append(ids, b.OfToolUse.ID)
		}
	}
	return ids
}

// helper: extract tool_result IDs (in order) from a user MessageParam.
func toolResultIDs(t *testing.T, m anthropic.MessageParam) []string {
	t.Helper()
	require.Equal(t, anthropic.MessageParamRoleUser, m.Role)
	var ids []string
	for _, b := range m.Content {
		if b.OfToolResult != nil {
			ids = append(ids, b.OfToolResult.ToolUseID)
		}
	}
	return ids
}

// TestAnthropic_ConsecutiveToolResultsCoalesce verifies that when the
// LLM emits N tool_use blocks in a single assistant turn (e.g. parallel
// MCP calls), the N resulting tool_result user messages produced by the
// agent are coalesced into ONE user message with N tool_result content
// blocks. The Anthropic Messages API requires this — it rejects each
// tool_result needing the immediately preceding message to contain the
// matching tool_use.
func TestAnthropic_ConsecutiveToolResultsCoalesce(t *testing.T) {
	in := []llm.Message{
		{Role: "user", Content: "do three things in parallel"},
		{
			Role:    "assistant",
			Content: "ok",
			ToolCalls: []llm.ToolCall{
				{ID: "A", Name: "search", Input: []byte(`{"q":"a"}`)},
				{ID: "B", Name: "search", Input: []byte(`{"q":"b"}`)},
				{ID: "C", Name: "search", Input: []byte(`{"q":"c"}`)},
			},
		},
		// Results may arrive in any order — agent dispatches in parallel.
		{Role: "user", ToolCallID: "B", Content: "result B"},
		{Role: "user", ToolCallID: "A", Content: "result A"},
		{Role: "user", ToolCallID: "C", Content: "result C"},
		{Role: "assistant", Content: "done"},
	}

	got := buildAnthropicMessages(in, false)

	// Expected shape: user(text), assistant(3 tool_uses), user(3 tool_results), assistant
	require.Len(t, got, 4, "expected 4 messages, got %d", len(got))

	assert.Equal(t, anthropic.MessageParamRoleUser, got[0].Role)
	assert.Equal(t, []string{"A", "B", "C"}, toolUseIDs(t, got[1]))
	assert.Equal(t, []string{"B", "A", "C"}, toolResultIDs(t, got[2]),
		"three tool_results must coalesce into one user message preserving order")
	assert.Equal(t, anthropic.MessageParamRoleAssistant, got[3].Role)
}

// TestAnthropic_SingleToolResultStillSeparate pins the existing
// behavior: a single tool_result still produces a single user message
// containing one tool_result block.
func TestAnthropic_SingleToolResultStillSeparate(t *testing.T) {
	in := []llm.Message{
		{Role: "user", Content: "search"},
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "X", Name: "search", Input: []byte(`{"q":"x"}`)},
			},
		},
		{Role: "user", ToolCallID: "X", Content: "result X"},
		{Role: "assistant", Content: "done"},
	}

	got := buildAnthropicMessages(in, false)

	require.Len(t, got, 4)
	assert.Equal(t, []string{"X"}, toolUseIDs(t, got[1]))
	assert.Equal(t, []string{"X"}, toolResultIDs(t, got[2]),
		"single tool_result still produces one user message with one block")
}

// TestAnthropic_ToolResultRunInterspersedWithText verifies that
// coalescing only spans a contiguous run of tool_result messages and
// stops at non-tool-result boundaries (e.g. a plain user text turn).
func TestAnthropic_ToolResultRunInterspersedWithText(t *testing.T) {
	in := []llm.Message{
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "T1", Name: "f", Input: []byte(`{}`)},
			},
		},
		{Role: "user", ToolCallID: "T1", Content: "r1"},
		{Role: "user", Content: "interjection"},
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "T2", Name: "f", Input: []byte(`{}`)},
				{ID: "T3", Name: "f", Input: []byte(`{}`)},
			},
		},
		{Role: "user", ToolCallID: "T2", Content: "r2"},
		{Role: "user", ToolCallID: "T3", Content: "r3"},
		{Role: "user", Content: "trailing text"},
	}

	got := buildAnthropicMessages(in, false)

	// Expected: assistant, user(1 tr), user(text), assistant, user(2 tr), user(text)
	require.Len(t, got, 6)
	assert.Equal(t, []string{"T1"}, toolUseIDs(t, got[0]))
	assert.Equal(t, []string{"T1"}, toolResultIDs(t, got[1]))

	// got[2] is the plain text interjection — user role, no tool_result.
	assert.Equal(t, anthropic.MessageParamRoleUser, got[2].Role)
	require.Len(t, got[2].Content, 1)
	require.NotNil(t, got[2].Content[0].OfText)
	assert.Equal(t, "interjection", got[2].Content[0].OfText.Text)

	assert.Equal(t, []string{"T2", "T3"}, toolUseIDs(t, got[3]))
	assert.Equal(t, []string{"T2", "T3"}, toolResultIDs(t, got[4]),
		"the 2-element tool_result run must coalesce")

	// got[5] is the trailing plain text.
	assert.Equal(t, anthropic.MessageParamRoleUser, got[5].Role)
	require.Len(t, got[5].Content, 1)
	require.NotNil(t, got[5].Content[0].OfText)
	assert.Equal(t, "trailing text", got[5].Content[0].OfText.Text)
}

// TestAnthropic_ToolResultsWithImages verifies that a coalesced run
// containing both plain-text and image-bearing tool_results preserves
// each block's images. The image-bearing branch builds a structured
// content slice; the plain branch uses the SDK helper. Both must end up
// inside the same user message.
func TestAnthropic_ToolResultsWithImages(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4E, 0x47}
	in := []llm.Message{
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "P1", Name: "snap", Input: []byte(`{}`)},
				{ID: "P2", Name: "snap", Input: []byte(`{}`)},
				{ID: "P3", Name: "snap", Input: []byte(`{}`)},
			},
		},
		// Plain text result.
		{Role: "user", ToolCallID: "P1", Content: "no image"},
		// Image-bearing result with caption.
		{
			Role:       "user",
			ToolCallID: "P2",
			Content:    "see attached",
			Images: []llm.ImageContent{
				{MimeType: "image/png", Data: pngBytes},
			},
		},
		// Image-bearing result, error.
		{
			Role:       "user",
			ToolCallID: "P3",
			IsError:    true,
			Images: []llm.ImageContent{
				{MimeType: "image/png", Data: pngBytes},
			},
		},
	}

	got := buildAnthropicMessages(in, false)

	// Expected: assistant(3 tool_uses), user(3 tool_results coalesced).
	require.Len(t, got, 2)
	assert.Equal(t, []string{"P1", "P2", "P3"}, toolUseIDs(t, got[0]))
	assert.Equal(t, []string{"P1", "P2", "P3"}, toolResultIDs(t, got[1]))

	// P1: no image — built via NewToolResultBlock helper, content is
	// either nil/empty or carries text — we don't assert on its shape
	// beyond ID. The key assertion is presence in the same user message.
	require.NotNil(t, got[1].Content[0].OfToolResult)

	// P2: image + text caption.
	p2 := got[1].Content[1].OfToolResult
	require.NotNil(t, p2)
	require.Len(t, p2.Content, 2, "image + text caption -> 2 content items")
	require.NotNil(t, p2.Content[0].OfImage)
	assert.Equal(t,
		base64.StdEncoding.EncodeToString(pngBytes),
		p2.Content[0].OfImage.Source.OfBase64.Data)
	require.NotNil(t, p2.Content[1].OfText)
	assert.Equal(t, "see attached", p2.Content[1].OfText.Text)

	// P3: image only, IsError set.
	p3 := got[1].Content[2].OfToolResult
	require.NotNil(t, p3)
	require.Len(t, p3.Content, 1, "image only -> 1 content item")
	require.NotNil(t, p3.Content[0].OfImage)
	assert.True(t, p3.IsError.Value)
}

func TestAnthropicSystemPromptPartsEmitCacheControl(t *testing.T) {
	got := buildAnthropicSystem(llm.ChatRequest{
		SystemPromptParts: []llm.SystemPromptPart{
			{Text: "static-cached", Cache: true},
			{Text: "dynamic", Cache: false},
		},
	})
	require.Len(t, got, 2)
	require.Equal(t, "static-cached", got[0].Text)
	require.Equal(t, "ephemeral", string(got[0].CacheControl.Type))
	require.Equal(t, "dynamic", got[1].Text)
	require.Empty(t, string(got[1].CacheControl.Type), "second block must not be cache-marked")
}

func TestAnthropicSystemPromptStringFallback(t *testing.T) {
	got := buildAnthropicSystem(llm.ChatRequest{SystemPrompt: "legacy"})
	require.Len(t, got, 1)
	require.Equal(t, "legacy", got[0].Text)
	require.Empty(t, string(got[0].CacheControl.Type))
}

func TestAnthropicSystemEmptyWhenBothEmpty(t *testing.T) {
	got := buildAnthropicSystem(llm.ChatRequest{})
	require.Empty(t, got)
}

func TestAnthropicSystemSkipsEmptyParts(t *testing.T) {
	got := buildAnthropicSystem(llm.ChatRequest{
		SystemPromptParts: []llm.SystemPromptPart{
			{Text: ""},
			{Text: "real", Cache: true},
		},
	})
	require.Len(t, got, 1)
	require.Equal(t, "real", got[0].Text)
}

func TestBuildAnthropicMessagesCacheLastTextBlock(t *testing.T) {
	in := []llm.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "second"},
	}
	got := buildAnthropicMessages(in, true)
	require.Len(t, got, 3)
	last := got[len(got)-1]
	blocks := last.Content
	require.NotEmpty(t, blocks)
	cc := blocks[len(blocks)-1].GetCacheControl()
	require.NotNil(t, cc)
	require.Equal(t, "ephemeral", string(cc.Type))
}

func TestBuildAnthropicMessagesNoMarkerWhenCacheLastFalse(t *testing.T) {
	in := []llm.Message{{Role: "user", Content: "hi"}}
	got := buildAnthropicMessages(in, false)
	require.Len(t, got, 1)
	blocks := got[0].Content
	require.NotEmpty(t, blocks)
	cc := blocks[len(blocks)-1].GetCacheControl()
	if cc != nil {
		require.Empty(t, string(cc.Type), "no cache_control should be emitted when CacheLastMessage=false")
	}
}

func TestBuildAnthropicMessagesCacheLastToolResult(t *testing.T) {
	in := []llm.Message{
		{Role: "assistant", Content: "thinking"},
		{Role: "user", ToolCallID: "tc_1", Content: "tool output"},
	}
	got := buildAnthropicMessages(in, true)
	last := got[len(got)-1]
	require.NotEmpty(t, last.Content)
	cc := last.Content[len(last.Content)-1].GetCacheControl()
	require.NotNil(t, cc)
	require.Equal(t, "ephemeral", string(cc.Type))
}

func TestBuildAnthropicMessagesCacheLastImageBlock(t *testing.T) {
	// Image-bearing user message: the SDK lays out blocks as [image..., text?].
	// With Content="" the trailing block is the image, so cache_control must
	// land on the OfImage variant.
	in := []llm.Message{{
		Role: "user",
		Images: []llm.ImageContent{
			{MimeType: "image/png", Data: []byte{0x89, 'P', 'N', 'G'}},
		},
	}}
	got := buildAnthropicMessages(in, true)
	require.Len(t, got, 1)
	last := got[0]
	require.NotEmpty(t, last.Content)
	tail := last.Content[len(last.Content)-1]
	require.NotNil(t, tail.OfImage, "expected last block to be an image variant")
	cc := tail.GetCacheControl()
	require.NotNil(t, cc)
	require.Equal(t, "ephemeral", string(cc.Type))
}

// TestAnthropicStreamSurfacesCacheTokens points the SDK at an httptest
// server that serves a canned SSE response with cache_creation_input_tokens
// and cache_read_input_tokens populated, and asserts the emitted
// llm.Usage carries them through.
func TestAnthropicStreamSurfacesCacheTokens(t *testing.T) {
	const sseBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":42,"cache_read_input_tokens":17}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody))
	}))
	t.Cleanup(srv.Close)

	p := NewAnthropicProvider("test-key", srv.URL)
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "claude-test"})
	require.NoError(t, err)

	var done *llm.ChatEvent
	for ev := range stream {
		if ev.Type == llm.EventDone {
			done = &ev
		}
	}
	require.NotNil(t, done, "expected llm.EventDone")
	require.NotNil(t, done.Usage, "expected llm.Usage on llm.EventDone")
	require.Equal(t, 42, done.Usage.CacheCreationInputTokens)
	require.Equal(t, 17, done.Usage.CacheReadInputTokens)
	require.Equal(t, 5, done.Usage.OutputTokens)
	require.Equal(t, 10, done.Usage.InputTokens)
}

// TestBuildMessageParamsIsPure protects the cache-prefix invariant
// behind the stream-fallback feature: when a stream dies mid-flight
// the runtime retries via ChatNonStreaming with the SAME llm.ChatRequest,
// and both call sites share buildMessageParams. If buildMessageParams
// were non-deterministic (timestamp, RNG, map iteration leaking into
// output) the retry would send different bytes and Anthropic would
// invalidate the prompt cache. We compare wire-form JSON to catch
// drift the Go struct's `==` would miss (e.g. nil vs empty slice).
func TestBuildMessageParamsIsPure(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	req := llm.ChatRequest{
		Model:       "claude-sonnet-4-5",
		MaxTokens:   4096,
		Temperature: 0.7,
		SystemPromptParts: []llm.SystemPromptPart{
			{Text: "you are an assistant", Cache: true},
			{Text: "today is 2026-05-01"},
		},
		Messages: []llm.Message{
			{Role: "user", Content: "hello"},
			{
				Role:    "assistant",
				Content: "ok",
				ToolCalls: []llm.ToolCall{
					{ID: "A", Name: "search", Input: []byte(`{"q":"a"}`)},
					{ID: "B", Name: "search", Input: []byte(`{"q":"b"}`)},
				},
			},
			{Role: "tool", Content: "ra", ToolCallID: "A"},
			{Role: "tool", Content: "rb", ToolCallID: "B"},
		},
		Tools: []llm.ToolDef{
			{Name: "search", Description: "search the web", Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)},
		},
		CacheLastMessage: true,
	}

	first := p.buildMessageParams(req)
	second := p.buildMessageParams(req)

	firstJSON, err := json.Marshal(first)
	require.NoError(t, err)
	secondJSON, err := json.Marshal(second)
	require.NoError(t, err)

	assert.Equal(t, string(firstJSON), string(secondJSON),
		"buildMessageParams must be a pure function — the stream-fallback "+
			"non-streaming retry depends on byte-identical params to keep the "+
			"prompt cache prefix valid")
}

// TestAnthropicStream_ToolUseInputOnContentBlockStart verifies the
// proxy-fallback path: some Anthropic-shaped gateways (e.g. platformai
// routing Gemini-Pro / Flash through the Anthropic API) emit the FULL
// tool input on content_block_start with NO subsequent input_json_delta
// events, because Gemini doesn't natively stream tool arguments.
//
// Without the fallback, the parser would emit `Input: ""` and the
// downstream MCP adapter would forward an empty/null `arguments`
// payload — which the gateway rejects with HTTP 400 (Bad Request).
// This was the actual production failure for the assistantai server
// only-when-using-Gemini.
func TestAnthropicStream_ToolUseInputOnContentBlockStart(t *testing.T) {
	const sseBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_g","type":"message","role":"assistant","content":[],"model":"gemini-3.1-pro","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_abc","name":"nl_search_ms_mail","input":{"query":"latest","limit":5}}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":11}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody))
	}))
	t.Cleanup(srv.Close)

	p := NewAnthropicProvider("test-key", srv.URL)
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "gemini-test"})
	require.NoError(t, err)

	var done *llm.ChatEvent
	var toolDone *llm.ChatEvent
	for ev := range stream {
		switch ev.Type {
		case llm.EventToolCallDone:
			ev := ev // capture
			toolDone = &ev
		case llm.EventDone:
			ev := ev
			done = &ev
		}
	}
	require.NotNil(t, done, "expected llm.EventDone")
	require.NotNil(t, toolDone, "expected llm.EventToolCallDone")
	require.NotNil(t, toolDone.ToolCall)
	assert.Equal(t, "call_abc", toolDone.ToolCall.ID)
	assert.Equal(t, "nl_search_ms_mail", toolDone.ToolCall.Name)
	assert.JSONEq(t, `{"query":"latest","limit":5}`, string(toolDone.ToolCall.Input),
		"input from content_block_start must be captured when no input_json_delta arrives — "+
			"otherwise Gemini-via-Anthropic-proxy tool calls reach MCP with empty args and 400")
}

// TestAnthropicStream_EmptyToolInputDefaultsToObject verifies that a
// tool_use that genuinely has no arguments still produces a valid empty
// object instead of an empty string, so MCP adapters always send a
// well-formed `arguments` field over the wire.
func TestAnthropicStream_EmptyToolInputDefaultsToObject(t *testing.T) {
	const sseBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_e","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_e","name":"ping","input":{}}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody))
	}))
	t.Cleanup(srv.Close)

	p := NewAnthropicProvider("test-key", srv.URL)
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "claude-test"})
	require.NoError(t, err)

	var toolDone *llm.ChatEvent
	for ev := range stream {
		if ev.Type == llm.EventToolCallDone {
			ev := ev
			toolDone = &ev
		}
	}
	require.NotNil(t, toolDone)
	require.NotNil(t, toolDone.ToolCall)
	assert.Equal(t, `{}`, string(toolDone.ToolCall.Input),
		"argument-less tool call must serialize as {} so MCP transport accepts it")
}

// --- Adaptive-thinking model mapping ---------------------------------
//
// Claude Fable 5, Claude Mythos 5, and Opus 4.7/4.8 reject the legacy
// thinking config: `thinking: {type:"enabled", budget_tokens:N}` returns
// HTTP 400 ("thinking.type.enabled is not supported for this model").
// They also removed sampling params — sending `temperature` is a 400.
// For these models the provider must emit `thinking: {type:"adaptive"}`
// plus `output_config.effort`, and drop temperature.

func wireJSON(t *testing.T, p *AnthropicProvider, req llm.ChatRequest) string {
	t.Helper()
	params := p.buildMessageParams(req)
	b, err := json.Marshal(params)
	require.NoError(t, err)
	return string(b)
}

func TestAdaptiveThinkingModels_EmitAdaptiveNotEnabled(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	models := []string{
		"claude-fable-5",
		"claude-fable-5-global",   // LiteLLM / gateway model-group names
		"claude-mythos-5",
		"claude-opus-4-7",
		"claude-opus-4-8",
		"us.anthropic.claude-opus-4-8-v1", // Bedrock-style ID
	}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			got := wireJSON(t, p, llm.ChatRequest{
				Model:     model,
				Reasoning: llm.ReasoningHigh,
				Messages:  []llm.Message{{Role: "user", Content: "hi"}},
			})
			assert.Contains(t, got, `"thinking":{"type":"adaptive"}`,
				"adaptive-only models must send thinking.type=adaptive")
			assert.NotContains(t, got, `"budget_tokens"`,
				"budget_tokens is rejected with 400 on adaptive-only models")
			assert.Contains(t, got, `"output_config":{"effort":"high"}`,
				"reasoning mode maps to output_config.effort on adaptive-only models")
		})
	}
}

func TestAdaptiveThinkingModels_EffortMapping(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	cases := []struct {
		mode   llm.ReasoningMode
		effort string
	}{
		{llm.ReasoningLow, "low"},
		{llm.ReasoningMedium, "medium"},
		{llm.ReasoningHigh, "high"},
	}
	for _, c := range cases {
		t.Run(string(c.mode), func(t *testing.T) {
			got := wireJSON(t, p, llm.ChatRequest{
				Model:     "claude-opus-4-8",
				Reasoning: c.mode,
				Messages:  []llm.Message{{Role: "user", Content: "hi"}},
			})
			assert.Contains(t, got, `"output_config":{"effort":"`+c.effort+`"}`)
		})
	}
}

func TestAdaptiveThinkingModels_ReasoningOffOmitsThinking(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	got := wireJSON(t, p, llm.ChatRequest{
		Model:    "claude-fable-5",
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	assert.NotContains(t, got, `"thinking"`,
		"reasoning off must omit the thinking param entirely — Fable 5 "+
			"rejects an explicit {type:disabled} with 400")
	assert.NotContains(t, got, `"output_config"`)
}

func TestAdaptiveThinkingModels_TemperatureDropped(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	for _, model := range []string{"claude-fable-5", "claude-opus-4-7", "claude-opus-4-8"} {
		t.Run(model, func(t *testing.T) {
			got := wireJSON(t, p, llm.ChatRequest{
				Model:       model,
				Temperature: 0.7,
				Reasoning:   llm.ReasoningMedium,
				Messages:    []llm.Message{{Role: "user", Content: "hi"}},
			})
			assert.NotContains(t, got, `"temperature"`,
				"sampling params are removed on these models — sending temperature is a 400")
		})
	}
}

func TestLegacyThinkingModels_KeepBudgetTokens(t *testing.T) {
	p := NewAnthropicProvider("test-key", "")
	got := wireJSON(t, p, llm.ChatRequest{
		Model:     "claude-sonnet-4-5",
		Reasoning: llm.ReasoningHigh,
		Messages:  []llm.Message{{Role: "user", Content: "hi"}},
	})
	assert.Contains(t, got, `"budget_tokens":16384`,
		"pre-4.7 models keep the enabled+budget_tokens config")
	assert.NotContains(t, got, `"output_config"`)
}

// TestAnthropicStream_RefusalStopReasonSurfaced reproduces the Claude
// Fable 5 classifier refusal: HTTP 200, a message_delta carrying
// stop_reason=refusal + stop_details.category, and NO content blocks.
// The parser previously dropped stop_reason entirely, so the runtime
// saw an empty-but-successful turn and the user saw a silent nothing.
func TestAnthropicStream_RefusalStopReasonSurfaced(t *testing.T) {
	const sseBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_r","type":"message","role":"assistant","content":[],"model":"claude-fable-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":29,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"bio","explanation":null}},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody))
	}))
	t.Cleanup(srv.Close)

	p := NewAnthropicProvider("test-key", srv.URL)
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "claude-fable-5"})
	require.NoError(t, err)

	var done *llm.ChatEvent
	for ev := range stream {
		if ev.Type == llm.EventDone {
			ev := ev
			done = &ev
		}
	}
	require.NotNil(t, done, "stream must emit EventDone")
	assert.Equal(t, "refusal", done.StopReason)
	assert.Equal(t, "bio", done.StopCategory)
}

// Normal end_turn streams must carry the stop reason too (and no category).
func TestAnthropicStream_EndTurnStopReason(t *testing.T) {
	const sseBody = `event: message_start
data: {"type":"message_start","message":{"id":"msg_n","type":"message","role":"assistant","content":[],"model":"claude-fable-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseBody))
	}))
	t.Cleanup(srv.Close)

	p := NewAnthropicProvider("test-key", srv.URL)
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{Model: "claude-fable-5"})
	require.NoError(t, err)

	var done *llm.ChatEvent
	for ev := range stream {
		if ev.Type == llm.EventDone {
			ev := ev
			done = &ev
		}
	}
	require.NotNil(t, done)
	assert.Equal(t, "end_turn", done.StopReason)
	assert.Empty(t, done.StopCategory)
}

func TestAnthropicInputAccumulation(t *testing.T) {
	var b strings.Builder
	for _, part := range []string{`{"a"`, `:1`, `}`} {
		b.WriteString(part)
	}
	require.Equal(t, `{"a":1}`, b.String())
}
