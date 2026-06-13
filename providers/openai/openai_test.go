package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

// captureOpenAIRequest points an OpenAI provider at an httptest server,
// drives one ChatStream call, and returns the captured request body.
// Surfaces any unmarshal error after the stream drains so a silent SDK
// shape change shows up as a clean test failure rather than a nil-pointer
// panic at the call site.
func captureOpenAIRequest(t *testing.T, req llm.ChatRequest) *openai.ChatCompletionRequest {
	t.Helper()
	var (
		captured     openai.ChatCompletionRequest
		unmarshalErr error
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		unmarshalErr = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	p := NewOpenAIProviderWithKind("test-key", srv.URL+"/v1", "openai-compatible")
	stream, err := p.ChatStream(context.Background(), req)
	require.NoError(t, err)
	for range stream {
	}
	require.NoError(t, unmarshalErr, "captured request body did not parse as openai.ChatCompletionRequest")
	return &captured
}

func TestOpenAIChatStreamUsesSystemPromptParts(t *testing.T) {
	captured := captureOpenAIRequest(t, llm.ChatRequest{
		SystemPromptParts: []llm.SystemPromptPart{
			{Text: "static"},
			{Text: "dynamic"},
		},
	})
	require.NotNil(t, captured)
	require.GreaterOrEqual(t, len(captured.Messages), 1)
	require.Equal(t, "system", string(captured.Messages[0].Role))
	require.Equal(t, "static\ndynamic", captured.Messages[0].Content)
}

func TestOpenAIChatStreamFallsBackToSystemPromptString(t *testing.T) {
	captured := captureOpenAIRequest(t, llm.ChatRequest{
		SystemPrompt: "legacy",
	})
	require.NotNil(t, captured)
	require.GreaterOrEqual(t, len(captured.Messages), 1)
	require.Equal(t, "system", string(captured.Messages[0].Role))
	require.Equal(t, "legacy", captured.Messages[0].Content)
}

func TestOpenAIChatStreamPartsBeatString(t *testing.T) {
	captured := captureOpenAIRequest(t, llm.ChatRequest{
		SystemPrompt:      "legacy",
		SystemPromptParts: []llm.SystemPromptPart{{Text: "new"}},
	})
	require.NotNil(t, captured)
	require.GreaterOrEqual(t, len(captured.Messages), 1)
	require.Equal(t, "new", captured.Messages[0].Content)
}

func TestEmitToolCalls_OrderedByIndex(t *testing.T) {
	mk := func(id, name string) *pendingTC {
		p := &pendingTC{id: id, name: name}
		p.args.WriteString("{}")
		return p
	}
	toolCalls := map[int]*pendingTC{2: mk("c", "third"), 0: mk("a", "first"), 1: mk("b", "second")}
	ch := make(chan llm.ChatEvent, 10)
	emitToolCalls(ch, toolCalls)
	close(ch)
	var ids []string
	for ev := range ch {
		if ev.Type == llm.EventToolCallDone {
			ids = append(ids, ev.ToolCall.ID)
		}
	}
	require.Equal(t, []string{"a", "b", "c"}, ids, "tool calls must emit in index order")
}
