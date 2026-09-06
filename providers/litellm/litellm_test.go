package litellm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

// TestNewLiteLLMProvider_HitsGivenBaseURL confirms the provider actually
// targets the baseURL it was constructed with, and that a streamed
// response comes back through it — not re-testing OpenAI wire-format
// translation itself (already covered by providers/openai's own tests),
// just that this thin constructor wires the base URL through correctly.
func TestNewLiteLLMProvider_HitsGivenBaseURL(t *testing.T) {
	var hit bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	p := NewLiteLLMProvider("test-key", srv.URL+"/v1")
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{SystemPrompt: "hi"})
	require.NoError(t, err)
	for range stream {
	}
	require.True(t, hit, "provider did not hit the configured baseURL")
}
