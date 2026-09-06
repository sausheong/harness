package openrouter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/sausheong/harness/llm"
)

func TestDefaultBaseURL(t *testing.T) {
	require.Equal(t, "https://openrouter.ai/api/v1", DefaultBaseURL)
}

// TestNewOpenRouterProvider_HitsGivenBaseURL confirms a custom baseURL
// overrides the default and is actually what the provider talks to.
func TestNewOpenRouterProvider_HitsGivenBaseURL(t *testing.T) {
	var hit bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	p := NewOpenRouterProvider("test-key", srv.URL+"/v1")
	stream, err := p.ChatStream(context.Background(), llm.ChatRequest{SystemPrompt: "hi"})
	require.NoError(t, err)
	for range stream {
	}
	require.True(t, hit, "provider did not hit the configured baseURL")
}

// TestNewOpenRouterProvider_EmptyBaseURLDoesNotPanic confirms the empty
// -> DefaultBaseURL fallback path constructs a usable provider (the
// actual outbound host isn't inspectable from outside the openai
// package — OpenAIProvider's client config is unexported — so this
// checks the fallback doesn't panic or return nil, and DefaultBaseURL's
// own value is asserted separately above).
func TestNewOpenRouterProvider_EmptyBaseURLDoesNotPanic(t *testing.T) {
	p := NewOpenRouterProvider("test-key", "")
	require.NotNil(t, p)
}
