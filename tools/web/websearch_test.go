package web

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestSearchClient_IsSafe(t *testing.T) {
	require.NotNil(t, searchClient)
	require.NotNil(t, searchClient.Transport)
	require.NotNil(t, searchClient.CheckRedirect)
	require.Equal(t, searchTimeout, searchClient.Timeout)
}

func TestSearxng_BlocksPrivateBaseURL(t *testing.T) {
	b := newSearxngBackend("http://127.0.0.1:8080")
	_, err := b.Search(testCtx(t), "hello", 3)
	require.Error(t, err)
}
