package web

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func newExaTestTool(t *testing.T, handler http.HandlerFunc) *ExaSearchTool {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return &ExaSearchTool{APIKey: "test-key", BaseURL: srv.URL, client: srv.Client()}
}

func TestExaSearch_RequestAndOutput(t *testing.T) {
	var gotKey string
	var gotBody map[string]any
	tl := newExaTestTool(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/search", r.URL.Path)
		gotKey = r.Header.Get("x-api-key")
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.Write([]byte(`{"results":[
			{"title":"HDB prices","url":"https://example.com/a","publishedDate":"2026-09-01T00:00:00.000Z","highlights":["Resale prices rose\n 2%.","Second  point."]},
			{"title":"No date","url":"https://example.com/b"}
		]}`))
	})

	res, err := tl.Execute(context.Background(), json.RawMessage(`{
		"query":"HDB resale prices","num_results":50,"category":"news",
		"include_domains":["straitstimes.com"],"start_published_date":"2026-01-01"}`))
	require.NoError(t, err)
	require.Empty(t, res.Error)

	require.Equal(t, "test-key", gotKey)
	require.Equal(t, "HDB resale prices", gotBody["query"])
	require.Equal(t, "auto", gotBody["type"])
	require.EqualValues(t, 10, gotBody["numResults"], "num_results is capped at 10")
	require.Equal(t, "news", gotBody["category"])
	require.Equal(t, []any{"straitstimes.com"}, gotBody["includeDomains"])
	require.Equal(t, "2026-01-01", gotBody["startPublishedDate"])
	require.NotContains(t, gotBody, "excludeDomains")
	require.Equal(t, map[string]any{"highlights": true}, gotBody["contents"])

	require.Contains(t, res.Output, "1. **HDB prices**\n   https://example.com/a\n   Published: 2026-09-01\n   Resale prices rose 2%. … Second point.")
	require.Contains(t, res.Output, "2. **No date**\n   https://example.com/b\n\n")
	require.Equal(t, 2, res.Metadata["num_results"])
}

func TestExaSearch_HTTPError(t *testing.T) {
	tl := newExaTestTool(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"invalid key"}`, http.StatusUnauthorized)
	})
	res, err := tl.Execute(context.Background(), json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)
	require.Contains(t, res.Error, "HTTP 401")
	require.Contains(t, res.Error, "invalid key")
}

func TestExaSearch_NoResults(t *testing.T) {
	tl := newExaTestTool(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"results":[]}`))
	})
	res, err := tl.Execute(context.Background(), json.RawMessage(`{"query":"nothing"}`))
	require.NoError(t, err)
	require.Equal(t, "No results found for: nothing", res.Output)
}

func TestExaSearch_InvalidInput(t *testing.T) {
	called := false
	tl := newExaTestTool(t, func(w http.ResponseWriter, r *http.Request) { called = true })
	for _, in := range []string{
		`{"query":"  "}`,
		`{"query":"x","type":"deep-reasoning"}`,
		`{"query":"x","category":"recipes"}`,
		`not json`,
	} {
		res, err := tl.Execute(context.Background(), json.RawMessage(in))
		require.NoError(t, err)
		require.NotEmpty(t, res.Error, in)
	}
	require.False(t, called, "invalid input must not reach the API")

	noKey := &ExaSearchTool{}
	res, _ := noKey.Execute(context.Background(), json.RawMessage(`{"query":"x"}`))
	require.Contains(t, res.Error, "no Exa API key")
}

func TestExaSearch_DefaultClientIsSafe(t *testing.T) {
	// The zero-value tool must use the SSRF-safe client, which refuses
	// loopback addresses.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()
	tl := &ExaSearchTool{APIKey: "k", BaseURL: srv.URL}
	res, err := tl.Execute(context.Background(), json.RawMessage(`{"query":"x"}`))
	require.NoError(t, err)
	require.NotEmpty(t, res.Error)
}

func TestExaHighlightText_Truncates(t *testing.T) {
	text := exaHighlightText([]string{strings.Repeat("a", exaMaxResultChars+100)})
	require.Equal(t, exaMaxResultChars+1, len([]rune(text)))
	require.True(t, strings.HasSuffix(text, "…"))
}
