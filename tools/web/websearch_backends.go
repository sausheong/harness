package web

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// WebSearchBackend is the strategy interface used by WebSearchTool.
// Implementations turn (query, maxResults) into ranked results.
//
// The interface deliberately does not include configuration — backends
// are constructed with their config (API key, base URL, etc.) at
// registration time and remain immutable thereafter. This keeps the tool
// dispatch path zero-allocation in the hot loop.
//
// Implementations that make outbound HTTP calls MUST use the package
// searchClient (SafeHTTPClient) so a configured/typo'd internal base URL
// cannot be used for SSRF. (N4)
type WebSearchBackend interface {
	Name() string
	Search(ctx context.Context, query string, maxResults int) ([]searchResult, error)
}

// ddgBackend is the default backend — scrapes html.duckduckgo.com.
// Fragile (DDG can change their HTML), no API key required, the original
// pre-S1 behavior. Kept as the no-config default so a fresh install with
// no env vars still has working search.
type ddgBackend struct{}

func newDDGBackend() WebSearchBackend                      { return &ddgBackend{} }
func (b *ddgBackend) Name() string                         { return "duckduckgo" }
func (b *ddgBackend) Search(ctx context.Context, q string, n int) ([]searchResult, error) {
	return duckDuckGoSearch(ctx, q, n)
}

// braveBackend uses the Brave Search REST API.
// Docs: https://api.search.brave.com/app/documentation/web-search/get-started
// Auth: header X-Subscription-Token: <key>
type braveBackend struct {
	apiKey string
}

func newBraveBackend(apiKey string) WebSearchBackend { return &braveBackend{apiKey: apiKey} }
func (b *braveBackend) Name() string                 { return "brave" }
func (b *braveBackend) Search(ctx context.Context, q string, n int) ([]searchResult, error) {
	if b.apiKey == "" {
		return nil, fmt.Errorf("brave backend: no API key configured")
	}
	if n > 20 {
		n = 20
	}
	u := "https://api.search.brave.com/res/v1/web/search?q=" + url.QueryEscape(q) +
		fmt.Sprintf("&count=%d", n)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", b.apiKey)
	resp, err := searchClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("brave returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("brave decode: %w", err)
	}
	out := make([]searchResult, 0, len(parsed.Web.Results))
	for _, r := range parsed.Web.Results {
		out = append(out, searchResult{
			Title:   cleanHTMLText(r.Title),
			URL:     r.URL,
			Snippet: cleanHTMLText(r.Description),
		})
	}
	if n > 0 && len(out) > n {
		out = out[:n]
	}
	return out, nil
}

// tavilyBackend uses the Tavily Search API.
// Docs: https://docs.tavily.com/docs/rest-api/api-reference
// Auth: JSON body field "api_key"; POST to /search.
type tavilyBackend struct {
	apiKey string
}

func newTavilyBackend(apiKey string) WebSearchBackend { return &tavilyBackend{apiKey: apiKey} }
func (b *tavilyBackend) Name() string                 { return "tavily" }
func (b *tavilyBackend) Search(ctx context.Context, q string, n int) ([]searchResult, error) {
	if b.apiKey == "" {
		return nil, fmt.Errorf("tavily backend: no API key configured")
	}
	if n > 20 {
		n = 20
	}
	body, _ := json.Marshal(map[string]any{
		"api_key":     b.apiKey,
		"query":       q,
		"max_results": n,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.tavily.com/search", strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := searchClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("tavily returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("tavily decode: %w", err)
	}
	out := make([]searchResult, 0, len(parsed.Results))
	for _, r := range parsed.Results {
		out = append(out, searchResult{
			Title:   r.Title,
			URL:     r.URL,
			Snippet: r.Content,
		})
	}
	return out, nil
}

// searxngBackend talks to a self-hosted SearXNG instance — the
// self-sufficiency-aligned option. JSON output must be enabled on the
// SearXNG server (`enabled_engines` and `formats: [json]` in settings.yml).
type searxngBackend struct {
	baseURL string // e.g. "http://localhost:8080"
}

func newSearxngBackend(baseURL string) WebSearchBackend {
	return &searxngBackend{baseURL: strings.TrimRight(baseURL, "/")}
}
func (b *searxngBackend) Name() string { return "searxng" }
func (b *searxngBackend) Search(ctx context.Context, q string, n int) ([]searchResult, error) {
	if b.baseURL == "" {
		return nil, fmt.Errorf("searxng backend: no base URL configured")
	}
	u := b.baseURL + "/search?format=json&q=" + url.QueryEscape(q)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := searchClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("searxng returned HTTP %d", resp.StatusCode)
	}
	var parsed struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("searxng decode: %w", err)
	}
	out := make([]searchResult, 0, len(parsed.Results))
	for _, r := range parsed.Results {
		out = append(out, searchResult{
			Title:   r.Title,
			URL:     r.URL,
			Snippet: r.Content,
		})
	}
	if n > 0 && len(out) > n {
		out = out[:n]
	}
	return out, nil
}

// WebSearchConfig selects the backend at registration time. Empty Backend
// or Backend == "duckduckgo" yields the default DDG scraper (no API key).
// "brave" + APIKey, "tavily" + APIKey, and "searxng" + BaseURL each
// activate their respective backends.
type WebSearchConfig struct {
	Backend string // "duckduckgo" (default) | "brave" | "tavily" | "searxng"
	APIKey  string // for brave / tavily
	BaseURL string // for searxng
}

// NewWebSearchBackend resolves a WebSearchConfig into a concrete backend.
// Returns the DDG fallback when the config is empty or names an unknown
// backend; never returns nil. Logging the fallback is the caller's
// responsibility (registerWebSearch in the gateway wires that).
func NewWebSearchBackend(cfg WebSearchConfig) WebSearchBackend {
	switch strings.ToLower(strings.TrimSpace(cfg.Backend)) {
	case "", "duckduckgo", "ddg":
		return newDDGBackend()
	case "brave":
		return newBraveBackend(cfg.APIKey)
	case "tavily":
		return newTavilyBackend(cfg.APIKey)
	case "searxng":
		return newSearxngBackend(cfg.BaseURL)
	default:
		return newDDGBackend()
	}
}
