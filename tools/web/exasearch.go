package web

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/sausheong/harness/tool"
)

// exaBaseURL is the Exa API origin. Docs: https://exa.ai/docs/search/quickstart
const exaBaseURL = "https://api.exa.ai"

// exaMaxResultChars bounds the highlight text kept per result so a search
// result list stays small in the model's context.
const exaMaxResultChars = 1500

// exaTypes are the search types exa_search accepts. The deep and
// deep-reasoning modes (4-40s) are left out: agents that need that depth
// can iterate with fetches instead of blocking on one call.
var exaTypes = map[string]bool{"auto": true, "fast": true, "instant": true, "deep-lite": true}

// exaCategories are Exa's known data categories.
var exaCategories = map[string]bool{
	"company": true, "publication": true, "news": true,
	"personal site": true, "financial report": true, "people": true,
}

// ExaSearchTool searches the web with the Exa Search API, which returns
// ranked results with query-relevant highlights extracted from each page.
// It requires an Exa API key (https://dashboard.exa.ai/api-keys).
type ExaSearchTool struct {
	// APIKey authenticates requests (sent as the x-api-key header).
	APIKey string
	// BaseURL overrides the API origin; empty means https://api.exa.ai.
	BaseURL string

	// client overrides the HTTP client (tests); nil means the shared
	// SSRF-safe searchClient.
	client *http.Client
}

type exaSearchInput struct {
	Query              string   `json:"query"`
	NumResults         int      `json:"num_results,omitempty"`
	Type               string   `json:"type,omitempty"`
	Category           string   `json:"category,omitempty"`
	IncludeDomains     []string `json:"include_domains,omitempty"`
	ExcludeDomains     []string `json:"exclude_domains,omitempty"`
	StartPublishedDate string   `json:"start_published_date,omitempty"`
}

func (t *ExaSearchTool) Name() string { return "exa_search" }

func (t *ExaSearchTool) Description() string {
	return "Search the web with Exa, a search engine built for agents. Write the query in natural language, describing what you want to find. Returns ranked results with titles, URLs, publication dates, and the passages of each page most relevant to the query. Optionally filter by category, domains, or publication date."
}

func (t *ExaSearchTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "A natural-language description of what to find, e.g. \"recent news on Singapore's HDB resale prices\""
			},
			"num_results": {
				"type": "integer",
				"description": "Maximum number of results to return (default: 5, max: 10)"
			},
			"type": {
				"type": "string",
				"enum": ["auto", "fast", "instant", "deep-lite"],
				"description": "Search mode: auto (default, ~1s), fast (~0.5s), instant (~0.25s), or deep-lite (~4s, light multi-step research)"
			},
			"category": {
				"type": "string",
				"enum": ["company", "publication", "news", "personal site", "financial report", "people"],
				"description": "Focus on one kind of source. company and people do not support exclude_domains or start_published_date."
			},
			"include_domains": {
				"type": "array",
				"items": {"type": "string"},
				"description": "Only return results from these domains or path prefixes, e.g. [\"straitstimes.com\", \"*.gov.sg\"]"
			},
			"exclude_domains": {
				"type": "array",
				"items": {"type": "string"},
				"description": "Never return results from these domains or path prefixes"
			},
			"start_published_date": {
				"type": "string",
				"description": "Only return pages published on or after this date (ISO 8601, e.g. 2026-01-01)"
			}
		},
		"required": ["query"]
	}`)
}

// IsConcurrencySafe returns true — exa_search is a pure read.
func (t *ExaSearchTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }

func (t *ExaSearchTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in exaSearchInput
	if err := json.Unmarshal(input, &in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
	}
	in.Query = strings.TrimSpace(in.Query)
	if in.Query == "" {
		return tool.ToolResult{Error: "query is required"}, nil
	}
	if t.APIKey == "" {
		return tool.ToolResult{Error: "exa_search: no Exa API key configured"}, nil
	}
	if in.Type == "" {
		in.Type = "auto"
	}
	if !exaTypes[in.Type] {
		return tool.ToolResult{Error: fmt.Sprintf("invalid type %q (want auto, fast, instant or deep-lite)", in.Type)}, nil
	}
	if in.Category != "" && !exaCategories[in.Category] {
		return tool.ToolResult{Error: fmt.Sprintf("invalid category %q", in.Category)}, nil
	}
	n := in.NumResults
	if n <= 0 {
		n = 5
	}
	if n > 10 {
		n = 10
	}

	body := map[string]any{
		"query":      in.Query,
		"type":       in.Type,
		"numResults": n,
		"contents":   map[string]any{"highlights": true},
	}
	if in.Category != "" {
		body["category"] = in.Category
	}
	if len(in.IncludeDomains) > 0 {
		body["includeDomains"] = in.IncludeDomains
	}
	if len(in.ExcludeDomains) > 0 {
		body["excludeDomains"] = in.ExcludeDomains
	}
	if in.StartPublishedDate != "" {
		body["startPublishedDate"] = in.StartPublishedDate
	}

	ctx, cancel := context.WithTimeout(ctx, searchTimeout)
	defer cancel()

	results, err := t.search(ctx, body)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("exa search failed: %v", err)}, nil
	}
	if len(results) == 0 {
		return tool.ToolResult{Output: "No results found for: " + in.Query}, nil
	}

	var sb strings.Builder
	for i, r := range results {
		fmt.Fprintf(&sb, "%d. **%s**\n   %s\n", i+1, r.Title, r.URL)
		if r.PublishedDate != "" {
			date, _, _ := strings.Cut(r.PublishedDate, "T")
			fmt.Fprintf(&sb, "   Published: %s\n", date)
		}
		if text := exaHighlightText(r.Highlights); text != "" {
			fmt.Fprintf(&sb, "   %s\n", text)
		}
		sb.WriteString("\n")
	}
	return tool.ToolResult{
		Output: sb.String(),
		Metadata: map[string]any{
			"query":       in.Query,
			"num_results": len(results),
		},
	}, nil
}

type exaResult struct {
	Title         string   `json:"title"`
	URL           string   `json:"url"`
	PublishedDate string   `json:"publishedDate"`
	Highlights    []string `json:"highlights"`
}

func (t *ExaSearchTool) search(ctx context.Context, body map[string]any) ([]exaResult, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	base := t.BaseURL
	if base == "" {
		base = exaBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(base, "/")+"/search", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", t.APIKey)
	client := t.client
	if client == nil {
		client = searchClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("exa returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Results []exaResult `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("exa decode: %w", err)
	}
	return parsed.Results, nil
}

// exaHighlightText joins a result's highlights into one line, collapsing
// whitespace and truncating to exaMaxResultChars.
func exaHighlightText(highlights []string) string {
	text := strings.Join(strings.Fields(strings.Join(highlights, " … ")), " ")
	if r := []rune(text); len(r) > exaMaxResultChars {
		text = string(r[:exaMaxResultChars]) + "…"
	}
	return text
}
