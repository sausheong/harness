package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
	"sync"
)

// normalizedEntry is one cached normalization result.
type normalizedEntry struct {
	Tools []ToolDef
	Diags []Diagnostic
}

// stripCache memoizes ApplyStripList by a hash of (fields, tools). The
// agent runtime calls NormalizeToolSchema once per turn with a
// deterministic tool list (alphabetically sorted by Registry.ToolDefs),
// which means turn 1 computes the strip and turns 2..N hit the cache.
//
// Cache size is bounded by the cardinality of distinct tool sets across
// the process. In practice this is small: one entry per (provider strip
// list, agent tool allowlist). A 256-entry hard cap with random
// eviction protects against pathological churn.
var (
	stripCacheMu sync.RWMutex
	stripCache   = map[string]normalizedEntry{}
)

const stripCacheMax = 256

// ApplyStripList runs StripFields over each tool's Parameters with the
// given field-strip list, preserving input order and accumulating
// diagnostics across all tools. The per-provider NormalizeToolSchema
// methods (OpenAI, Qwen, Gemini) are thin wrappers over this helper —
// only the strip list differs. Anthropic doesn't use this (identity).
//
// Memoized: identical inputs return cached output without re-walking
// the JSON Schema. Cache key is a SHA-256 over (sorted fields, tool
// names+parameters). Returned slices are copies so callers can safely
// mutate them.
func ApplyStripList(tools []ToolDef, fields []string) ([]ToolDef, []Diagnostic) {
	if len(tools) == 0 {
		return tools, nil
	}
	key := stripCacheKey(tools, fields)

	stripCacheMu.RLock()
	if hit, ok := stripCache[key]; ok {
		stripCacheMu.RUnlock()
		return cloneToolDefs(hit.Tools), cloneDiags(hit.Diags)
	}
	stripCacheMu.RUnlock()

	out := make([]ToolDef, len(tools))
	var allDiags []Diagnostic
	for i, t := range tools {
		newParams, diags := StripFields(t.Name, t.Parameters, fields)
		td := t
		td.Parameters = newParams
		out[i] = td
		allDiags = append(allDiags, diags...)
	}

	stripCacheMu.Lock()
	if len(stripCache) >= stripCacheMax {
		// Drop one arbitrary entry to keep the map bounded. Map iteration
		// order is randomized, so this is effectively random eviction.
		for k := range stripCache {
			delete(stripCache, k)
			break
		}
	}
	stripCache[key] = normalizedEntry{Tools: cloneToolDefs(out), Diags: cloneDiags(allDiags)}
	stripCacheMu.Unlock()

	return out, allDiags
}

// stripCacheKey hashes the inputs into a stable string. Sorts the fields
// list because the per-provider strip lists are stable but defensively
// supports unsorted input.
func stripCacheKey(tools []ToolDef, fields []string) string {
	sortedFields := make([]string, len(fields))
	copy(sortedFields, fields)
	sort.Strings(sortedFields)
	h := sha256.New()
	h.Write([]byte(strings.Join(sortedFields, "\x00")))
	h.Write([]byte("\x01"))
	for _, t := range tools {
		h.Write([]byte(t.Name))
		h.Write([]byte("\x02"))
		h.Write(t.Parameters)
		if t.CacheControl != nil {
			h.Write([]byte(t.CacheControl.Type))
			h.Write([]byte("\x04"))
			h.Write([]byte(t.CacheControl.TTL))
		}
		h.Write([]byte("\x03"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func cloneToolDefs(in []ToolDef) []ToolDef {
	if in == nil {
		return nil
	}
	out := make([]ToolDef, len(in))
	copy(out, in)
	for i := range out {
		if in[i].CacheControl != nil {
			cache := *in[i].CacheControl
			out[i].CacheControl = &cache
		}
	}
	return out
}

func cloneDiags(in []Diagnostic) []Diagnostic {
	if in == nil {
		return nil
	}
	out := make([]Diagnostic, len(in))
	copy(out, in)
	return out
}

// ResetStripCache is exported for tests so they can verify cache
// behavior without process restart. Production code should not call
// this — the cache is bounded by stripCacheMax and otherwise lives for
// the process lifetime.
func ResetStripCache() {
	stripCacheMu.Lock()
	stripCache = map[string]normalizedEntry{}
	stripCacheMu.Unlock()
}

// StripFields removes the given field names from a JSON Schema document
// recursively. It descends into "properties.*", "items", and
// "additionalProperties" (the standard JSON Schema schema-bearing
// positions). Returns the rewritten schema as JSON bytes and one
// Diagnostic per stripped occurrence (ToolName set, Field set to the
// dotted JSON path).
//
// Determinism: the walker visits map keys in sorted order so output
// and diagnostics are reproducible across calls. Required for prompt
// cache stability — the agent runtime calls NormalizeToolSchema once
// per turn, and any non-determinism here would invalidate the cache.
//
// Malformed schemas are returned unchanged with no diagnostics; the
// provider SDK will produce its own error.
func StripFields(toolName string, schema json.RawMessage, fields []string) (json.RawMessage, []Diagnostic) {
	if len(fields) == 0 || len(schema) == 0 {
		return schema, nil
	}
	stripSet := make(map[string]bool, len(fields))
	for _, f := range fields {
		stripSet[f] = true
	}
	var doc any
	if err := json.Unmarshal(schema, &doc); err != nil {
		return schema, nil
	}
	var diags []Diagnostic
	stripped := walkStrip(doc, "", toolName, stripSet, &diags)
	out, err := json.Marshal(stripped)
	if err != nil {
		return schema, nil
	}
	return out, diags
}

// walkStrip is the recursive worker. Returns the (possibly rewritten)
// node and appends any diagnostics for stripped fields.
func walkStrip(node any, path string, toolName string, stripSet map[string]bool, diags *[]Diagnostic) any {
	m, ok := node.(map[string]any)
	if !ok {
		return node
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fieldPath := joinPath(path, k)
		if stripSet[k] {
			*diags = append(*diags, Diagnostic{
				ToolName: toolName,
				Field:    fieldPath,
				Action:   "stripped",
				Reason:   "field not supported by provider",
			})
			delete(m, k)
			continue
		}
		switch k {
		case "properties":
			if props, ok := m[k].(map[string]any); ok {
				propKeys := make([]string, 0, len(props))
				for pk := range props {
					propKeys = append(propKeys, pk)
				}
				sort.Strings(propKeys)
				for _, pk := range propKeys {
					props[pk] = walkStrip(props[pk], joinPath(fieldPath, pk), toolName, stripSet, diags)
				}
			}
		case "items", "additionalProperties":
			m[k] = walkStrip(m[k], fieldPath, toolName, stripSet, diags)
		}
	}
	return m
}

func joinPath(base, leaf string) string {
	if base == "" {
		return leaf
	}
	return base + "." + leaf
}
