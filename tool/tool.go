package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/sausheong/harness/llm"
)

// ExpandHome rewrites a leading "~" or "~/" in p to the user's home directory.
// Other tildes (mid-path, "~user/...") are left alone — the harness never
// escalates privileges, so cross-user tilde expansion would silently fail
// anyway.
//
// The shell ($BASH/zsh) does this expansion before exec, which is why the
// bash tool works without it. read/write/edit_file call os.Open directly so
// they need to handle "~/" themselves.
func ExpandHome(p string) string {
	if p == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
		return p
	}
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// ResolveExistingPath returns a path that actually exists on disk, recovering
// from Unicode-whitespace mismatches between what the LLM supplies and what
// the filesystem holds. Resolution order:
//  1. The path as given.
//  2. The Unicode-sanitized variant (NBSP / narrow-NBSP / ideographic /
//     en/em/figure-space → ASCII space; zero-width chars stripped).
//  3. The parent directory is scanned and an entry whose own sanitized name
//     equals the sanitized basename of the requested path is used — but only
//     if the match is unambiguous.
//
// If nothing resolves, p is returned so the caller's error message reflects
// what the LLM actually supplied. Never used for write paths — writing to a
// "resolved" path could create files in unintended locations.
func ResolveExistingPath(p string) string {
	if _, err := os.Stat(p); err == nil {
		return p
	}
	if alt := SanitizeLLMText(p); alt != p {
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	dir, base := filepath.Split(p)
	if dir == "" {
		dir = "."
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return p
	}
	target := SanitizeLLMText(base)
	var matches []string
	for _, e := range entries {
		if SanitizeLLMText(e.Name()) == target {
			matches = append(matches, e.Name())
		}
	}
	if len(matches) == 1 {
		return filepath.Join(dir, matches[0])
	}
	return p
}

// ResolveExistingPathStrict is like ResolveExistingPath but the dir-scan
// fallback only fires when the matched on-disk entry actually contains a
// non-ASCII whitespace character. Used by the bash tool, where freeform
// commands include both read-paths (which we want to recover) and create-
// paths like `mkdir /tmp/newdir` (which must NOT be silently substituted
// with a similarly-named pre-existing entry).
func ResolveExistingPathStrict(p string) string {
	if _, err := os.Stat(p); err == nil {
		return p
	}
	if alt := SanitizeLLMText(p); alt != p {
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	dir, base := filepath.Split(p)
	if dir == "" {
		dir = "."
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return p
	}
	target := SanitizeLLMText(base)
	var matches []string
	for _, e := range entries {
		if SanitizeLLMText(e.Name()) == target && HasUnicodeWhitespace(e.Name()) {
			matches = append(matches, e.Name())
		}
	}
	if len(matches) == 1 {
		return filepath.Join(dir, matches[0])
	}
	return p
}

// HasUnicodeWhitespace reports whether s contains any of the whitespace or
// invisible characters that SanitizeLLMText normalizes away.
func HasUnicodeWhitespace(s string) bool {
	for _, r := range s {
		switch r {
		case '\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF', '\u2028', '\u2029':
			return true
		}
		if r != ' ' && r != '\t' && r != '\n' && r != '\r' && unicode.IsSpace(r) {
			return true
		}
	}
	return false
}

// SanitizeLLMText normalizes Unicode whitespace lookalikes that small or
// quantized LLMs sometimes emit in place of ASCII space and newline, and
// strips zero-width characters that have no shell or filesystem meaning.
// Used as a fallback by ResolveExistingPath, not applied eagerly to LLM
// input (eager sanitization breaks the case where a file genuinely
// contains NBSP in its name).
func SanitizeLLMText(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch r {
		case '\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF':
			continue // zero-width / BOM: drop
		case '\u2028', '\u2029':
			b.WriteByte('\n') // line / paragraph separator → newline
		default:
			if r != ' ' && r != '\t' && r != '\n' && r != '\r' && unicode.IsSpace(r) {
				b.WriteByte(' ') // any other Unicode whitespace → ASCII space
				continue
			}
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Tool is the interface every harness tool must implement.
type Tool interface {
	Name() string
	Description() string
	Parameters() json.RawMessage // JSON Schema
	Execute(ctx context.Context, input json.RawMessage) (ToolResult, error)
	// IsConcurrencySafe reports whether this tool can be invoked in parallel
	// with other concurrency-safe tools in the same dispatch batch. Pure-read
	// tools should return true; anything that mutates state or has ordering
	// sensitivity should return false. The input parameter is plumbed for
	// future input-aware classifiers (e.g., bash subcommand-specific safety);
	// current impls may ignore it. Implementations MUST NOT panic on weird
	// input — the partitioner treats panics as unsafe via recover().
	IsConcurrencySafe(input json.RawMessage) bool
}

// ToolResult holds the output of a tool execution.
type ToolResult struct {
	Output   string             `json:"output"`
	Error    string             `json:"error,omitempty"`
	Metadata map[string]any     `json:"metadata,omitempty"`
	Images   []llm.ImageContent `json:"-"` // image attachments (not JSON-serialized)
}

// Executor is the interface used by agent runtime for tool operations.
// Implemented by Registry.
type Executor interface {
	Execute(ctx context.Context, name string, input json.RawMessage) (ToolResult, error)
	ToolDefs() []llm.ToolDef
	Names() []string
	Get(name string) (Tool, bool)
}

// Registry manages a collection of available tools.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry creates a new tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry.
func (r *Registry) Register(t Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[t.Name()] = t
}

// Get returns a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// Execute runs a tool by name with the given input.
func (r *Registry) Execute(ctx context.Context, name string, input json.RawMessage) (ToolResult, error) {
	t, ok := r.Get(name)
	if !ok {
		return ToolResult{}, fmt.Errorf("unknown tool: %q", name)
	}
	return t.Execute(ctx, input)
}

// ToolDefs returns the tool definitions for the LLM API.
// Output is sorted by tool name so the LLM-request prefix is stable across
// turns — required for prompt-cache hits and for compaction-summary
// reproducibility. Map iteration in Go is randomized; without sorting,
// every turn would invalidate the cache and a single tool reordering
// could change a generated summary's content.
func (r *Registry) ToolDefs() []llm.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	defs := make([]llm.ToolDef, 0, len(names))
	for _, name := range names {
		t := r.tools[name]
		defs = append(defs, llm.ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		})
	}
	return defs
}

// Names returns the names of all registered tools, sorted alphabetically.
// See ToolDefs() for why ordering matters.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// ValidatePathInWorkDir ensures that the resolved path is within the workspace.
// It resolves symlinks and normalizes the path to prevent traversal attacks.
// Exported for the concrete-tool packages (file, bash, etc.) that share
// this workspace-clamp invariant.
func ValidatePathInWorkDir(path, workDir string) error {
	if workDir == "" {
		return nil
	}

	absWork, err := filepath.Abs(workDir)
	if err != nil {
		return fmt.Errorf("invalid workspace: %w", err)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path: %w", err)
	}

	// Resolve symlinks to prevent symlink-based traversal
	realWork, err := filepath.EvalSymlinks(absWork)
	if err != nil {
		realWork = absWork // workspace might not exist yet
	}

	// For the target path, resolve the parent directory (file may not exist yet)
	parentDir := filepath.Dir(absPath)
	realParent, err := filepath.EvalSymlinks(parentDir)
	if err != nil {
		// Parent doesn't exist — use the unresolved absolute path
		realParent = parentDir
	}
	realPath := filepath.Join(realParent, filepath.Base(absPath))

	if !strings.HasPrefix(realPath, realWork+string(filepath.Separator)) && realPath != realWork {
		return fmt.Errorf("path %q is outside workspace %q", path, workDir)
	}

	return nil
}

// RegisterCron registers the cron tool with the given scheduler. agentID is
// baked into the tool so jobs the agent schedules later run as the same
// agent that scheduled them.
func RegisterCron(reg *Registry, agentID string, scheduler JobScheduler) {
	reg.Register(&CronTool{AgentID: agentID, Scheduler: scheduler})
}

