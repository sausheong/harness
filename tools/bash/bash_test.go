package bash

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sausheong/harness/tool"
)

// runAllowlist executes cmd under an allowlist policy and returns the
// ToolResult.Error (empty string = the command was permitted by policy).
func runAllowlist(t *testing.T, allow []string, command string) string {
	t.Helper()
	bt := &BashTool{ExecPolicy: &ExecPolicy{Level: "allowlist", Allowlist: allow}}
	in, err := json.Marshal(bashInput{Command: command})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	res, err := bt.Execute(context.Background(), in)
	if err != nil {
		t.Fatalf("Execute returned a Go error (should use ToolResult.Error): %v", err)
	}
	return res.Error
}

func TestExtractCommands_BackgroundAmpersand(t *testing.T) {
	// "ls & curl ..." must validate BOTH sides; curl is not allowed →
	// rejected by policy BEFORE any command executes. Asserting the policy
	// rejection message (not just a non-empty Error) is essential: without
	// the fix, curl actually runs and its own runtime stderr lands in
	// res.Error, which would mask the bypass.
	got := runAllowlist(t, []string{"ls"}, "ls & curl http://evil")
	if !strings.Contains(got, "not in the exec allowlist") {
		t.Fatalf("background-& bypass: expected allowlist rejection of curl, got %q", got)
	}
	// "ls && echo hi" must still split on && (both allowed) → permitted.
	if got := runAllowlist(t, []string{"ls", "echo"}, "ls && echo hi"); got != "" {
		t.Fatalf("&& chain wrongly rejected: %q", got)
	}
}

func TestSanitizeLLMText(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"ascii passthrough", "ls -la /tmp", "ls -la /tmp"},
		{"nbsp between words", "open /Users/me/SGQR\u00a0Specs.pdf", "open /Users/me/SGQR Specs.pdf"},
		{"narrow nbsp", "echo a\u202fb", "echo a b"},
		{"ideographic space", "echo a\u3000b", "echo a b"},
		{"en space", "echo a\u2002b", "echo a b"},
		{"zero-width joiner stripped", "echo foo\u200dbar", "echo foobar"},
		{"bom stripped", "\ufeffls", "ls"},
		{"line separator to newline", "ls\u2028pwd", "ls\npwd"},
		{"preserves real tab and newline", "ls\t-l\npwd", "ls\t-l\npwd"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tool.SanitizeLLMText(tt.in); got != tt.want {
				t.Errorf("tool.SanitizeLLMText(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestResolveBashCommandPaths(t *testing.T) {
	dir := t.TempDir()

	// File on disk with NBSP in its name.
	nbspPath := filepath.Join(dir, "SGQR\u00a0Specifications.pdf")
	if err := os.WriteFile(nbspPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	asciiVariant := filepath.Join(dir, "SGQR Specifications.pdf")

	// A file with a plain ASCII-space name; it must NOT be substituted
	// because its on-disk name has no Unicode whitespace.
	plainPath := filepath.Join(dir, "plain.txt")
	if err := os.WriteFile(plainPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name      string
		cmd       string
		wantCmd   string
		wantSubs  int
	}{
		{
			name:     "backslash-escaped path with ascii spaces resolves to nbsp file",
			cmd:      "pdftotext " + strings.ReplaceAll(asciiVariant, " ", `\ `) + " /tmp/out.txt",
			wantCmd:  "pdftotext " + shellSingleQuote(nbspPath) + " /tmp/out.txt",
			wantSubs: 1,
		},
		{
			name:     "double-quoted path resolves",
			cmd:      `pdftotext "` + asciiVariant + `" /tmp/out.txt`,
			wantCmd:  "pdftotext " + shellSingleQuote(nbspPath) + " /tmp/out.txt",
			wantSubs: 1,
		},
		{
			name:     "single-quoted path resolves",
			cmd:      `pdftotext '` + asciiVariant + `' /tmp/out.txt`,
			wantCmd:  "pdftotext " + shellSingleQuote(nbspPath) + " /tmp/out.txt",
			wantSubs: 1,
		},
		{
			name:     "create-style command on missing path is left alone",
			cmd:      "mkdir " + filepath.Join(dir, "newdir"),
			wantCmd:  "mkdir " + filepath.Join(dir, "newdir"),
			wantSubs: 0,
		},
		{
			name:     "existing path is left alone",
			cmd:      "cat " + plainPath,
			wantCmd:  "cat " + plainPath,
			wantSubs: 0,
		},
		{
			name:     "no absolute paths is a no-op",
			cmd:      "echo hello",
			wantCmd:  "echo hello",
			wantSubs: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, subs := resolveBashCommandPaths(tt.cmd)
			if got != tt.wantCmd {
				t.Errorf("cmd:\n  got:  %q\n  want: %q", got, tt.wantCmd)
			}
			if len(subs) != tt.wantSubs {
				t.Errorf("subs count: got %d, want %d (subs=%v)", len(subs), tt.wantSubs, subs)
			}
		})
	}
}

func TestResolveExistingPath(t *testing.T) {
	dir := t.TempDir()

	// File on disk has a real NBSP in its name.
	nbspPath := filepath.Join(dir, "SGQR\u00a0Specifications.pdf")
	if err := os.WriteFile(nbspPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	// File on disk has plain ASCII space.
	asciiPath := filepath.Join(dir, "plain space.txt")
	if err := os.WriteFile(asciiPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	asciiVariantOfNBSP := filepath.Join(dir, "SGQR Specifications.pdf")
	missing := filepath.Join(dir, "does-not-exist.txt")

	tests := []struct {
		name string
		in   string
		want string
	}{
		{"existing nbsp path returned unchanged", nbspPath, nbspPath},
		{"existing ascii path returned unchanged", asciiPath, asciiPath},
		{"nbsp emitted by LLM for ascii-space file resolves", filepath.Join(dir, "plain\u00a0space.txt"), asciiPath},
		{"ascii-space LLM input recovers real nbsp file via dir scan", asciiVariantOfNBSP, nbspPath},
		{"missing path returned unchanged", missing, missing},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tool.ResolveExistingPath(tt.in); got != tt.want {
				t.Errorf("tool.ResolveExistingPath(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
