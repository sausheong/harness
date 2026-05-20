package runtime

import (
	"os"
	"strconv"
)

// maxToolResultLen returns the per-result truncation cap (chars) for tool
// results in the in-context message history. Precedence:
//  1. Runtime.AgentLoop.MaxToolResultLen (>0) — config wins.
//  2. HARNESS_MAX_TOOL_RESULT_LEN env var (>0) — env fallback.
//  3. defaultMaxToolResultLen (4000).
//
// Tool results longer than this are truncated (or spilled to disk via
// spillConfig when supplied). 4000 is conservative; engineering-style
// agents that read source files and run tests typically want 16000-25000.
func (r *Runtime) maxToolResultLen() int {
	if r.AgentLoop.MaxToolResultLen > 0 {
		return r.AgentLoop.MaxToolResultLen
	}
	if v := os.Getenv("HARNESS_MAX_TOOL_RESULT_LEN"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultMaxToolResultLen
}
