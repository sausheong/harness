package runtime

import (
	"os"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
)

// streamingToolsEnabled reports whether streaming tool kickoff is on for
// this Runtime. Precedence:
//  1. Runtime.AgentLoop.StreamingTools == true → on (config wins).
//  2. Otherwise HARNESS_STREAMING_TOOLS=="1" → on.
//  3. Otherwise off.
//
// Strict "1" for the env fallback (rather than truthy parsing) keeps the
// env contract simple and unambiguous.
// Note: an explicit `false` in JSON5 behaves the same as the field being
// absent — to disable, leave both unset.
func (r *Runtime) streamingToolsEnabled() bool {
	if r.AgentLoop.StreamingTools {
		return true
	}
	return os.Getenv("HARNESS_STREAMING_TOOLS") == "1"
}

// kickoffResult is the channel payload sent by a streaming-kickoff goroutine
// once executeToolKickoff returns. The goroutine has already emitted the
// live EventToolResult (via r.emitToolResult); the main goroutine consumes
// this struct post-stream to write the paired session entries (ToolCall +
// ToolResult) IN STREAM ORDER, AFTER the assistant text is saved.
//
// Deferring session writes to the main goroutine preserves the API
// invariant that every tool_result message follows an assistant message
// containing the matching tool_use. Phase D's parallelism win (tools
// running during the stream) is preserved because Execute still happens
// inside the kickoff goroutine.
type kickoffResult struct {
	tc      llm.ToolCall // the tool call we executed (carried so the main goroutine doesn't have to look it up)
	result  tool.ToolResult
	aborted bool
}

// drainKickoffs blocks until every kickoff channel has received a value, then
// returns. Used on early-return paths (LLM error) so kickoff goroutines
// fully settle before Run() returns and r.events closes — preventing leaks.
// (The abort path inlines its own walk so it can append paired session
// entries in stream order.)
func drainKickoffs(kickoffs map[string]chan kickoffResult) {
	for _, ch := range kickoffs {
		<-ch
	}
}
