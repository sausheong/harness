package runtime

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

const maxToolResultLen = 4000 // truncate tool results longer than this

// detectImageMIME returns the actual MIME type based on magic bytes.
// Falls back to the provided hint if the format is unrecognized.
func detectImageMIME(data []byte, hint string) string {
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg"
	}
	if len(data) >= 4 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' {
		return "image/png"
	}
	if len(data) >= 4 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F' && data[3] == '8' {
		return "image/gif"
	}
	if len(data) >= 4 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' {
		return "image/webp"
	}
	return hint
}

const defaultIdentityBase = `You are an AI agent. Conduct yourself professionally and politely. Be concise and direct. When executing tasks, think step by step and use your tools to accomplish the user's goals. When you need to call multiple independent tools to gather information, emit them in a single response (parallel tool calls) rather than waiting for each one — this cuts response latency on local models.`

// toolHints maps tool names to usage guidance injected into the default identity.
var toolHints = map[string]string{
	"read_file":    "You can read files. You have vision capabilities — you can see and analyze images by using read_file on image files. Do not say you cannot see or analyze images.",
	"write_file":   "You can create or overwrite files.",
	"edit_file":    "You can make targeted edits to existing files.",
	"bash":         "You can execute bash commands on the user's machine. ALWAYS wrap file paths in double quotes when invoking bash (e.g. ls \"/path/with spaces/file.txt\") so paths with spaces or special characters survive shell tokenization.",
	"web_fetch":    "You can fetch web pages using the web_fetch tool.",
	"web_search":   "You can search the web using the web_search tool.",
	"browser":      "You can automate a headless browser for interactive pages using the browser tool.",
	"send_message": "You can send messages to other users or channels using the send_message tool.",
	"cron":         "You can schedule recurring tasks using the cron tool.",
	"todo_write":   "You have a todo_write tool, but use it sparingly. Start working on the user's request directly — do NOT pre-plan with a sequence of todo_write calls. Reserve todo_write for genuinely long, multi-stage work (roughly 5+ independent subtasks that will span many turns). When you do initialize a list, emit every `add` as a parallel tool call in a single assistant response — never one item per turn.",
	"load_skill":   "You can load a skill body on demand by name via load_skill. Consult the Skills Index in your system prompt to pick the right skill name; the body is returned as the tool output.",
	"load_memory":  "You can load a memory entry body on demand by id via load_memory. Consult the Memory Index in your system prompt to pick the right entry id.",
}

// buildDefaultIdentity constructs the default identity prompt tailored to
// the tools actually available to this agent.
func buildDefaultIdentity(toolNames []string) string {
	if len(toolNames) == 0 {
		return defaultIdentityBase
	}
	available := make(map[string]bool, len(toolNames))
	for _, name := range toolNames {
		available[name] = true
	}
	var hints []string
	for _, name := range toolNames {
		if h, ok := toolHints[name]; ok {
			hints = append(hints, h)
		}
	}
	if len(hints) == 0 {
		return defaultIdentityBase
	}
	return defaultIdentityBase + " " + strings.Join(hints, " ")
}

// BuildStaticSystemPrompt assembles the portion of the system prompt that
// does not change across turns within a Run: identity (from systemPrompt
// arg, IDENTITY.md in workspace, or the built-in default tailored to
// toolNames), agent self-identity, then the four caller-provided context
// blocks in order — configSummary, skillsIndex, memoryIndex, memoryFiles.
//
// Pure with one allowed exception: it reads IDENTITY.md from workspace
// when systemPrompt is empty. Caller pre-resolves configSummary,
// skillsIndex, memoryIndex, and memoryFiles so neither config loading
// nor skill/memory index assembly nor memory-file disk reads happen
// per-turn. Suitable to call once at Runtime construction.
//
// configSummary is the consumer's place to advertise things like the
// configured agents, channel bindings, configuration paths, or any
// other hot-reload-bound context. Leave empty if you have nothing to
// surface.
//
// memoryFiles is the consumer's place to inject project / user memory
// files. A common pattern is to concatenate AGENTS.md from the
// workspace and the user's $HOME, but the harness only treats this as
// opaque text to append.
func BuildStaticSystemPrompt(
	workspace, systemPrompt, agentID, agentName string,
	toolNames []string,
	configSummary string,
	skillsIndex string,
	memoryIndex string,
	memoryFiles string,
) string {
	var base string
	if systemPrompt != "" {
		base = systemPrompt
	} else {
		identityPath := filepath.Join(workspace, "IDENTITY.md")
		data, err := os.ReadFile(identityPath)
		if err != nil {
			base = buildDefaultIdentity(toolNames)
		} else {
			base = string(data)
		}
	}

	if agentID != "" {
		base += fmt.Sprintf("\n\nYou are the %q agent (id: %s).", agentName, agentID)
	}

	if configSummary != "" {
		base += "\n\n" + configSummary
	}
	if skillsIndex != "" {
		base += skillsIndex
	}
	if memoryIndex != "" {
		base += memoryIndex
	}
	if memoryFiles != "" {
		base += memoryFiles
	}

	return base
}

// buildDynamicSystemPromptSuffix concatenates the per-turn dynamic context
// — the date line and the cortex hint — into a single string the runtime
// sends as the second (un-cached) SystemPromptPart. The date line, when
// non-empty, appears at the top so the model anchors on "today" before
// the per-turn content. Returns "" when all inputs are empty/nil.
//
// As of sub-project 5, skill bodies and memory entries are no longer
// auto-injected here. Their indices live in the cached static prompt
// and the agent loads bodies on demand via the load_skill / load_memory
// tools. This trims 5–15 KB of speculative skill bodies and 5–10 KB of
// speculative memory bodies from every prefill.
func buildDynamicSystemPromptSuffix(dateLine, cortexContext string) string {
	var sb strings.Builder
	if dateLine != "" {
		sb.WriteString(dateLine)
	}
	if cortexContext != "" {
		sb.WriteString(cortexContext)
	}
	return sb.String()
}

// assembleMessages converts session history into LLM messages.
// It ensures that every tool_use block in an assistant message has a
// corresponding tool_result in the next user message. Orphaned tool calls
// (e.g. from interrupted sessions) get synthetic error results injected.
func assembleMessages(history []session.SessionEntry) []llm.Message {
	// First pass: collect tool result IDs so we can detect orphaned tool calls.
	resultIDs := make(map[string]bool)
	for _, entry := range history {
		if entry.Type == session.EntryTypeToolResult {
			var tr session.ToolResultData
			if err := json.Unmarshal(entry.Data, &tr); err == nil {
				resultIDs[tr.ToolCallID] = true
			}
		}
	}

	var msgs []llm.Message

	for _, entry := range history {
		switch entry.Type {
		case session.EntryTypeCompaction:
			var cd session.CompactionData
			if err := json.Unmarshal(entry.Data, &cd); err != nil {
				continue
			}
			// The summary is followed by an explicit continuation directive
			// so the model resumes the conversation rather than treating it
			// as a fresh start. Without this, models tend to reply with
			// openers like "I'm ready to help! Our previous conversation
			// covered X" — which loses the in-flight task context that
			// the user's next message implicitly relies on.
			content := "[Previous conversation summary]\n\n" + cd.Summary +
				"\n\nContinue the conversation from where it left off without asking the user any further questions. " +
				"Resume directly — do not acknowledge the summary, do not recap what was happening, " +
				"do not preface with \"I'll continue\" or similar. Pick up the last task as if the break never happened."
			msgs = append(msgs, llm.Message{
				Role:    "user",
				Content: content,
			})

		case session.EntryTypeMeta:
			// Meta entries (e.g. compaction summaries) are treated as system context
			var md session.MessageData
			if err := json.Unmarshal(entry.Data, &md); err != nil {
				continue
			}
			msgs = append(msgs, llm.Message{
				Role:    "user",
				Content: "[Session Summary]\n" + md.Text,
			})

		case session.EntryTypeMessage:
			var md session.MessageData
			if err := json.Unmarshal(entry.Data, &md); err != nil {
				continue
			}
			// Before appending a new message, check if the last assistant
			// message has orphaned tool calls that need synthetic results.
			msgs = injectMissingToolResults(msgs)
			msg := llm.Message{
				Role:    entry.Role,
				Content: md.Text,
			}
			// Convert session images to LLM image content
			if entry.Role == "user" {
				for _, img := range md.Images {
					data, err := base64.StdEncoding.DecodeString(img.Data)
					if err != nil {
						continue
					}
					msg.Images = append(msg.Images, llm.ImageContent{
						MimeType: detectImageMIME(data, img.MimeType),
						Data:     data,
					})
				}
			}
			msgs = append(msgs, msg)

		case session.EntryTypeToolCall:
			var td session.ToolCallData
			if err := json.Unmarshal(entry.Data, &td); err != nil {
				continue
			}
			// Skip a corrupted tool_call entry that has no ID — the
			// API requires every tool_use block to have a non-empty
			// tool_use_id that the matching tool_result references.
			// Reaches here when an old session file was written with
			// the pre-fix ToolCallEntry that swallowed marshal errors
			// on an empty json.RawMessage input and persisted
			// "data":null on disk. Skipping leaves the tool unpaired,
			// which assembleMessages's later injectMissingToolResults
			// pass would otherwise paper over with a synthetic error
			// result — but here both call AND result are absent from
			// the assistant message, which is the only safe shape
			// because the tool_result entry under it has no tool_use
			// to anchor to.
			if td.ID == "" {
				continue
			}
			// Tool calls are part of the assistant turn — merge into the last assistant message
			// or create one if needed
			if len(msgs) == 0 || msgs[len(msgs)-1].Role != "assistant" {
				msgs = append(msgs, llm.Message{Role: "assistant"})
			}
			msgs[len(msgs)-1].ToolCalls = append(msgs[len(msgs)-1].ToolCalls, llm.ToolCall{
				ID:    td.ID,
				Name:  td.Tool,
				Input: td.Input,
			})

		case session.EntryTypeToolResult:
			var tr session.ToolResultData
			if err := json.Unmarshal(entry.Data, &tr); err != nil {
				continue
			}
			// Skip an orphan tool_result whose matching tool_call was
			// dropped above (e.g. corrupted "data":null tool_call from
			// the pre-fix ToolCallEntry). Sending the tool_result alone
			// would trigger Anthropic's "messages.N.content.0:
			// unexpected tool_use_id" 400 because no preceding
			// assistant message contains a tool_use with this ID.
			if !lastAssistantHasToolCall(msgs, tr.ToolCallID) {
				continue
			}
			content := tr.Output
			if tr.Error != "" {
				content = tr.Error
			}
			if content == "" {
				content = "(no output)"
			}
			msg := llm.Message{
				Role:       "user",
				Content:    content,
				ToolCallID: tr.ToolCallID,
				IsError:    tr.IsError,
			}
			// Convert session images to LLM image content
			for _, img := range tr.Images {
				data, err := base64.StdEncoding.DecodeString(img.Data)
				if err != nil {
					continue
				}
				msg.Images = append(msg.Images, llm.ImageContent{
					MimeType: detectImageMIME(data, img.MimeType),
					Data:     data,
				})
			}
			msgs = append(msgs, msg)
		}
	}

	// Final check: handle orphaned tool calls at the end of history.
	msgs = injectMissingToolResults(msgs)

	return msgs
}

// injectMissingToolResults walks the message sequence and inserts synthetic
// tool_result user messages for any assistant tool_calls that lack a matching
// tool_result in the immediately-following user messages. Handles both
// end-of-history orphans (the original case) and mid-history orphans (which
// can occur when a Phase B parallel dispatch crashes mid-batch and the
// session is later /resume'd).
//
// Algorithm: scan left-to-right; for each assistant message with ToolCalls,
// collect the next k user messages' ToolCallIDs (where k = len(ToolCalls)).
// Any tc.ID missing from that set gets a synthetic error tool_result inserted
// immediately after the assistant message.
func injectMissingToolResults(msgs []llm.Message) []llm.Message {
	if len(msgs) == 0 {
		return msgs
	}
	out := make([]llm.Message, 0, len(msgs))
	i := 0
	for i < len(msgs) {
		m := msgs[i]
		out = append(out, m)
		if m.Role != "assistant" || len(m.ToolCalls) == 0 {
			i++
			continue
		}
		// Collect tool_call_ids present in the next user-role messages
		// immediately after this assistant turn. Stop when we hit a non-user
		// message or a user message lacking ToolCallID (which means it's a
		// regular user prompt, not a tool result).
		seen := map[string]bool{}
		j := i + 1
		for j < len(msgs) && msgs[j].Role == "user" && msgs[j].ToolCallID != "" {
			seen[msgs[j].ToolCallID] = true
			j++
		}
		// For each tool_call without a matching result, append a synthetic.
		for _, tc := range m.ToolCalls {
			if !seen[tc.ID] {
				out = append(out, llm.Message{
					Role:       "user",
					Content:    "(tool execution was interrupted)",
					ToolCallID: tc.ID,
					IsError:    true,
				})
			}
		}
		i++ // advance past this assistant; the for-loop will copy the user tool_results next iteration
	}
	return out
}

// lastAssistantHasToolCall reports whether the most recent assistant
// message in msgs contains a tool_call with the given ID. Used by
// assembleMessages to decide whether a tool_result entry has a
// preceding tool_use to anchor to. Walks backward through msgs and
// stops at the first assistant message it finds — there can only be
// one preceding assistant block before any given tool_result run.
func lastAssistantHasToolCall(msgs []llm.Message, id string) bool {
	if id == "" {
		return false
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "assistant" {
			continue
		}
		for _, tc := range msgs[i].ToolCalls {
			if tc.ID == id {
				return true
			}
		}
		return false // first assistant we hit didn't have it; no earlier one will count
	}
	return false
}

// truncationMarker / spillMarker uniquely identify content that
// pruneToolResults has already shortened (truncated in-place) or spilled
// to disk, so re-runs across turns become cheap no-ops instead of
// re-scanning multi-hundred-KB tool outputs each time. Both markers are
// recognised on idempotency check; only spillMarker is written when a
// spillConfig is supplied and the disk write succeeds.
const (
	truncationMarker = "[truncated — "
	spillMarker      = "[spilled — "
)

// spillConfig configures disk-spillover behaviour for pruneToolResults.
// Zero value (Workspace == "") disables spillover and falls back to the
// legacy in-place truncation path — keeping tests and any code path
// without a workspace working unchanged.
//
// When Workspace is set, oversized tool results are written to
// <Workspace>/.harness/spill/<SessionKey>/<ToolCallID>.txt and the message
// is rewritten as: head preview (first maxLen chars cut at newline) +
// spillMarker pointing the model at the absolute path so it can recover
// the full output via read_file (which gates on Workspace).
type spillConfig struct {
	Workspace  string
	SessionKey string
}

// spillToolResult writes content to the workspace-local spill directory
// and returns the absolute path. Caller is responsible for handing that
// path to the model. Returns an error on any I/O failure so callers can
// fall back to truncation.
func spillToolResult(cfg spillConfig, toolCallID, content string) (string, error) {
	if cfg.Workspace == "" || cfg.SessionKey == "" || toolCallID == "" {
		return "", fmt.Errorf("spillToolResult: workspace, session key, and tool call id are required")
	}
	dir := filepath.Join(cfg.Workspace, ".harness", "spill", cfg.SessionKey)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("spill mkdir: %w", err)
	}
	path := filepath.Join(dir, toolCallID+".txt")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("spill write: %w", err)
	}
	return path, nil
}

// pruneToolResults bounds oversized tool results in the message history
// to prevent context window overflow and bound prefill growth. Only
// affects tool result messages (identified by having a ToolCallID).
// Idempotent: messages already shortened in a prior turn are skipped via
// marker detection (truncationMarker or spillMarker).
//
// Two modes:
//   - With cfg.Workspace + cfg.SessionKey set: the FULL original content
//     is written to <Workspace>/.harness/spill/<SessionKey>/<ToolCallID>.txt
//     and the message becomes "<head>\n\n[spilled — N of M chars saved
//     to <abs path>; use read_file to access the full output]". The
//     model can recover the entire output via read_file, eliminating the
//     "re-run the tool" round-trip.
//   - Without spill config (or on spill write failure): falls back to
//     the legacy in-place truncation with truncationMarker. This keeps
//     callers without a workspace (tests, edge paths) working.
//
// Either way the retained head is the FIRST maxLen chars cut at the
// nearest newline boundary — most tool outputs (file reads, command
// output) are front-loaded with the most relevant info, so the head
// preview is usually enough on its own.
func pruneToolResults(msgs []llm.Message, maxLen int, cfg spillConfig) {
	for i := range msgs {
		if msgs[i].ToolCallID == "" || len(msgs[i].Content) <= maxLen {
			continue
		}
		// Already-shortened content carries one of the markers near the
		// end; skip the expensive LastIndex scan over hundreds of KB.
		if strings.Contains(msgs[i].Content, truncationMarker) ||
			strings.Contains(msgs[i].Content, spillMarker) {
			continue
		}
		originalLen := len(msgs[i].Content)
		head := msgs[i].Content[:maxLen]
		if idx := strings.LastIndex(head, "\n"); idx > maxLen/2 {
			head = head[:idx]
		}

		// Try spill first if configured. On any failure, fall through to
		// legacy truncation so the prefill stays bounded either way.
		if cfg.Workspace != "" && cfg.SessionKey != "" {
			if path, err := spillToolResult(cfg, msgs[i].ToolCallID, msgs[i].Content); err == nil {
				msgs[i].Content = fmt.Sprintf("%s\n\n%s%d of %d chars saved to %s; use read_file to access the full output]",
					head, spillMarker, len(head), originalLen, path)
				continue
			}
		}

		msgs[i].Content = fmt.Sprintf("%s\n\n%s%d of %d chars; re-run the tool with offset/limit to see more]",
			head, truncationMarker, len(head), originalLen)
	}
}


// FormatDateLine returns the canonical date line injected into the
// dynamic system suffix every Run. Single-line, deterministic format.
//
//	"Today's date is YYYY-MM-DD."
//
// Uses the caller's process timezone (no UTC normalization) so "today"
// matches the user's local sense of the day.
func FormatDateLine(now time.Time) string {
	return fmt.Sprintf("Today's date is %s.", now.Format("2006-01-02"))
}

// PostCompactRestoreFiles caps how many recently-touched files
// buildPostCompactRestore re-injects after a successful compaction.
// 5 mirrors Claude Code's harness behaviour (harness.md §3) and keeps
// the restore message bounded.
const PostCompactRestoreFiles = 5

// PostCompactRestoreBytesPerFile bounds the per-file byte budget for
// the restore message. ~5 KB per file × 5 files ≈ 25 KB total — modest
// next to a 200K-token context window, large enough to carry a typical
// source file's relevant region.
const PostCompactRestoreBytesPerFile = 5 * 1024

// extractPathFromInput returns the "path" field from a tool call's
// JSON input, or "" if absent / unparseable. Used by Runtime to record
// which files the agent has touched (read_file/write_file/edit_file
// all share the same {"path": "..."} schema).
func extractPathFromInput(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	var probe struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return ""
	}
	return probe.Path
}

// buildPostCompactRestore produces a single user-role llm.Message
// containing the latest contents of the most-recent K touched files,
// or returns an empty message (Content == "") when there's nothing
// useful to inject.
//
// Selection: walks files newest-to-oldest (caller passes them in
// touch order, so we range from the back), takes up to maxFiles paths
// that exist and read successfully. Files larger than maxBytesPerFile
// are read up to the cap and a short "[truncated]" marker appended —
// truncating from the head matches our tool-result spillover convention
// and keeps imports/declarations visible.
//
// Format is wrapped in <system-reminder>...<file path="...">...</file>
// tags so the model recognises it as out-of-band context, not a fresh
// user request. Mirrors how Claude Code re-injects POST_COMPACT files.
func buildPostCompactRestore(files []string, maxFiles, maxBytesPerFile int) llm.Message {
	if len(files) == 0 || maxFiles <= 0 || maxBytesPerFile <= 0 {
		return llm.Message{}
	}
	var sb strings.Builder
	sb.WriteString("<system-reminder>\nFiles you were recently working with — full contents below for context restoration after history compaction. The file system is the source of truth; re-read with the read_file tool if you need updated content.\n\n")
	picked := 0
	for i := len(files) - 1; i >= 0 && picked < maxFiles; i-- {
		path := files[i]
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		truncated := false
		if len(data) > maxBytesPerFile {
			data = data[:maxBytesPerFile]
			if idx := strings.LastIndex(string(data), "\n"); idx > maxBytesPerFile/2 {
				data = data[:idx]
			}
			truncated = true
		}
		fmt.Fprintf(&sb, "<file path=%q>\n", path)
		sb.Write(data)
		if !strings.HasSuffix(string(data), "\n") {
			sb.WriteString("\n")
		}
		if truncated {
			fmt.Fprintf(&sb, "[truncated — over %d bytes]\n", maxBytesPerFile)
		}
		sb.WriteString("</file>\n\n")
		picked++
	}
	if picked == 0 {
		return llm.Message{}
	}
	sb.WriteString("</system-reminder>")
	return llm.Message{Role: "user", Content: sb.String()}
}

// prependPostCompactRestore is the runtime-facing helper: builds the
// restore message with the standard caps and prepends it to msgs when
// non-empty. Returns msgs unchanged when there are no touched files,
// when every file fails to read, or when buildPostCompactRestore
// otherwise produces an empty message — keeps the call sites in
// runtime.go to a single line and centralises the cap policy.
func prependPostCompactRestore(msgs []llm.Message, touched []string) []llm.Message {
	restore := buildPostCompactRestore(touched, PostCompactRestoreFiles, PostCompactRestoreBytesPerFile)
	if restore.Content == "" {
		return msgs
	}
	out := make([]llm.Message, 0, len(msgs)+1)
	out = append(out, restore)
	out = append(out, msgs...)
	return out
}
