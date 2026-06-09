package runtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

// TurnResult describes the outcome of a single RunTurn call.
type TurnResult struct {
	Done       bool                   // true when no tool calls were produced (loop should stop)
	StopReason string                 // "continue" | "completed" | "error" | "aborted"
	Entries    []session.SessionEntry // entries appended during THIS turn, in order
	Err        error
	Usage      *llm.Usage
}

// TurnEmit is an optional live-streaming callback. Pass nil for headless
// execution (e.g. durable replay, where live emission must be suppressed).
type TurnEmit func(AgentEvent)

// RunTurn executes exactly one turn of the agent loop against the current
// session and returns the entries it produced. The first call of a session
// passes the user message; continuation calls pass "".
//
// RunTurn is the durable unit for a per-turn driver: each call can be wrapped
// as a checkpointed step whose return value (TurnResult.Entries) rebuilds
// session state on replay. It differs from Run by design: it is headless
// (live events go only through the emit callback, never an internal channel),
// it dispatches tool calls serially (Run parallelizes concurrency-safe
// batches), and it does NOT perform compaction, knowledge-graph recall/ingest,
// streaming-tool kickoff, or the trace/slog emissions that Run's emitToolResult
// produces. Those remain Run's responsibility.
func (r *Runtime) RunTurn(ctx context.Context, userMsg string, images []llm.ImageContent, emit TurnEmit) (TurnResult, error) {
	if emit == nil {
		emit = func(AgentEvent) {}
	}
	startLen := len(r.Session.Entries())

	if userMsg != "" || len(images) > 0 {
		if len(images) > 0 {
			var imgData []session.ImageData
			for _, img := range images {
				imgData = append(imgData, session.ImageData{
					MimeType: img.MimeType,
					Data:     base64.StdEncoding.EncodeToString(img.Data),
				})
			}
			r.Session.Append(session.UserMessageWithImagesEntry(userMsg, imgData))
		} else {
			r.Session.Append(session.UserMessageEntry(userMsg))
		}
	}

	if ctx.Err() != nil {
		return r.turnSlice(startLen, true, "aborted", nil, ctx.Err()), nil
	}

	history := r.Session.View()
	msgs := assembleMessages(history)
	toolDefs := r.Tools.ToolDefs()
	if r.Permission != nil {
		toolDefs = r.Permission.FilterToolDefs(toolDefs, r.AgentID)
	}
	sort.SliceStable(toolDefs, func(i, j int) bool { return toolDefs[i].Name < toolDefs[j].Name })
	toolDefs, _ = r.LLM.NormalizeToolSchema(toolDefs)

	// KG recall (first round of the exchange only): bounded-synchronous so a
	// slow embedder cannot stall the turn. The hint is a non-cached suffix —
	// it varies per query, so caching it would poison the static prefix cache.
	var kgHint string
	if r.KG != nil && (userMsg != "" || len(images) > 0) && r.KG.ShouldRecall(userMsg) {
		rctx, cancel := context.WithTimeout(ctx, 800*time.Millisecond)
		kgHint = r.KG.Recall(rctx, userMsg)
		cancel()
	}

	parts := []llm.SystemPromptPart{{Text: r.StaticSystemPrompt, Cache: true}}
	if kgHint != "" {
		parts = append(parts, llm.SystemPromptPart{Text: kgHint, Cache: false})
	}
	req := llm.ChatRequest{
		Model:             r.Model,
		Messages:          msgs,
		Tools:             toolDefs,
		MaxTokens:         8192,
		SystemPromptParts: parts,
		CacheLastMessage:  r.providerSupportsCaching(),
		Reasoning:         r.Reasoning,
	}

	stream, err := r.LLM.ChatStream(ctx, req)
	if err != nil {
		if r.FallbackModel != "" && r.FallbackModel != req.Model && llm.IsRetryableModelError(err) {
			req.Model = r.FallbackModel
			stream, err = r.LLM.ChatStream(ctx, req)
		}
		if err != nil {
			emit(AgentEvent{Type: EventError, Error: fmt.Errorf("llm error: %w", err)})
			return r.turnSlice(startLen, true, "error", nil, err), nil
		}
	}

	var textContent strings.Builder
	var toolCalls []llm.ToolCall
	var lastUsage *llm.Usage
	for event := range stream {
		switch event.Type {
		case llm.EventTextDelta:
			textContent.WriteString(event.Text)
			emit(AgentEvent{Type: EventTextDelta, Text: event.Text})
		case llm.EventToolCallStart:
			emit(AgentEvent{Type: EventToolCallStart, ToolCall: event.ToolCall})
		case llm.EventToolCallDone:
			if event.ToolCall != nil {
				toolCalls = append(toolCalls, *event.ToolCall)
			}
		case llm.EventDone:
			if event.Usage != nil {
				lastUsage = event.Usage
			}
		case llm.EventError:
			emit(AgentEvent{Type: EventError, Error: event.Error})
			return r.turnSlice(startLen, true, "error", lastUsage, event.Error), nil
		}
	}

	if textContent.Len() > 0 {
		r.Session.Append(session.AssistantMessageEntry(textContent.String()))
	}

	if len(toolCalls) == 0 {
		// Exchange complete: ingest the full thread once. Background + best-effort
		// (the KG spawns its own bounded goroutine), so it never delays the turn.
		// context.Background() because the request ctx may be cancelling and the
		// ingest goroutine deliberately outlives the turn. Unlike Run, RunTurn does
		// not pre-filter trivial threads or gate on IngestSource: the KG impl owns
		// its own growth gate (so single-message threads are skipped there), and the
		// only RunTurn caller today is the chat serve loop. TODO: if a reviewer or
		// subagent path is ever migrated to RunTurn, add an IngestSource-style gate
		// here so non-chat transcripts are not ingested.
		if r.KG != nil {
			r.KG.Ingest(context.Background(), sessionThread(r.Session.View()))
		}
		emit(AgentEvent{Type: EventDone, Usage: lastUsage})
		return r.turnSlice(startLen, true, "completed", lastUsage, nil), nil
	}

	// RunTurn dispatches tool calls serially by design: deterministic
	// ordering keeps durable replay simple, and parallel/concurrency-safe
	// batching deliberately stays in Run (see the doc comment). dispatchTool
	// appends the tool_call and tool_result entries to the session.
	for _, tc := range toolCalls {
		result, aborted := r.dispatchTool(ctx, tc, nil)
		emit(AgentEvent{Type: EventToolResult, ToolCall: &tc, Result: &result})
		if aborted {
			return r.turnSlice(startLen, true, "aborted", lastUsage, ctx.Err()), nil
		}
	}

	return r.turnSlice(startLen, false, "continue", lastUsage, nil), nil
}

// turnSlice builds a TurnResult from the entries appended since startLen.
// It assumes a single active turn per session: the delta entries[startLen:]
// is correct only because no other goroutine appends to this session
// concurrently during the turn.
func (r *Runtime) turnSlice(startLen int, done bool, reason string, usage *llm.Usage, err error) TurnResult {
	all := r.Session.Entries()
	var delta []session.SessionEntry
	if startLen <= len(all) {
		delta = append(delta, all[startLen:]...)
	}
	return TurnResult{Done: done, StopReason: reason, Entries: delta, Usage: usage, Err: err}
}

// sessionThread converts session history into the minimal []Message the
// KnowledgeGraph ingests, mirroring how Run accumulates its thread: user and
// assistant messages by role+text, tool calls as "[tool: name]\n<input>"
// (assistant), tool results as their output or "[error] <err>" (user).
// Compaction/meta and non-user/assistant message entries are skipped — the
// thread carries only conversation turns and tool exchanges.
// On the runtime's per-turn serve path each call sees one turn's fresh session,
// so a compaction summary's absence here is intentional, not a gap.
func sessionThread(history []session.SessionEntry) []Message {
	var thread []Message
	for _, e := range history {
		switch e.Type {
		case session.EntryTypeMessage:
			if e.Role != "user" && e.Role != "assistant" {
				continue // skip system/summary messages
			}
			var d session.MessageData
			if json.Unmarshal(e.Data, &d) != nil {
				continue
			}
			thread = append(thread, Message{Role: e.Role, Content: d.Text})
		case session.EntryTypeToolCall:
			var d session.ToolCallData
			if json.Unmarshal(e.Data, &d) != nil {
				continue
			}
			thread = append(thread, Message{
				Role:    "assistant",
				Content: fmt.Sprintf("[tool: %s]\n%s", d.Tool, string(d.Input)),
			})
		case session.EntryTypeToolResult:
			var d session.ToolResultData
			if json.Unmarshal(e.Data, &d) != nil {
				continue
			}
			content := d.Output
			if d.Error != "" {
				content = "[error] " + d.Error
			}
			thread = append(thread, Message{Role: "user", Content: content})
		}
	}
	return thread
}
