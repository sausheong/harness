package runtime

import (
	"context"
	"encoding/base64"
	"fmt"
	"sort"
	"strings"

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

	parts := []llm.SystemPromptPart{{Text: r.StaticSystemPrompt, Cache: true}}
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
