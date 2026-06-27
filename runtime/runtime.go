package runtime

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sausheong/harness/compaction"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tokens"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

// EventType identifies the kind of agent event.
type EventType int

const (
	EventTextDelta EventType = iota
	EventToolCallStart
	EventToolResult
	EventDone
	EventError
	EventAborted
	EventCompactionStart
	EventCompactionDone
	EventCompactionSkipped
)

// AgentEvent is a single streaming event from the agent.
type AgentEvent struct {
	Type EventType
	// AgentID is the emitter's agent identifier. Empty for top-level
	// (Parent==nil) runtimes; populated by Runtime.emit when forwarding a
	// subagent's event up to its parent.
	AgentID    string
	Text       string
	ToolCall   *llm.ToolCall
	Result     *tool.ToolResult
	Error      error
	Compaction *compaction.Result // populated for EventCompaction* events
	Usage      *llm.Usage         // on the terminal EventDone, the token total accumulated across all turns of the run (nil if the provider never reported usage); per-turn figures are available via RunTurn's TurnResult.Usage
}

// Runtime is the agent think-act loop.
type Runtime struct {
	LLM       llm.LLMProvider
	Tools     tool.Executor
	Session   *session.Session
	AgentID   string
	AgentName string
	Model     string
	Reasoning llm.ReasoningMode // optional; zero value = ReasoningOff
	Workspace string
	MaxTurns  int
	// AgentLoop carries the loop tunables (concurrency cap, depth cap,
	// streaming-tools toggle). Zero value → readers fall back to env vars
	// then compiled-in defaults.
	AgentLoop    LoopConfig
	SystemPrompt string              // optional: inline system prompt
	Skills       SkillProvider       // optional: skill index + load_skill backend
	Memory       MemoryProvider      // optional: memory index + load_memory backend
	KG           KnowledgeGraph      // optional: knowledge-graph plug point
	Compaction   *compaction.Manager // optional; nil → no compaction

	// Provider is the LLM provider name parsed from the agent's
	// "provider/model" model string (e.g., "anthropic", "openai", "local").
	Provider string

	// FallbackModel is the bare model name to retry against on a synchronous
	// ChatStream error matching llm.IsRetryableModelError.
	FallbackModel string

	// ContextWindow overrides the auto-detected window. 0 = auto-detect.
	ContextWindow int

	// StaticSystemPrompt is the cacheable portion of the system prompt.
	// Built once at BuildRuntime time; reused verbatim every turn so the
	// prompt cache hits.
	StaticSystemPrompt string

	// Permission gates tool execution at dispatch time. nil → allow-all.
	Permission tool.PermissionChecker

	// Depth is the recursion level for subagents. 0 for top-level agents.
	Depth int

	// Parent points to the Runtime that invoked this Runtime as a subagent.
	Parent *Runtime

	events chan AgentEvent

	// kgMu serializes appends to the per-Run KG thread when parallel
	// dispatchTool calls happen from multiple goroutines.
	kgMu sync.Mutex

	// touchedFiles is the in-order list of file paths the agent has
	// successfully read/written/edited during this Runtime's lifetime.
	touchedMu    sync.Mutex
	touchedFiles []string

	// spilledIDs records ToolCallIDs whose oversized results have already
	// been written to the deterministic spill path during this Runtime's
	// lifetime. Content is immutable per ToolCallID, so once spilled we
	// skip the (multi-MB) disk re-write on subsequent turns and only
	// rebuild the in-memory marker.
	spilledMu  sync.Mutex
	spilledIDs map[string]bool

	// IngestSource controls whether this run's thread is ingested into the KG.
	// "chat" (or empty) ingests; any other value skips ingest.
	IngestSource string

	// CalibratorStore persists the per-session token Calibrator. nil = off.
	CalibratorStore *tokens.CalibratorStore

	calibrator *tokens.Calibrator

	// calibratorDirty records that the calibrator was updated during this
	// Run and must be flushed to CalibratorStore once at Run end. Accessed
	// only on the Run goroutine (EventDone handling + deferred flush), so no
	// lock is needed.
	calibratorDirty bool

	// mcpClients holds the MCP client sessions resolved from
	// AgentSpec.MCPServers. Released by Close. Untouched when MCP is
	// not used.
	mcpClients []*mcp.Client
}

// Close releases external resources owned by the Runtime — currently
// just the MCP client sessions resolved from AgentSpec.MCPServers.
// Optional: callers who never use MCP can skip it. Safe to call
// multiple times; safe to call before or after Run completes.
// Returns the first Close error encountered, if any; remaining
// clients are still closed.
func (r *Runtime) Close() error {
	var firstErr error
	for _, c := range r.mcpClients {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	r.mcpClients = nil
	return firstErr
}

// emit sends ev to this runtime's events channel and forwards a copy to
// Parent.events when this is a subagent. Forward is non-blocking — drops
// when the parent's channel is full.
func (r *Runtime) emit(ev AgentEvent) {
	if r.Parent != nil {
		forward := ev
		forward.AgentID = r.AgentID
		select {
		case r.Parent.events <- forward:
		default:
		}
	}
	r.events <- ev
}

// maybeKickoffAsyncCompaction conditionally fires Compaction.MaybeCompactAsync
// when the just-finished turn left the session close enough to the trigger
// threshold that the NEXT turn would compact synchronously.
func (r *Runtime) maybeKickoffAsyncCompaction(msgs []llm.Message, parts []llm.SystemPromptPart, toolDefs []llm.ToolDef) {
	if r.Compaction == nil || r.Model == "" {
		return
	}
	if r.Compaction.HasInFlight(r.Session) {
		return
	}
	if r.calibrator == nil {
		r.calibrator = tokens.NewCalibrator()
	}
	estimate := r.calibrator.Adjust(tokens.Estimate(msgs, llm.JoinSystemPromptParts(parts), toolDefs))
	window := tokens.ContextWindowFor(r.Model, r.ContextWindow)
	threshold := 0.6
	if r.Compaction.Threshold > 0 {
		threshold = r.Compaction.Threshold
	}
	preemptThresholdHit := window > 0 && float64(estimate) > 0.8*threshold*float64(window)
	msgCap := r.Compaction.MessageCap
	preemptCountHit := msgCap > 0 && len(msgs) > int(0.8*float64(msgCap))
	if !preemptThresholdHit && !preemptCountHit {
		return
	}
	r.Compaction.MaybeCompactAsync(r.Session, compaction.ReasonPreventive)
}

// providerSupportsCaching returns true when the runtime's provider implements
// Anthropic-style explicit prompt caching. The capability interface wins:
// a provider block named "platformai" (or "bedrock", or anything else)
// wrapping the Anthropic implementation still caches. The name check is
// the legacy fallback for custom LLMProvider impls that predate the
// capability interface.
func (r *Runtime) providerSupportsCaching() bool {
	if pc, ok := r.LLM.(llm.PromptCachingProvider); ok {
		return pc.SupportsPromptCaching()
	}
	return r.Provider == "anthropic"
}

// providerSupportsMidLoopCompaction returns true for hosted frontier
// providers that handle mid-loop summary injection cleanly. Same
// capability-over-name rule as providerSupportsCaching: an
// Anthropic-shaped provider under a custom name qualifies.
func (r *Runtime) providerSupportsMidLoopCompaction() bool {
	if pc, ok := r.LLM.(llm.PromptCachingProvider); ok && pc.SupportsPromptCaching() {
		return true
	}
	switch r.Provider {
	case "anthropic", "openai", "gemini":
		return true
	default:
		return false
	}
}

// recordFileTouch appends path to the touched-files list, deduping by moving
// an existing entry to the end.
func (r *Runtime) recordFileTouch(path string) {
	if path == "" || r == nil {
		return
	}
	r.touchedMu.Lock()
	defer r.touchedMu.Unlock()
	for i, p := range r.touchedFiles {
		if p == path {
			r.touchedFiles = append(append(r.touchedFiles[:i:i], r.touchedFiles[i+1:]...), path)
			return
		}
	}
	r.touchedFiles = append(r.touchedFiles, path)
}

// snapshotTouchedFiles returns a copy of the current touched-files list.
func (r *Runtime) snapshotTouchedFiles() []string {
	r.touchedMu.Lock()
	defer r.touchedMu.Unlock()
	out := make([]string, len(r.touchedFiles))
	copy(out, r.touchedFiles)
	return out
}

// prune shortens oversized tool results in msgs, spilling to disk where
// configured. It carries the Runtime's per-Run spilled-ID set so a result
// already spilled this Run is not re-written to disk on subsequent turns.
func (r *Runtime) prune(msgs []llm.Message, spillCfg spillConfig) {
	r.spilledMu.Lock()
	if r.spilledIDs == nil {
		r.spilledIDs = map[string]bool{}
	}
	pruneToolResults(msgs, r.maxToolResultLen(), spillCfg, r.spilledIDs)
	r.spilledMu.Unlock()
}

// isFileTool reports whether a tool name's input contains a "path" field.
func isFileTool(name string) bool {
	switch name {
	case "read_file", "write_file", "edit_file":
		return true
	}
	return false
}

// Run executes the agent loop for a user message, returning a channel of events.
func (r *Runtime) Run(ctx context.Context, userMsg string, images []llm.ImageContent) (<-chan AgentEvent, error) {
	r.events = make(chan AgentEvent, 100)
	tr := TraceFrom(ctx)
	tr.Mark("agent.run.start", "user_msg_len", len(userMsg), "images", len(images))

	go func() {
		defer close(r.events)
		defer tr.Summary()

		// Flush the per-session calibrator once at Run end (all exit paths:
		// normal, error, abort) instead of writing to disk on every LLM
		// round. EventDone sets calibratorDirty when it folds in a usage
		// sample; this defer persists the final state. Registered here so it
		// runs regardless of which return below fires.
		defer func() {
			if r.calibratorDirty && r.CalibratorStore != nil && r.Session != nil && r.calibrator != nil {
				ratio, count := r.calibrator.Snapshot()
				r.CalibratorStore.Save(r.AgentID, r.Session.Key, ratio, count)
			}
		}()

		// stopReason is read by the OnStop deferred hook below. Each early
		// return path reassigns it before returning. Default is "max_turns"
		// so the natural fall-through (loop exhausted MaxTurns) records the
		// right reason without an explicit assignment at that exit.
		stopReason := "max_turns"
		if hook := r.AgentLoop.Hooks.OnStop; hook != nil {
			defer func() { hook(ctx, stopReason) }()
		}

		// OnUserPromptSubmit may rewrite prompt and/or images BEFORE the
		// session sees them. Returning err aborts Run with EventError.
		if hook := r.AgentLoop.Hooks.OnUserPromptSubmit; hook != nil {
			newMsg, newImgs, err := hook(ctx, userMsg, images)
			if err != nil {
				stopReason = "error"
				r.emit(AgentEvent{Type: EventError, Error: err})
				return
			}
			userMsg = newMsg
			images = newImgs
		}

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

		if hook := r.AgentLoop.Hooks.OnSessionStart; hook != nil {
			hook(ctx, r.Session)
		}

		// Initialise KG thread and recall (once, before the loop). Recall
		// runs in a background goroutine so it overlaps with message
		// assembly. The main goroutine waits for it (with an 800ms cap)
		// before invoking the LLM. Ingest is gated by IngestSource.
		var thread []Message
		kgContext := ""
		var kgCh chan string
		var kgCancel context.CancelFunc
		shouldIngest := r.IngestSource == "" || r.IngestSource == "chat"
		if r.KG != nil {
			thread = []Message{{Role: "user", Content: userMsg}}
			if r.KG.ShouldRecall(userMsg) {
				kgCh = make(chan string, 1)
				var kgCtx context.Context
				kgCtx, kgCancel = context.WithCancel(ctx)
				kgStart := time.Now()
				kg := r.KG
				go func() {
					hint := kg.Recall(kgCtx, userMsg)
					tr.Mark("kg.recall", "len", len(hint), "dur_ms_local", time.Since(kgStart).Milliseconds())
					kgCh <- hint
				}()
			} else {
				tr.Mark("kg.recall.skipped", "reason", "trivial")
			}

			kg := r.KG
			defer func() {
				if kgCancel != nil {
					kgCancel()
				}
				if shouldIngest && len(thread) > 1 {
					kg.Ingest(context.Background(), thread)
				}
			}()
		}

		maxTurns := r.MaxTurns
		if maxTurns == 0 {
			maxTurns = 25
		}

		// Computed once per Run so the date stays stable across turns.
		dateLine := FormatDateLine(time.Now())

		// Tool defs are invariant for the whole Run (tools and permission don't
		// change mid-Run), so normalize once instead of per turn (P3).
		toolDefs := r.Tools.ToolDefs()
		if r.Permission != nil {
			toolDefs = r.Permission.FilterToolDefs(toolDefs, r.AgentID)
		}
		sort.SliceStable(toolDefs, func(i, j int) bool {
			return toolDefs[i].Name < toolDefs[j].Name
		})
		toolDefs, diags := r.LLM.NormalizeToolSchema(toolDefs)
		for _, d := range diags {
			slog.Info("tool schema normalized",
				"tool", d.ToolName,
				"field", d.Field,
				"action", d.Action,
				"reason", d.Reason)
		}

		var runUsage *llm.Usage // accumulated across all turns; nil until first reported

		for turn := 0; turn < maxTurns; turn++ {
			if ctx.Err() != nil {
				stopReason = "aborted"
				r.emit(AgentEvent{Type: EventAborted})
				return
			}

			phaseStart := time.Now()

			// KG hint: on first turn wait for the background Recall (with
			// 800ms cap so a slow embedder can't stall the request).
			if kgCh != nil && kgContext == "" {
				select {
				case kgContext = <-kgCh:
				case <-time.After(800 * time.Millisecond):
					tr.Mark("kg.recall.timeout", "turn", turn, "budget_ms", 800)
					if kgCancel != nil {
						kgCancel()
						kgCancel = nil
					}
				}
				kgCh = nil
			}

			dynamicSuffix := buildDynamicSystemPromptSuffix(dateLine, kgContext)

			staticText := r.StaticSystemPrompt
			parts := []llm.SystemPromptPart{
				{Text: staticText, Cache: true},
			}
			if dynamicSuffix != "" {
				parts = append(parts, llm.SystemPromptPart{Text: dynamicSuffix, Cache: false})
			}

			history := r.Session.View()
			msgs := assembleMessages(history)

			spillCfg := spillConfig{Workspace: r.Workspace, SessionKey: r.Session.Key}
			r.prune(msgs, spillCfg)

			tr.Mark("context.assemble", "turn", turn, "msgs", len(msgs), "tools", len(toolDefs), "sysprompt_chars", len(staticText)+len(dynamicSuffix), "dur_ms_local", time.Since(phaseStart).Milliseconds())

			// Wait briefly on any in-flight async compaction kicked off by
			// the previous turn.
			if turn == 0 && r.Compaction != nil {
				if res, ok := r.Compaction.WaitForInFlight(r.Session, 8*time.Second); ok && res.Compacted {
					r.emit(AgentEvent{Type: EventCompactionDone, Compaction: &res})
					history = r.Session.View()
					msgs = assembleMessages(history)
					r.prune(msgs, spillCfg)
					msgs = prependPostCompactRestore(msgs, r.snapshotTouchedFiles())
				}
			}

			compactionAllowed := turn == 0 || r.providerSupportsMidLoopCompaction()
			if compactionAllowed && r.Compaction != nil && r.Model != "" {
				if r.calibrator == nil {
					r.calibrator = tokens.NewCalibrator()
				}
				estimate := r.calibrator.Adjust(tokens.Estimate(msgs, llm.JoinSystemPromptParts(parts), toolDefs))
				window := tokens.ContextWindowFor(r.Model, r.ContextWindow)
				threshold := 0.6
				if r.Compaction != nil && r.Compaction.Threshold > 0 {
					threshold = r.Compaction.Threshold
				}
				thresholdHit := window > 0 && estimate > int(threshold*float64(window))
				msgCap := r.Compaction.MessageCap
				countHit := msgCap > 0 && len(msgs) > msgCap
				if thresholdHit || countHit {
					r.emit(AgentEvent{Type: EventCompactionStart})
					res, _ := r.Compaction.MaybeCompact(ctx, r.Session, compaction.ReasonPreventive, "")
					if res.Compacted {
						r.emit(AgentEvent{Type: EventCompactionDone, Compaction: &res})
						history = r.Session.View()
						msgs = assembleMessages(history)
						r.prune(msgs, spillCfg)
						msgs = prependPostCompactRestore(msgs, r.snapshotTouchedFiles())
					} else {
						r.emit(AgentEvent{Type: EventCompactionSkipped, Compaction: &res})
					}
				}
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

			llmStart := time.Now()
			prefillChars := len(staticText) + len(dynamicSuffix)
			for _, m := range msgs {
				prefillChars += len(m.Content)
			}
			tr.Mark("llm.request_sent", "turn", turn, "model", r.Model, "prefill_chars", prefillChars)
			stream, err := r.LLM.ChatStream(ctx, req)
			if err != nil {
				if compaction.IsContextOverflow(err) && r.Compaction != nil {
					r.emit(AgentEvent{Type: EventCompactionStart})
					res, _ := r.Compaction.MaybeCompact(ctx, r.Session, compaction.ReasonReactive, "")
					if res.Compacted {
						r.emit(AgentEvent{Type: EventCompactionDone, Compaction: &res})
						history = r.Session.View()
						msgs = assembleMessages(history)
						r.prune(msgs, spillCfg)
						msgs = prependPostCompactRestore(msgs, r.snapshotTouchedFiles())
						req.Messages = msgs
						stream, err = r.LLM.ChatStream(ctx, req)
					} else {
						r.emit(AgentEvent{Type: EventCompactionSkipped, Compaction: &res})
					}
				}
				if err != nil && r.FallbackModel != "" && r.FallbackModel != req.Model && llm.IsRetryableModelError(err) {
					slog.Info("llm fallback model engaged",
						"agent", r.AgentID,
						"primary", req.Model,
						"fallback", r.FallbackModel,
						"err", err.Error())
					tr.Mark("llm.fallback", "turn", turn, "primary", req.Model, "fallback", r.FallbackModel)
					req.Model = r.FallbackModel
					stream, err = r.LLM.ChatStream(ctx, req)
				}
				if err != nil {
					stopReason = "error"
					r.emit(AgentEvent{Type: EventError, Error: fmt.Errorf("llm error: %w", err)})
					return
				}
			}

			var textContent strings.Builder
			var toolCalls []llm.ToolCall
			gotFirstToken := false

			streamingOn := r.streamingToolsEnabled()
			kickoffs := map[string]chan kickoffResult{}
			kickoffStopped := false

			streamSource := stream
			retriedNonStreaming := false
			preTokenRetried := false
			retriedRefusal := false
			var refused bool
			var refusalCategory string
		streamLoop:
			for {
				for event := range streamSource {
					switch event.Type {
					case llm.EventTextDelta:
						if !gotFirstToken {
							gotFirstToken = true
							tr.Mark("llm.first_token", "turn", turn, "ttft_ms", time.Since(llmStart).Milliseconds())
						}
						textContent.WriteString(event.Text)
						r.emit(AgentEvent{Type: EventTextDelta, Text: event.Text})

					case llm.EventToolCallStart:
						if !gotFirstToken {
							gotFirstToken = true
							tr.Mark("llm.first_token", "turn", turn, "ttft_ms", time.Since(llmStart).Milliseconds(), "kind", "tool_call")
						}
						r.emit(AgentEvent{Type: EventToolCallStart, ToolCall: event.ToolCall})

					case llm.EventToolCallDone:
						if event.ToolCall == nil {
							continue
						}
						tc := *event.ToolCall
						toolCalls = append(toolCalls, tc)
						if !streamingOn || kickoffStopped {
							continue
						}
						if !isCallConcurrencySafe(tc, r.Tools) {
							kickoffStopped = true
							continue
						}
						ch := make(chan kickoffResult, 1)
						kickoffs[tc.ID] = ch
						tcCopy := tc
						go func() {
							result, aborted := r.executeToolKickoff(ctx, tcCopy)
							r.emitToolResult(tr, turn, tcCopy, result, aborted)
							ch <- kickoffResult{tc: tcCopy, result: result, aborted: aborted}
						}()

					case llm.EventDone:
						refused = event.StopReason == llm.StopReasonRefusal
						refusalCategory = event.StopCategory
						if event.Usage != nil {
							runUsage = addUsage(runUsage, event.Usage)
						}
						if event.Usage != nil && r.calibrator != nil {
							// Recompute the estimate on the CURRENT msgs (which may
							// have been reassembled by mid-turn compaction) so the
							// calibration ratio matches event.Usage.InputTokens, as
							// the pre-debounce code did. The P8 change only debounces
							// the disk write (calibratorDirty + deferred flush), not
							// the estimate computation.
							r.calibrator.Update(event.Usage.InputTokens, tokens.Estimate(msgs, llm.JoinSystemPromptParts(parts), toolDefs))
							r.calibratorDirty = true
						}

					case llm.EventError:
						if !gotFirstToken && !preTokenRetried {
							if newStream, ok := r.recoverFromPreTokenError(ctx, event.Error, &req, &msgs, spillCfg); ok {
								preTokenRetried = true
								drainKickoffs(kickoffs)
								kickoffs = map[string]chan kickoffResult{}
								textContent.Reset()
								toolCalls = nil
								kickoffStopped = false
								streamSource = newStream
								continue streamLoop
							}
						}
						if gotFirstToken && !retriedNonStreaming {
							if ns, ok := r.LLM.(llm.NonStreamingProvider); ok {
								slog.Warn("stream died mid-flight; retrying as non-streaming",
									"agent", r.AgentID, "turn", turn, "err", event.Error)
								tr.Mark("llm.stream_fallback", "turn", turn, "err", event.Error.Error())
								textContent.Reset()
								toolCalls = nil
								drainKickoffs(kickoffs)
								kickoffs = map[string]chan kickoffResult{}
								gotFirstToken = false
								kickoffStopped = false
								nsStream, retryErr := ns.ChatNonStreaming(ctx, req)
								if retryErr != nil {
									stopReason = "error"
									r.emit(AgentEvent{Type: EventError, Error: retryErr})
									return
								}
								retriedNonStreaming = true
								streamSource = nsStream
								continue streamLoop
							}
						}
						drainKickoffs(kickoffs)
						stopReason = "error"
						r.emit(AgentEvent{Type: EventError, Error: event.Error})
						return
					}
				}

				// Classifier/model refusal (Claude Fable 5+): HTTP 200,
				// stop_reason=refusal, usually no content. Without handling,
				// the turn ends as a silent empty success. Retry once on the
				// fallback model (refusals are model-specific — Opus-tier
				// has no bio/cyber classifiers); otherwise synthesize a
				// visible explanation. Mid-stream refusals discard partial
				// output per the API contract.
				if refused {
					if !retriedRefusal && r.FallbackModel != "" && r.FallbackModel != req.Model {
						slog.Warn("model refused; retrying on fallback model",
							"agent", r.AgentID,
							"primary", req.Model,
							"fallback", r.FallbackModel,
							"category", refusalCategory)
						tr.Mark("llm.refusal_fallback", "turn", turn,
							"primary", req.Model, "fallback", r.FallbackModel,
							"category", refusalCategory)
						textContent.Reset()
						toolCalls = nil
						drainKickoffs(kickoffs)
						kickoffs = map[string]chan kickoffResult{}
						gotFirstToken = false
						kickoffStopped = false
						refused = false
						retriedRefusal = true
						req.Model = r.FallbackModel
						retryStream, retryErr := r.LLM.ChatStream(ctx, req)
						if retryErr != nil {
							stopReason = "error"
							r.emit(AgentEvent{Type: EventError, Error: retryErr})
							return
						}
						streamSource = retryStream
						continue streamLoop
					}
					textContent.Reset()
					toolCalls = nil
					drainKickoffs(kickoffs)
					note := "The model declined this request (safety classifier"
					if refusalCategory != "" {
						note += ", category: " + refusalCategory
					}
					note += "). Try rephrasing, or switch this agent to a model without these classifiers."
					textContent.WriteString(note)
					r.emit(AgentEvent{Type: EventTextDelta, Text: note})
				}
				break streamLoop
			}
			tr.Mark("llm.stream_end", "turn", turn,
				"total_ms", time.Since(llmStart).Milliseconds(),
				"text_chars", textContent.Len(),
				"tool_calls", len(toolCalls))

			if textContent.Len() > 0 {
				r.Session.Append(session.AssistantMessageEntry(textContent.String()))
				if r.KG != nil {
					r.kgMu.Lock()
					thread = append(thread, Message{
						Role:    "assistant",
						Content: textContent.String(),
					})
					r.kgMu.Unlock()
				}
			}

			if len(toolCalls) == 0 {
				if len(kickoffs) > 0 {
					drainKickoffs(kickoffs)
				}
				tr.Mark("agent.done", "turn", turn, "reason", "no_tool_calls")
				stopReason = "completed"
				r.emit(AgentEvent{Type: EventDone, Usage: runUsage})
				r.maybeKickoffAsyncCompaction(msgs, parts, toolDefs)
				return
			}

			var pending []llm.ToolCall
			for _, tc := range toolCalls {
				if ch, ok := kickoffs[tc.ID]; ok {
					kp := <-ch
					r.Session.Append(session.ToolCallEntry(kp.tc.ID, kp.tc.Name, kp.tc.Input))
					if r.KG != nil {
						r.kgMu.Lock()
						thread = append(thread, Message{
							Role:    "assistant",
							Content: fmt.Sprintf("[tool: %s]\n%s", kp.tc.Name, string(kp.tc.Input)),
						})
						r.kgMu.Unlock()
					}
					if kp.aborted {
						r.Session.Append(session.AbortedToolResultEntry(kp.tc.ID))
						if r.KG != nil {
							r.kgMu.Lock()
							thread = append(thread, Message{Role: "user", Content: "[error] aborted by user"})
							r.kgMu.Unlock()
						}
						for _, tc2 := range toolCalls {
							if tc2.ID == kp.tc.ID {
								continue
							}
							ch2, ok := kickoffs[tc2.ID]
							if !ok {
								continue
							}
							kp2 := <-ch2
							r.Session.Append(session.ToolCallEntry(kp2.tc.ID, kp2.tc.Name, kp2.tc.Input))
							r.Session.Append(session.AbortedToolResultEntry(kp2.tc.ID))
							if r.KG != nil {
								r.kgMu.Lock()
								thread = append(thread, Message{
									Role:    "assistant",
									Content: fmt.Sprintf("[tool: %s]\n%s", kp2.tc.Name, string(kp2.tc.Input)),
								})
								thread = append(thread, Message{Role: "user", Content: "[error] aborted by user"})
								r.kgMu.Unlock()
							}
						}
						stopReason = "aborted"
						r.emit(AgentEvent{Type: EventAborted})
						return
					}
					imgData := convertToolResultImages(kp.result.Images)
					r.Session.Append(session.ToolResultEntry(kp.tc.ID, kp.result.Output, kp.result.Error, imgData))
					if r.KG != nil {
						content := kp.result.Output
						if kp.result.Error != "" {
							content = "[error] " + kp.result.Error
						}
						r.kgMu.Lock()
						thread = append(thread, Message{Role: "user", Content: content})
						r.kgMu.Unlock()
					}
					continue
				}
				pending = append(pending, tc)
			}

			batches := partitionToolCalls(pending, r.Tools)
			for _, b := range batches {
				if r.runBatch(ctx, b, kgThreadOrNil(r.KG, &thread), turn, tr) {
					stopReason = "aborted"
					r.emit(AgentEvent{Type: EventAborted})
					return
				}
			}
		}

		// stopReason already "max_turns" by default — fall through to
		// the EventError below records "agent exceeded maximum turns".
		r.emit(AgentEvent{
			Type:  EventError,
			Error: fmt.Errorf("agent exceeded maximum turns (%d)", maxTurns),
		})
	}()

	return r.events, nil
}

// recoverFromPreTokenError attempts the same recovery the synchronous
// ChatStream-error path performs, for an error delivered via EventError
// before any token arrived (Anthropic/Gemini deliver HTTP errors this way).
// Returns a fresh stream to resume from and true if recovered; false if the
// error is not recoverable (caller should abort). May mutate *req (model
// swap) and the session (compaction), and reassigns *msgs on compaction.
func (r *Runtime) recoverFromPreTokenError(ctx context.Context, err error, req *llm.ChatRequest, msgs *[]llm.Message, spillCfg spillConfig) (<-chan llm.ChatEvent, bool) {
	if compaction.IsContextOverflow(err) && r.Compaction != nil {
		r.emit(AgentEvent{Type: EventCompactionStart})
		res, _ := r.Compaction.MaybeCompact(ctx, r.Session, compaction.ReasonReactive, "")
		if res.Compacted {
			r.emit(AgentEvent{Type: EventCompactionDone, Compaction: &res})
			history := r.Session.View()
			newMsgs := assembleMessages(history)
			r.prune(newMsgs, spillCfg)
			newMsgs = prependPostCompactRestore(newMsgs, r.snapshotTouchedFiles())
			*msgs = newMsgs
			req.Messages = newMsgs
			if s, e := r.LLM.ChatStream(ctx, *req); e == nil {
				return s, true
			}
		} else {
			r.emit(AgentEvent{Type: EventCompactionSkipped, Compaction: &res})
		}
	}
	if r.FallbackModel != "" && r.FallbackModel != req.Model && llm.IsRetryableModelError(err) {
		slog.Info("llm fallback model engaged (stream error)",
			"agent", r.AgentID, "primary", req.Model, "fallback", r.FallbackModel, "err", err.Error())
		req.Model = r.FallbackModel
		if s, e := r.LLM.ChatStream(ctx, *req); e == nil {
			return s, true
		}
	}
	return nil, false
}

// RunSync is a convenience method that runs the agent and collects the
// full text response.
func (r *Runtime) RunSync(ctx context.Context, userMsg string, images []llm.ImageContent) (string, error) {
	events, err := r.Run(ctx, userMsg, images)
	if err != nil {
		return "", err
	}

	var response strings.Builder
	for event := range events {
		switch event.Type {
		case EventTextDelta:
			response.WriteString(event.Text)
		case EventAborted:
			return response.String(), context.Canceled
		case EventError:
			return response.String(), event.Error
		}
	}

	return response.String(), nil
}

// dispatchTool executes one tool call with strict tool_use ↔ tool_result
// pairing. Always appends a ToolCallEntry then exactly one ToolResultEntry
// (real, error, denial, or aborted) before returning.
func (r *Runtime) dispatchTool(
	ctx context.Context,
	tc llm.ToolCall,
	kgThread *[]Message,
) (result tool.ToolResult, aborted bool) {
	r.Session.Append(session.ToolCallEntry(tc.ID, tc.Name, tc.Input))
	if kgThread != nil {
		r.kgMu.Lock()
		*kgThread = append(*kgThread, Message{
			Role:    "assistant",
			Content: fmt.Sprintf("[tool: %s]\n%s", tc.Name, string(tc.Input)),
		})
		r.kgMu.Unlock()
	}

	// BeforeToolUse fires before the PermissionChecker. A hook denial
	// has the same observable shape as a Permission.Check denial.
	if hook := r.AgentLoop.Hooks.BeforeToolUse; hook != nil {
		d, err := hook(ctx, tc.Name, tc.Input)
		if err != nil {
			res := r.appendDenialResult(tc.ID, err.Error(), kgThread)
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
		if !d.Allow {
			res := r.appendDenialResult(tc.ID, d.Reason, kgThread)
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
	}

	if r.Permission != nil {
		if d := r.Permission.Check(ctx, r.AgentID, tc.Name, tc.Input); d.Behavior == tool.DecisionDeny {
			res := r.appendDenialResult(tc.ID, d.Reason, kgThread)
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
	}

	if ctx.Err() != nil {
		return r.appendAbortedResult(tc.ID, kgThread), true
	}

	result, err := r.Tools.Execute(ctx, tc.Name, tc.Input)
	if err != nil {
		result = tool.ToolResult{Error: err.Error()}
	}

	if ctx.Err() != nil {
		return r.appendAbortedResult(tc.ID, kgThread), true
	}

	imgData := convertToolResultImages(result.Images)
	r.Session.Append(session.ToolResultEntry(tc.ID, result.Output, result.Error, imgData))
	if kgThread != nil {
		content := result.Output
		if result.Error != "" {
			content = "[error] " + result.Error
		}
		r.kgMu.Lock()
		*kgThread = append(*kgThread, Message{Role: "user", Content: content})
		r.kgMu.Unlock()
	}

	if result.Error == "" && isFileTool(tc.Name) {
		r.recordFileTouch(extractPathFromInput(tc.Input))
	}

	if hook := r.AgentLoop.Hooks.AfterToolUse; hook != nil {
		hook(ctx, tc.Name, tc.Input, result)
	}

	return result, false
}

// executeToolKickoff runs the tool's permission gate, ctx checks, and
// Execute call WITHOUT touching session or KG thread. Used by the
// streaming kickoff goroutines.
func (r *Runtime) executeToolKickoff(ctx context.Context, tc llm.ToolCall) (result tool.ToolResult, aborted bool) {
	if hook := r.AgentLoop.Hooks.BeforeToolUse; hook != nil {
		d, err := hook(ctx, tc.Name, tc.Input)
		if err != nil {
			res := tool.ToolResult{Error: err.Error()}
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
		if !d.Allow {
			res := tool.ToolResult{Error: d.Reason}
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
	}
	if r.Permission != nil {
		if d := r.Permission.Check(ctx, r.AgentID, tc.Name, tc.Input); d.Behavior == tool.DecisionDeny {
			res := tool.ToolResult{Error: d.Reason}
			if after := r.AgentLoop.Hooks.AfterToolUse; after != nil {
				after(ctx, tc.Name, tc.Input, res)
			}
			return res, false
		}
	}
	if ctx.Err() != nil {
		return tool.ToolResult{Error: "aborted by user"}, true
	}
	result, err := r.Tools.Execute(ctx, tc.Name, tc.Input)
	if err != nil {
		result = tool.ToolResult{Error: err.Error()}
	}
	if ctx.Err() != nil {
		return tool.ToolResult{Error: "aborted by user"}, true
	}
	if result.Error == "" && isFileTool(tc.Name) {
		r.recordFileTouch(extractPathFromInput(tc.Input))
	}
	if hook := r.AgentLoop.Hooks.AfterToolUse; hook != nil {
		hook(ctx, tc.Name, tc.Input, result)
	}
	return result, false
}

// appendDenialResult writes the tool-result entry for a denied tool call.
func (r *Runtime) appendDenialResult(toolCallID, reason string, kgThread *[]Message) tool.ToolResult {
	r.Session.Append(session.ToolResultEntry(toolCallID, "", reason, nil))
	if kgThread != nil {
		r.kgMu.Lock()
		*kgThread = append(*kgThread, Message{
			Role: "user", Content: "[error] " + reason,
		})
		r.kgMu.Unlock()
	}
	return tool.ToolResult{Error: reason}
}

// appendAbortedResult writes the synthetic abort entry.
func (r *Runtime) appendAbortedResult(toolCallID string, kgThread *[]Message) tool.ToolResult {
	r.Session.Append(session.AbortedToolResultEntry(toolCallID))
	if kgThread != nil {
		r.kgMu.Lock()
		*kgThread = append(*kgThread, Message{
			Role: "user", Content: "[error] aborted by user",
		})
		r.kgMu.Unlock()
	}
	return tool.ToolResult{Error: "aborted by user"}
}

// convertToolResultImages adapts tool image attachments to session ImageData.
func convertToolResultImages(imgs []llm.ImageContent) []session.ImageData {
	if len(imgs) == 0 {
		return nil
	}
	out := make([]session.ImageData, 0, len(imgs))
	for _, img := range imgs {
		out = append(out, session.ImageData{
			MimeType: img.MimeType,
			Data:     base64.StdEncoding.EncodeToString(img.Data),
		})
	}
	return out
}

// kgThreadOrNil returns a pointer to the KG thread if KG is enabled, else nil.
func kgThreadOrNil(kg KnowledgeGraph, thread *[]Message) *[]Message {
	if kg == nil {
		return nil
	}
	return thread
}
