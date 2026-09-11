package compaction

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

// ErrEmptySummary is returned when the LLM emits no usable summary text.
var ErrEmptySummary = errors.New("compaction: empty summary returned")

// Summarizer wraps an llm.LLMProvider with the prompt and call shape used
// for compaction. The provider is expected to be the bundled Ollama in
// production but any LLMProvider works (used for tests).
type Summarizer struct {
	Route llm.CallRoute // identifies the independently selected provider client

	Provider        llm.LLMProvider
	MaxOutputTokens int           // 0 defaults to 4096; otherwise 1-32768
	Model           string        // bare model id, e.g. "qwen2.5:3b-instruct"
	Timeout         time.Duration // per-call deadline; 0 → 60s
}

// Summarize sends entries through the configured provider and returns the
// trimmed, formatted summary text. additionalInstructions is appended to
// the prompt when non-empty (used by manual /compact <focus...>).
//
// A full transcript is attempted first. Overflow/stream failures may retry
// with oversized messages explicitly elided. If neither attempt produces a
// summary, return the error so the manager preserves effective history.
func (s *Summarizer) Summarize(ctx context.Context, entries []session.SessionEntry, additionalInstructions string) (string, error) {
	return s.summarizeWithFallback(ctx, entries, additionalInstructions)
}

func (s *Summarizer) summarizeWithFallback(ctx context.Context, entries []session.SessionEntry, additionalInstructions string) (string, error) {
	// Stage 1: full transcript.
	out, err := s.callOnce(ctx, BuildTranscript(entries), additionalInstructions)
	if err == nil && out != "" {
		return out, nil
	}
	stage1Err := err

	// Cancellation and per-call deadlines stop retries and retain their
	// classification for the manager.
	if ctxErr := ctx.Err(); ctxErr != nil || errors.Is(stage1Err, context.DeadlineExceeded) {
		return "", stage1Err
	}

	// Stage 2: drop oversized messages and retry. Only meaningful when
	// buildSmallOnlyTranscript actually elides something — otherwise we'd
	// be re-sending the same transcript with the same prompt against the
	// same provider (predictable failure, wasted call). Fall through to
	// the final error in that case.
	if isOverflowError(stage1Err) || isStreamError(stage1Err) {
		small, droppedCount := buildSmallOnlyTranscript(entries)
		if droppedCount > 0 {
			small += fmt.Sprintf("\n[oversized message(s) elided: %d]\n", droppedCount)
			out, err = s.callOnce(ctx, small, additionalInstructions)
			if err == nil && out != "" {
				return out, nil
			}
		}
	}

	if failure := errors.Join(stage1Err, err); failure != nil {
		return "", failure
	}
	return "", ErrEmptySummary
}

// callOnce performs a single summarizer invocation against a pre-built
// transcript. Returns the formatted summary text or an error.
//
// Prompt-cache strategy: the static instruction header goes into a
// cache-marked SystemPromptPart so Anthropic caches the ~2 KB prefix.
// The transcript + per-call focus go into the user message. The first
// compaction pays cache_creation; every subsequent compaction in the
// 5-minute TTL window hits cache_read for the prefix, dropping a few
// seconds off TTFT.
func (s *Summarizer) callOnce(ctx context.Context, transcript, additionalInstructions string) (string, error) {
	maxTokens := s.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	if maxTokens < 1 || maxTokens > 32768 {
		return "", errors.New("summariser output token limit must be 1-32768")
	}
	if s.Provider == nil {
		return "", errors.New("summariser provider unavailable")
	}
	systemPrompt, userMessage := BuildPromptParts(transcript, additionalInstructions)
	timeout := s.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req := llm.ChatRequest{
		Route:    s.Route,
		Model:    s.Model,
		Messages: []llm.Message{{Role: "user", Content: userMessage}},
		// One cache-marked system part: providers that support caching
		// (Anthropic) emit a cache marker on it; the OpenAI/Gemini/Qwen
		// path concatenates the Text fields and ignores Cache.
		SystemPromptParts: []llm.SystemPromptPart{
			{Text: systemPrompt, Cache: true},
		},
		MaxTokens: maxTokens,
	}
	stream, err := llm.ObserveChat(callCtx, req, llm.CallCompaction, s.Provider.ChatStream)
	if err != nil {
		return "", fmt.Errorf("compaction: chat stream: %w", err)
	}

	var sb strings.Builder
	var terminalReason string
	for ev := range stream {
		switch ev.Type {
		case llm.EventTextDelta:
			if len(ev.Text) > 256<<10-sb.Len() {
				return "", errors.New("summariser response exceeds 256 KiB")
			}
			sb.WriteString(ev.Text)
		case llm.EventDone:
			terminalReason = ev.StopReason
		case llm.EventError:
			return "", fmt.Errorf("compaction: stream error: %w", ev.Error)
		}
	}
	if err := callCtx.Err(); err != nil {
		return "", err
	}
	if terminalReason == "length" || terminalReason == "max_tokens" {
		return "", errors.New("summariser reached its output limit; original history retained")
	}
	out := strings.TrimSpace(sb.String())
	if out == "" {
		return "", ErrEmptySummary
	}
	out = FormatCompactSummary(out)
	if out == "" {
		return "", ErrEmptySummary
	}
	return out, nil
}

// maxSmallEntryLen is the per-entry size threshold for stage 2's
// drop-the-oversized-stuff transcript. Entries larger than this are
// elided. The threshold tracks maxTranscriptToolResultLen so a single
// hot-path tool result doesn't get re-elided here — only entries that
// blow far past the per-result cap are dropped.
const maxSmallEntryLen = maxTranscriptToolResultLen

// buildSmallOnlyTranscript renders entries while skipping any single-entry
// payload larger than maxSmallEntryLen. Returns the transcript and the
// count of dropped entries so the caller can append a note.
func buildSmallOnlyTranscript(entries []session.SessionEntry) (string, int) {
	var dropped int
	kept := make([]session.SessionEntry, 0, len(entries))
	for _, e := range entries {
		if len(e.Data) > maxSmallEntryLen {
			dropped++
			continue
		}
		kept = append(kept, e)
	}
	return BuildTranscript(kept), dropped
}

// isOverflowError reports whether err looks like a "your prompt is too big"
// signal from any provider. Re-uses the overflow signature list rather
// than depending on package-level state.
func isOverflowError(err error) bool {
	return err != nil && IsContextOverflow(err)
}

// isStreamError reports whether err originated from the streaming layer
// (vs being a wrapper of context.DeadlineExceeded etc.). Used to decide
// whether the second stage is worth attempting.
func isStreamError(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(err.Error(), "stream error")
}
