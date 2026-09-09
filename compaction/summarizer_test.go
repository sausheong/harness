package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
)

// fakeProvider is an llm.LLMProvider stub that emits a fixed text response.
type fakeProvider struct {
	llmtest.Base
	text string
	err  error
}

func (f *fakeProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	if f.err != nil {
		return nil, f.err
	}
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: f.text}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

func TestSummarizerReturnsModelOutput(t *testing.T) {
	s := &Summarizer{
		Provider: &fakeProvider{text: "we picked option B for X."},
		Model:    "qwen2.5:3b-instruct",
		Timeout:  5 * time.Second,
	}
	entries := []session.SessionEntry{session.UserMessageEntry("hello")}
	got, err := s.Summarize(context.Background(), entries, "")
	require.NoError(t, err)
	assert.Equal(t, "we picked option B for X.", got)
}

func TestSummarizerTrimsWhitespace(t *testing.T) {
	s := &Summarizer{
		Provider: &fakeProvider{text: "  \n  summary text  \n  "},
		Model:    "qwen2.5:3b-instruct",
		Timeout:  5 * time.Second,
	}
	got, err := s.Summarize(context.Background(), []session.SessionEntry{session.UserMessageEntry("hi")}, "")
	require.NoError(t, err)
	assert.Equal(t, "summary text", got)
}

func TestSummarizerEmptyResponseReturnsError(t *testing.T) {
	s := &Summarizer{
		Provider: &fakeProvider{text: "   \n  "},
		Model:    "qwen2.5:3b-instruct",
		Timeout:  5 * time.Second,
	}
	got, err := s.Summarize(context.Background(), []session.SessionEntry{session.UserMessageEntry("hi")}, "")
	require.Error(t, err)
	assert.Empty(t, got)
}

func TestSummarizerProviderErrorReturnsError(t *testing.T) {
	s := &Summarizer{
		Provider: &fakeProvider{err: errors.New("ollama down")},
		Model:    "qwen2.5:3b-instruct",
		Timeout:  5 * time.Second,
	}
	got, err := s.Summarize(context.Background(), []session.SessionEntry{session.UserMessageEntry("hi")}, "")
	require.Error(t, err)
	assert.Empty(t, got)
}

// flakyProvider returns ChatStream errors a configured number of times,
// then succeeds. Used to exercise the fallback chain.
type flakyProvider struct {
	llmtest.Base
	failsRemaining int
	successText    string
	failureErr     error
}

func (f *flakyProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	if f.failsRemaining > 0 {
		f.failsRemaining--
		ch := make(chan llm.ChatEvent, 2)
		go func() {
			defer close(ch)
			err := f.failureErr
			if err == nil {
				err = errors.New("input is too long")
			}
			ch <- llm.ChatEvent{Type: llm.EventError, Error: err}
		}()
		return ch, nil
	}
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: f.successText}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

func TestSummarizeFallbackFullStageSucceeds(t *testing.T) {
	s := &Summarizer{
		Provider: &fakeProvider{text: "stage 1 summary"},
		Model:    "m",
		Timeout:  time.Second,
	}
	entries := []session.SessionEntry{session.UserMessageEntry("hi")}
	got, err := s.Summarize(context.Background(), entries, "")
	require.NoError(t, err)
	assert.Contains(t, got, "stage 1 summary")
}

func TestSummarizeFallbackToSmallOnlyOnOverflow(t *testing.T) {
	huge := strings.Repeat("X", 50000)
	entries := []session.SessionEntry{
		session.UserMessageEntry("small 1"),
		session.AssistantMessageEntry(huge),
		session.UserMessageEntry("small 2"),
	}
	s := &Summarizer{
		Provider: &flakyProvider{failsRemaining: 1, successText: "stage 2 summary"},
		Model:    "m",
		Timeout:  time.Second,
	}
	got, err := s.Summarize(context.Background(), entries, "")
	require.NoError(t, err)
	assert.Contains(t, got, "stage 2 summary",
		"second-stage success must produce the summary")
}

func TestSummarizeReturnsErrorWhenAllStagesFail(t *testing.T) {
	entries := []session.SessionEntry{session.UserMessageEntry("hi")}
	s := &Summarizer{
		Provider: &flakyProvider{failsRemaining: 99, successText: ""},
		Model:    "m",
		Timeout:  time.Second,
	}
	got, err := s.Summarize(context.Background(), entries, "")
	require.Error(t, err)
	assert.Empty(t, got)
}

func TestSummarizePerCallTimeoutPropagates(t *testing.T) {
	// A per-call timeout (Summarizer.Timeout shorter than provider delay)
	// must surface to the caller as context.DeadlineExceeded — NOT silently
	// degrade to a placeholder. The Manager classifies this as "timeout"
	// for telemetry; the placeholder path would mask that as a successful
	// stub.
	s := &Summarizer{
		Provider: &delayedProvider{
			text:    "never reached",
			delay:   500 * time.Millisecond,
			started: make(chan struct{}),
		},
		Model:   "m",
		Timeout: 50 * time.Millisecond,
	}
	entries := []session.SessionEntry{session.UserMessageEntry("hi")}
	_, err := s.Summarize(context.Background(), entries, "")
	require.Error(t, err, "per-call timeout must propagate, not be swallowed by placeholder")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error must be (or wrap) context.DeadlineExceeded; got %v", err)
}
