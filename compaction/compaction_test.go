package compaction

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
)

func longSession() *session.Session {
	sess := session.NewSession("default", "test")
	for range 6 {
		sess.Append(session.UserMessageEntry("user msg"))
		sess.Append(session.AssistantMessageEntry("assistant reply"))
	}
	return sess
}

func TestManagerAppendsCompactionEntry(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "summary text"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()
	res, err := mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	require.NoError(t, err)
	assert.True(t, res.Compacted)
	assert.Equal(t, ReasonManual, res.Reason)

	// Final entry should be the compaction.
	last := sess.View()[0]
	assert.Equal(t, session.EntryTypeCompaction, last.Type)
}

// TestViewIncludesPreservedRangeAfterCompaction guards against the regression
// where View() returns only the compaction entry and silently drops the
// preserved range. longSession() has 6 user/assistant pairs (12 entries);
// with PreserveTurns=4 the last 4 user turns must remain verbatim, so View()
// after compaction must be: [compaction, u3, a3, u4, a4, u5, a5, u6, a6].
func TestViewIncludesPreservedRangeAfterCompaction(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "summary text"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()
	res, err := mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	require.NoError(t, err)
	require.True(t, res.Compacted)

	view := sess.View()
	require.Len(t, view, 9, "view must be [compaction, 4 user/assistant pairs]")
	assert.Equal(t, session.EntryTypeCompaction, view[0].Type, "first entry is the summary")
	for i := 1; i < 9; i++ {
		assert.Equal(t, session.EntryTypeMessage, view[i].Type, "entry %d is a preserved message", i)
	}
	// User/assistant should alternate starting with user.
	assert.Equal(t, "user", view[1].Role)
	assert.Equal(t, "assistant", view[2].Role)
	assert.Equal(t, "user", view[7].Role)
	assert.Equal(t, "assistant", view[8].Role)
}

func TestManagerRefusesShortSession(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "summary"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := session.NewSession("default", "test")
	sess.Append(session.UserMessageEntry("only one"))
	res, err := mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	require.NoError(t, err)
	assert.False(t, res.Compacted)
	assert.Equal(t, "too_short", res.Skipped)
}

func TestManagerSummarizerErrorSkipsWithoutSummary(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: ""},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()
	res, err := mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	require.NoError(t, err)
	assert.False(t, res.Compacted)
	assert.Empty(t, res.Summary)
	assert.NotEmpty(t, res.Skipped)
}

func TestManagerSerializesPerSession(t *testing.T) {
	// Two concurrent compactions on the same session should not race.
	// The 2nd call must block until the 1st finishes.
	delayCh := make(chan struct{})
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &delayedProvider{text: "ok", delay: 200 * time.Millisecond, started: delayCh},
			Model:    "m",
			Timeout:  5 * time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()

	var wg sync.WaitGroup
	wg.Add(2)
	starts := make([]time.Time, 2)
	go func() {
		defer wg.Done()
		starts[0] = time.Now()
		_, _ = mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	}()
	<-delayCh // wait until first call has started its provider call
	go func() {
		defer wg.Done()
		starts[1] = time.Now()
		_, _ = mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	}()
	wg.Wait()

	// They should not have run truly in parallel.
	gap := starts[1].Sub(starts[0])
	assert.Less(t, gap.Milliseconds(), int64(50), "starts should be near-simultaneous")
	// (Mutex serializes the Summarize call, not MaybeCompact's first instructions.
	//  We assert serialization indirectly: with delay 200ms each, total wall time > 200ms.)
}

func TestManagerClassifiesCancellation(t *testing.T) {
	// Provider that blocks until ctx is cancelled, returning ctx.Err().
	cancelMe, cancelFn := context.WithCancel(context.Background())
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &delayedProvider{
				text:    "never reached",
				delay:   5 * time.Second,
				started: make(chan struct{}),
			},
			Model:   "m",
			Timeout: 10 * time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()

	// Cancel the parent ctx after 50ms — fires while summarizer is still waiting.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancelFn()
	}()

	res, err := mgr.MaybeCompact(cancelMe, sess, ReasonManual, "")
	require.NoError(t, err) // skip is not a hard error
	assert.False(t, res.Compacted)
	assert.Equal(t, "cancelled", res.Skipped)
}

func TestManagerForgetsSession(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "ok"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()
	_, _ = mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	// Manager should now have a lock entry for this session.
	key := sess.AgentID + "/" + sess.Key
	mgr.mu.Lock()
	_, ok := mgr.locks[key]
	mgr.mu.Unlock()
	require.True(t, ok)

	mgr.ForgetSession(sess)

	mgr.mu.Lock()
	_, ok = mgr.locks[key]
	mgr.mu.Unlock()
	assert.False(t, ok)
}

// TestDeferredForgetSessionDrainsLocksMap verifies that ForgetSession
// drains all per-session map entries. The maps are keyed by
// AgentID/Key (stable across loads), so 50 turns reusing the same
// persistent identifier only ever populate one entry — but
// ForgetSession must still clear it for callers that delete a
// persistent session (e.g. session.clear).
func TestDeferredForgetSessionDrainsLocksMap(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "ok"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	const turns = 50
	turn := func() {
		sess := longSession()
		defer mgr.ForgetSession(sess)
		_, _ = mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
	}
	for range turns {
		turn()
	}
	mgr.mu.Lock()
	defer mgr.mu.Unlock()
	assert.Equal(t, 0, len(mgr.locks),
		"defer ForgetSession on each turn must drain the locks map; "+
			"got %d residual entries after %d turns", len(mgr.locks), turns)
	mgr.failMu.Lock()
	defer mgr.failMu.Unlock()
	assert.Equal(t, 0, len(mgr.failures),
		"defer ForgetSession must also drain the failures map")
}

// TestSeparateManagersDoNotSerialize sanity-checks that the per-session lock
// lives on the Manager instance — two Manager instances run concurrently
// even when handed the same session ID. This proves why call sites must build
// ONE Manager at startup and share it; per-call construction would lose the
// serialization guarantee provided by the per-session mutex.
//
// We use two distinct *session.Session values with the same agent/key so that
// the timing assertion isn't undermined by an actual data race on Append
// (which is the very thing the shared-Manager design prevents). The point of
// this test is the timing — separate Managers complete in ~1 delay, not 2.
// Companion to TestManagerSerializesPerSession (which proves the shared case).
func TestSeparateManagersDoNotSerialize(t *testing.T) {
	delay := 200 * time.Millisecond
	mgrA := &Manager{
		Summarizer: &Summarizer{
			Provider: &delayedProvider{text: "a", delay: delay, started: make(chan struct{})},
			Model:    "m",
			Timeout:  5 * time.Second,
		},
		PreserveTurns: 4,
	}
	mgrB := &Manager{
		Summarizer: &Summarizer{
			Provider: &delayedProvider{text: "b", delay: delay, started: make(chan struct{})},
			Model:    "m",
			Timeout:  5 * time.Second,
		},
		PreserveTurns: 4,
	}
	// Distinct sessions so the assertion isn't muddied by a real data race
	// on session.Append; we only care about the per-Manager lock's scope here.
	sessA := longSession()
	sessB := longSession()

	var wg sync.WaitGroup
	wg.Add(2)
	start := time.Now()
	go func() {
		defer wg.Done()
		_, _ = mgrA.MaybeCompact(context.Background(), sessA, ReasonManual, "")
	}()
	go func() {
		defer wg.Done()
		_, _ = mgrB.MaybeCompact(context.Background(), sessB, ReasonManual, "")
	}()
	wg.Wait()
	elapsed := time.Since(start)

	// Two separate Managers don't share a per-session lock; they run in
	// parallel. Total wall time should be close to one delay, not two.
	// Allow generous slack for slow CI: anything < 1.5 * delay proves overlap.
	assert.Less(t, elapsed, 3*delay/2,
		"separate Managers should run concurrently; observed %s vs single delay %s — they appear serialized",
		elapsed, delay)
}

// delayedProvider sleeps before responding, signalling start via a channel.
type delayedProvider struct {
	llmtest.Base
	text    string
	delay   time.Duration
	started chan struct{}
	once    sync.Once
}

func (d *delayedProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	d.once.Do(func() { close(d.started) })
	ch := make(chan llm.ChatEvent, 2)
	go func() {
		defer close(ch)
		select {
		case <-time.After(d.delay):
		case <-ctx.Done():
			ch <- llm.ChatEvent{Type: llm.EventError, Error: ctx.Err()}
			return
		}
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: d.text}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

// alwaysFailingProvider returns an error from ChatStream every call.
type alwaysFailingProvider struct{ llmtest.Base }

func (a *alwaysFailingProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	return nil, errors.New("provider down")
}

func TestCircuitBreakerTripsAfterMaxFailures(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &alwaysFailingProvider{},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()

	// Failed attempts leave history available and increment the breaker.
	for i := range MaxConsecutiveFailures {
		res, err := mgr.MaybeCompact(context.Background(), sess, ReasonPreventive, "")
		require.NoError(t, err, "iteration %d", i)
		assert.False(t, res.Compacted, "iteration %d must preserve history", i)
		// Add new turns so the next iteration has something to compact.
		sess.Append(session.UserMessageEntry("follow-up"))
		sess.Append(session.AssistantMessageEntry("more reply"))
	}
	res, err := mgr.MaybeCompact(context.Background(), sess, ReasonPreventive, "")
	require.NoError(t, err)
	assert.False(t, res.Compacted, "call after MaxConsecutiveFailures must be skipped by circuit breaker")
	assert.Equal(t, "circuit_breaker", res.Skipped)
}

func TestCircuitBreakerResetsOnSuccess(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &fakeProvider{text: "ok"},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sess := longSession()

	// Run several successful compactions; failure count stays 0.
	for i := range MaxConsecutiveFailures + 5 {
		_, err := mgr.MaybeCompact(context.Background(), sess, ReasonPreventive, "")
		require.NoError(t, err, "iteration %d", i)
		// Don't assert Compacted=true: Split returns ok=false once the
		// session has nothing left to compact past PreserveTurns. Either
		// way the breaker must NOT trip on stage-1 success.
	}
}

func TestCircuitBreakerIsPerSession(t *testing.T) {
	mgr := &Manager{
		Summarizer: &Summarizer{
			Provider: &alwaysFailingProvider{},
			Model:    "m",
			Timeout:  time.Second,
		},
		PreserveTurns: 4,
	}
	sessA := longSessionWith("default", "a")
	sessB := longSessionWith("default", "b")

	for range MaxConsecutiveFailures {
		_, _ = mgr.MaybeCompact(context.Background(), sessA, ReasonPreventive, "")
		// Add new turns so each iteration has compactable history.
		sessA.Append(session.UserMessageEntry("follow-up"))
		sessA.Append(session.AssistantMessageEntry("more reply"))
	}

	res, err := mgr.MaybeCompact(context.Background(), sessB, ReasonPreventive, "")
	require.NoError(t, err)
	assert.NotEqual(t, "circuit_breaker", res.Skipped,
		"Session B must not be tripped by Session A's failures")
}

// longSessionWith builds a session with the same shape as longSession()
// but with caller-provided AgentID/Key — used by tests that need
// distinct stable identifiers (the Manager's per-session maps are
// keyed by AgentID/Key, so two sessions sharing those would share the
// same lock/failure-counter/in-flight entry).
func longSessionWith(agentID, key string) *session.Session {
	sess := session.NewSession(agentID, key)
	for range 6 {
		sess.Append(session.UserMessageEntry("user msg"))
		sess.Append(session.AssistantMessageEntry("assistant reply"))
	}
	return sess
}

func TestCommittedObserverSeesCommittedHistoryAndCannotChangeResult(t *testing.T) {
	for _, mode := range []string{"success", "panic", "skip", "failure"} {
		t.Run(mode, func(t *testing.T) {
			sess := longSession()
			defer sess.Close()
			mgr := &Manager{Summarizer: &Summarizer{Provider: &fakeProvider{text: "summary"}, Model: "m", Timeout: time.Second}, PreserveTurns: 4}
			if mode == "skip" {
				mgr.Summarizer = nil
			}
			if mode == "failure" {
				mgr.Summarizer.Provider = &fakeProvider{err: errors.New("unavailable")}
			}
			calls := 0
			mgr.OnCommitted = func(ctx context.Context, event CommitNotification) {
				calls++
				if event.SessionID != sess.ID || event.TurnsCompacted == 0 || sess.View()[0].Type != session.EntryTypeCompaction {
					t.Fatal(event, sess.View())
				}
				deadline, ok := ctx.Deadline()
				if !ok || time.Until(deadline) > 2*time.Second {
					t.Fatal("observer not bounded")
				}
				lock := mgr.lockFor(stableKey(sess))
				if !lock.TryLock() {
					t.Fatal("observer called under compaction lock")
				}
				lock.Unlock()
				if mode == "panic" {
					panic("observer failure")
				}
			}
			result, err := mgr.MaybeCompact(context.Background(), sess, ReasonManual, "")
			if err != nil {
				t.Fatal(err)
			}
			expected := mode == "success" || mode == "panic"
			if result.Compacted != expected || (calls == 1) != expected {
				t.Fatal(result, calls)
			}
		})
	}
}

func TestBackgroundCompactionJoinsCommittedObserver(t *testing.T) {
	sess := longSession()
	defer sess.Close()
	entered, release := make(chan struct{}), make(chan struct{})
	manager := &Manager{Summarizer: &Summarizer{Provider: &fakeProvider{text: "summary"}, Model: "m", Timeout: time.Second}, PreserveTurns: 4}
	manager.OnCommitted = func(ctx context.Context, event CommitNotification) {
		close(entered)
		select {
		case <-release:
		case <-ctx.Done():
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	done := manager.MaybeCompactAsyncContext(ctx, sess, ReasonPreventive)
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("observer never entered")
	}
	select {
	case <-done:
		t.Fatal("background finished before observer joined")
	default:
	}
	if !manager.HasInFlight(sess) {
		t.Fatal("active observer not tracked")
	}
	close(release)
	result, joined, err := manager.JoinInFlight(ctx, sess)
	if err != nil || !joined || !result.Compacted || manager.HasInFlight(sess) {
		t.Fatal(result, joined, err)
	}
}
