package compaction

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/sausheong/harness/session"
)

// MaxConsecutiveFailures is the per-session circuit-breaker threshold.
// After this many consecutive autocompact attempts that drop to the
// placeholder stage (stage 3), MaybeCompact stops attempting compaction
// for the session and returns Skipped="circuit_breaker".
//
// The breaker resets on any genuine summarizer success (stage 1 or 2
// returning real content). It exists to prevent a session whose context
// is irrecoverably over the limit from hammering the API on every turn.
const MaxConsecutiveFailures = 3

// Reason identifies why compaction was triggered.
type Reason string

const (
	ReasonPreventive Reason = "preventive"
	ReasonReactive   Reason = "reactive"
	ReasonManual     Reason = "manual"
)

// Result describes the outcome of a MaybeCompact call. When Compacted is
// false, Skipped names the reason ("too_short", "empty_summary",
// "ollama_down", "model_missing", "timeout", "summarizer_error").
type Result struct {
	Compacted      bool
	Reason         Reason
	Skipped        string
	TurnsCompacted int
	TokensBefore   int
	TokensAfter    int
	Summary        string
	DurationMs     int64
}

// Manager orchestrates compaction for sessions. One Manager is shared across
// the whole agent runtime; it tracks per-session mutexes internally.
type Manager struct {
	Summarizer    *Summarizer
	PreserveTurns int     // K; default 4 if zero
	Threshold     float64 // fraction of context window that triggers preventive compaction (e.g. 0.6); 0 means use caller default
	// MessageCap is a hard backstop on total message count before compaction
	// fires, regardless of token threshold. 0 disables the cap. See
	// CompactionConfig.MessageCap for the rationale.
	MessageCap int

	mu    sync.Mutex             // guards locks map
	locks map[string]*sync.Mutex // session.ID → mutex

	failMu   sync.Mutex     // guards failures map; separate from mu so the breaker check (called before lockFor) doesn't serialize on the per-session lock-map allocator
	failures map[string]int // session.ID → consecutive-failure count

	// inFlightMu guards the inFlight map. Separate from the other
	// mutexes so awaiting an in-flight compaction (WaitForInFlight)
	// never serializes on the per-session lock-map or failure-counter.
	inFlightMu sync.Mutex
	inFlight   map[string]*inFlightCompaction // session.ID → in-flight handle
}

// inFlightCompaction tracks a background compaction goroutine spawned
// via MaybeCompactAsync. The done channel is closed when the goroutine
// returns; result holds the outcome for any waiter that wants it.
type inFlightCompaction struct {
	done   chan struct{}
	result Result
	err    error
}

// MaybeCompact runs a compaction pass on sess if the session has more than
// K user turns. It is safe to call concurrently from multiple goroutines on
// the same session; calls serialize per-session.
//
// Errors are returned only for true unexpected failures. Routine "skip"
// outcomes (too short, empty summary, provider error) come back via
// Result.Skipped with err == nil so callers can treat them uniformly.
//
// Note: MaybeCompact holds the per-session mutex for the entire summarizer
// call (default 60s timeout). A second concurrent call on the same session
// will block until the first completes. This is intentional — it prevents
// two compactions from racing on session.Append — but callers triggering
// manual compactions while a preventive one is in flight should expect a wait.
func (m *Manager) MaybeCompact(ctx context.Context, sess *session.Session, reason Reason, instructions string) (Result, error) {
	if m == nil || m.Summarizer == nil {
		return Result{Reason: reason, Skipped: "no_summarizer"}, nil
	}

	key := stableKey(sess)
	if fc := m.failureCount(key); fc >= MaxConsecutiveFailures {
		slog.Info("compaction skipped",
			"session_id", sess.ID,
			"reason", string(reason),
			"skipped", "circuit_breaker",
			"consecutive_failures", fc)
		return Result{Reason: reason, Skipped: "circuit_breaker"}, nil
	}

	K := m.PreserveTurns
	if K <= 0 {
		K = 4
	}

	mu := m.lockFor(key)
	mu.Lock()
	defer mu.Unlock()

	start := time.Now()
	view := sess.View()
	toCompact, _, ok := Split(view, K)
	if !ok {
		slog.Debug("compaction skipped", "session_id", sess.ID, "reason", string(reason), "skipped", "too_short")
		return Result{Reason: reason, Skipped: "too_short"}, nil
	}

	slog.Info("compaction triggered", "session_id", sess.ID, "reason", string(reason))

	summary, err := m.Summarizer.Summarize(ctx, toCompact, instructions)
	if err != nil {
		skipReason := classifySummarizerError(err)
		slog.Warn("compaction skipped", "session_id", sess.ID, "reason", string(reason), "skipped", skipReason, "detail", err.Error())
		// A hard error from Summarize means even stage 3 didn't run.
		// Count it as a failure for breaker accounting.
		m.incrementFailure(key)
		return Result{Reason: reason, Skipped: skipReason}, nil
	}

	// summarizeWithFallback's stage 3 returns a placeholder summary
	// (no error) when both stage 1 and stage 2 failed. We detect
	// placeholders by their stable marker phrase and treat them as
	// failures for breaker accounting; real (stage-1 or stage-2)
	// summaries reset the counter.
	isPlaceholder := strings.Contains(summary, "compaction failed and the summary could not be generated")
	if isPlaceholder {
		m.incrementFailure(key)
	} else {
		m.resetFailures(key)
	}

	first := toCompact[0]
	last := toCompact[len(toCompact)-1]
	_, toPreserve, _ := Split(view, K)
	entry := session.CompactionEntry(summary, first.ID, last.ID, m.Summarizer.Model, 0, 0, len(toCompact))
	// Splice the compaction entry between the to-be-compacted range and the
	// preserved range so View()'s walk-back from leaf hits: leaf → ... →
	// preserved[0] → compaction → STOP. Without re-linking, Append would put
	// compaction at the leaf and View() would terminate on it immediately,
	// silently dropping every preserved turn.
	entry.ParentID = toPreserve[0].ParentID
	sess.Append(entry)
	for i, e := range toPreserve {
		if i == 0 {
			e.ParentID = entry.ID
		}
		// Re-append with same ID — Session.Append's entryMap overwrite and
		// the loader's last-write-wins behaviour both make this safe.
		sess.Append(e)
	}

	dur := time.Since(start).Milliseconds()
	slog.Info("compaction complete", "session_id", sess.ID, "reason", string(reason), "turns_compacted", len(toCompact), "duration_ms", dur)

	return Result{
		Compacted:      true,
		Reason:         reason,
		TurnsCompacted: len(toCompact),
		Summary:        summary,
		DurationMs:     dur,
	}, nil
}

// MaybeCompactAsync starts a compaction goroutine for sess if one is
// not already in flight for that session ID, then returns immediately.
// The returned channel closes when the goroutine completes — most
// callers ignore it; the existence is what matters (WaitForInFlight
// looks it up via session ID).
//
// Designed for the "between turns" pattern: at the end of a chat turn
// the runtime checks if the session is approaching the compaction
// threshold; if so, it calls MaybeCompactAsync and returns. The next
// chat turn calls WaitForInFlight at the top of its loop and either
// finds the work already done (zero added latency) or briefly waits
// for the in-flight goroutine to finish.
//
// No-op when m is nil, the session already has an in-flight
// compaction, or the manager has no Summarizer. The goroutine uses a
// fresh background context bounded by 2× the summarizer timeout (long
// enough for the three-stage fallback path) so that a returning
// caller's cancellation can't kill the in-flight summary.
func (m *Manager) MaybeCompactAsync(sess *session.Session, reason Reason) <-chan struct{} {
	if m == nil || m.Summarizer == nil || sess == nil {
		ch := make(chan struct{})
		close(ch)
		return ch
	}
	key := stableKey(sess)
	m.inFlightMu.Lock()
	if existing, ok := m.inFlight[key]; ok {
		m.inFlightMu.Unlock()
		return existing.done
	}
	if m.inFlight == nil {
		m.inFlight = make(map[string]*inFlightCompaction)
	}
	fl := &inFlightCompaction{done: make(chan struct{})}
	m.inFlight[key] = fl
	m.inFlightMu.Unlock()

	timeout := m.Summarizer.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	go func() {
		defer close(fl.done)
		ctx, cancel := context.WithTimeout(context.Background(), 2*timeout)
		defer cancel()
		res, err := m.MaybeCompact(ctx, sess, reason, "")
		fl.result = res
		fl.err = err
		// Drop our entry from the map so a subsequent burst can fire a
		// fresh compaction (the next turn's WaitForInFlight will see no
		// entry and proceed straight to its own threshold check).
		m.inFlightMu.Lock()
		delete(m.inFlight, key)
		m.inFlightMu.Unlock()
	}()
	return fl.done
}

// WaitForInFlight blocks until any in-flight async compaction for the
// given session completes, or until the timeout elapses. Returns
// (result, true) on completion within the budget, (zero, false)
// otherwise. No-op (returns false immediately) when no compaction is
// in flight for the session.
//
// Wire this at the top of the chat loop, before assembling messages,
// so the next turn picks up the freshly-compacted view when the prior
// turn kicked off async compaction.
//
// Takes *session.Session (not Session.ID) because Session.ID is a
// per-load instance ID; the in-flight map is keyed by AgentID/Key so
// the handoff survives the Runtime rebuild between chat.send calls.
func (m *Manager) WaitForInFlight(sess *session.Session, timeout time.Duration) (Result, bool) {
	if m == nil || sess == nil {
		return Result{}, false
	}
	key := stableKey(sess)
	m.inFlightMu.Lock()
	fl, ok := m.inFlight[key]
	m.inFlightMu.Unlock()
	if !ok {
		return Result{}, false
	}
	select {
	case <-fl.done:
		return fl.result, true
	case <-time.After(timeout):
		// Timed out waiting; the goroutine keeps running. The next
		// preventive check on this turn will likely fall through to
		// synchronous compaction since nothing's been spliced in yet.
		return Result{}, false
	}
}

// HasInFlight reports whether an async compaction is currently running
// for the given session. Used by the runtime to decide whether to
// spawn another async compaction at end-of-turn.
func (m *Manager) HasInFlight(sess *session.Session) bool {
	if m == nil || sess == nil {
		return false
	}
	key := stableKey(sess)
	m.inFlightMu.Lock()
	defer m.inFlightMu.Unlock()
	_, ok := m.inFlight[key]
	return ok
}

// ForgetSession removes the per-session lock, failure counter, and
// any in-flight tracking entry for the given session. Safe to call on
// a nil Manager and on unknown sessions.
//
// The internal maps are keyed by AgentID+Key (the persistent
// identifier), so calling ForgetSession is only strictly required when
// a persistent session is being deleted (e.g. session.clear). The old
// "defer ForgetSession on every chat.send" pattern is no longer
// necessary for unbounded-growth protection — the maps are now bounded
// by the number of distinct persistent sessions, not by agent turns.
// Calling it on every turn is still harmless.
func (m *Manager) ForgetSession(sess *session.Session) {
	if m == nil || sess == nil {
		return
	}
	key := stableKey(sess)
	m.mu.Lock()
	delete(m.locks, key)
	m.mu.Unlock()
	m.failMu.Lock()
	delete(m.failures, key)
	m.failMu.Unlock()
	// Drop any in-flight tracking entry too. The goroutine itself
	// already deletes its entry on completion; this covers the rare
	// case of a session being forgotten mid-compaction.
	m.inFlightMu.Lock()
	delete(m.inFlight, key)
	m.inFlightMu.Unlock()
}

// stableKey returns the per-session identifier used as the map key for
// locks, failure counters, and in-flight async compactions. We use
// AgentID + Key (both persistent across loads) rather than Session.ID
// (a per-load instance ID), so async compaction kicked off at the end
// of one chat.send is observable by the next chat.send's
// WaitForInFlight call — the Runtime is rebuilt each chat.send and
// the freshly-loaded Session has a different Session.ID even though
// the persistent file on disk is the same.
//
// Falls back to Session.ID when AgentID and Key are both empty (some
// tests use NewSession("","") and rely on per-instance scoping).
func stableKey(sess *session.Session) string {
	if sess.AgentID == "" && sess.Key == "" {
		return sess.ID
	}
	return sess.AgentID + "/" + sess.Key
}

// incrementFailure bumps the per-session consecutive-failure count and
// returns the new count.
func (m *Manager) incrementFailure(sessionID string) int {
	m.failMu.Lock()
	defer m.failMu.Unlock()
	if m.failures == nil {
		m.failures = make(map[string]int)
	}
	m.failures[sessionID]++
	return m.failures[sessionID]
}

// resetFailures clears the per-session counter on a genuine success.
func (m *Manager) resetFailures(sessionID string) {
	m.failMu.Lock()
	defer m.failMu.Unlock()
	if m.failures != nil {
		delete(m.failures, sessionID)
	}
}

// failureCount returns the current per-session consecutive-failure count.
func (m *Manager) failureCount(sessionID string) int {
	m.failMu.Lock()
	defer m.failMu.Unlock()
	return m.failures[sessionID]
}

func (m *Manager) lockFor(sessionID string) *sync.Mutex {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.locks == nil {
		m.locks = make(map[string]*sync.Mutex)
	}
	if mu, ok := m.locks[sessionID]; ok {
		return mu
	}
	mu := &sync.Mutex{}
	m.locks[sessionID] = mu
	return mu
}

func classifySummarizerError(err error) string {
	switch {
	case errors.Is(err, ErrEmptySummary):
		return "empty_summary"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	default:
		// Network failure to localhost Ollama → "ollama_down" (best effort).
		// More specific classification can come later.
		return "summarizer_error"
	}
}
