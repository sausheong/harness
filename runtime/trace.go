package runtime

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// otelTracer is the package-level tracer used for OTel span emission
// alongside the existing slog perf logs. nil → no OTel emission. Set
// once at startup via SetOTelTracer; not safe to swap mid-flight.
var otelTracer trace.Tracer

// SetOTelTracer installs a tracer for the agent package. Call once
// from startup wiring after building the OTel provider; thereafter
// every NewTrace creates an OTel root span and each Mark adds a span
// event. Pass nil to disable.
func SetOTelTracer(t trace.Tracer) { otelTracer = t }

// Trace records phase timings across a single chat request so we can spot
// where latency comes from. Designed to be cheap: each Mark allocates a small
// struct and emits one slog.Info record. A typical request produces ~10 lines.
//
// Usage:
//
//	tr := agent.NewTrace(agentID, model)
//	ctx = agent.WithTrace(ctx, tr)
//	tr.Mark("ws.received")
//	... runtime calls tr.Mark("llm.first_token", "turn", n) ...
//	tr.Summary()
type Trace struct {
	ID      string
	AgentID string
	Model   string
	Started time.Time

	mu     sync.Mutex
	last   time.Time
	phases []phaseRecord

	// onMark, when set, is invoked synchronously on every Mark call so
	// the gateway WebSocket can forward live phase events to subscribed
	// clients (the chat-UI trace panel). Nil means no live forwarding,
	// which preserves the original "slog only" behavior for tests and
	// non-gateway call paths.
	onMark func(phase string, durMs, atMs int64, attrs []any)

	// span is the root OTel span for this Trace, created by NewTrace
	// when otelTracer is non-nil. Each Mark adds an event to this span
	// and Summary ends it. nil when OTel is disabled — every method on
	// trace.Span is nil-safe so no extra guards needed at call sites.
	span trace.Span
}

// SetOnMark registers a callback fired on every Mark. Safe to call once
// before the trace is shared with a goroutine; later calls overwrite the
// previous callback. The callback is invoked under the trace's lock so
// receivers must be quick (e.g., non-blocking channel send).
func (t *Trace) SetOnMark(fn func(phase string, durMs, atMs int64, attrs []any)) {
	if t == nil {
		return
	}
	t.onMark = fn
}

type phaseRecord struct {
	Name  string
	DurMs int64 // since previous Mark
	AtMs  int64 // since trace start
}

// NewTrace creates a Trace seeded with the start time. Returns nil-safe value;
// all methods are no-ops on a nil receiver so callers don't need nil checks.
//
// When the package-level otelTracer is set, NewTrace also starts a root
// OTel span "agent.run" tagged with trace_id, agent.id, and llm.model.
// The span is ended by Summary().
func NewTrace(agentID, model string) *Trace {
	now := time.Now()
	id := newTraceID()
	t := &Trace{
		ID:      id,
		AgentID: agentID,
		Model:   model,
		Started: now,
		last:    now,
	}
	if otelTracer != nil {
		_, t.span = otelTracer.Start(context.Background(), "agent.run",
			trace.WithAttributes(
				attribute.String("harness.trace_id", id),
				attribute.String("agent.id", agentID),
				attribute.String("llm.model", model),
			),
		)
	}
	return t
}

// Mark records a phase boundary and emits a slog.Info entry tagged "perf".
// extraAttrs are key/value pairs appended to the log record (e.g. "turn", 3).
//
// When the Trace has an attached OTel span (because otelTracer was set
// at construction time), Mark also adds a span event so the phase
// timeline is visible in OTel UIs (Tempo / Jaeger / etc).
func (t *Trace) Mark(phase string, extraAttrs ...any) {
	if t == nil {
		return
	}
	t.mu.Lock()
	now := time.Now()
	dur := now.Sub(t.last).Milliseconds()
	at := now.Sub(t.Started).Milliseconds()
	t.phases = append(t.phases, phaseRecord{Name: phase, DurMs: dur, AtMs: at})
	t.last = now
	cb := t.onMark
	span := t.span
	t.mu.Unlock()

	attrs := []any{
		"trace_id", t.ID,
		"agent", t.AgentID,
		"phase", phase,
		"dur_ms", dur,
		"at_ms", at,
	}
	attrs = append(attrs, extraAttrs...)
	slog.Info("perf", attrs...)

	if cb != nil {
		cb(phase, dur, at, extraAttrs)
	}

	if span != nil && span.IsRecording() {
		span.AddEvent(phase, trace.WithAttributes(buildOTelAttrs(dur, at, extraAttrs)...))
	}
}

// buildOTelAttrs converts the slog-style key/value pairs used throughout
// the trace package into the typed attribute.KeyValue slice the OTel
// SDK wants. Unknown types fall through to fmt.Sprint so the attribute
// always lands as a string rather than being dropped.
func buildOTelAttrs(dur, at int64, kvs []any) []attribute.KeyValue {
	out := make([]attribute.KeyValue, 0, 2+len(kvs)/2)
	out = append(out, attribute.Int64("dur_ms", dur), attribute.Int64("at_ms", at))
	for i := 0; i+1 < len(kvs); i += 2 {
		key, ok := kvs[i].(string)
		if !ok {
			continue
		}
		switch v := kvs[i+1].(type) {
		case string:
			out = append(out, attribute.String(key, v))
		case bool:
			out = append(out, attribute.Bool(key, v))
		case int:
			out = append(out, attribute.Int(key, v))
		case int64:
			out = append(out, attribute.Int64(key, v))
		case float64:
			out = append(out, attribute.Float64(key, v))
		default:
			out = append(out, attribute.String(key, fmt.Sprint(v)))
		}
	}
	return out
}

// Summary emits one final slog.Info "perf summary" line with the total
// elapsed time and the top three slowest phases. Useful for at-a-glance
// triage without grep.
func (t *Trace) Summary() {
	if t == nil {
		return
	}
	t.mu.Lock()
	total := time.Since(t.Started).Milliseconds()
	// Aggregate dur by phase name (some phases recur per turn).
	agg := map[string]int64{}
	for _, p := range t.phases {
		agg[p.Name] += p.DurMs
	}
	type kv struct {
		Name string
		Dur  int64
	}
	flat := make([]kv, 0, len(agg))
	for k, v := range agg {
		flat = append(flat, kv{k, v})
	}
	t.mu.Unlock()
	sort.Slice(flat, func(i, j int) bool { return flat[i].Dur > flat[j].Dur })
	top := flat
	if len(top) > 3 {
		top = top[:3]
	}
	attrs := []any{
		"trace_id", t.ID,
		"agent", t.AgentID,
		"model", t.Model,
		"total_ms", total,
		"phase_count", len(t.phases),
	}
	for i, kv := range top {
		attrs = append(attrs,
			"top"+itoa(i+1)+"_phase", kv.Name,
			"top"+itoa(i+1)+"_ms", kv.Dur,
		)
	}
	slog.Info("perf summary", attrs...)

	if t.span != nil {
		t.span.SetAttributes(
			attribute.Int64("harness.total_ms", total),
			attribute.Int("harness.phase_count", len(t.phases)),
		)
		t.span.End()
	}
}

type traceKey struct{}

// WithTrace stashes the Trace in ctx so deeper layers can call Mark.
func WithTrace(ctx context.Context, t *Trace) context.Context {
	return context.WithValue(ctx, traceKey{}, t)
}

// TraceFrom retrieves the Trace from ctx, or nil if none. nil is safe — all
// Trace methods tolerate a nil receiver.
func TraceFrom(ctx context.Context) *Trace {
	if v := ctx.Value(traceKey{}); v != nil {
		if t, ok := v.(*Trace); ok {
			return t
		}
	}
	return nil
}

func newTraceID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [4]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
