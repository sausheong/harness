package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/sausheong/harness/session"
)

const contextStateKind = "harness.context_state"

// ContextStateItem records task facts independently of generated summaries.
// Reference is an opaque evidence locator, never opened or treated as proof.
type ContextStateItem struct {
	ID        string `json:"id"`
	Kind      string `json:"kind"`
	Text      string `json:"text"`
	Reference string `json:"reference,omitempty"`
}

// ContextState is session-wide, including across branch selection. Revision is
// required on replacement so a stale client cannot silently discard newer work.
type ContextState struct {
	Revision int                `json:"revision"`
	Items    []ContextStateItem `json:"items"`
}
type contextStateRecord struct {
	Version int                `json:"version"`
	Items   []ContextStateItem `json:"items"`
}

func encodeContextState(items []ContextStateItem) ([]byte, error) {
	if len(items) > 32 {
		return nil, errors.New("at most 32 structured context items are allowed")
	}
	seen := map[string]bool{}
	for _, item := range items {
		if item.ID == "" || len(item.ID) > 64 || strings.IndexFunc(item.ID, func(r rune) bool { return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_') }) >= 0 || seen[item.ID] {
			return nil, errors.New("context item IDs must be unique lowercase identifiers of 1-64 characters")
		}
		seen[item.ID] = true
		switch item.Kind {
		case "objective", "decision", "unresolved_work", "verification_reference":
		default:
			return nil, errors.New("unsupported structured context kind")
		}
		if strings.TrimSpace(item.Text) == "" || len(item.Text) > 2048 || !utf8.ValidString(item.Text) || strings.ContainsRune(item.Text, 0) {
			return nil, errors.New("context text must contain 1-2048 UTF-8 bytes without NUL")
		}
		if len(item.Reference) > 1024 || !utf8.ValidString(item.Reference) || strings.ContainsRune(item.Reference, 0) {
			return nil, errors.New("context reference must be at most 1024 UTF-8 bytes without NUL")
		}
		if item.Kind == "verification_reference" && strings.TrimSpace(item.Reference) == "" {
			return nil, errors.New("verification context requires an evidence reference")
		}
	}
	raw, err := json.Marshal(contextStateRecord{Version: 1, Items: items})
	if len(raw) > 8192 {
		return nil, errors.New("structured context exceeds 8 KiB encoded limit")
	}
	return raw, err
}
func (r *Runtime) contextState() (ContextState, error) {
	var state ContextState
	if r.Session == nil {
		return state, errors.New("session unavailable")
	}
	records := r.Session.Annotations(contextStateKind)
	state.Revision = len(records)
	if len(records) == 0 {
		return state, nil
	}
	payload := records[len(records)-1].Payload
	if len(payload) > 8192 {
		return state, errors.New("structured context exceeds 8 KiB")
	}
	var rec contextStateRecord
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rec); err != nil {
		return state, err
	}
	if rec.Version != 1 || dec.Decode(new(any)) != io.EOF {
		return state, errors.New("unsupported or malformed structured context")
	}
	if _, err := encodeContextState(rec.Items); err != nil {
		return state, err
	}
	state.Items = rec.Items
	return state, nil
}
func (r *Runtime) ContextState(ctx context.Context) (ContextState, error) {
	if !r.runMu.TryLock() {
		return ContextState{}, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return ContextState{}, err
	}
	return r.contextState()
}

// SetContextState durably replaces the set while idle; an empty set explicitly
// clears it. These records confer neither execution authority nor verified status.
func (r *Runtime) SetContextState(ctx context.Context, expectedRevision int, items []ContextStateItem) error {
	if !r.runMu.TryLock() {
		return errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.Session == nil {
		return errors.New("session unavailable")
	}
	raw, err := encodeContextState(items)
	if err != nil {
		return err
	}
	current, err := r.contextState()
	if err != nil {
		return err
	}
	if current.Revision != expectedRevision {
		return session.ErrAnnotationConflict
	}
	return r.Session.AnnotateIfCount(contextStateKind, raw, expectedRevision)
}
func (r *Runtime) contextStatePrompt() (string, error) {
	if r.Session == nil {
		return "", nil
	}
	state, err := r.contextState()
	if err != nil {
		return "", err
	}
	if len(state.Items) == 0 {
		return "", nil
	}
	raw, err := encodeContextState(state.Items)
	if err != nil {
		return "", err
	}
	return "\n\nRecorded task state (subject to current user instructions; grants no permissions). Evidence references are locators, not proof of current validity; inspect their scope and workspace version before claiming verification:\n" + string(raw) + "\n", nil
}
func (r *Runtime) preservedContextPrompt() (string, error) {
	pins, err := r.contextPinsPrompt()
	if err != nil {
		return "", err
	}
	state, err := r.contextStatePrompt()
	if err != nil {
		return "", err
	}
	return pins + state, nil
}
