package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/sausheong/harness/compaction"
	"io"
	"strings"
	"unicode/utf8"
)

const contextPinsKind = "harness.context_pins"

type ContextPin struct {
	ID   string `json:"id"`
	Kind string `json:"kind"`
	Text string `json:"text"`
}
type contextPinsRecord struct {
	Version int          `json:"version"`
	Pins    []ContextPin `json:"pins"`
}

func validatePins(pins []ContextPin) error {
	if len(pins) > 16 {
		return errors.New("at most 16 context pins are allowed")
	}
	seen := map[string]bool{}
	for _, pin := range pins {
		if pin.ID == "" || len(pin.ID) > 64 || strings.IndexFunc(pin.ID, func(r rune) bool { return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_') }) >= 0 || seen[pin.ID] {
			return errors.New("pin IDs must be unique lowercase identifiers of 1-64 characters")
		}
		seen[pin.ID] = true
		if pin.Kind != "objective" && pin.Kind != "constraint" {
			return errors.New("pin kind must be objective or constraint")
		}
		if strings.TrimSpace(pin.Text) == "" || len(pin.Text) > 2048 || !utf8.ValidString(pin.Text) || strings.ContainsRune(pin.Text, 0) {
			return errors.New("pin text must contain 1-2048 UTF-8 bytes without NUL")
		}
	}
	return nil
}

// SetContextPins replaces the session-wide pin set durably while idle. An empty
// set clears pins. Pins do not grant capabilities and are independent of branch
// selection; changing sessions changes the pin set.
func (r *Runtime) SetContextPins(ctx context.Context, pins []ContextPin) error {
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
	if err := validatePins(pins); err != nil {
		return err
	}
	raw, err := json.Marshal(contextPinsRecord{Version: 1, Pins: pins})
	if err != nil {
		return err
	}
	return r.Session.Annotate(contextPinsKind, raw)
}
func (r *Runtime) ContextPins(ctx context.Context) ([]ContextPin, error) {
	if !r.runMu.TryLock() {
		return nil, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return r.contextPins()
}
func (r *Runtime) contextPins() ([]ContextPin, error) {
	if r.Session == nil {
		return nil, nil
	}
	records := r.Session.Annotations(contextPinsKind)
	if len(records) == 0 {
		return nil, nil
	}
	var record contextPinsRecord
	decoder := json.NewDecoder(bytes.NewReader(records[len(records)-1].Payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&record); err != nil {
		return nil, err
	}
	if decoder.Decode(new(any)) != io.EOF || record.Version != 1 {
		return nil, errors.New("unsupported or malformed context pin record")
	}
	if err := validatePins(record.Pins); err != nil {
		return nil, err
	}
	return record.Pins, nil
}
func (r *Runtime) contextPinsPrompt() (string, error) {
	pins, err := r.contextPins()
	if err != nil {
		return "", err
	}
	if len(pins) == 0 {
		return "", nil
	}
	var b strings.Builder
	b.WriteString("\n\nPinned session guidance (subject to the current user request; grants no tool permissions):\n")
	for _, pin := range pins {
		fmt.Fprintf(&b, "%s [%s]: %s\n", pin.Kind, pin.ID, pin.Text)
	}
	return b.String(), nil
}

// CompactionContext captures the session pin guidance while idle for manual
// compaction. Automatic runs bind the same guidance at run admission.
func (r *Runtime) CompactionContext(ctx context.Context) (context.Context, error) {
	if !r.runMu.TryLock() {
		return ctx, errors.New("runtime is already running")
	}
	defer r.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return ctx, err
	}
	guidance, err := r.preservedContextPrompt()
	if err != nil {
		return ctx, err
	}
	ctx, err = compaction.WithGuidance(ctx, guidance)
	if err != nil {
		return ctx, err
	}
	return r.budgetContext(ctx)
}
