// Package budget provides durable provider-attempt admission policies.
package budget

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

var ErrExhausted = errors.New("budget_exhausted")

const tokenKind = "harness.budget.tokens.v1"
const maxRecords = 10000
const maxTokens int64 = 1 << 50

type tokenEvent struct {
	Version  int              `json:"version"`
	Kind     string           `json:"kind"`
	ID       string           `json:"id,omitempty"`
	Model    string           `json:"model,omitempty"`
	Category llm.CallCategory `json:"category,omitempty"`
	Tokens   int64            `json:"tokens"`
	Status   string           `json:"status,omitempty"`
}

type TokenAttempt struct {
	ID       string           `json:"id"`
	Model    string           `json:"model"`
	Category llm.CallCategory `json:"category"`
	Reserved int64            `json:"reserved"`
	Charged  int64            `json:"charged"`
	Status   string           `json:"status"`
}

type TokenState struct {
	Limit     int64          `json:"limit"`
	Committed int64          `json:"committed"`
	Exhausted bool           `json:"exhausted"`
	Attempts  []TokenAttempt `json:"attempts"`
}

// TokenLedger shares a session-wide limit across every attached operation and
// branch. Use a disk session loaded exclusively for durability across processes.
// Reopening never changes limits or releases interrupted reservations. Multiple
// ledgers over the same Session coordinate through conditional durable appends.
// This is token admission, not a financial billing cap.
type TokenLedger struct {
	session *session.Session
	kind    string
}

func OpenTokenLedger(s *session.Session) (*TokenLedger, error) {
	return openTokenLedger(s, tokenKind)
}

func openTokenLedger(s *session.Session, kind string) (*TokenLedger, error) {
	if s == nil {
		return nil, errors.New("budget requires a session")
	}
	l := &TokenLedger{session: s, kind: kind}
	_, _, err := l.read()
	return l, err
}

func (l *TokenLedger) read() (TokenState, int, error) {
	var state TokenState
	records := l.session.Annotations(l.kind)
	if len(records) > maxRecords {
		return state, 0, errors.New("budget journal exceeds record limit")
	}
	index := map[string]int{}
	for _, record := range records {
		var e tokenEvent
		if err := decodeBudgetJSON(record.Payload, &e); err != nil {
			return state, 0, err
		}
		if e.Version != 1 || e.Tokens < 0 || e.Tokens > maxTokens {
			return state, 0, errors.New("invalid budget record")
		}
		switch e.Kind {
		case "limit":
			if e.Tokens <= state.Committed {
				return state, 0, errors.New("budget decision leaves no available capacity")
			}
			state.Limit = e.Tokens
			state.Exhausted = false
		case "reserve":
			if _, exists := index[e.ID]; exists || e.ID == "" || state.Limit == 0 || state.Exhausted || e.Tokens <= 0 || e.Tokens > state.Limit-state.Committed {
				return state, 0, errors.New("invalid budget reservation")
			}
			index[e.ID] = len(state.Attempts)
			state.Attempts = append(state.Attempts, TokenAttempt{ID: e.ID, Model: e.Model, Category: e.Category, Reserved: e.Tokens, Charged: e.Tokens, Status: "reserved"})
			state.Committed += e.Tokens
		case "settle":
			i, ok := index[e.ID]
			if !ok || state.Attempts[i].Status != "reserved" {
				return state, 0, errors.New("invalid budget settlement")
			}
			a := &state.Attempts[i]
			if e.Status != "reported" && e.Status != "unknown" && e.Status != "not_dispatched" {
				return state, 0, errors.New("invalid settlement status")
			}
			if (e.Status == "unknown" && e.Tokens < a.Reserved) || (e.Status == "not_dispatched" && e.Tokens != 0) {
				return state, 0, errors.New("invalid settlement charge")
			}
			state.Committed -= a.Charged
			if e.Tokens > maxTokens-state.Committed {
				return state, 0, errors.New("budget total overflow")
			}
			state.Committed += e.Tokens
			a.Charged = e.Tokens
			a.Status = e.Status
			if state.Committed >= state.Limit {
				state.Exhausted = true
			}
		case "exhausted":
			state.Exhausted = true
		default:
			return state, 0, errors.New("unknown budget event")
		}
	}
	return state, len(records), nil
}

func (l *TokenLedger) append(e tokenEvent, revision int) error {
	state, current, err := l.read()
	if err != nil {
		return err
	}
	if current != revision {
		return session.ErrAnnotationConflict
	}
	pending := 0
	for _, a := range state.Attempts {
		if a.Status == "reserved" {
			pending++
		}
	}
	if e.Kind == "reserve" {
		pending++
	}
	if e.Kind == "settle" {
		pending--
	}
	if revision+1+pending > maxRecords {
		return errors.New("budget journal lacks settlement capacity")
	}
	if revision >= maxRecords {
		return errors.New("budget journal full")
	}
	e.Version = 1
	raw, err := json.Marshal(e)
	if err != nil {
		return err
	}
	return l.session.AnnotateIfCount(l.kind, raw, revision)
}
func (l *TokenLedger) State() (TokenState, error) { s, _, err := l.read(); return s, err }

// Decide explicitly sets an absolute session token ceiling above all committed
// and uncertain usage. It resumes a previously exhausted ledger without erasing
// charges. Callers must obtain the user's budget decision before invoking it.
func (l *TokenLedger) Decide(ctx context.Context, limit int64) error {
	if limit <= 0 || limit > maxTokens {
		return errors.New("invalid token ceiling")
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		s, n, err := l.read()
		if err != nil {
			return err
		}
		if limit <= s.Committed {
			return errors.New("ceiling must exceed committed and reserved tokens")
		}
		err = l.append(tokenEvent{Kind: "limit", Tokens: limit}, n)
		if errors.Is(err, session.ErrAnnotationConflict) {
			continue
		}
		return err
	}
}

// Admission reserves inputEstimate(req)+MaxTokens before dispatch. The supplied
// estimator must include system text, tool schemas, framing and multimodal
// input; its provenance must be disclosed by the application. Estimation error
// and provider output overruns can exceed the limit and latch exhaustion.
func (l *TokenLedger) Admission(inputEstimate func(llm.ChatRequest) (int64, error)) llm.CallAdmission {
	return func(ctx context.Context, req llm.ChatRequest, category llm.CallCategory, id string) (func(llm.RequestUsage) error, error) {
		if inputEstimate == nil || id == "" || len(id) > 128 || len(req.Model) > 1024 || req.MaxTokens <= 0 {
			return nil, errors.New("bounded request and estimator required")
		}
		input, err := inputEstimate(req)
		if err != nil {
			return nil, err
		}
		if input < 0 || input > maxTokens-int64(req.MaxTokens) {
			return nil, errors.New("invalid token estimate")
		}
		reserved := input + int64(req.MaxTokens)
		for {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			s, n, err := l.read()
			if err != nil {
				return nil, err
			}
			for _, a := range s.Attempts {
				if a.ID == id {
					return nil, errors.New("duplicate budget attempt")
				}
			}
			if s.Limit == 0 || s.Exhausted {
				return nil, ErrExhausted
			}
			if reserved > s.Limit-s.Committed {
				err = l.append(tokenEvent{Kind: "exhausted"}, n)
				if errors.Is(err, session.ErrAnnotationConflict) {
					continue
				}
				return nil, errors.Join(ErrExhausted, err)
			}
			// Leave one record for each outstanding settlement. Journal exhaustion
			// must not admit a request whose terminal record cannot fit.
			pending := 0
			for _, a := range s.Attempts {
				if a.Status == "reserved" {
					pending++
				}
			}
			if n+pending+2 > maxRecords {
				return nil, errors.New("budget journal lacks settlement capacity")
			}
			err = l.append(tokenEvent{Kind: "reserve", ID: id, Model: req.Model, Category: category, Tokens: reserved}, n)
			if errors.Is(err, session.ErrAnnotationConflict) {
				continue
			}
			if err != nil {
				return nil, err
			}
			break
		}
		var once sync.Once
		var settledErr error
		return func(r llm.RequestUsage) error {
			once.Do(func() { settledErr = l.settle(id, reserved, r) })
			return settledErr
		}, nil
	}
}
func (l *TokenLedger) settle(id string, reserved int64, r llm.RequestUsage) error {
	if r.ID != id {
		return errors.New("settlement identity mismatch")
	}
	charge, status := reserved, "unknown"
	if r.Status == "not_dispatched" {
		charge, status = 0, "not_dispatched"
	} else if r.Usage != nil {
		u := r.Usage
		if u.InputTokens < 0 || u.OutputTokens < 0 || int64(u.InputTokens) > maxTokens-int64(u.OutputTokens) {
			return errors.New("invalid reported tokens")
		}
		charge = int64(u.InputTokens) + int64(u.OutputTokens)
		if r.Status == "completed" {
			status = "reported"
		} else if charge < reserved {
			charge = reserved
		}
	}
	for {
		state, n, err := l.read()
		if err != nil {
			return err
		}
		if charge > maxTokens || charge > maxTokens-(state.Committed-reserved) {
			return errors.New("settled tokens exceed representable ledger balance; reservation retained")
		}
		err = l.append(tokenEvent{Kind: "settle", ID: id, Tokens: charge, Status: status}, n)
		if errors.Is(err, session.ErrAnnotationConflict) {
			continue
		}
		if err != nil {
			return fmt.Errorf("persist token settlement: %w", err)
		}
		state, err = l.State()
		if err != nil {
			return err
		}
		if state.Exhausted {
			return ErrExhausted
		}
		return nil
	}
}
