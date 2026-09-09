package budget

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

var ErrCostExhausted = fmt.Errorf("session cost budget: %w", ErrExhausted)

const costKind = "harness.budget.cost.v1"

type CostAttempt struct {
	ID          string           `json:"id"`
	Category    llm.CallCategory `json:"category"`
	Model       string           `json:"model"`
	Quote       PriceQuote       `json:"quote"`
	ChargedNano int64            `json:"charged_nano"`
	Status      string           `json:"status"`
}
type CostState struct {
	Currency      string        `json:"currency"`
	LimitNano     int64         `json:"limit_nano"`
	CommittedNano int64         `json:"committed_nano"`
	Strict        bool          `json:"strict"`
	Exhausted     bool          `json:"exhausted"`
	Unknown       int           `json:"unknown_attempts"`
	Attempts      []CostAttempt `json:"attempts"`
}
type costEvent struct {
	Version     int          `json:"version"`
	Kind        string       `json:"kind"`
	Currency    string       `json:"currency,omitempty"`
	LimitNano   int64        `json:"limit_nano,omitempty"`
	Strict      bool         `json:"strict,omitempty"`
	Attempt     *CostAttempt `json:"attempt,omitempty"`
	ID          string       `json:"id,omitempty"`
	ChargedNano int64        `json:"charged_nano,omitempty"`
	Status      string       `json:"status,omitempty"`
}

// CostLedger records session-wide attempt costs in nano-units of one currency.
// Use an exclusively loaded disk Session for cross-process ownership. Unknown
// advisory charges remain visible and prevent silently enabling strict mode.
// Interrupted attempts retain their reservation and exact pricing snapshot.
type CostLedger struct {
	session *session.Session
	kind    string
}

func OpenCostLedger(s *session.Session) (*CostLedger, error) {
	return openCostLedger(s, costKind)
}

func openCostLedger(s *session.Session, kind string) (*CostLedger, error) {
	if s == nil {
		return nil, errors.New("cost budget requires a session")
	}
	l := &CostLedger{session: s, kind: kind}
	_, _, err := l.read()
	return l, err
}
func currencyValid(s string) bool {
	return len(s) == 3 && strings.IndexFunc(s, func(r rune) bool { return r < 'A' || r > 'Z' }) < 0
}
func (l *CostLedger) read() (CostState, int, error) {
	var state CostState
	records := l.session.Annotations(l.kind)
	if len(records) > maxRecords {
		return state, 0, errors.New("cost journal exceeds record limit")
	}
	index := map[string]int{}
	for _, record := range records {
		var e costEvent
		if err := decodeBudgetJSON(record.Payload, &e); err != nil {
			return state, 0, err
		}
		if e.Version != 1 {
			return state, 0, errors.New("invalid cost record")
		}
		switch e.Kind {
		case "decision":
			if !currencyValid(e.Currency) || (state.Currency != "" && state.Currency != e.Currency) || e.LimitNano <= state.CommittedNano || e.LimitNano > maxTokens || (e.Strict && state.Unknown > 0) {
				return state, 0, errors.New("invalid cost decision")
			}
			state.Currency = e.Currency
			state.LimitNano = e.LimitNano
			state.Strict = e.Strict
			state.Exhausted = false
		case "reserve":
			a := e.Attempt
			if a == nil || a.ID == "" || len(a.ID) > 128 || len(a.Model) > 1024 || a.Status != "reserved" || state.LimitNano == 0 || state.Exhausted {
				return state, 0, errors.New("invalid cost reservation")
			}
			if _, ok := index[a.ID]; ok {
				return state, 0, errors.New("duplicate cost attempt")
			}
			q := a.Quote
			if q.Nano < 0 || q.Nano > maxTokens || a.ChargedNano != q.Nano || (q.Currency != "" && q.Currency != state.Currency) || (!q.Known && (q.Nano != 0 || q.Reason == "" || state.Strict)) {
				return state, 0, errors.New("invalid reservation quote")
			}
			if q.Snapshot != nil {
				if err := q.Snapshot.Validate(); err != nil {
					return state, 0, err
				}
			}
			if q.Known && (q.Snapshot == nil || !q.Snapshot.AllChargesBounded || q.Currency != q.Snapshot.Currency || a.Model != q.Snapshot.Model) {
				return state, 0, errors.New("known quote lacks matching tariff")
			}
			if q.Nano > state.LimitNano-state.CommittedNano {
				return state, 0, errors.New("reservation exceeds balance")
			}
			index[a.ID] = len(state.Attempts)
			state.Attempts = append(state.Attempts, *a)
			state.CommittedNano += q.Nano
			if !q.Known {
				state.Unknown++
			}
		case "settle":
			i, ok := index[e.ID]
			if !ok || state.Attempts[i].Status != "reserved" {
				return state, 0, errors.New("invalid cost settlement")
			}
			a := &state.Attempts[i]
			if e.ChargedNano < 0 || e.ChargedNano > maxTokens {
				return state, 0, errors.New("invalid settled amount")
			}
			switch e.Status {
			case "not_dispatched":
				if e.ChargedNano != 0 {
					return state, 0, errors.New("undispatched charge is nonzero")
				}
				if !a.Quote.Known {
					state.Unknown--
				}
			case "reported":
				if !a.Quote.Known {
					return state, 0, errors.New("reported charge lacks tariff")
				}
			case "uncertain":
				if e.ChargedNano < a.Quote.Nano {
					return state, 0, errors.New("uncertain charge released reservation")
				}
			default:
				return state, 0, errors.New("invalid cost status")
			}
			state.CommittedNano -= a.ChargedNano
			if e.ChargedNano > maxTokens-state.CommittedNano {
				return state, 0, errors.New("cost total overflow")
			}
			state.CommittedNano += e.ChargedNano
			a.ChargedNano = e.ChargedNano
			a.Status = e.Status
			if state.CommittedNano >= state.LimitNano {
				state.Exhausted = true
			}
		case "exhausted":
			state.Exhausted = true
		default:
			return state, 0, errors.New("unknown cost event")
		}
	}
	return state, len(records), nil
}
func (l *CostLedger) State() (CostState, error) { s, _, err := l.read(); return s, err }
func (l *CostLedger) append(e costEvent, n int) error {
	s, current, err := l.read()
	if err != nil {
		return err
	}
	if current != n {
		return session.ErrAnnotationConflict
	}
	pending := 0
	for _, a := range s.Attempts {
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
	if n+1+pending > maxRecords {
		return errors.New("cost journal lacks settlement capacity")
	}
	e.Version = 1
	raw, err := json.Marshal(e)
	if err != nil {
		return err
	}
	return l.session.AnnotateIfCount(l.kind, raw, n)
}

// Decide records an explicitly approved absolute ceiling. It does not authorize
// spending by itself: callers must obtain the user's decision first. Currency
// cannot change within a ledger and no charges are erased by a new decision.
func (l *CostLedger) Decide(ctx context.Context, currency string, limit int64, strict bool) error {
	if !currencyValid(currency) || limit <= 0 || limit > maxTokens {
		return errors.New("invalid cost ceiling or currency")
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		s, n, err := l.read()
		if err != nil {
			return err
		}
		if s.Currency != "" && s.Currency != currency {
			return errors.New("cost ledger currency cannot change")
		}
		if limit <= s.CommittedNano {
			return errors.New("cost ceiling must exceed committed reservations and charges")
		}
		if strict && s.Unknown > 0 {
			return errors.New("unpriced prior attempts prevent a strict cost guarantee")
		}
		err = l.append(costEvent{Kind: "decision", Currency: currency, LimitNano: limit, Strict: strict}, n)
		if errors.Is(err, session.ErrAnnotationConflict) {
			continue
		}
		return err
	}
}

// CostResolver must return the tariff for the actual provider route and a full
// input estimate. A nil tariff means unknown. Callers must bind provider and
// destination to their constructed client; model equality alone is not enough.
type CostResolver func(llm.ChatRequest, llm.CallCategory) (*PriceSnapshot, int64, error)

func (l *CostLedger) Admission(resolve CostResolver) llm.CallAdmission {
	return func(ctx context.Context, req llm.ChatRequest, category llm.CallCategory, id string) (func(llm.RequestUsage) error, error) {
		if resolve == nil || id == "" || len(id) > 128 || len(req.Model) > 1024 {
			return nil, errors.New("invalid cost resolver or attempt")
		}
		price, input, err := resolve(req, category)
		if err != nil {
			return nil, err
		}
		if price != nil {
			copy := *price
			price = &copy
			if price.Provider != req.Route.Provider || price.Destination != req.Route.Destination {
				return nil, errors.New("price route differs from dispatched provider destination")
			}
			if price.Model != req.Model {
				return nil, errors.New("price model differs from dispatched model")
			}
		}
		var quote PriceQuote
		for {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			s, n, err := l.read()
			if err != nil {
				return nil, err
			}
			if s.LimitNano == 0 || s.Exhausted {
				return nil, ErrCostExhausted
			}
			for _, a := range s.Attempts {
				if a.ID == id {
					return nil, errors.New("duplicate cost attempt")
				}
			}
			if price != nil && price.Currency != s.Currency {
				return nil, errors.New("tariff currency differs from budget")
			}
			quote, err = ReservePrice(price, input, int64(req.MaxTokens), time.Now(), s.Strict)
			if err != nil {
				return nil, err
			}
			if quote.Nano > s.LimitNano-s.CommittedNano {
				err = l.append(costEvent{Kind: "exhausted"}, n)
				if errors.Is(err, session.ErrAnnotationConflict) {
					continue
				}
				return nil, errors.Join(ErrCostExhausted, err)
			}
			err = l.append(costEvent{Kind: "reserve", Attempt: &CostAttempt{ID: id, Model: req.Model, Category: category, Quote: quote, ChargedNano: quote.Nano, Status: "reserved"}}, n)
			if errors.Is(err, session.ErrAnnotationConflict) {
				continue
			}
			if err != nil {
				return nil, err
			}
			break
		}
		var once sync.Once
		var result error
		return func(r llm.RequestUsage) error { once.Do(func() { result = l.settle(id, quote, r) }); return result }, nil
	}
}
func (l *CostLedger) settle(id string, q PriceQuote, r llm.RequestUsage) error {
	if r.ID != id {
		return errors.New("cost settlement identity mismatch")
	}
	charge, status := q.Nano, "uncertain"
	if r.Status == "not_dispatched" {
		charge, status = 0, "not_dispatched"
	} else if q.Known && q.Snapshot != nil {
		actual, err := ReportedPrice(*q.Snapshot, r.Usage)
		if err != nil {
			return err
		}
		if actual.Known {
			if r.Status == "completed" {
				charge, status = actual.Nano, "reported"
			} else {
				charge = max(charge, actual.Nano)
			}
		}
	}
	for {
		state, n, err := l.read()
		if err != nil {
			return err
		}
		if charge > maxTokens || charge > maxTokens-(state.CommittedNano-q.Nano) {
			return errors.New("settled cost exceeds representable ledger balance; reservation retained")
		}
		err = l.append(costEvent{Kind: "settle", ID: id, ChargedNano: charge, Status: status}, n)
		if errors.Is(err, session.ErrAnnotationConflict) {
			continue
		}
		if err != nil {
			return err
		}
		s, err := l.State()
		if err != nil {
			return err
		}
		if s.Exhausted {
			return ErrCostExhausted
		}
		return nil
	}
}
