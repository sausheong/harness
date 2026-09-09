package budget

import (
	"errors"
	"fmt"
	"math/big"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sausheong/harness/llm"
)

var ErrUnknownPrice = errors.New("request price is unknown or unbounded")

// PriceSnapshot is an immutable, caller-reviewed tariff for a specific route.
// Rates are billionths of Currency per million tokens; FixedNano is charged
// once per provider attempt. No implicit currency conversion is performed.
// AllChargesBounded must be false when tool, reasoning, image, tiered-cache or
// other provider charges are not fully represented by these counters/rates.
// Input, cache creation and cache read counters must be disjoint subsets of
// total input under the provider adapter's usage contract.
type PriceSnapshot struct {
	Provider                 string    `json:"provider"`
	Model                    string    `json:"model"`
	Destination              string    `json:"destination"`
	Currency                 string    `json:"currency"`
	Source                   string    `json:"source"`
	Version                  string    `json:"version"`
	EffectiveAt              time.Time `json:"effective_at"`
	ExpiresAt                time.Time `json:"expires_at"`
	InputNanoPerMillion      int64     `json:"input_nano_per_million"`
	OutputNanoPerMillion     int64     `json:"output_nano_per_million"`
	CacheWriteNanoPerMillion int64     `json:"cache_write_nano_per_million"`
	CacheReadNanoPerMillion  int64     `json:"cache_read_nano_per_million"`
	FixedNano                int64     `json:"fixed_nano"`
	AllChargesBounded        bool      `json:"all_charges_bounded"`
}

func (p PriceSnapshot) Validate() error {
	for _, s := range []string{p.Provider, p.Model, p.Destination, p.Source, p.Version} {
		if strings.TrimSpace(s) == "" || len(s) > 2048 || !utf8.ValidString(s) || strings.ContainsAny(s, "\x00\r\n") {
			return errors.New("pricing requires bounded route, source and version metadata")
		}
	}
	if len(p.Currency) != 3 || strings.IndexFunc(p.Currency, func(r rune) bool { return r < 'A' || r > 'Z' }) >= 0 {
		return errors.New("pricing currency requires three uppercase letters")
	}
	if p.EffectiveAt.IsZero() || p.ExpiresAt.IsZero() || !p.ExpiresAt.After(p.EffectiveAt) {
		return errors.New("pricing requires an explicit validity interval")
	}
	for _, rate := range []int64{p.InputNanoPerMillion, p.OutputNanoPerMillion, p.CacheWriteNanoPerMillion, p.CacheReadNanoPerMillion, p.FixedNano} {
		if rate < 0 {
			return errors.New("negative price")
		}
	}
	return nil
}

// PriceQuote separates a known amount from unknown cost. Nano must never be
// interpreted as a free request when Known is false. Snapshot is copied into
// the quote for durable request accounting by the caller.
type PriceQuote struct {
	Known    bool           `json:"known"`
	Nano     int64          `json:"nano"`
	Currency string         `json:"currency,omitempty"`
	Reason   string         `json:"reason,omitempty"`
	Snapshot *PriceSnapshot `json:"snapshot,omitempty"`
}

// ReservePrice prices an estimated input bound plus a finite output cap. The
// maximum input/cache rate prevents optimistic cache-hit assumptions. Strict
// mode refuses missing, expired or unbounded pricing. Advisory mode returns an
// explicitly unknown quote. Invalid metadata/negative values always fail.
func ReservePrice(p *PriceSnapshot, input, output int64, now time.Time, strict bool) (PriceQuote, error) {
	if input < 0 || output <= 0 {
		return PriceQuote{}, errors.New("reservation requires nonnegative input and a positive output cap")
	}
	unknown := func(reason string) (PriceQuote, error) {
		q := PriceQuote{Reason: reason}
		if p != nil {
			copy := *p
			q.Snapshot = &copy
			q.Currency = p.Currency
		}
		if strict {
			return q, fmt.Errorf("%w: %s", ErrUnknownPrice, reason)
		}
		return q, nil
	}
	if p == nil {
		return unknown("no route price supplied")
	}
	if err := p.Validate(); err != nil {
		return PriceQuote{}, err
	}
	if now.IsZero() || now.Before(p.EffectiveAt) || !now.Before(p.ExpiresAt) {
		return unknown("price is outside its validity interval")
	}
	if !p.AllChargesBounded {
		return unknown("tariff does not bound all provider charges")
	}
	rate := max(p.InputNanoPerMillion, p.CacheWriteNanoPerMillion, p.CacheReadNanoPerMillion)
	amount, err := priceSum(p.FixedNano, [][2]int64{{input, rate}, {output, p.OutputNanoPerMillion}})
	if err != nil {
		return PriceQuote{}, err
	}
	copy := *p
	return PriceQuote{Known: true, Nano: amount, Currency: p.Currency, Snapshot: &copy}, nil
}

// ReportedPrice uses the admission-time snapshot, even if its validity interval
// has since expired. Nil usage remains unknown. Cache tokens are subsets of
// InputTokens, not extra input charges. Invalid/inconsistent counters fail so
// callers retain their conservative reservation instead of undercharging.
func ReportedPrice(p PriceSnapshot, u *llm.Usage) (PriceQuote, error) {
	if err := p.Validate(); err != nil {
		return PriceQuote{}, err
	}
	q := PriceQuote{Currency: p.Currency, Snapshot: &p}
	if u == nil {
		q.Reason = "provider usage unavailable"
		return q, nil
	}
	if !p.AllChargesBounded {
		q.Reason = "tariff does not bound all provider charges"
		return q, nil
	}
	input, output, write, read := int64(u.InputTokens), int64(u.OutputTokens), int64(u.CacheCreationInputTokens), int64(u.CacheReadInputTokens)
	if input < 0 || output < 0 || write < 0 || read < 0 || write > input || read > input-write {
		return q, errors.New("inconsistent provider token counters")
	}
	amount, err := priceSum(p.FixedNano, [][2]int64{{input - write - read, p.InputNanoPerMillion}, {write, p.CacheWriteNanoPerMillion}, {read, p.CacheReadNanoPerMillion}, {output, p.OutputNanoPerMillion}})
	if err != nil {
		return q, err
	}
	q.Known = true
	q.Nano = amount
	return q, nil
}

// Integer accumulation rounds the final token charge up once to a nano-unit.
// big.Int avoids multiplication overflow before checking the representable sum.
func priceSum(fixed int64, terms [][2]int64) (int64, error) {
	total := new(big.Int)
	for _, term := range terms {
		if term[0] < 0 || term[1] < 0 || fixed < 0 {
			return 0, errors.New("negative pricing operand")
		}
		total.Add(total, new(big.Int).Mul(big.NewInt(term[0]), big.NewInt(term[1])))
	}
	total.Add(total, big.NewInt(999999))
	total.Quo(total, big.NewInt(1000000))
	total.Add(total, big.NewInt(fixed))
	if !total.IsInt64() {
		return 0, errors.New("request price exceeds representable amount")
	}
	return total.Int64(), nil
}
