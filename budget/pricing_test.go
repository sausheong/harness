package budget

import (
	"errors"
	"github.com/sausheong/harness/llm"
	"math"
	"testing"
	"time"
)

func fixturePrice() PriceSnapshot {
	return PriceSnapshot{Provider: "fixture", Model: "scripted", Destination: "http://127.0.0.1/v1", Currency: "USD", Source: "test fixture, not a market tariff", Version: "fixture-1", EffectiveAt: time.Unix(1000, 0), ExpiresAt: time.Unix(2000, 0), InputNanoPerMillion: 3000000000, OutputNanoPerMillion: 15000000000, CacheWriteNanoPerMillion: 3750000000, CacheReadNanoPerMillion: 300000000, FixedNano: 7, AllChargesBounded: true}
}
func TestPriceReservationAndCacheReconciliation(t *testing.T) {
	p := fixturePrice()
	q, err := ReservePrice(&p, 100, 20, time.Unix(1500, 0), true)
	if err != nil || !q.Known || q.Nano != 675007 || q.Currency != "USD" {
		t.Fatalf("quote %+v err %v", q, err)
	}
	actual, err := ReportedPrice(p, &llm.Usage{InputTokens: 100, OutputTokens: 20, CacheCreationInputTokens: 20, CacheReadInputTokens: 30})
	// 50 regular + 20 cache write + 30 cache read; no double-counted cache.
	if err != nil || actual.Nano != 534007 || !actual.Known {
		t.Fatalf("actual %+v err %v", actual, err)
	}
	p.Source = "changed"
	if q.Snapshot.Source == p.Source {
		t.Fatal("quote retained caller-owned snapshot")
	}
}
func TestUnknownPricesAreNotFree(t *testing.T) {
	for _, name := range []string{"missing", "expired", "future", "unbounded"} {
		t.Run(name, func(t *testing.T) {
			p := fixturePrice()
			ptr := &p
			now := time.Unix(1500, 0)
			switch name {
			case "missing":
				ptr = nil
			case "expired":
				now = p.ExpiresAt
			case "future":
				now = p.EffectiveAt.Add(-time.Second)
			case "unbounded":
				p.AllChargesBounded = false
			}
			advisory, err := ReservePrice(ptr, 1, 1, now, false)
			if err != nil || advisory.Known || advisory.Reason == "" {
				t.Fatalf("advisory %+v %v", advisory, err)
			}
			strict, err := ReservePrice(ptr, 1, 1, now, true)
			if !errors.Is(err, ErrUnknownPrice) || strict.Known {
				t.Fatalf("strict %+v %v", strict, err)
			}
		})
	}
	p := fixturePrice()
	q, err := ReportedPrice(p, nil)
	if err != nil || q.Known || q.Reason == "" {
		t.Fatal("missing usage charged as free")
	}
}
func TestPricingRefusesInvalidCountersAndOverflow(t *testing.T) {
	p := fixturePrice()
	for _, u := range []llm.Usage{{InputTokens: -1}, {OutputTokens: -1}, {InputTokens: 10, CacheCreationInputTokens: 8, CacheReadInputTokens: 3}, {InputTokens: 10, CacheReadInputTokens: -1}} {
		if _, err := ReportedPrice(p, &u); err == nil {
			t.Fatalf("accepted %+v", u)
		}
	}
	p.OutputNanoPerMillion = math.MaxInt64
	if _, err := ReservePrice(&p, 0, math.MaxInt64, time.Unix(1500, 0), true); err == nil {
		t.Fatal("overflow accepted")
	}
	p = fixturePrice()
	p.InputNanoPerMillion = -1
	if _, err := ReservePrice(&p, 1, 1, time.Unix(1500, 0), false); err == nil {
		t.Fatal("advisory accepted negative tariff")
	}
}
func TestPriceRoundingAndExplicitZeroTariff(t *testing.T) {
	amount, err := priceSum(0, [][2]int64{{1, 1}, {1, 1}})
	if err != nil || amount != 1 {
		t.Fatalf("rounding %d %v", amount, err)
	}
	p := fixturePrice()
	p.InputNanoPerMillion = 0
	p.OutputNanoPerMillion = 0
	p.CacheReadNanoPerMillion = 0
	p.CacheWriteNanoPerMillion = 0
	p.FixedNano = 0
	q, err := ReservePrice(&p, 10, 10, time.Unix(1500, 0), true)
	if err != nil || !q.Known || q.Nano != 0 {
		t.Fatalf("explicit free tariff %+v %v", q, err)
	}
}
