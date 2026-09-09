package budget

import (
	"bytes"
	"encoding/json"
	"testing"
)

func FuzzBudgetRecordJSON(f *testing.F) {
	f.Add(byte(0), []byte(`{"version":1,"kind":"limit","tokens":100}`))
	f.Add(byte(0), []byte(`{"version":1,"kind":"settle","id":"one","status":"reported","tokens":0}`))
	f.Add(byte(0), []byte(`{"version":1,"kind":"limit","tokens":1,"to\u006bens":1000}`))
	f.Add(byte(1), []byte(`{"version":1,"kind":"decision","currency":"USD","limit_nano":100,"strict":true}`))
	f.Add(byte(2), []byte(`{"version":1,"decided_at":"2026-09-09T00:00:00Z","deadline":"2026-09-09T01:00:00Z"}`))
	f.Add(byte(2), []byte(`{"version":1,"decided_at":"2026-09-09T00:00:00+00:00","deadline":"2026-09-09T01:00:00+00:00"}`))
	f.Fuzz(func(t *testing.T, kind byte, raw []byte) {
		if len(raw) > 128<<10 {
			return
		}
		var first, second any
		switch kind % 3 {
		case 0:
			first, second = new(tokenEvent), new(tokenEvent)
		case 1:
			first, second = new(costEvent), new(costEvent)
		default:
			first, second = new(DeadlineState), new(DeadlineState)
		}
		if err := decodeBudgetJSON(raw, first); err != nil {
			return
		}
		canonical, err := json.Marshal(first)
		if err != nil {
			t.Fatal(err)
		}
		if err := decodeBudgetJSON(canonical, second); err != nil {
			t.Fatalf("accepted record cannot round trip: %v", err)
		}
		// Compare persisted semantics, not time.Time's internal Location
		// pointer, which may normalise a zero offset to UTC on encoding.
		repeated, err := json.Marshal(second)
		if err != nil || !bytes.Equal(canonical, repeated) {
			t.Fatal("canonical round trip changed budget record", err)
		}
	})
}
