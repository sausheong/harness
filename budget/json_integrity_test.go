package budget

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestBudgetJSONAmbiguityRefusesReopenedAdmission(t *testing.T) {
	cases := []struct{ name, kind, payload string }{
		{"token-duplicate", tokenKind, `{"version":1,"kind":"limit","tokens":1,"tokens":1000}`},
		{"token-escaped", tokenKind, `{"version":1,"kind":"limit","tokens":1,"to\u006bens":1000}`},
		{"token-alias", tokenKind, `{"version":1,"kind":"limit","tokens":1,"TOKENS":1000}`},
		{"cost-duplicate", costKind, `{"version":1,"kind":"decision","currency":"USD","limit_nano":1,"limit_nano":1000}`},
		{"cost-alias", costKind, `{"version":1,"kind":"decision","currency":"USD","limit_nano":1,"LIMIT_NANO":1000}`},
		{"cost-null-strict", costKind, `{"version":1,"kind":"decision","currency":"USD","limit_nano":1000,"strict":null}`},
		{"deadline-duplicate", deadlineKind, `{"version":1,"decided_at":"2026-09-09T00:00:00Z","deadline":"2026-09-09T00:00:01Z","deadline":"2026-09-10T00:00:00Z"}`},
		{"deadline-alias", deadlineKind, `{"version":1,"decided_at":"2026-09-09T00:00:00Z","deadline":"2026-09-09T00:00:01Z","DEADLINE":"2026-09-10T00:00:00Z"}`},
	}
	for _, tc := range cases {
		for _, run := range []bool{false, true} {
			name := tc.name
			if run {
				name += "-run"
			}
			t.Run(name, func(t *testing.T) {
				kind := tc.kind
				if run {
					var err error
					kind, err = runKind(kind, "logical")
					if err != nil {
						t.Fatal(err)
					}
				}
				root := t.TempDir()
				store := session.NewStore(root)
				sess, err := store.LoadExclusive("hand", "key")
				if err != nil {
					t.Fatal(err)
				}
				if err := sess.Annotate(kind, json.RawMessage(tc.payload)); err != nil {
					t.Fatal(err)
				}
				if err := sess.Close(); err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(root, "hand", "key.jsonl")
				before, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				sess, err = store.LoadExclusive("hand", "key")
				if err != nil {
					t.Fatal(err)
				}
				defer sess.Close()
				switch tc.kind {
				case tokenKind:
					ledger, err := openTokenLedger(sess, kind)
					if err == nil {
						t.Fatal("ambiguous token ceiling accepted")
					}
					if finish, err := ledger.Admission(func(llm.ChatRequest) (int64, error) { return 1, nil })(context.Background(), llm.ChatRequest{Model: "m", MaxTokens: 2}, llm.CallGeneration, "attempt"); err == nil || finish != nil {
						t.Fatal("admitted corrupt token state")
					}
				case costKind:
					ledger, err := openCostLedger(sess, kind)
					if err == nil {
						t.Fatal("ambiguous cost ceiling accepted")
					}
					if err := ledger.Decide(context.Background(), "USD", 2000, false); err == nil {
						t.Fatal("appended over corrupt cost state")
					}
				case deadlineKind:
					if _, err := readDeadline(sess, kind); err == nil {
						t.Fatal("ambiguous deadline accepted")
					}
				}
				if err := sess.Close(); err != nil {
					t.Fatal(err)
				}
				after, err := os.ReadFile(path)
				if err != nil || !bytes.Equal(before, after) {
					t.Fatal("corrupt budget changed", err)
				}
			})
		}
	}
}

func TestBudgetNestedQuoteIntegrity(t *testing.T) {
	price := currentFixturePrice()
	event := costEvent{Version: 1, Kind: "reserve", Attempt: &CostAttempt{ID: "one", Model: price.Model, Category: llm.CallGeneration, Quote: PriceQuote{Known: true, Nano: 1, Currency: "USD", Snapshot: &price}, ChargedNano: 1, Status: "reserved"}}
	raw, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"valid", "duplicate-price", "alias-price", "null-price", "duplicate-quote"} {
		t.Run(name, func(t *testing.T) {
			payload := string(raw)
			switch name {
			case "duplicate-price":
				payload = strings.Replace(payload, `"fixed_nano":`, `"fixed_nano":999,"fixed_nano":`, 1)
			case "alias-price":
				payload = strings.Replace(payload, `"fixed_nano":`, `"FIXED_NANO":999,"fixed_nano":`, 1)
			case "null-price":
				payload = strings.Replace(payload, `"fixed_nano":`+strconv.FormatInt(price.FixedNano, 10), `"fixed_nano":null`, 1)
			case "duplicate-quote":
				payload = strings.Replace(payload, `"nano":1`, `"nano":999,"nano":1`, 1)
			}
			sess := session.NewSession("a", "nested")
			ledger, err := OpenCostLedger(sess)
			if err != nil {
				t.Fatal(err)
			}
			if err := ledger.Decide(context.Background(), "USD", 1000000, true); err != nil {
				t.Fatal(err)
			}
			if err := sess.Annotate(costKind, json.RawMessage(payload)); err != nil {
				t.Fatal(err)
			}
			state, err := ledger.State()
			if name == "valid" {
				if err != nil || state.CommittedNano != 1 {
					t.Fatalf("valid nested record failed: %+v %v", state, err)
				}
			} else if err == nil {
				t.Fatal("ambiguous nested price accepted")
			}
		})
	}
}

func TestMissingTokenSettlementChargeDoesNotReleaseReservation(t *testing.T) {
	root := t.TempDir()
	store := session.NewStore(root)
	sess, err := store.LoadExclusive("a", "key")
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := OpenTokenLedger(sess)
	if err != nil {
		t.Fatal(err)
	}
	if err := ledger.Decide(context.Background(), 100); err != nil {
		t.Fatal(err)
	}
	if _, err := ledger.Admission(func(llm.ChatRequest) (int64, error) { return 10, nil })(context.Background(), llm.ChatRequest{Model: "m", MaxTokens: 10}, llm.CallGeneration, "one"); err != nil {
		t.Fatal(err)
	}
	if err := sess.Annotate(tokenKind, json.RawMessage(`{"version":1,"kind":"settle","id":"one","status":"reported"}`)); err != nil {
		t.Fatal(err)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "a", "key.jsonl")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sess, err = store.LoadExclusive("a", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	ledger, err = OpenTokenLedger(sess)
	if err == nil {
		state, _ := ledger.State()
		t.Fatalf("missing charge released reserved tokens: %+v", state)
	}
	if finish, err := ledger.Admission(func(llm.ChatRequest) (int64, error) { return 1, nil })(context.Background(), llm.ChatRequest{Model: "m", MaxTokens: 1}, llm.CallGeneration, "next"); err == nil || finish != nil {
		t.Fatal("corrupt settlement admitted request")
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("corrupt journal changed", err)
	}
}

func TestBudgetRequiredNestedFieldsAndExplicitZero(t *testing.T) {
	price := currentFixturePrice()
	event := costEvent{Version: 1, Kind: "reserve", Attempt: &CostAttempt{ID: "one", Model: price.Model, Category: llm.CallGeneration, Quote: PriceQuote{Known: true, Nano: 0, Currency: "USD", Snapshot: &price}, ChargedNano: 0, Status: "reserved"}}
	raw, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range [][]string{
		{"attempt", "charged_nano"},
		{"attempt", "quote", "nano"},
		{"attempt", "quote", "known"},
		{"attempt", "quote", "snapshot", "fixed_nano"},
		{"attempt", "quote", "snapshot", "input_nano_per_million"},
		{"attempt", "quote", "snapshot", "all_charges_bounded"},
	} {
		t.Run(strings.Join(path, "/"), func(t *testing.T) {
			var object map[string]any
			if err := json.Unmarshal(raw, &object); err != nil {
				t.Fatal(err)
			}
			current := object
			for _, key := range path[:len(path)-1] {
				current = current[key].(map[string]any)
			}
			delete(current, path[len(path)-1])
			corrupted, err := json.Marshal(object)
			if err != nil {
				t.Fatal(err)
			}
			var decoded costEvent
			if err := decodeBudgetJSON(corrupted, &decoded); err == nil {
				t.Fatal("missing required pricing field accepted")
			}
		})
	}
	// Explicit zero is still meaningful; optional false/empty fields emitted
	// with omitempty remain compatible with the existing writer contract.
	for _, value := range []any{event, costEvent{Version: 1, Kind: "decision", Currency: "USD", LimitNano: 100}, tokenEvent{Version: 1, Kind: "settle", ID: "one", Status: "reported", Tokens: 0}} {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		switch value.(type) {
		case costEvent:
			var decoded costEvent
			if err := decodeBudgetJSON(encoded, &decoded); err != nil {
				t.Fatal("writer output rejected", err)
			}
		case tokenEvent:
			var decoded tokenEvent
			if err := decodeBudgetJSON(encoded, &decoded); err != nil {
				t.Fatal("explicit zero rejected", err)
			}
		}
	}
}
