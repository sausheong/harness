package session

import (
	"context"
	"errors"
	"os"
	"testing"
)

func TestAmbiguousAnnotationEnvelopeCannotDisappearDuringLoad(t *testing.T) {
	for name, data := range map[string]string{
		"duplicate_kind":     `{"version":1,"kind":"budget","kind":"ignored","payload":{}}`,
		"escaped_kind":       `{"version":1,"kind":"budget","\u006bind":"ignored","payload":{}}`,
		"case_alias_kind":    `{"version":1,"kind":"budget","Kind":"ignored","payload":{}}`,
		"case_alias_version": `{"Version":1,"kind":"budget","payload":{}}`,
		"duplicate_version":  `{"version":2,"version":1,"kind":"budget","payload":{}}`,
		"duplicate_payload":  `{"version":1,"kind":"budget","payload":{"limit":1},"payload":{"limit":1000}}`,
		"case_alias_payload": `{"version":1,"kind":"budget","payload":{"limit":1},"Payload":{"limit":1000}}`,
		"unknown_field":      `{"version":1,"kind":"budget","payload":{},"unexpected":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			raw := `{"id":"a","type":"annotation","data":` + data + "}\n"
			store, path := recoveryFixture(t, raw)
			for _, exclusive := range []bool{false, true} {
				var sess *Session
				var err error
				if exclusive {
					sess, err = store.LoadExclusive("agent", "key")
				} else {
					sess, err = store.Load("agent", "key")
				}
				if sess != nil {
					sess.Close()
				}
				if err == nil || sess != nil {
					t.Errorf("ambiguous annotation loaded (exclusive=%v): %v", exclusive, err)
				}
			}
			sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
			if sess != nil {
				sess.Close()
			}
			var record *RecordError
			if sess != nil || !errors.As(err, &record) || record.RecoverableTail || report.Repaired {
				t.Errorf("ambiguous annotation accepted or repaired: %v", err)
			}
			after, err := os.ReadFile(path)
			if err != nil || string(after) != raw {
				t.Fatal("load changed corrupt evidence", err)
			}
		})
	}
}
