package session

import (
	"context"
	"errors"
	"os"
	"testing"
)

func TestSelectionRecordAmbiguityCannotRedirectRestoredBranch(t *testing.T) {
	prefix := `{"id":"a","type":"message","data":{"text":"branch a"}}` + "\n" + `{"id":"b","parentId":"a","type":"message","data":{"text":"branch b"}}` + "\n"
	for name, raw := range map[string]string{
		"duplicate_parent":        `{"id":"s","parentId":"a","parentId":"b","type":"selection","data":{"version":1}}`,
		"escaped_parent":          `{"id":"s","parentId":"a","\u0070arentId":"b","type":"selection","data":{"version":1}}`,
		"case_alias_parent":       `{"id":"s","parentId":"a","ParentId":"b","type":"selection","data":{"version":1}}`,
		"duplicate_type":          `{"id":"s","parentId":"a","type":"message","type":"selection","data":{"version":1}}`,
		"duplicate_data":          `{"id":"s","parentId":"a","type":"selection","data":{"version":2},"data":{"version":1}}`,
		"case_alias_data":         `{"id":"s","parentId":"a","type":"selection","Data":{"version":1}}`,
		"duplicate_version":       `{"id":"s","parentId":"a","type":"selection","data":{"version":2,"version":1}}`,
		"case_alias_version":      `{"id":"s","parentId":"a","type":"selection","data":{"Version":1}}`,
		"unknown_selection_field": `{"id":"s","parentId":"a","type":"selection","data":{"version":1,"other":true}}`,
	} {
		t.Run(name, func(t *testing.T) {
			content := prefix + raw + "\n"
			store, path := recoveryFixture(t, content)
			sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
			if sess != nil {
				sess.Close()
			}
			var record *RecordError
			if sess != nil || !errors.As(err, &record) || record.RecoverableTail || report.Repaired {
				t.Fatalf("ambiguous selection accepted or repaired: %v", err)
			}
			after, err := os.ReadFile(path)
			if err != nil || string(after) != content {
				t.Fatal("corrupt branch evidence changed", err)
			}
		})
	}
}
