package session

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSessionRecordErrorsDistinguishTailFromInterior(t *testing.T) {
	prefix := `{"id":"first","type":"message","data":{"text":"retained"}}` + "\n"
	cases := []struct {
		name, body string
		tail       bool
		line       int
		offset     int64
	}{
		{"unfinished tail", prefix + `{"id":"second","data":`, true, 2, int64(len(prefix))},
		{"unfinished interior", prefix + `{"id":` + "\n" + prefix, false, 2, int64(len(prefix))},
		{"terminated corrupt tail", prefix + `{"id":` + "\n", false, 2, int64(len(prefix))},
		{"invalid tail", prefix + `not JSON`, false, 2, int64(len(prefix))},
		{"trailing object", prefix + `{} {}`, false, 2, int64(len(prefix))},
		{"null record", `null`, false, 1, 0},
		{"oversized", strings.Repeat("x", MaxSessionRecordBytes+1), false, 1, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "agent")
			if err := os.MkdirAll(dir, 0700); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, "key.jsonl")
			if err := os.WriteFile(path, []byte(tc.body), 0600); err != nil {
				t.Fatal(err)
			}
			sess, err := NewStore(root).Load("agent", "key")
			var record *RecordError
			if sess != nil || !errors.As(err, &record) || record.RecoverableTail != tc.tail || record.Line != tc.line || record.Offset != tc.offset {
				t.Fatalf("wrong corruption classification: %v", err)
			}
			raw, _ := os.ReadFile(path)
			if string(raw) != tc.body {
				t.Fatal("read silently modified session")
			}
		})
	}
}

func TestCompleteFinalRecordWithoutNewlineLoads(t *testing.T) {
	entries, err := decodeSessionRecords(strings.NewReader("\n" + `{"id":"first","type":"message"}`))
	if err != nil || len(entries) != 1 || entries[0].ID != "first" {
		t.Fatal(entries, err)
	}
}

func TestAppendDelimitsCompleteUnterminatedRecord(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "agent")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "key.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"first","type":"message"}`), 0600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(root)
	s, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	s.Append(UserMessageEntry("next"))
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	s.Close()
	loaded, err := store.Load("agent", "key")
	if err != nil || len(loaded.Entries()) != 2 {
		t.Fatal("append corrupted unterminated final record", err)
	}
}

func FuzzSessionRecords(f *testing.F) {
	for _, seed := range []string{
		`{"id":"entry","type":"message"}` + "\n", `{"id":`, "null\n", `{} {}`, "\xff",
		`{"id":"root","type":"message"}` + "\n" + `{"id":"s","parentId":"root","type":"selection","data":{"version":1}}` + "\n",
		`{"id":"a","type":"annotation","data":{"version":1,"kind":"budget","payload":{"limit":1}}}` + "\n",
		`{"id":"a","type":"annotation","data":{"version":1,"kind":"budget","kind":"ignored","payload":{}}}` + "\n",
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		_, err := decodeSessionRecords(bytes.NewReader(raw))
		if err == nil {
			return
		}
		var record *RecordError
		if !errors.As(err, &record) || record.Offset < 0 || record.Offset > int64(len(raw)) || record.Line < 1 {
			t.Fatal("invalid error location", err)
		}
		if record.RecoverableTail {
			tail := raw[record.Offset:]
			if bytes.Contains(tail, []byte{'\n'}) || !errors.Is(record.Cause, io.ErrUnexpectedEOF) || !utf8.Valid(tail) {
				t.Fatal("non-tail corruption classified recoverable", err)
			}
		}
	})
}
