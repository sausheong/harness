package session

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGraphCorruptionRejectedWithoutTailRepair(t *testing.T) {
	root := `{"id":"root","type":"message"}` + "\n"
	for name, raw := range map[string]string{
		"missing ID":               `{}`,
		"duplicate ID":             root + root,
		"self cycle":               `{"id":"a","parentId":"a"}`,
		"future parent":            `{"id":"a","parentId":"b"}` + "\n" + `{"id":"b","parentId":"a"}`,
		"selection missing target": root + `{"id":"s","type":"selection","data":{"version":1}}`,
		"selection future version": root + `{"id":"s","parentId":"root","type":"selection","data":{"version":2}}`,
		"selection invalid data":   root + `{"id":"s","parentId":"root","type":"selection","data":"bad"}`,
		"selection parent":         root + `{"id":"s","parentId":"root","type":"selection","data":{"version":1}}` + "\n" + `{"id":"child","parentId":"s"}`,
	} {
		t.Run(name, func(t *testing.T) {
			store, path := recoveryFixture(t, raw)
			sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
			var record *RecordError
			if sess != nil || !errors.As(err, &record) || record.RecoverableTail || report.Repaired {
				t.Fatal("graph corruption accepted or repaired", err)
			}
			after, _ := os.ReadFile(path)
			if string(after) != raw {
				t.Fatal("graph corruption modified")
			}
		})
	}
}

func TestAppendRejectsDuplicateAndMissingParent(t *testing.T) {
	for _, duplicate := range []bool{false, true} {
		sess := NewSession("agent", "key")
		sess.Append(UserMessageEntry("root"))
		before := sess.Entries()
		entry := UserMessageEntry("invalid")
		if duplicate {
			entry.ID = sess.LeafID()
		} else {
			entry.ParentID = "missing"
		}
		sess.Append(entry)
		if sess.PersistenceError() == nil || !reflect.DeepEqual(before, sess.Entries()) {
			t.Fatal("invalid append changed graph")
		}
	}
}

func TestCommitCompactionPreservesOriginalGraphAndRestart(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	for _, text := range []string{"one", "two", "three"} {
		sess.Append(UserMessageEntry(text))
	}
	original := sess.History()
	summary := CompactionEntry("summary", original[0].ID, original[1].ID, "model", 0, 0, 2)
	summary.ParentID = original[1].ID
	if err := sess.CommitCompaction(original[2].ID, summary, original[2:]); err != nil {
		t.Fatal(err)
	}
	view := sess.View()
	if len(view) != 2 || view[0].Type != EntryTypeCompaction || view[1].ID == original[2].ID || view[1].ParentID != view[0].ID {
		t.Fatal("compaction view invalid", view)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || !reflect.DeepEqual(loaded.View(), view) {
		t.Fatal("compaction restart differs", err)
	}
	if err := sess.Branch(original[2].ID); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sess.History(), original) {
		t.Fatal("compaction mutated old graph")
	}
}

func TestCommitCompactionRejectsStaleView(t *testing.T) {
	sess := NewSession("agent", "key")
	sess.Append(UserMessageEntry("original"))
	previous := sess.LeafID()
	sess.Append(UserMessageEntry("new work"))
	before := sess.Entries()
	err := sess.CommitCompaction(previous, CompactionEntry("obsolete", previous, previous, "model", 0, 0, 1), nil)
	if !errors.Is(err, ErrSessionChanged) || !reflect.DeepEqual(sess.Entries(), before) {
		t.Fatal("stale compaction changed graph", err)
	}
}

func TestSelectionControlVersionRoundTrip(t *testing.T) {
	sess := NewSession("agent", "key")
	sess.Append(UserMessageEntry("root"))
	root := sess.LeafID()
	if err := sess.Branch(root); err != nil {
		t.Fatal(err)
	}
	marker := sess.Entries()[1]
	if marker.Type != EntryTypeSelection || string(marker.Data) != `{"version":1}` {
		t.Fatal("invalid marker", marker)
	}
	if err := sess.Branch(marker.ID); err == nil {
		t.Fatal("selected control record")
	}
	if _, err := decodeSessionRecords(strings.NewReader(`{"id":"legacy","type":"message"}`)); err != nil {
		t.Fatal("ordinary legacy record rejected", err)
	}
}
