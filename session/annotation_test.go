package session

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestAnnotationsPreserveSelectedLeafAcrossRestartAndCompaction(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Annotate("usage", json.RawMessage(`{"tokens":1}`)); err != nil {
		t.Fatal(err)
	}
	if sess.LeafID() != "" || len(sess.History()) != 0 {
		t.Fatal("annotation created conversation")
	}
	sess.Append(UserMessageEntry("original"))
	leaf := sess.LeafID()
	if err := sess.Annotate("usage", json.RawMessage(`{"tokens":2}`)); err != nil {
		t.Fatal(err)
	}
	if sess.LeafID() != leaf || len(sess.History()) != 1 {
		t.Fatal("annotation changed branch")
	}
	annotationID := sess.Entries()[len(sess.Entries())-1].ID
	if err := sess.Branch(annotationID); err == nil {
		t.Fatal("annotation selected as conversation")
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	sess, err = store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	if sess.LeafID() != leaf || len(sess.Annotations("usage")) != 2 || len(sess.History()) != 1 {
		t.Fatal("restart lost annotation or branch")
	}
	records := sess.Annotations("usage")
	records[0].Payload[0] = 'x'
	if !json.Valid(sess.Annotations("usage")[0].Payload) {
		t.Fatal("annotation getter leaked mutable payload")
	}
	sess.Compact("summary", 0)
	if err := sess.Flush(); err != nil {
		t.Fatal(err)
	}
	if len(sess.Annotations("usage")) != 2 {
		t.Fatal("compaction discarded accounting")
	}
	if err := sess.Branch(leaf); err != nil {
		t.Fatal(err)
	}
	if len(sess.History()) != 1 {
		t.Fatal("annotation leaked into original history")
	}
}

func TestAnnotationValidation(t *testing.T) {
	sess := NewSession("agent", "key")
	for _, tc := range []struct{ kind, payload string }{{"bad kind", "{}"}, {"", "{}"}, {"usage", "invalid"}, {"usage", `"` + strings.Repeat("x", MaxAnnotationBytes) + `"`}} {
		if err := sess.Annotate(tc.kind, json.RawMessage(tc.payload)); err == nil {
			t.Fatal("invalid annotation accepted")
		}
	}
	if len(sess.Entries()) != 0 || sess.PersistenceError() != nil {
		t.Fatal("invalid input mutated or degraded session")
	}
	if _, err := decodeSessionRecords(strings.NewReader(`{"id":"a","type":"annotation","data":{"version":2,"kind":"usage","payload":{}}}`)); err == nil {
		t.Fatal("future annotation version accepted")
	}
}

func TestAnnotationConcurrentAppendAndClosedWriter(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	workers.Add(2)
	go func() {
		defer workers.Done()
		for i := 0; i < 25; i++ {
			sess.Append(UserMessageEntry("message"))
		}
	}()
	go func() {
		defer workers.Done()
		for i := 0; i < 25; i++ {
			if err := sess.Annotate("usage", json.RawMessage(`{"tokens":1}`)); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	workers.Wait()
	leaf := sess.LeafID()
	if len(sess.History()) != 25 || len(sess.Annotations("usage")) != 25 {
		t.Fatal("concurrent append lost records")
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	if err := sess.Annotate("usage", json.RawMessage(`{}`)); err == nil {
		t.Fatal("annotation bypassed closed writer")
	}
	reopened, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if reopened.LeafID() != leaf || len(reopened.History()) != 25 || len(reopened.Annotations("usage")) != 25 {
		t.Fatal("durable graph differs after concurrent annotations")
	}
}
