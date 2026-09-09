package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestPersistenceErrorIsStickyAndSessionScoped(t *testing.T) {
	root := filepath.Join(t.TempDir(), "store")
	if err := os.WriteFile(root, []byte("obstruction"), 0600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(root)
	broken := NewSession("agent", "broken")
	broken.SetStore(store)
	broken.Append(UserMessageEntry("cannot persist"))
	first := broken.PersistenceError()
	if first == nil {
		t.Fatal("append failure hidden")
	}
	if err := os.Remove(root); err != nil {
		t.Fatal(err)
	}
	healthy := NewSession("agent", "healthy")
	healthy.SetStore(store)
	healthy.Append(UserMessageEntry("retained"))
	if err := healthy.Flush(); err != nil {
		t.Fatal(err)
	}
	broken.Append(UserMessageEntry("later write works"))
	if broken.PersistenceError() != first || broken.Flush() != first {
		t.Fatal("later write erased degraded state")
	}
	loaded, err := store.Load("agent", "healthy")
	if err != nil || len(loaded.Entries()) != 1 || loaded.PersistenceError() != nil {
		t.Fatal(loaded, err)
	}
}

func TestPersistenceCapturesMarshalAndFlushFailures(t *testing.T) {
	s := NewSession("agent", "invalid")
	s.SetStore(NewStore(t.TempDir()))
	s.Append(SessionEntry{Type: EntryTypeMessage, Data: json.RawMessage(`invalid-json`)})
	if s.PersistenceError() == nil {
		t.Fatal("marshal failure hidden")
	}
	missing := NewSession("agent", "missing")
	missing.SetStore(NewStore(t.TempDir()))
	if missing.Flush() == nil || missing.PersistenceError() == nil {
		t.Fatal("missing file sync accepted")
	}
	if err := NewSession("agent", "memory").Flush(); err != nil {
		t.Fatal(err)
	}
}

func TestPersistenceErrorConcurrentReaders(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "missing-parent", "store"))
	s := NewSession("agent", "session")
	s.SetStore(store)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				s.PersistenceError()
			}
		}()
	}
	s.Append(UserMessageEntry("saved"))
	wg.Wait()
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
}
