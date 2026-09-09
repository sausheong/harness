package session

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestRewriteMarshalFailurePreservesOriginal(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	original := NewSession("agent", "key")
	original.SetStore(store)
	original.Append(UserMessageEntry("recoverable original"))
	path := filepath.Join(root, "agent", "key.jsonl")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	replacement := NewSession("agent", "key")
	replacement.Append(UserMessageEntry("replacement prefix"))
	replacement.Append(SessionEntry{Type: EntryTypeMessage, Data: json.RawMessage("invalid JSON")})
	store.Rewrite(replacement)
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("failed rewrite destroyed original session")
	}
	if replacement.PersistenceError() == nil {
		t.Fatal("rewrite failure was hidden")
	}
	files, _ := os.ReadDir(filepath.Dir(path))
	regular := 0
	for _, file := range files {
		if !file.IsDir() {
			regular++
		}
	}
	if regular != 1 {
		t.Fatal("temporary rewrite leaked", files)
	}
}

func TestAtomicRewriteIsPrivateAndReloadable(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	old := NewSession("agent", "key")
	old.SetStore(store)
	old.Append(UserMessageEntry("old"))
	replacement := NewSession("agent", "key")
	replacement.Append(UserMessageEntry("replacement"))
	store.Rewrite(replacement)
	if err := replacement.PersistenceError(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "agent", "key.jsonl")
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Mode().Perm() != 0600 {
		t.Fatal("rewrite permissions", stat.Mode())
	}
	loaded, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Entries(), replacement.Entries()) {
		t.Fatal("rewrite did not survive reload")
	}
}

func TestConcurrentRewriteAndAppendPreserveDiskOrder(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	s := NewSession("agent", "key")
	s.SetStore(store)
	s.Append(UserMessageEntry("initial"))
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			s.Append(UserMessageEntry(fmt.Sprint(i)))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 40; i++ {
			store.Rewrite(s)
		}
	}()
	wg.Wait()
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Entries(), s.Entries()) || len(loaded.Entries()) != 101 {
		t.Fatal("concurrent rewrite lost/duplicated entries")
	}
}

func TestConcurrentLegacyCompactionAndAppendRemainConsistent(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	s := NewSession("agent", "key")
	s.SetStore(store)
	s.Append(UserMessageEntry("initial"))
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			s.Append(UserMessageEntry(fmt.Sprint(i)))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 40; i++ {
			s.Compact("summary", 4)
			s.View()
		}
	}()
	wg.Wait()
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Entries(), s.Entries()) {
		t.Fatal("compaction snapshot diverged from disk")
	}
}
