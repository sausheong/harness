package session

import (
	"os"
	"reflect"
	"sync"
	"testing"
)

func TestSelectedBranchSurvivesRestartAndRewrite(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Append(UserMessageEntry("root"))
	root := sess.LeafID()
	sess.Append(AssistantMessageEntry("first branch"))
	first := sess.LeafID()
	if err := sess.Branch(root); err != nil {
		t.Fatal(err)
	}
	sess.Append(AssistantMessageEntry("second branch"))
	second := sess.LeafID()
	if err := sess.Branch(first); err != nil {
		t.Fatal(err)
	}
	for _, rewrite := range []bool{false, true} {
		if rewrite {
			store.Rewrite(sess)
		}
		loaded, err := store.Load("agent", "key")
		if err != nil || loaded.LeafID() != first {
			t.Fatal("selected branch lost on restart", err)
		}
		history := loaded.History()
		if len(history) != 2 || history[0].ID != root || history[1].ID != first {
			t.Fatal("selection leaked into history", history)
		}
		if !reflect.DeepEqual(loaded.View(), history) {
			t.Fatal("view differs from selected branch")
		}
	}
	sess.Append(UserMessageEntry("continue first"))
	loaded, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	history := loaded.History()
	if len(history) != 3 || history[1].ID != first || history[2].ParentID != first {
		t.Fatal("append ignored selected parent")
	}
	if err := sess.Branch(second); err != nil {
		t.Fatal("other branch lost", err)
	}
}

func TestLegacyCompactionPreservesOriginalBranches(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	for _, msg := range []string{"root", "middle", "last"} {
		sess.Append(UserMessageEntry(msg))
	}
	original := sess.History()
	if err := sess.Branch(original[0].ID); err != nil {
		t.Fatal(err)
	}
	sess.Append(AssistantMessageEntry("alternative"))
	other := sess.LeafID()
	if err := sess.Branch(original[2].ID); err != nil {
		t.Fatal(err)
	}
	sess.Compact("summary", 1)
	if err := sess.Flush(); err != nil {
		t.Fatal(err)
	}
	current := sess.History()
	if len(current) != 2 || current[1].ID == original[2].ID {
		t.Fatal("compaction must clone retained nodes")
	}
	loaded, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.Branch(original[2].ID); err == nil {
		t.Fatal("legacy writer bypassed active lease")
	}
	sess.Close()
	loaded, err = store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if err := loaded.Branch(original[2].ID); err != nil {
		t.Fatal("original history discarded", err)
	}
	if !reflect.DeepEqual(loaded.History(), original) {
		t.Fatal("original graph mutated")
	}
	if err := loaded.Branch(other); err != nil || len(loaded.History()) != 2 {
		t.Fatal("unselected branch discarded", err)
	}
}

func TestFailedBranchPersistenceIsVisible(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Append(UserMessageEntry("root"))
	root := sess.LeafID()
	sess.Append(UserMessageEntry("tail"))
	previous := sess.LeafID()
	path := store.sessionPath("agent", "key")
	if err := os.Rename(path, path+".original"); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	if err := sess.Branch(root); err == nil || sess.PersistenceError() == nil {
		t.Fatal("branch storage failure hidden")
	}
	if sess.LeafID() != previous {
		t.Fatal("failed selection changed memory leaf")
	}
}

func TestConcurrentBranchAppendAndViews(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Append(UserMessageEntry("root"))
	root := sess.LeafID()
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			if err := sess.Branch(root); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			sess.Append(UserMessageEntry("append"))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			sess.History()
			sess.View()
			sess.Entries()
		}
	}()
	wg.Wait()
	if err := sess.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || !reflect.DeepEqual(loaded.Entries(), sess.Entries()) || loaded.LeafID() != sess.LeafID() {
		t.Fatal("concurrent graph differs on disk", err)
	}
}
