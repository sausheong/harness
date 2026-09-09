package attachment

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func openTest(t *testing.T, dir string, quota int64) *Store {
	t.Helper()
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir, quota)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestLargeAttachmentRoundTripAndDeduplication(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir, DefaultStoreBytes)
	data := bytes.Repeat([]byte{0x42}, 12<<20)
	ref, err := s.Put(context.Background(), data)
	if err != nil {
		t.Fatal(err)
	}
	duplicate, err := s.Put(context.Background(), data)
	if err != nil || duplicate != ref {
		t.Fatal(duplicate, err)
	}
	other := openTest(t, dir, DefaultStoreBytes)
	got, err := other.Read(context.Background(), ref)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatal("reopened blob changed", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatal("deduplication left extra files", entries)
	}
	info, err := os.Stat(filepath.Join(dir, ref.SHA256))
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatal(info, err)
	}
}

func TestAttachmentIntegrityAndTraversal(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir, 100)
	ref, err := s.Put(context.Background(), []byte("original"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ref.SHA256), []byte("modified"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Read(context.Background(), ref); err == nil {
		t.Fatal("tampering accepted")
	}
	if _, err := s.Put(context.Background(), []byte("original")); err == nil {
		t.Fatal("corrupt existing blob reused")
	}
	if _, err := s.Read(context.Background(), Ref{SHA256: "../outside", Size: 1}); err == nil {
		t.Fatal("path accepted as digest")
	}
	if err := os.Remove(filepath.Join(dir, ref.SHA256)); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "secret")
	if err := os.WriteFile(outside, []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, ref.SHA256)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Read(context.Background(), ref); err == nil {
		t.Fatal("symlink accepted")
	}
}

func TestAttachmentQuotaSerialisesIndependentStores(t *testing.T) {
	dir := t.TempDir()
	stores := []*Store{openTest(t, dir, 10), openTest(t, dir, 10)}
	var wg sync.WaitGroup
	errs := make([]error, 2)
	refs := make([]Ref, 2)
	for i := range stores {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			refs[i], errs[i] = stores[i].Put(context.Background(), bytes.Repeat([]byte{byte(i)}, 8))
		}(i)
	}
	wg.Wait()
	success := 0
	for i, err := range errs {
		if err == nil {
			success++
			if _, err := stores[i].Read(context.Background(), refs[i]); err != nil {
				t.Fatal(err)
			}
		}
	}
	if success != 1 {
		t.Fatalf("quota accepted %d independent writes: %v", success, errs)
	}
}

func TestAttachmentCancellationAndPendingCleanup(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir, 100)
	guard, err := s.root.OpenFile(".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer guard.Close()
	if err := lock(context.Background(), guard); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err = s.Put(ctx, []byte("cancelled"))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	unlock(guard)
	if err := os.WriteFile(filepath.Join(dir, ".pending-orphan"), []byte("unfinished"), 0600); err != nil {
		t.Fatal(err)
	}
	ref, err := s.Put(context.Background(), []byte("complete"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, ".pending-orphan")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("orphan retained", err)
	}
	cancelled, stop := context.WithCancel(context.Background())
	stop()
	if _, err := s.Read(cancelled, ref); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.Read(context.Background(), ref); err != nil {
		t.Fatal("cancellation damaged content", err)
	}
}

func TestAttachmentLimitsAndPrivateDirectory(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir, 100)
	if _, err := s.Put(context.Background(), make([]byte, MaxBlobBytes+1)); err == nil {
		t.Fatal("oversized blob accepted")
	}
	if _, err := s.Put(context.Background(), make([]byte, 101)); err == nil {
		t.Fatal("quota exceeded")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Name() != ".lock" {
			t.Fatal("failed put published file", entry.Name())
		}
	}
	if _, err := Open(t.TempDir(), 0); err == nil {
		t.Fatal("invalid quota accepted")
	}
	public := t.TempDir()
	if err := os.Chmod(public, 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(public, 100); err == nil {
		t.Fatal("public directory accepted")
	}
}
