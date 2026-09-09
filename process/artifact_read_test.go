//go:build unix

package process

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestArtifactReadVerifiedAndTampered(t *testing.T) {
	store := testStore(t)
	capture := store.Capture(4)
	data := bytes.Repeat([]byte("captured output\n"), 100)
	capture.Write(data)
	if _, err := store.Read(context.Background(), capture.Info()); err == nil {
		t.Fatal("active unfinalised capture accepted")
	}
	if err := capture.Close(); err != nil {
		t.Fatal(err)
	}
	info := capture.Info()
	got, err := store.Read(context.Background(), info)
	if err != nil || !bytes.Equal(got, data) || len(info.SHA256) != 64 {
		t.Fatal("verified artifact unavailable", err)
	}
	tampered := bytes.Repeat([]byte("x"), len(data))
	if err := os.WriteFile(info.Path, tampered, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Read(context.Background(), info); err == nil {
		t.Fatal("same-size tampering accepted")
	}
	if err := os.Remove(info.Path); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Read(context.Background(), info); !os.IsNotExist(err) {
		t.Fatal("missing artifact not reported", err)
	}
}
func TestArtifactReadScopeLockAndCancellation(t *testing.T) {
	store := testStore(t)
	capture := store.Capture(1)
	capture.Write([]byte("complete output"))
	capture.Close()
	info := capture.Info()
	outside := info
	outside.Path = filepath.Join(t.TempDir(), filepath.Base(info.Path))
	if _, err := store.Read(context.Background(), outside); err == nil {
		t.Fatal("outside path accepted")
	}
	link := filepath.Join(store.dir, "output-link.log")
	if err := os.Symlink(info.Path, link); err != nil {
		t.Fatal(err)
	}
	alias := info
	alias.Path = link
	if _, err := store.Read(context.Background(), alias); err == nil {
		t.Fatal("symlink accepted")
	}
	file, err := os.OpenFile(info.Path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := lockArtifact(file, false); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Read(context.Background(), info); err == nil {
		t.Fatal("locked artifact accepted")
	}
	file.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Read(ctx, info); err != context.Canceled {
		t.Fatal(err)
	}
	legacy := info
	legacy.SHA256 = ""
	if _, err := store.Read(context.Background(), legacy); err == nil {
		t.Fatal("unverified legacy record accepted")
	}
}
