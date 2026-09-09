package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkerSnapshotPinsBytesAcrossSourceReplacement(t *testing.T) {
	source := filepath.Join(t.TempDir(), "worker")
	original := []byte("#!/bin/sh\nprintf original\n")
	if err := os.WriteFile(source, original, 0700); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(original)
	digest := hex.EncodeToString(hash[:])
	frozen, err := snapshotWorker(context.Background(), source, digest, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(source, []byte("#!/bin/sh\nprintf replaced\n"), 0700); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(frozen)
	if err != nil || string(raw) != string(original) {
		t.Fatal("snapshot changed with source")
	}
	dir := t.TempDir()
	if _, err = snapshotWorker(context.Background(), source, digest, dir); err == nil {
		t.Fatal("changed worker accepted")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatal("failed snapshot retained")
	}
}
func TestWorkerSnapshotRejectsCancellationAndInvalidDigest(t *testing.T) {
	source := filepath.Join(t.TempDir(), "worker")
	if err := os.WriteFile(source, []byte("worker"), 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotWorker(context.Background(), source, "invalid", t.TempDir()); err == nil {
		t.Fatal("invalid digest accepted")
	}
	hash := sha256.Sum256([]byte("worker"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := snapshotWorker(ctx, source, hex.EncodeToString(hash[:]), t.TempDir()); err != context.Canceled {
		t.Fatal(err)
	}
}
