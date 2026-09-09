package tool

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreationPathResolvesExistingAncestors(t *testing.T) {
	work, outside := t.TempDir(), t.TempDir()
	alias := filepath.Join(t.TempDir(), "workspace-link")
	if err := os.Symlink(work, alias); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePathInWorkDir(filepath.Join(alias, "new/deep/file"), alias); err != nil {
		t.Fatal("safe missing parents rejected", err)
	}
	if err := os.Symlink(outside, filepath.Join(work, "escape")); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePathInWorkDir(filepath.Join(work, "escape/new/deep/file"), work); err == nil {
		t.Fatal("missing parents hid outside ancestor")
	}
	if err := os.Symlink(filepath.Join(outside, "missing"), filepath.Join(work, "dangling")); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePathInWorkDir(filepath.Join(work, "dangling/new/file"), work); err == nil {
		t.Fatal("dangling symlink accepted")
	}
}
