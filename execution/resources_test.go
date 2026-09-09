package execution

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedResourceSnapshotIntegrity(t *testing.T) {
	source := filepath.Join(t.TempDir(), "asset")
	if err := os.WriteFile(source, []byte("reviewed"), 0600); err != nil {
		t.Fatal(err)
	}
	pin := fmt.Sprintf("%x", sha256.Sum256([]byte("reviewed")))
	root := t.TempDir()
	files := []ReadOnlyResource{{Path: source, Relative: "package/asset.txt", SHA256: pin}}
	frozen, err := snapshotResources(context.Background(), files, root)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(source, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(frozen, "package/asset.txt"))
	if err != nil || string(b) != "reviewed" {
		t.Fatal(string(b), err)
	}
	if err = os.RemoveAll(frozen); err != nil {
		t.Fatal("snapshot not removable", err)
	}
	if _, err = snapshotResources(context.Background(), files, root); err == nil {
		t.Fatal("changed source accepted")
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 0 {
		t.Fatal("partial snapshot retained", entries, err)
	}
	for _, paths := range [][]string{{"../escape"}, {"a", "a/b"}, {"A", "a"}, {"bad,path"}} {
		var invalid []ReadOnlyResource
		for _, path := range paths {
			invalid = append(invalid, ReadOnlyResource{Path: source, Relative: path, SHA256: pin})
		}
		if _, err = snapshotResources(context.Background(), invalid, root); err == nil {
			t.Fatal("invalid layout accepted", paths)
		}
	}
	link := filepath.Join(t.TempDir(), "link")
	if err = os.Symlink(source, link); err != nil {
		t.Fatal(err)
	}
	files[0].Path = link
	if _, err = snapshotResources(context.Background(), files, root); err == nil {
		t.Fatal("symlink source accepted")
	}
}
