package execution

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/sausheong/harness/process"
)

func TestHandleCompletionIncludesBackendCleanup(t *testing.T) {
	store, err := process.NewArtifactStore(filepath.Join(t.TempDir(), "capture"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := process.StartHandle(context.Background(), store, t.TempDir(), "/bin/sh", "-c", "exit 0")
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	h := ownedHandle(raw, func() error { close(entered); <-release; return nil })
	<-entered
	if !h.Snapshot().Running {
		t.Fatal("terminal visible before cleanup")
	}
	close(release)
	if err = h.Wait(); err != nil {
		t.Fatal(err)
	}
	if h.Snapshot().Running {
		t.Fatal("joined handle remains running")
	}
}
