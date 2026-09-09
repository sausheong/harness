//go:build unix

package process

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestHandleSendReadWait(t *testing.T) {
	h, err := StartHandle(context.Background(), testStore(t), t.TempDir(), "sh", "-c", "read line; printf 'received:%s' \"$line\"")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := h.Send(ctx, []byte("hello\n")); err != nil {
		t.Fatal(err)
	}
	if err := h.Wait(); err != nil {
		t.Fatal(err)
	}
	result := h.Snapshot()
	if result.Running || result.ExitCode != 0 || result.Stdout != "received:hello" {
		t.Fatal(result)
	}
	if err := h.Send(ctx, []byte("late")); err == nil {
		t.Fatal("input accepted after exit")
	}
}
func TestHandleBoundedCaptureAndOwnerCancellation(t *testing.T) {
	store := testStore(t)
	h, err := StartHandle(context.Background(), store, t.TempDir(), "sh", "-c", "head -c 70000 /dev/zero")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if err := h.Wait(); err != nil {
		t.Fatal(err)
	}
	result := h.Snapshot()
	if len(result.Stdout) != 64<<10 || result.StdoutBytes != 70000 || !result.StdoutTruncated {
		t.Fatal("capture bound failed")
	}
	data, err := store.Read(context.Background(), result.StdoutArtifact)
	if err != nil || len(data) != 70000 {
		t.Fatal("full capture unavailable", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	server, err := StartHandle(ctx, store, t.TempDir(), "sh", "-c", "sleep 30 & wait")
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	cancel()
	done := make(chan error, 1)
	go func() { done <- server.Wait() }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("owned process did not stop")
	}
	if server.Snapshot().Running {
		t.Fatal("cancelled process still running")
	}
}
func TestHandleRejectsOversizedInput(t *testing.T) {
	h, err := StartHandle(context.Background(), testStore(t), t.TempDir(), "sh", "-c", "sleep 30")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	if err := h.Send(context.Background(), []byte(strings.Repeat("x", HandleInputLimit+1))); err == nil {
		t.Fatal("oversized input accepted")
	}
}

func TestHandleCancelledInputStopsAndJoinsProcess(t *testing.T) {
	h, err := StartHandle(context.Background(), testStore(t), t.TempDir(), "sh", "-c", "sleep 30")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	payload := []byte(strings.Repeat("x", HandleInputLimit))
	// A process that never reads eventually fills its pipe; delivery cancellation
	// must stop it rather than leave a queued or partially delivered write alive.
	for i := 0; i < 32; i++ {
		err = h.Send(ctx, payload)
		if err != nil {
			break
		}
	}
	if err != context.DeadlineExceeded {
		t.Fatal("blocked input did not observe deadline", err)
	}
	if h.Snapshot().Running {
		t.Fatal("cancelled input returned before process joined")
	}
}
