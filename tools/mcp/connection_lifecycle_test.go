//go:build darwin || linux

package mcp

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestCancelledHandshakeReapsChildPromptly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pid")
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	cli, err := Connect(ctx, ServerConfig{Name: "hung", Command: "sh", Args: []string{"-c", `echo $$ > "$1"; exec sleep 30`, "fixture", path}})
	elapsed := time.Since(start)
	if cli != nil {
		cli.Close()
		t.Fatal("hung handshake connected")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("connect error %v", err)
	}
	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	pid, parseErr := strconv.Atoi(strings.TrimSpace(string(data)))
	if parseErr != nil {
		t.Fatal(parseErr)
	}
	aliveErr := syscall.Kill(pid, 0)
	if aliveErr == nil {
		syscall.Kill(pid, syscall.SIGKILL)
		t.Fatal("connection returned with child still alive")
	}
	if !errors.Is(aliveErr, syscall.ESRCH) {
		t.Fatalf("child status %v", aliveErr)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("cancelled handshake took %s", elapsed)
	}
}
