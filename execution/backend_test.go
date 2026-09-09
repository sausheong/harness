package execution

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHostExplicitEnvironmentAndExactArguments(t *testing.T) {
	t.Setenv("EXECUTION_TEST_SECRET", "must-not-inherit")
	h := Host{Workspace: t.TempDir()}
	result, err := h.Run(context.Background(), Request{Argv: []string{"/bin/sh", "-c", `test -z "$EXECUTION_TEST_SECRET"; printf '%s' "$1"`, "sh", "a; echo injected"}})
	if err != nil || result.Stdout != "a; echo injected" {
		t.Fatalf("%+v %v", result, err)
	}
	if !strings.Contains(h.Boundary(), "unrestricted host") {
		t.Fatal("host boundary hidden")
	}
}
func TestMissingContainerNeverExecutesHostCommand(t *testing.T) {
	path := filepath.Join(t.TempDir(), "must-not-exist")
	c := Container{Docker: "/missing/docker", Socket: "/missing/socket", Image: "sha256:" + strings.Repeat("a", 64), Workspace: t.TempDir()}
	if _, err := c.Run(context.Background(), Request{Argv: []string{"/bin/sh", "-c", "touch " + path}}); err == nil {
		t.Fatal("missing backend accepted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("host fallback ran")
	}
}

// This is an opt-in real-container suite. Its skip never qualifies isolation.
func TestNativeContainerBoundary(t *testing.T) {
	image, socket := os.Getenv("HARNESS_TEST_CONTAINER_IMAGE"), os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native container fixture not configured")
	}
	workspace := t.TempDir()
	if err := os.Chmod(workspace, 0777); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(secret, []byte("outside"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(workspace, "escape")); err != nil {
		t.Fatal(err)
	}
	c := Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: workspace}
	script := `set -eu; command -v /sbin/ip >/dev/null; command -v nc >/dev/null; test -z "${EXECUTION_TEST_SECRET-}"; test "$EXPLICIT_VALUE" = allowed; test ! -e /var/run/docker.sock; test ! -e /workspace/escape; ! touch /workspace/denied; ! touch /etc/denied; printf scratch >/tmp/allowed; sh -c 'test ! -e /workspace/escape; ! touch /etc/child-denied'; test -z "$(/sbin/ip route)"; ! nc -z -w 1 192.0.2.1 80; printf bounded`
	t.Setenv("EXECUTION_TEST_SECRET", "must-not-inherit")
	result, err := c.Run(context.Background(), Request{Argv: []string{"/bin/sh", "-c", script}, Env: map[string]string{"EXPLICIT_VALUE": "allowed"}})
	if err != nil || result.Stdout != "bounded" {
		t.Fatalf("boundary: %+v %v", result, err)
	}
	c.Writable = true
	result, err = c.Run(context.Background(), Request{Argv: []string{"/bin/sh", "-c", "cat > /workspace/allowed"}, Stdin: []byte("explicit write")})
	if err != nil {
		t.Fatalf("write: %+v %v", result, err)
	}
	raw, err := os.ReadFile(filepath.Join(workspace, "allowed"))
	if err != nil || string(raw) != "explicit write" {
		t.Fatalf("write missing: %q %v", raw, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	start := time.Now()
	_, err = c.Run(ctx, Request{Argv: []string{"/bin/sh", "-c", "sleep 60 & wait"}})
	if err == nil || time.Since(start) > 6*time.Second {
		t.Fatalf("cancel not bounded: %v %s", err, time.Since(start))
	}
}

func TestImageImplicitVolumesRejected(t *testing.T) {
	for _, raw := range []string{`{"/data":{}}`, `{"/tmp":{}}`, `[]`, `"unknown"`, `{} garbage`} {
		if err := validateImageVolumes(raw); err == nil {
			t.Fatalf("accepted implicit or malformed volumes: %s", raw)
		}
	}
	for _, raw := range []string{`null`, `{}`} {
		if err := validateImageVolumes(raw); err != nil {
			t.Fatal(err)
		}
	}
}
func TestNativeImageVolumeRejection(t *testing.T) {
	image, socket := os.Getenv("HARNESS_TEST_VOLUME_IMAGE"), os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native volume image fixture not configured")
	}
	c := Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: t.TempDir()}
	_, err := c.Run(context.Background(), Request{Argv: []string{"/bin/sh", "-c", "exit 99"}})
	if err == nil || !strings.Contains(err.Error(), "implicit volumes") {
		t.Fatalf("image volume declarations were not rejected: %v", err)
	}
}

func TestNativeCancellationStopsChildWrites(t *testing.T) {
	image, socket := os.Getenv("HARNESS_TEST_CONTAINER_IMAGE"), os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native container fixture not configured")
	}
	workspace := t.TempDir()
	if err := os.Chmod(workspace, 0777); err != nil {
		t.Fatal(err)
	}
	c := Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: workspace, Writable: true}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	done := make(chan error, 1)
	joined := false
	defer func() {
		cancel()
		if !joined {
			select {
			case <-done:
			case <-time.After(6 * time.Second):
				t.Error("container invocation did not join during test cleanup")
			}
		}
	}()
	go func() {
		_, err := c.Run(ctx, Request{Argv: []string{"/bin/sh", "-c", `sh -c 'while :; do printf tick >> /workspace/heartbeat; sleep 0.05; done' & wait`}})
		done <- err
	}()
	heartbeat := filepath.Join(workspace, "heartbeat")
	for {
		if b, err := os.ReadFile(heartbeat); err == nil && len(b) > 0 {
			break
		}
		select {
		case err := <-done:
			joined = true
			t.Fatalf("container exited before child became active: %v", err)
		case <-ctx.Done():
			t.Fatal("child did not become active before deadline")
		case <-time.After(20 * time.Millisecond):
		}
	}
	start := time.Now()
	cancel()
	select {
	case err := <-done:
		joined = true
		if err == nil {
			t.Fatal("cancelled invocation reported success")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("container cleanup exceeded five seconds")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("child cleanup exceeded five seconds")
	}
	before, err := os.ReadFile(heartbeat)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(250 * time.Millisecond)
	after, err := os.ReadFile(heartbeat)
	if err != nil || string(before) != string(after) {
		t.Fatalf("child continued writing after cancellation: before=%d after=%d err=%v", len(before), len(after), err)
	}
}
