package execution

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHostProtocolStreamRoundTrip(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s, err := (Host{Workspace: t.TempDir()}).OpenStream(ctx, Request{Argv: []string{"/bin/sh", "-c", "cat; printf diagnostic >&2"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	errout := make(chan string, 1)
	go func() { b, _ := io.ReadAll(s.Stderr); errout <- string(b) }()
	if _, err = s.Stdin.Write([]byte("protocol\n")); err != nil {
		t.Fatal(err)
	}
	if err = s.Stdin.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(s.Stdout)
	if err != nil || string(b) != "protocol\n" {
		t.Fatal(string(b), err)
	}
	if err = s.Wait(); err != nil {
		t.Fatal(err)
	}
	if got := <-errout; got != "diagnostic" {
		t.Fatal(got)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestHostProtocolStreamCancellationAndValidation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s, err := (Host{Workspace: t.TempDir()}).OpenStream(ctx, Request{Argv: []string{"/bin/sh", "-c", "printf ready; sleep 20"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	b := make([]byte, 5)
	if _, err = io.ReadFull(s.Stdout, b); err != nil || string(b) != "ready" {
		t.Fatal(string(b), err)
	}
	start := time.Now()
	cancel()
	_ = s.Close()
	if time.Since(start) > 5*time.Second {
		t.Fatal("stream cleanup exceeded deadline")
	}
	if _, err = (Host{}).OpenStream(context.Background(), Request{Argv: []string{"/bin/sh"}, Stdin: []byte("x")}); err == nil {
		t.Fatal("buffered input accepted")
	}
	_, err = (Container{}).OpenStream(context.Background(), Request{Argv: []string{"/bin/sh"}})
	if err == nil || !strings.Contains(err.Error(), "container requires") {
		t.Fatal("invalid container did not fail closed", err)
	}
}

func TestContainerProtocolStreamRoundTrip(t *testing.T) {
	image := os.Getenv("HARNESS_TEST_CONTAINER_IMAGE")
	socket := os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native container fixture required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	c := Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: t.TempDir()}
	s, err := c.OpenStream(ctx, Request{Argv: []string{"/bin/sh", "-c", "cat; printf isolated >&2"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	diagnostic := make(chan string, 1)
	go func() { b, _ := io.ReadAll(s.Stderr); diagnostic <- string(b) }()
	if _, err = s.Stdin.Write([]byte("container protocol\n")); err != nil {
		t.Fatal(err)
	}
	if err = s.Stdin.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(s.Stdout)
	if err != nil || string(b) != "container protocol\n" {
		t.Fatal(string(b), err)
	}
	if err = s.Wait(); err != nil {
		t.Fatal(err)
	}
	if got := <-diagnostic; got != "isolated" {
		t.Fatal(got)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestContainerProtocolPinnedResources(t *testing.T) {
	image, socket := os.Getenv("HARNESS_TEST_CONTAINER_IMAGE"), os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native container fixture required")
	}
	source := filepath.Join(t.TempDir(), "asset.txt")
	if err := os.WriteFile(source, []byte("pinned-content"), 0600); err != nil {
		t.Fatal(err)
	}
	c := Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: t.TempDir(), Resources: []ReadOnlyResource{{Path: source, Relative: "package/asset.txt", SHA256: fmt.Sprintf("%x", sha256.Sum256([]byte("pinned-content")))}}}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	s, err := c.OpenStream(ctx, Request{Argv: []string{"/bin/sh", "-c", "cat /harness-resources/package/asset.txt; if printf changed > /harness-resources/package/asset.txt 2>/dev/null; then exit 7; fi"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	go io.Copy(io.Discard, s.Stderr)
	b, err := io.ReadAll(s.Stdout)
	if err != nil || string(b) != "pinned-content" {
		t.Fatal(string(b), err)
	}
	if err = s.Wait(); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(source)
	if err != nil || string(original) != "pinned-content" {
		t.Fatal(string(original), err)
	}
}
