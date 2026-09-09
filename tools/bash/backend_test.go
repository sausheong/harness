package bash

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/sausheong/harness/execution"
	"github.com/sausheong/harness/process"
)

type recordingBackend struct {
	requests []execution.Request
	fail     bool
}

func (*recordingBackend) Boundary() string { return "fixture boundary" }
func (b *recordingBackend) Run(_ context.Context, r execution.Request) (execution.Result, error) {
	b.requests = append(b.requests, r)
	if b.fail {
		return execution.Result{ExitCode: -1}, errors.New("backend unavailable")
	}
	return execution.Result{Stdout: "captured", ExitCode: 0}, nil
}
func TestBackendReceivesExactApprovedCommandAndNeverFallsBack(t *testing.T) {
	workspace := t.TempDir()
	actual := filepath.Join(workspace, "narrow\u00a0space")
	if err := os.WriteFile(actual, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	command := "cat '" + filepath.Join(workspace, "narrow space") + "'"
	backend := &recordingBackend{}
	bt := &BashTool{WorkDir: workspace, Backend: backend}
	input, _ := json.Marshal(bashInput{Command: command})
	result, err := bt.Execute(context.Background(), input)
	if err != nil || result.Error != "" {
		t.Fatalf("%+v %v", result, err)
	}
	if len(backend.requests) != 1 || !reflect.DeepEqual(backend.requests[0].Argv, []string{"/bin/bash", "-c", command}) {
		t.Fatal("approved source was rewritten")
	}
	if backend.requests[0].OutputStore == nil || result.Metadata["execution_boundary"] != "fixture boundary" {
		t.Fatal("capture or boundary missing")
	}
	backend.fail = true
	marker := filepath.Join(workspace, "must-not-exist")
	input, _ = json.Marshal(bashInput{Command: "touch '" + marker + "'"})
	result, _ = bt.Execute(context.Background(), input)
	if !strings.Contains(result.Error, "backend unavailable") {
		t.Fatal(result)
	}
	if _, err = os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("fallback ran on host")
	}
	bt.ExecPolicy = &ExecPolicy{Level: "deny"}
	bt.Execute(context.Background(), input)
	if len(backend.requests) != 2 {
		t.Fatal("policy denial reached backend")
	}
}
func TestBashNativeContainerWithCapture(t *testing.T) {
	image, socket := os.Getenv("HARNESS_TEST_BASH_IMAGE"), os.Getenv("HARNESS_TEST_CONTAINER_SOCKET")
	if image == "" || socket == "" {
		t.Skip("native bash container fixture not configured")
	}
	workspace := t.TempDir()
	if err := os.Chmod(workspace, 0777); err != nil {
		t.Fatal(err)
	}
	store, err := process.NewArtifactStore(filepath.Join(t.TempDir(), "capture"))
	if err != nil {
		t.Fatal(err)
	}
	bt := &BashTool{OutputStore: store, Backend: execution.Container{Docker: "/usr/local/bin/docker", Socket: socket, Image: image, Workspace: workspace, Writable: true}}
	input, _ := json.Marshal(bashInput{Command: `set -eu; test ! -e /var/run/docker.sock; ! touch /etc/outside; printf allowed >/workspace/allowed; head -c 100000 /dev/zero`})
	result, err := bt.Execute(context.Background(), input)
	if err != nil || result.Error != "" {
		t.Fatalf("%+v %v", result, err)
	}
	if result.Metadata["stdout_bytes"] != int64(100000) || result.Metadata["output_truncated"] != true {
		t.Fatal("capture limits missing", result.Metadata)
	}
	info, ok := result.Metadata["stdout_artifact"].(process.ArtifactInfo)
	if !ok || info.Bytes != 100000 || info.Error != "" {
		t.Fatal("full artifact missing", info)
	}
	raw, err := store.Read(context.Background(), info)
	if err != nil || len(raw) != 100000 {
		t.Fatalf("artifact: %d %v", len(raw), err)
	}
	raw, err = os.ReadFile(filepath.Join(workspace, "allowed"))
	if err != nil || string(raw) != "allowed" {
		t.Fatal("workspace write absent")
	}
}
