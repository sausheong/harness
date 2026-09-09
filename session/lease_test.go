package session

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestExclusiveLeaseBlocksOtherWritersAndMutations(t *testing.T) {
	store := NewStore(t.TempDir())
	owner, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	owner.Append(UserMessageEntry("retained"))
	if err := owner.Flush(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadExclusive("agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal(err)
	}
	if err := store.Delete("agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal("delete bypassed lease", err)
	}
	if err := store.Rename("agent", "key", "other"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal("rename bypassed lease", err)
	}
	other, err := store.Load("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	other.Append(UserMessageEntry("must not persist"))
	if !errors.Is(other.PersistenceError(), ErrSessionBusy) {
		t.Fatal("legacy append bypassed lease", other.PersistenceError())
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || len(loaded.Entries()) != 1 {
		t.Fatal("unauthorised entry persisted", err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal("close not idempotent", err)
	}
	owner.Append(UserMessageEntry("closed"))
	if !errors.Is(owner.PersistenceError(), ErrSessionClosed) {
		t.Fatal("closed writer accepted append")
	}
	resumed, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Close()
	if len(resumed.Entries()) != 1 {
		t.Fatal("lease lifecycle damaged session")
	}
}

func TestWriterLeaseCannotBeReusedForAnotherKey(t *testing.T) {
	store := NewStore(t.TempDir())
	s, err := store.LoadExclusive("agent", "original")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Key = "different"
	s.Append(UserMessageEntry("must not write"))
	if s.PersistenceError() == nil {
		t.Fatal("changed key retained write capability")
	}
}

func TestWriterLeaseHelper(t *testing.T) {
	root := os.Getenv("HARNESS_LEASE_HELPER_ROOT")
	if root == "" {
		return
	}
	s, err := NewStore(root).LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	fmt.Println("LEASE_READY")
	io.Copy(io.Discard, os.Stdin)
}

func TestWriterLeaseRecoversAfterProcessDeath(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestWriterLeaseHelper$")
	cmd.Env = append(os.Environ(), "HARNESS_LEASE_HELPER_ROOT="+root)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if cmd.ProcessState == nil {
			cmd.Process.Kill()
			cmd.Wait()
		}
	}()
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() || scanner.Text() != "LEASE_READY" {
		t.Fatal("child did not acquire lease")
	}
	store := NewStore(root)
	if _, err := store.LoadExclusive("agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal("second process acquired live writer lease", err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	cmd.Wait()
	recovered, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal("stale lease blocked recovery", err)
	}
	defer recovered.Close()
	recovered.Append(UserMessageEntry("after process death"))
	if err := recovered.Flush(); err != nil {
		t.Fatal(err)
	}
}
