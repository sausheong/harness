package session

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func TestRecoveryProcessHelper(t *testing.T) {
	root := os.Getenv("HARNESS_RECOVERY_HELPER_ROOT")
	if root == "" {
		return
	}
	stage, err := strconv.Atoi(os.Getenv("HARNESS_RECOVERY_HELPER_STAGE"))
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	sess, _, err := NewStore(root).loadRecovering(context.Background(), "agent", "key", func(dir string) error {
		if err := syncRecoveryDirectory(dir); err != nil {
			return err
		}
		calls++
		if calls == stage {
			fmt.Println("RECOVERY_READY")
			io.Copy(io.Discard, os.Stdin)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
}

func TestRecoverySurvivesProcessDeath(t *testing.T) {
	prefix := `{"id":"retained","future":true}` + "\n"
	original := prefix + `{"id":`
	for _, stage := range []int{1, 2} {
		t.Run(strconv.Itoa(stage), func(t *testing.T) {
			store, path := recoveryFixture(t, original)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRecoveryProcessHelper$")
			cmd.Env = append(os.Environ(), "HARNESS_RECOVERY_HELPER_ROOT="+store.baseDir, "HARNESS_RECOVERY_HELPER_STAGE="+strconv.Itoa(stage))
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
			if !scanner.Scan() || scanner.Text() != "RECOVERY_READY" {
				t.Fatal("child did not reach durability boundary")
			}
			if _, _, err := store.LoadRecovering(context.Background(), "agent", "key"); !errors.Is(err, ErrSessionBusy) {
				t.Fatal("live repair bypassed", err)
			}
			if err := cmd.Process.Kill(); err != nil {
				t.Fatal(err)
			}
			cmd.Wait()
			raw, err := os.ReadFile(path)
			expected := original
			if stage == 2 {
				expected = prefix
			}
			if err != nil || string(raw) != expected {
				t.Fatal("process death left partial replacement", err)
			}
			backups, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.recovery-backup-*"))
			if err != nil || len(backups) != 1 {
				t.Fatal("backup missing after death", backups, err)
			}
			backup, err := os.ReadFile(backups[0])
			if err != nil || string(backup) != original {
				t.Fatal("backup corrupted after death", err)
			}
			sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
			if err != nil {
				t.Fatal("restart failed", err)
			}
			defer sess.Close()
			if report.Repaired != (stage == 1) || len(sess.Entries()) != 1 {
				t.Fatal("restart wrong repair", report)
			}
			sess.Append(UserMessageEntry("after restart"))
			if err := sess.Flush(); err != nil {
				t.Fatal(err)
			}
			loaded, err := store.Load("agent", "key")
			if err != nil || len(loaded.Entries()) != 2 {
				t.Fatal("restart not writable", err)
			}
		})
	}
}
