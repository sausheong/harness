package session

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestMigrationProcessHelper(t *testing.T) {
	root := os.Getenv("HARNESS_MIGRATION_HELPER_ROOT")
	if root == "" {
		return
	}
	stage, err := strconv.Atoi(os.Getenv("HARNESS_MIGRATION_HELPER_STAGE"))
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	sess, _, err := NewStore(root).migrateLegacyWithRecovery(context.Background(), "agent", "key", func(dir string) error {
		if err := syncRecoveryDirectory(dir); err != nil {
			return err
		}
		calls++
		if calls == stage {
			fmt.Println("MIGRATION_READY")
			io.Copy(io.Discard, os.Stdin)
		}
		return nil
	}, os.Getenv("HARNESS_MIGRATION_RECOVER_TAIL") == "true")
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
}

func TestMigrationSurvivesProcessDeath(t *testing.T)         { migrationSurvivesProcessDeath(t, false) }
func TestCombinedMigrationSurvivesProcessDeath(t *testing.T) { migrationSurvivesProcessDeath(t, true) }
func migrationSurvivesProcessDeath(t *testing.T, recoverTail bool) {
	original := legacyCompactionFixture()
	converted, _, err := ConvertLegacySession(strings.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	prefix := string(converted)
	if recoverTail {
		original += `{"id":"unfinished"`
	}
	for _, stage := range []int{1, 2} {
		t.Run(strconv.Itoa(stage), func(t *testing.T) {
			store, path := recoveryFixture(t, original)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestMigrationProcessHelper$")
			cmd.Env = append(os.Environ(), "HARNESS_MIGRATION_HELPER_ROOT="+store.baseDir, "HARNESS_MIGRATION_HELPER_STAGE="+strconv.Itoa(stage), "HARNESS_MIGRATION_RECOVER_TAIL="+strconv.FormatBool(recoverTail))
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
			if !scanner.Scan() || scanner.Text() != "MIGRATION_READY" {
				t.Fatal("child did not reach durability boundary")
			}
			if _, _, err := store.MigrateLegacy(context.Background(), "agent", "key"); !errors.Is(err, ErrSessionBusy) {
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
			if stage == 2 {
				entries, decodeErr := decodeSessionRecords(bytes.NewReader(raw))
				if decodeErr != nil || len(entries) != 7 || entries[0].Type != EntryTypeHeader {
					t.Fatal("missing durable format header", decodeErr)
				}
				raw = raw[bytes.IndexByte(raw, '\n')+1:]
			}
			if err != nil || string(raw) != expected {
				t.Fatal("process death left partial replacement", err)
			}
			backups, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.migration-backup-*"))
			if err != nil || len(backups) != 1 {
				t.Fatal("backup missing after death", backups, err)
			}
			backup, err := os.ReadFile(backups[0])
			if err != nil || string(backup) != original {
				t.Fatal("backup corrupted after death", err)
			}
			var sess *Session
			var report MigrationReport
			if recoverTail {
				sess, report, err = store.MigrateRecovering(context.Background(), "agent", "key")
			} else {
				sess, report, err = store.MigrateLegacy(context.Background(), "agent", "key")
			}
			if err != nil {
				t.Fatal("restart failed", err)
			}
			defer sess.Close()
			if report.Migrated != (stage == 1) || report.TailRecovered != (recoverTail && stage == 1) || len(sess.Entries()) != 6 {
				t.Fatal("restart wrong repair", report)
			}
			sess.Append(UserMessageEntry("after restart"))
			if err := sess.Flush(); err != nil {
				t.Fatal(err)
			}
			loaded, err := store.Load("agent", "key")
			if err != nil || len(loaded.Entries()) != 7 {
				t.Fatal("restart not writable", err)
			}
		})
	}
}
