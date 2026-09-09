package session

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func recoveryFixture(t *testing.T, body string) (*Store, string) {
	t.Helper()
	store := NewStore(t.TempDir())
	path := store.sessionPath("agent", "key")
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	return store, path
}

func TestRecoveryPreservesBackupAndExactPrefix(t *testing.T) {
	prefix := "\n" + ` {"id":"first", "type":"message", "future":{"unknown":true}} ` + "\n\n"
	tail := `{"id":"second","data":`
	store, path := recoveryFixture(t, prefix+tail)
	sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	if !report.Repaired || report.DroppedBytes != int64(len(tail)) || report.BackupPath == "" {
		t.Fatal(report)
	}
	backup, err := os.ReadFile(report.BackupPath)
	if err != nil || string(backup) != prefix+tail {
		t.Fatal("original backup mismatch", err)
	}
	repaired, err := os.ReadFile(path)
	if err != nil || string(repaired) != prefix {
		t.Fatal("valid bytes changed", err)
	}
	for _, p := range []string{path, report.BackupPath} {
		info, err := os.Stat(p)
		if err != nil || info.Mode().Perm() != 0600 {
			t.Fatal("recovery file is not private", err)
		}
	}
	if len(sess.Entries()) != 1 || sess.Entries()[0].ID != "first" {
		t.Fatal("wrong recovered history")
	}
	if _, err := NewStore(store.baseDir).LoadExclusive("agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal("recovered writer lost lease", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	second, again, err := store.LoadRecovering(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if again != (RecoveryReport{}) {
		t.Fatal("valid repeat produced another repair", again)
	}
	second.Append(UserMessageEntry("after recovery"))
	if err := second.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || len(loaded.Entries()) != 2 {
		t.Fatal("repaired session not writable", err)
	}
	backups, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.recovery-backup-*"))
	if len(backups) != 1 {
		t.Fatal("backup proliferation", backups)
	}
	temps, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.recovery-replace-*"))
	if len(temps) != 0 {
		t.Fatal("replacement temporary leaked", temps)
	}
}

func TestRecoveryRejectsCorruptionWithoutMutation(t *testing.T) {
	for _, body := range []string{`{"id":` + "\n" + `{"id":"good"}`, `{"id":` + "\n", `not JSON`, `null`, `{} {}`, strings.Repeat("x", MaxSessionRecordBytes+1)} {
		t.Run(body[:min(len(body), 25)], func(t *testing.T) {
			store, path := recoveryFixture(t, body)
			sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
			if err == nil || sess != nil || report != (RecoveryReport{}) {
				t.Fatal("invalid repair accepted", err, report)
			}
			raw, _ := os.ReadFile(path)
			if string(raw) != body {
				t.Fatal("corrupt file changed")
			}
			files, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.recovery-*"))
			if len(files) != 0 {
				t.Fatal("refused recovery left files", files)
			}
			lease, err := store.acquireLease("agent", "key")
			if err != nil {
				t.Fatal("error leaked lease", err)
			}
			lease.close()
		})
	}
}

func TestRecoveryCancellationAndBusyLeaveOriginal(t *testing.T) {
	store, path := recoveryFixture(t, `{"id":`)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := store.LoadRecovering(ctx, "agent", "key"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	lease, err := store.acquireLease("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.close()
	if _, _, err := NewStore(store.baseDir).LoadRecovering(context.Background(), "agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	if string(raw) != `{"id":` {
		t.Fatal("busy/cancelled recovery changed source")
	}
}

func TestRecoveryRejectsSymlink(t *testing.T) {
	store, path := recoveryFixture(t, `{"id":`)
	target := path + ".target"
	if err := os.Rename(path, target); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.LoadRecovering(context.Background(), "agent", "key"); err == nil {
		t.Fatal("symlink accepted")
	}
	raw, _ := os.ReadFile(target)
	if string(raw) != `{"id":` {
		t.Fatal("symlink target changed")
	}
}

func TestRecoveryEmptyPrefixAndMissingSession(t *testing.T) {
	store, path := recoveryFixture(t, `{"id":`)
	sess, report, err := store.LoadRecovering(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) != 0 || !report.Repaired || len(sess.Entries()) != 0 {
		t.Fatal("first interrupted record", report, err)
	}
	missing, report, err := store.LoadRecovering(context.Background(), "agent", "missing")
	if err != nil {
		t.Fatal(err)
	}
	defer missing.Close()
	if report != (RecoveryReport{}) {
		t.Fatal("missing session counted as repair")
	}
}

func TestRecoveryDirectorySyncFailures(t *testing.T) {
	prefix := `{"id":"retained"}` + "\n"
	original := prefix + `{"id":`
	failure := errors.New("injected directory sync failure")
	for _, failAt := range []int{1, 2} {
		t.Run(string(rune('0'+failAt)), func(t *testing.T) {
			store, path := recoveryFixture(t, original)
			calls := 0
			sess, report, err := store.loadRecovering(context.Background(), "agent", "key", func(dir string) error {
				calls++
				if calls == failAt {
					return failure
				}
				return syncRecoveryDirectory(dir)
			})
			if sess != nil || !errors.Is(err, failure) {
				t.Fatal("sync failure hidden", err)
			}
			raw, _ := os.ReadFile(path)
			if failAt == 1 {
				if string(raw) != original || report.Repaired || report.BackupPath != "" {
					t.Fatal("replacement before durable backup", report)
				}
			} else {
				if string(raw) != prefix || !report.Repaired || report.BackupPath == "" {
					t.Fatal("post-rename failure misreported", report)
				}
				backup, err := os.ReadFile(report.BackupPath)
				if err != nil || string(backup) != original {
					t.Fatal("post-rename failure lost backup", err)
				}
			}
			lease, err := store.acquireLease("agent", "key")
			if err != nil {
				t.Fatal("failed repair leaked lease", err)
			}
			lease.close()
		})
	}
}

func TestRecoveryCancellationAfterBackupPreservesOriginal(t *testing.T) {
	original := `{"id":"retained"}` + "\n" + `{"id":`
	store, path := recoveryFixture(t, original)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sess, report, err := store.loadRecovering(ctx, "agent", "key", func(dir string) error {
		err := syncRecoveryDirectory(dir)
		cancel()
		return err
	})
	if sess != nil || !errors.Is(err, context.Canceled) || report.Repaired || report.BackupPath == "" {
		t.Fatal("cancelled repair outcome", err, report)
	}
	for _, p := range []string{path, report.BackupPath} {
		raw, err := os.ReadFile(p)
		if err != nil || string(raw) != original {
			t.Fatal("cancelled repair lost original", p, err)
		}
	}
	temps, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.recovery-replace-*"))
	if len(temps) != 0 {
		t.Fatal("cancelled repair leaked replacement", temps)
	}
}
