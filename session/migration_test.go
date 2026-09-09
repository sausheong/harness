package session

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestLegacyMigrationBackupRestartAndIdempotence(t *testing.T) {
	original := legacyCompactionFixture()
	store, path := recoveryFixture(t, original)
	sess, report, err := store.MigrateLegacy(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	if !report.Migrated || report.RemappedIDs != 2 || report.BackupPath == "" {
		t.Fatal(report)
	}
	backup, err := os.ReadFile(report.BackupPath)
	if err != nil || string(backup) != original {
		t.Fatal("original backup differs", err)
	}
	for _, p := range []string{path, report.BackupPath} {
		info, err := os.Stat(p)
		if err != nil || info.Mode().Perm() != 0600 {
			t.Fatal("migration files not private", err)
		}
	}
	if _, _, err := store.MigrateLegacy(context.Background(), "agent", "key"); !errors.Is(err, ErrSessionBusy) {
		t.Fatal("migration bypassed writer", err)
	}
	if len(sess.View()) != 3 || sess.View()[0].Type != EntryTypeCompaction {
		t.Fatal("migrated view wrong", sess.View())
	}
	if err := sess.Branch("c"); err != nil || len(sess.History()) != 3 {
		t.Fatal("original branch unavailable", err)
	}
	sess.Close()
	before, _ := os.ReadFile(path)
	resumed, again, err := store.MigrateLegacy(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Close()
	after, _ := os.ReadFile(path)
	if again.Migrated || again.BackupPath != "" || string(before) != string(after) || resumed.LeafID() != "c" {
		t.Fatal("repeat migration changed data", again)
	}
	backups, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.migration-backup-*"))
	if len(backups) != 1 {
		t.Fatal("repeat migration duplicated backup")
	}
	resumed.Append(UserMessageEntry("after migration"))
	if err := resumed.Flush(); err != nil {
		t.Fatal(err)
	}
}

func TestLegacyMigrationFailureBoundaries(t *testing.T) {
	original := legacyCompactionFixture()
	failure := errors.New("injected migration directory sync failure")
	for _, stage := range []int{1, 2} {
		store, path := recoveryFixture(t, original)
		calls := 0
		sess, report, err := store.migrateLegacy(context.Background(), "agent", "key", func(dir string) error {
			calls++
			if calls == stage {
				return failure
			}
			return syncRecoveryDirectory(dir)
		})
		if sess != nil || !errors.Is(err, failure) {
			t.Fatal("migration failure hidden", err)
		}
		raw, _ := os.ReadFile(path)
		if stage == 1 {
			if string(raw) != original || report.Migrated || report.BackupPath != "" {
				t.Fatal("changed before durable backup")
			}
		} else {
			if !report.Migrated || report.BackupPath == "" {
				t.Fatal("post-replace failure misreported")
			}
			backup, _ := os.ReadFile(report.BackupPath)
			if string(backup) != original {
				t.Fatal("backup lost")
			}
		}
		lease, err := store.acquireLease("agent", "key")
		if err != nil {
			t.Fatal("failure leaked lease", err)
		}
		lease.close()
	}
}

func TestLegacyMigrationCancelledAfterBackup(t *testing.T) {
	original := legacyCompactionFixture()
	store, path := recoveryFixture(t, original)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sess, report, err := store.migrateLegacy(ctx, "agent", "key", func(dir string) error { err := syncRecoveryDirectory(dir); cancel(); return err })
	if sess != nil || !errors.Is(err, context.Canceled) || report.Migrated || report.BackupPath == "" {
		t.Fatal("cancelled migration outcome", err, report)
	}
	for _, p := range []string{path, report.BackupPath} {
		raw, _ := os.ReadFile(p)
		if string(raw) != original {
			t.Fatal("cancelled migration lost original")
		}
	}
	temps, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".key.migration-replace-*"))
	if len(temps) != 0 {
		t.Fatal("temporary file leaked")
	}
}
