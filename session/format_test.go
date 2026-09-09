package session

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

func TestPersistedSessionIdentity(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	first, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	id := first.ID
	empty, err := store.Load("agent", "key")
	if err != nil || empty.ID != id || len(empty.Entries()) != 0 || empty.LeafID() != "" {
		t.Fatal("empty identity or graph changed", err)
	}
	first.Append(UserMessageEntry("message"))
	store.Rewrite(first)
	if err := first.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || loaded.ID != id || len(loaded.Entries()) != 1 || len(loaded.History()) != 1 {
		t.Fatal("header identity/history lost", err)
	}
	first.Close()
	if err := store.Rename("agent", "key", "renamed"); err != nil {
		t.Fatal(err)
	}
	renamed, err := store.Load("agent", "renamed")
	if err != nil || renamed.ID != id {
		t.Fatal("rename changed session identity", err)
	}
	list, err := store.List("agent")
	if err != nil || len(list) != 1 || list[0].ID != id || list[0].EntryCount != 1 {
		t.Fatal("catalogue metadata counts header as message", list, err)
	}
}

func TestFirstAppendPersistsIdentity(t *testing.T) {
	store := NewStore(t.TempDir())
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Append(UserMessageEntry("first"))
	if err := sess.Flush(); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("agent", "key")
	if err != nil || loaded.ID != sess.ID {
		t.Fatal("first append lost identity", err)
	}
	raw, err := os.ReadFile(store.sessionPath("agent", "key"))
	if err != nil {
		t.Fatal(err)
	}
	entries, err := decodeSessionRecords(bytes.NewReader(raw))
	if err != nil || len(entries) != 2 || entries[0].Type != EntryTypeHeader || entries[0].SchemaVersion != 1 {
		t.Fatal("first append lacks format header", err)
	}
}

func TestFormatHeadersRejectedWithoutRecoveryOrMigration(t *testing.T) {
	for name, raw := range map[string]string{
		"future":                `{"id":"s","type":"session_header","schemaVersion":2}`,
		"missing version":       `{"id":"s","type":"session_header"}`,
		"late":                  `{"id":"a"}` + "\n" + `{"id":"s","type":"session_header","schemaVersion":1}`,
		"second header":         `{"id":"s","type":"session_header","schemaVersion":1}` + "\n" + `{"id":"s2","type":"session_header","schemaVersion":1}`,
		"header parent":         `{"id":"s","type":"session_header","schemaVersion":1,"parentId":"a"}`,
		"message parent header": `{"id":"s","type":"session_header","schemaVersion":1}` + "\n" + `{"id":"a","parentId":"s"}`,
		"version on message":    `{"id":"a","type":"message","schemaVersion":2}`,
	} {
		t.Run(name, func(t *testing.T) {
			store, path := recoveryFixture(t, raw)
			if _, err := store.Load("agent", "key"); err == nil {
				t.Fatal("invalid schema loaded")
			}
			if _, _, err := store.LoadRecovering(context.Background(), "agent", "key"); err == nil {
				t.Fatal("invalid schema repaired")
			}
			if _, _, err := store.MigrateLegacy(context.Background(), "agent", "key"); err == nil {
				t.Fatal("invalid schema migrated")
			}
			after, _ := os.ReadFile(path)
			if string(after) != raw {
				t.Fatal("invalid schema modified")
			}
		})
	}
}

func TestLegacyFormatUpgradeBackedUpAndStable(t *testing.T) {
	for _, raw := range []string{"", `{"id":"old","type":"message","timestamp":123,"future":true}` + "\n"} {
		store, path := recoveryFixture(t, raw)
		sess, report, err := store.MigrateLegacy(context.Background(), "agent", "key")
		if err != nil {
			t.Fatal(err)
		}
		id := sess.ID
		sess.Close()
		if !report.Migrated || !report.FormatUpgraded || report.RemappedIDs != 0 || report.BackupPath == "" {
			t.Fatal("legacy format upgrade unreported", report)
		}
		backup, _ := os.ReadFile(report.BackupPath)
		if string(backup) != raw {
			t.Fatal("format upgrade lost original")
		}
		before, _ := os.ReadFile(path)
		again, second, err := store.MigrateLegacy(context.Background(), "agent", "key")
		if err != nil {
			t.Fatal(err)
		}
		again.Close()
		after, _ := os.ReadFile(path)
		if again.ID != id || second.Migrated || second.FormatUpgraded || !bytes.Equal(before, after) {
			t.Fatal("format migration not idempotent", second)
		}
		if raw != "" && !strings.Contains(string(after), `"future":true`) {
			t.Fatal("format upgrade dropped unknown field")
		}
	}
}

func TestCreateDoesNotOverwriteExistingSession(t *testing.T) {
	store, path := recoveryFixture(t, `{"id":"existing"}`)
	if err := store.Create("agent", "key"); err == nil {
		t.Fatal("create overwrote existing session")
	}
	raw, _ := os.ReadFile(path)
	if string(raw) != `{"id":"existing"}` {
		t.Fatal("existing bytes changed")
	}
}

func TestAppendCannotReuseHeaderIdentity(t *testing.T) {
	sess := NewSession("agent", "key")
	entry := UserMessageEntry("invalid")
	entry.ID = sess.ID
	sess.Append(entry)
	if sess.PersistenceError() == nil || len(sess.Entries()) != 0 {
		t.Fatal("header ID reused as graph node")
	}
}
