package session

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestCombinedMigrationRecoveryValidatesEntirePrefix(t *testing.T) {
	const tail = `{"id":"unfinished","data":`
	for _, kind := range []string{"clones", "image_clones", "versioned"} {
		t.Run(kind, func(t *testing.T) {
			prefix := legacyCompactionFixture()
			if kind == "image_clones" {
				image := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{42}, 12<<20))
				prefix = strings.Replace(prefix, `"text":"root"`, `"text":"root","images":[{"mime_type":"image/png","data":"`+image+`"}]`, 1)
			}
			if kind == "versioned" {
				prefix = `{"id":"session","type":"session_header","schemaVersion":1,"timestamp":123}` + "\n" + `{"id":"a","type":"message","data":{"text":"kept"}}` + "\n"
			}
			raw := prefix + tail
			store, path := recoveryFixture(t, raw)
			if kind == "clones" {
				if _, _, err := store.MigrateLegacy(context.Background(), "agent", "key"); err == nil {
					t.Fatal("strict migration silently recovered")
				}
			}
			sess, report, err := store.MigrateRecovering(context.Background(), "agent", "key")
			if err != nil {
				t.Fatal(err)
			}
			if !report.TailRecovered || report.DiscardedTailBytes != int64(len(tail)) || !report.Migrated {
				t.Fatal(report)
			}
			backup, err := os.ReadFile(report.BackupPath)
			if err != nil || string(backup) != raw {
				t.Fatal("full original not backed up", err)
			}
			if kind != "versioned" && report.RemappedIDs != 2 {
				t.Fatal("legacy graph not migrated", report)
			}
			if kind == "image_clones" {
				if report.ExternalizedImages != 1 {
					t.Fatal(report)
				}
				if err := sess.Branch("a"); err != nil {
					t.Fatal(err)
				}
				replay, err := sess.ResolveImages(context.Background(), sess.History())
				if err != nil {
					t.Fatal(err)
				}
				var data MessageData
				json.Unmarshal(replay[0].Data, &data)
				pixels, err := base64.StdEncoding.DecodeString(data.Images[0].Data)
				if err != nil || len(pixels) != 12<<20 || pixels[0] != 42 {
					t.Fatal("image lost", err)
				}
			}
			id, leaf := sess.ID, sess.LeafID()
			sess.Close()
			before, _ := os.ReadFile(path)
			resumed, again, err := store.MigrateRecovering(context.Background(), "agent", "key")
			if err != nil {
				t.Fatal(err)
			}
			defer resumed.Close()
			after, _ := os.ReadFile(path)
			if again.Migrated || again.TailRecovered || again.DiscardedTailBytes != 0 || again.BackupPath != "" || resumed.ID != id || resumed.LeafID() != leaf || !bytes.Equal(before, after) {
				t.Fatal("repeat recovery changed state", again)
			}
		})
	}
}

func TestCombinedRecoveryRejectsCorruptPrefixesAndTerminatedDamage(t *testing.T) {
	for _, raw := range []string{
		`{"id":"a","type":"message"}` + "\n" + `{"id":"a","type":"message"}` + "\n" + `{"id":"tail"`,
		`{"id":"a","parentId":"missing","type":"message"}` + "\n" + `{"id":"tail"`,
		`{"id":"a","type":"message"}` + "\n" + `{"id":"broken"` + "\n",
		`{"id":"broken"` + "\n" + `{"id":"tail"`,
	} {
		store, path := recoveryFixture(t, raw)
		sess, report, err := store.MigrateRecovering(context.Background(), "agent", "key")
		if err == nil || sess != nil || report.Migrated || report.TailRecovered {
			t.Fatal("corruption accepted", report, err)
		}
		after, err := os.ReadFile(path)
		if err != nil || string(after) != raw {
			t.Fatal("corrupt source modified", err)
		}
		// No writer lease remains after rejection.
		lease, err := store.acquireLease("agent", "key")
		if errors.Is(err, ErrSessionBusy) || err != nil {
			t.Fatal(err)
		}
		lease.close()
	}
}
