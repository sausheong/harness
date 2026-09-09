package session

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestMigrateLargeLegacyImagePreservesOriginalAndRestarts(t *testing.T) {
	data := bytes.Repeat([]byte{42}, 12<<20)
	entry := UserMessageWithImagesEntry("legacy image", []ImageData{{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString(data)}})
	entry.ID = "legacy-image"
	entry.Timestamp = 123
	raw, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	raw = append(raw, '\n')
	store, path := recoveryFixture(t, string(raw))
	if _, err := store.Load("agent", "key"); !errors.Is(err, ErrSessionRecordTooLarge) {
		t.Fatal("normal reader limit changed", err)
	}
	sess, report, err := store.MigrateLegacy(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	if !report.Migrated || report.ExternalizedImages != 1 || report.SourceSHA256 != fmt.Sprintf("%x", sha256.Sum256(raw)) {
		t.Fatal(report)
	}
	backup, err := os.ReadFile(report.BackupPath)
	if err != nil || !bytes.Equal(backup, raw) {
		t.Fatal("original backup changed", err)
	}
	converted, err := os.ReadFile(path)
	if err != nil || len(converted) > 4096 {
		t.Fatal("image remains inline", err)
	}
	sess.Close()
	resumed, again, err := store.MigrateLegacy(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Close()
	if again.Migrated || again.ExternalizedImages != 0 || again.BackupPath != "" || resumed.LeafID() != entry.ID {
		t.Fatal(again)
	}
	replay, err := resumed.ResolveImages(context.Background(), resumed.History())
	if err != nil {
		t.Fatal(err)
	}
	var message MessageData
	if err := json.Unmarshal(replay[0].Data, &message); err != nil {
		t.Fatal(err)
	}
	restored, err := base64.StdEncoding.DecodeString(message.Images[0].Data)
	if err != nil || !bytes.Equal(restored, data) {
		t.Fatal("legacy pixels changed", err)
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(after, converted) {
		t.Fatal("repeat migration changed file", err)
	}
}

func TestImageMigrationRejectsOversizedTextAndMalformedImage(t *testing.T) {
	for _, kind := range []string{"text", "bad_image"} {
		t.Run(kind, func(t *testing.T) {
			entry := UserMessageEntry("old")
			if kind == "text" {
				entry = UserMessageEntry(string(bytes.Repeat([]byte("x"), MaxSessionRecordBytes)))
			} else {
				entry = UserMessageWithImagesEntry("bad", []ImageData{{MimeType: "image/png", Data: "%%%"}})
			}
			entry.ID = "old"
			raw, err := json.Marshal(entry)
			if err != nil {
				t.Fatal(err)
			}
			raw = append(raw, '\n')
			store, path := recoveryFixture(t, string(raw))
			if _, _, err := store.MigrateLegacy(context.Background(), "agent", "key"); err == nil {
				t.Fatal("invalid import accepted")
			}
			after, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(after, raw) {
				t.Fatal("failed migration altered original", err)
			}
		})
	}
}

func TestImageMigrationPreservesLegacyCompactionClones(t *testing.T) {
	raw := strings.ReplaceAll(legacyCompactionFixture(), `"text":"kept"`, `"text":"kept","images":[{"mime_type":"image/png","data":"aW1hZ2U="}]`)
	store, path := recoveryFixture(t, raw)
	sess, report, err := store.MigrateLegacy(context.Background(), "agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	if report.RemappedIDs != 2 || report.ExternalizedImages != 2 {
		t.Fatal(report)
	}
	refs, err := AttachmentReferences(sess.Entries())
	if err != nil || len(refs) != 1 {
		t.Fatal("clone attachment not deduplicated", refs, err)
	}
	if err := sess.Branch("b"); err != nil {
		t.Fatal(err)
	}
	replay, err := sess.ResolveImages(context.Background(), sess.History())
	if err != nil {
		t.Fatal(err)
	}
	var data MessageData
	if err := json.Unmarshal(replay[1].Data, &data); err != nil || len(data.Images) != 1 || data.Images[0].Data != "aW1hZ2U=" {
		t.Fatal(data, err)
	}
	converted, err := os.ReadFile(path)
	if err != nil || !bytes.Contains(converted, []byte(`"value":9007199254740993`)) {
		t.Fatal("unknown integer field lost", err)
	}
}
