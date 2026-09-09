package session

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"
)

func TestSessionAttachmentReferencesSurviveRestart(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	data := bytes.Repeat([]byte{42}, 12<<20)
	image := ImageData{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString(data)}
	if err := sess.AppendContext(context.Background(), UserMessageWithImagesEntry("large image", []ImageData{image})); err != nil {
		t.Fatal(err)
	}
	if err := sess.Flush(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(store.sessionPath("agent", "key"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 4096 || bytes.Contains(raw, []byte(image.Data[:128])) {
		t.Fatal("inline blob persisted")
	}
	leaf := sess.LeafID()
	sess.Close()
	sess, err = store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	var saved MessageData
	if err := json.Unmarshal(sess.History()[0].Data, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Images[0].Reference == nil || saved.Images[0].Data != "" {
		t.Fatal("reference missing")
	}
	resolved, err := sess.ResolveImages(context.Background(), sess.View())
	if err != nil {
		t.Fatal(err)
	}
	var replay MessageData
	if err := json.Unmarshal(resolved[0].Data, &replay); err != nil {
		t.Fatal(err)
	}
	decoded, err := base64.StdEncoding.DecodeString(replay.Images[0].Data)
	if err != nil || !bytes.Equal(decoded, data) || sess.LeafID() != leaf {
		t.Fatal("replay changed content or graph", err)
	}
	if bytes.Equal(resolved[0].Data, sess.History()[0].Data) {
		t.Fatal("replay mutated stored reference")
	}
}

func TestSessionAttachmentWriteFailurePreservesLeaf(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Append(UserMessageEntry("original"))
	leaf := sess.LeafID()
	err = sess.AppendContext(context.Background(), UserMessageWithImagesEntry("bad image", []ImageData{{MimeType: "image/png", Data: "%%%"}}))
	if err == nil || sess.PersistenceError() == nil || sess.LeafID() != leaf {
		t.Fatal("failed attachment published message", err)
	}
}

func TestToolAttachmentPreservesUnknownPayloadFields(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	entry := ToolResultEntry("call", "image", "", []ImageData{{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString([]byte("image"))}})
	var fields map[string]json.RawMessage
	json.Unmarshal(entry.Data, &fields)
	fields["future_field"] = json.RawMessage(`{"x":1}`)
	entry.Data, _ = json.Marshal(fields)
	if err := sess.AppendContext(context.Background(), entry); err != nil {
		t.Fatal(err)
	}
	var saved map[string]json.RawMessage
	json.Unmarshal(sess.History()[0].Data, &saved)
	if string(saved["future_field"]) != `{"x":1}` {
		t.Fatal("unknown payload lost")
	}
	replay, err := sess.ResolveImages(context.Background(), sess.History())
	if err != nil {
		t.Fatal(err)
	}
	var result ToolResultData
	json.Unmarshal(replay[0].Data, &result)
	if len(result.Images) != 1 || result.Images[0].Data != base64.StdEncoding.EncodeToString([]byte("image")) {
		t.Fatal(result)
	}
}

func TestImageAtExactAttachmentLimitIsAccepted(t *testing.T) {
	store := NewStore(t.TempDir())
	if err := store.Create("agent", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("agent", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	// 32 MiB has base64 padding: DecodedLen alone overestimates by one byte.
	data := make([]byte, 32<<20)
	entry := UserMessageWithImagesEntry("boundary", []ImageData{{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString(data)}})
	if err := sess.AppendContext(context.Background(), entry); err != nil {
		t.Fatal("exact limit rejected", err)
	}
	refs, err := AttachmentReferences(sess.Entries())
	if err != nil || len(refs) != 1 || refs[0].Size != int64(len(data)) {
		t.Fatal(refs, err)
	}
}
