package runtime

import (
	"context"
	"errors"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestSteeringImagesPersistAndRetryWithoutDuplicate(t *testing.T) {
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "key"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("a", "key")
	if err != nil {
		t.Fatal(err)
	}
	message := SteeringMessage{ID: "with-image", Text: "inspect this", Images: []llm.ImageContent{{MimeType: "image/png", Data: []byte("pixels")}}}
	ackErr := errors.New("acknowledgement unavailable")
	message.Acknowledge = func() error { return ackErr }
	ctx := WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) { return &message, nil })
	rt := &Runtime{Session: sess}
	if applied, err := rt.applySteering(ctx, nil); !applied || !errors.Is(err, ackErr) {
		t.Fatal(applied, err)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	sess, err = store.LoadExclusive("a", "key")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	rt.Session = sess
	before := len(sess.Entries())
	ack := false
	message.Acknowledge = func() error { ack = true; return nil }
	if applied, err := rt.applySteering(ctx, nil); err != nil || applied || !ack || len(sess.Entries()) != before {
		t.Fatal("image retry duplicated or conflicted", applied, err, ack)
	}
	message.Images[0].Data = []byte("different")
	if _, err := rt.applySteering(ctx, nil); err == nil {
		t.Fatal("conflicting image identity accepted")
	}
}

func TestSteeringImagesValidateBeforeSkippingTools(t *testing.T) {
	rt := &Runtime{Session: session.NewSession("a", "key")}
	message := SteeringMessage{ID: "bad-image", Text: "inspect", Images: []llm.ImageContent{{Data: []byte("pixels")}}}
	ctx := WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) { return &message, nil })
	if _, err := rt.applySteering(ctx, []llm.ToolCall{{ID: "pending", Name: "read"}}); err == nil {
		t.Fatal("invalid image accepted")
	}
	if len(rt.Session.Entries()) != 0 {
		t.Fatal("invalid steering changed proposed tool history")
	}
}
