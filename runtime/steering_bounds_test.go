package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/sausheong/harness/attachment"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestSteeringMismatchRejectedBeforeBlobAccess(t *testing.T) {
	// An in-memory session cannot resolve references. A mismatch must nevertheless
	// return false without attempting hydration or opening any attachment store.
	sess := session.NewSession("a", "key")
	data := []byte("expected")
	digest := sha256.Sum256(data)
	good := session.ImageData{MimeType: "image/png", Reference: &attachment.Ref{SHA256: hex.EncodeToString(digest[:]), Size: int64(len(data))}}
	message := SteeringMessage{ID: "x", Text: "inspect", Images: []llm.ImageContent{{MimeType: "image/png", Data: data}}}
	for _, mode := range []string{"count", "size", "digest", "mime", "text"} {
		t.Run(mode, func(t *testing.T) {
			image := good
			ref := *good.Reference
			image.Reference = &ref
			text := message.Text
			images := []session.ImageData{image}
			switch mode {
			case "count":
				images = append(images, image)
			case "size":
				ref.Size = 32 << 20
			case "digest":
				ref.SHA256 = hex.EncodeToString(make([]byte, 32))
			case "mime":
				images[0].MimeType = "image/jpeg"
			case "text":
				text = "different"
			}
			entry := session.UserMessageWithImagesEntry(text, images)
			entry.ID = "steering_x"
			matched, err := MatchesSteering(context.Background(), sess, message, entry)
			if err != nil || matched {
				t.Fatal("mismatch attempted blob access", matched, err)
			}
		})
	}
	entry := session.UserMessageWithImagesEntry(message.Text, []session.ImageData{good})
	entry.ID = "steering_x"
	if _, err := MatchesSteering(context.Background(), sess, message, entry); err == nil {
		t.Fatal("matching reference skipped integrity read")
	}
}
