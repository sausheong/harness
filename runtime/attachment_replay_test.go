package runtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
)

type attachmentReplayProvider struct {
	llmtest.Base
	calls  int
	images []llm.ImageContent
}

func (p *attachmentReplayProvider) ChatStream(_ context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.calls++
	for _, msg := range req.Messages {
		p.images = append(p.images, msg.Images...)
	}
	events := make(chan llm.ChatEvent, 2)
	events <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "Answer"}
	events <- llm.ChatEvent{Type: llm.EventDone}
	close(events)
	return events, nil
}

func TestRuntimeReplaysAttachmentsAndRejectsMissingBlobs(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "replay"
		if missing {
			name = "missing"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			store := session.NewStore(dir)
			if err := store.Create("agent", "key"); err != nil {
				t.Fatal(err)
			}
			sess, err := store.LoadExclusive("agent", "key")
			if err != nil {
				t.Fatal(err)
			}
			sess.Append(session.UserMessageWithImagesEntry("original", []session.ImageData{{MimeType: "image/png", Data: base64.StdEncoding.EncodeToString([]byte("original image"))}}))
			if err := sess.Flush(); err != nil {
				t.Fatal(err)
			}
			sess.Close()
			sess, err = store.LoadExclusive("agent", "key")
			if err != nil {
				t.Fatal(err)
			}
			defer sess.Close()
			if missing {
				var data session.MessageData
				json.Unmarshal(sess.History()[0].Data, &data)
				if err := os.Remove(filepath.Join(dir, "agent", ".attachments", data.Images[0].Reference.SHA256)); err != nil {
					t.Fatal(err)
				}
			}
			provider := &attachmentReplayProvider{}
			rt := &Runtime{Session: sess, LLM: provider, Tools: usageNoopExecutor{}, Model: "model", MaxTurns: 1}
			_, err = rt.RunSync(context.Background(), "continue", nil)
			if missing {
				if err == nil || provider.calls != 0 {
					t.Fatal("missing image silently omitted", err, provider.calls)
				}
				return
			}
			if err != nil || provider.calls != 1 || len(provider.images) != 1 || string(provider.images[0].Data) != "original image" {
				t.Fatal(provider, err)
			}
		})
	}
}
