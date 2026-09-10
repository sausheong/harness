package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"unicode/utf8"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

type steeringContextKey struct{}

// SteeringMessage identifies one correction. Source retains it until Acknowledge
// succeeds. ID must be stable across retry; Text must not change after delivery.
type SteeringMessage struct {
	ID          string
	Text        string
	Images      []llm.ImageContent
	Acknowledge func() error
}
type SteeringSource func(context.Context) (*SteeringMessage, error)

// WithSteering enables safe-boundary delivery for Run. Tool execution becomes
// serial and streaming kickoff is disabled so no eligible later tool can race a
// correction. Source is called after model output and between joined tools.
// A nil message means no correction. Source must return promptly and honour ctx.
func WithSteering(ctx context.Context, source SteeringSource) context.Context {
	return context.WithValue(ctx, steeringContextKey{}, source)
}
func steeringSource(ctx context.Context) SteeringSource {
	source, _ := ctx.Value(steeringContextKey{}).(SteeringSource)
	return source
}

func (r *Runtime) applySteering(ctx context.Context, pending []llm.ToolCall) (bool, error) {
	source := steeringSource(ctx)
	if source == nil {
		return false, nil
	}
	if err := ctx.Err(); err != nil {
		return false, context.Cause(ctx)
	}
	message, err := source(ctx)
	if err != nil || message == nil {
		return false, err
	}
	if message.ID == "" || len(message.ID) > 128 || strings.ContainsAny(message.ID, "\r\n\x00") || !utf8.ValidString(message.ID) || !utf8.ValidString(message.Text) || strings.TrimSpace(message.Text) == "" || len(message.Text) > 8<<20 {
		return false, errors.New("invalid steering identity or text")
	}
	if len(message.Images) > 16 {
		return false, errors.New("too many steering images")
	}
	totalImageBytes := 0
	for _, img := range message.Images {
		if len(img.Data) > (32<<20)-totalImageBytes {
			return false, errors.New("steering images exceed 32 MiB")
		}
		totalImageBytes += len(img.Data)
		if img.MimeType == "" {
			return false, errors.New("steering image MIME type missing")
		}
	}
	entry := session.UserMessageEntry(message.Text)
	if len(entry.Data) > 8<<20 {
		return false, errors.New("encoded steering text exceeds 8 MiB")
	}
	if len(message.Images) > 0 {
		entry = session.UserMessageWithImagesEntry(message.Text, convertToolResultImages(message.Images))
	}
	entry.ID = "steering_" + message.ID
	for _, prior := range r.Session.Entries() {
		if prior.ID != entry.ID {
			continue
		}
		matches, err := MatchesSteering(ctx, r.Session, *message, prior)
		if err != nil {
			return false, err
		}
		if !matches {
			return false, errors.New("conflicting delivered steering identity")
		}
		if err := r.Session.Flush(); err != nil {
			return false, err
		}
		if message.Acknowledge != nil {
			return false, message.Acknowledge()
		}
		return false, nil
	}
	// Reconcile every proposed call before injecting user input. Unexecuted tools
	// get explicit error results so provider histories never contain orphan calls.
	for _, tc := range pending {
		r.Session.AppendContext(ctx, session.ToolCallEntry(tc.ID, tc.Name, tc.Input))
		r.Session.AppendContext(ctx, session.ToolResultEntry(tc.ID, "", "skipped because user steering superseded this tool call", nil))
	}
	r.Session.AppendContext(ctx, entry)
	if err := r.Session.Flush(); err != nil {
		return false, err
	}
	if message.Acknowledge != nil {
		if err := message.Acknowledge(); err != nil {
			return true, err
		}
	}
	return true, nil
}

// MatchesSteering compares a delivered correction with its immutable source,
// hydrating and verifying referenced image bytes before comparing semantics.
// It is also used by hosts reconciling an interrupted acknowledgement.
func MatchesSteering(ctx context.Context, sess *session.Session, message SteeringMessage, prior session.SessionEntry) (bool, error) {
	if prior.ID != "steering_"+message.ID || prior.Type != session.EntryTypeMessage || prior.Role != "user" {
		return false, nil
	}
	// Reject count/size/digest mismatches before hydration. A small forged JSONL
	// record can otherwise reference many large blobs and expand without bound.
	if len(message.Images) > 16 {
		return false, errors.New("too many steering images")
	}
	total := 0
	for _, img := range message.Images {
		if len(img.Data) > (32<<20)-total {
			return false, errors.New("steering images exceed 32 MiB")
		}
		total += len(img.Data)
	}
	var stored session.MessageData
	if err := json.Unmarshal(prior.Data, &stored); err != nil {
		return false, err
	}
	if stored.Text != message.Text || len(stored.Images) != len(message.Images) {
		return false, nil
	}
	for i, img := range stored.Images {
		want := message.Images[i]
		if img.MimeType != want.MimeType {
			return false, nil
		}
		if img.Reference != nil {
			if img.Data != "" {
				return false, errors.New("ambiguous steering image representation")
			}
			if err := img.Reference.Validate(); err != nil {
				return false, err
			}
			digest := sha256.Sum256(want.Data)
			if img.Reference.Size != int64(len(want.Data)) || img.Reference.SHA256 != hex.EncodeToString(digest[:]) {
				return false, nil
			}
		} else if img.Data != base64.StdEncoding.EncodeToString(want.Data) {
			return false, nil
		}
	}
	entries, err := sess.ResolveImages(ctx, []session.SessionEntry{prior})
	if err != nil {
		return false, err
	}
	var actual session.MessageData
	if err := json.Unmarshal(entries[0].Data, &actual); err != nil {
		return false, err
	}
	expected := session.MessageData{Text: message.Text, Images: convertToolResultImages(message.Images)}
	return reflect.DeepEqual(actual, expected), nil
}
