package runtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

// TestConvertToolResultImages_PreservesPDFMimeType pins convertToolResultImages'
// behavior for a PDF attachment: the MimeType is carried through unchanged
// (it is not an image format convertToolResultImages inspects or rewrites).
func TestConvertToolResultImages_PreservesPDFMimeType(t *testing.T) {
	pdfBytes := []byte("%PDF-1.4 fake")
	out := convertToolResultImages([]llm.ImageContent{{MimeType: "application/pdf", Data: pdfBytes}})
	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}
	if out[0].MimeType != "application/pdf" {
		t.Fatalf("MimeType = %q, want application/pdf", out[0].MimeType)
	}
	decoded, err := base64.StdEncoding.DecodeString(out[0].Data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(decoded) != string(pdfBytes) {
		t.Fatalf("Data = %q, want %q", decoded, pdfBytes)
	}
}

// TestSessionRoundTrip_ToolResultPDFAttachment writes a
// ToolResultWithArtifactsEntry carrying a PDF attachment through
// session.ToolResultWithArtifactsEntry, round-trips it through JSON exactly
// as disk storage does, and confirms decoding it back reproduces the
// MimeType and bytes untouched.
func TestSessionRoundTrip_ToolResultPDFAttachment(t *testing.T) {
	pdfBytes := []byte("%PDF-1.4 fake pdf")
	images := []session.ImageData{{MimeType: "application/pdf", Data: base64.StdEncoding.EncodeToString(pdfBytes)}}
	entry := session.ToolResultWithArtifactsEntry("tc_1", "showing pages", "", images, nil)

	raw, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded session.SessionEntry
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var tr session.ToolResultData
	if err := json.Unmarshal(decoded.Data, &tr); err != nil {
		t.Fatalf("unmarshal tool result data: %v", err)
	}
	if len(tr.Images) != 1 || tr.Images[0].MimeType != "application/pdf" {
		t.Fatalf("tr.Images = %+v, want one application/pdf image", tr.Images)
	}
	gotBytes, err := base64.StdEncoding.DecodeString(tr.Images[0].Data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(gotBytes) != string(pdfBytes) {
		t.Fatalf("Data = %q, want %q", gotBytes, pdfBytes)
	}
}

// pdfToolResultProvider issues one tool call on its first turn, then on the
// second turn records the images (and their MimeTypes) it received in the
// tool_result message.
type pdfToolResultProvider struct {
	llmtest.Base
	calls  int
	images []llm.ImageContent
}

func (p *pdfToolResultProvider) ChatStream(_ context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.calls++
	events := make(chan llm.ChatEvent, 4)
	if p.calls == 1 {
		events <- llm.ChatEvent{Type: llm.EventToolCallStart, ToolCall: &llm.ToolCall{ID: "tc_1", Name: "read_document"}}
		events <- llm.ChatEvent{Type: llm.EventToolCallDone, ToolCall: &llm.ToolCall{ID: "tc_1", Name: "read_document", Input: json.RawMessage(`{}`)}}
		events <- llm.ChatEvent{Type: llm.EventDone, StopReason: "tool_use"}
		close(events)
		return events, nil
	}
	for _, msg := range req.Messages {
		p.images = append(p.images, msg.Images...)
	}
	events <- llm.ChatEvent{Type: llm.EventTextDelta, Text: "done"}
	events <- llm.ChatEvent{Type: llm.EventDone}
	close(events)
	return events, nil
}

// pdfExecutor is a minimal tool.Executor whose one tool ("read_document")
// always returns a PDF attachment, standing in for readdoc_tool.go's real
// behavior for this harness-side round-trip test.
type pdfExecutor struct{ pdfBytes []byte }

func newPDFTestRegistry(pdfBytes []byte) tool.Executor { return pdfExecutor{pdfBytes: pdfBytes} }

func (e pdfExecutor) Execute(_ context.Context, name string, _ json.RawMessage) (tool.ToolResult, error) {
	if name != "read_document" {
		return tool.ToolResult{Error: "unknown tool"}, nil
	}
	return tool.ToolResult{
		Output: "showing pages",
		Images: []llm.ImageContent{{MimeType: "application/pdf", Data: e.pdfBytes}},
	}, nil
}
func (e pdfExecutor) ToolDefs() []llm.ToolDef      { return []llm.ToolDef{{Name: "read_document"}} }
func (e pdfExecutor) Names() []string              { return []string{"read_document"} }
func (e pdfExecutor) Get(string) (tool.Tool, bool) { return nil, false }

// TestRuntimePDFAttachmentSurvivesSessionAppendAndReachesNextTurn drives a
// full tool-call round trip: a tool returns a PDF attachment, the runtime
// appends it to the session as a ToolResultWithArtifactsEntry, and the very
// next LLM request (assembled from that same session) must carry the PDF
// back with MimeType "application/pdf" intact.
func TestRuntimePDFAttachmentSurvivesSessionAppendAndReachesNextTurn(t *testing.T) {
	pdfBytes := []byte("%PDF-1.4 fake pdf pages")
	sess := session.NewSession("agent", "key")
	provider := &pdfToolResultProvider{}
	reg := newPDFTestRegistry(pdfBytes)

	rt := &Runtime{Session: sess, LLM: provider, Tools: reg, Model: "model", MaxTurns: 3}
	if _, err := rt.RunSync(context.Background(), "read this pdf", nil); err != nil {
		t.Fatalf("RunSync() error = %v", err)
	}

	if provider.calls != 2 {
		t.Fatalf("provider.calls = %d, want 2", provider.calls)
	}
	if len(provider.images) != 1 {
		t.Fatalf("len(provider.images) = %d, want 1", len(provider.images))
	}
	if provider.images[0].MimeType != "application/pdf" {
		t.Fatalf("MimeType = %q, want application/pdf", provider.images[0].MimeType)
	}
	if string(provider.images[0].Data) != string(pdfBytes) {
		t.Fatalf("Data mismatch: got %d bytes, want %d bytes", len(provider.images[0].Data), len(pdfBytes))
	}

	// Confirm the session entry itself carries the PDF MimeType, not just
	// the reconstructed message — i.e. the append path (not only assembly)
	// preserved it.
	var found bool
	for _, e := range sess.History() {
		if e.Type != session.EntryTypeToolResult {
			continue
		}
		var tr session.ToolResultData
		if err := json.Unmarshal(e.Data, &tr); err != nil {
			continue
		}
		for _, img := range tr.Images {
			if img.MimeType == "application/pdf" {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("no application/pdf image found in session history's tool_result entry")
	}
}
