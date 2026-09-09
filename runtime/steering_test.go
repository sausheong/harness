package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/session"
)

func TestSteeringStopsRemainingToolsAndAcknowledgesDurably(t *testing.T) {
	exec := newTimedExecutor()
	exec.addTool("read")
	exec.markSafe("read")
	first := &llm.ToolCall{ID: "one", Name: "read", Input: json.RawMessage(`{}`)}
	second := &llm.ToolCall{ID: "two", Name: "read", Input: json.RawMessage(`{}`)}
	provider := &scriptedStreamLLM{events: []scriptedStreamEvent{{typ: llm.EventToolCallDone, toolCall: first}, {typ: llm.EventToolCallDone, toolCall: second}, {typ: llm.EventDone}}}
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "k"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("a", "k")
	if err != nil {
		t.Fatal(err)
	}
	rt := &Runtime{LLM: provider, Tools: exec, Session: sess, AgentID: "a", Model: "test", MaxTurns: 3, AgentLoop: LoopConfig{StreamingTools: true}}
	ack := false
	ctx := WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) {
		if exec.callCount() == 0 || ack {
			return nil, nil
		}
		return &SteeringMessage{ID: "correction", Text: "Stop reading and answer", Acknowledge: func() error { ack = true; return nil }}, nil
	})
	if _, err := rt.RunSync(ctx, "go", nil); err != nil {
		t.Fatal(err)
	}
	if exec.callCount() != 1 || !ack {
		t.Fatal("remaining tool ran or correction lost", exec.callCount(), ack)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	sess, err = store.LoadExclusive("a", "k")
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	calls, results, corrections := 0, 0, 0
	for _, entry := range sess.Entries() {
		switch entry.Type {
		case session.EntryTypeToolCall:
			calls++
		case session.EntryTypeToolResult:
			results++
		}
		if entry.ID == "steering_correction" {
			corrections++
		}
	}
	if calls != 2 || results != 2 || corrections != 1 {
		t.Fatal(calls, results, corrections)
	}
	rt.Session = sess
	before := len(sess.Entries())
	ctx = WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) {
		return &SteeringMessage{ID: "correction", Text: "Stop reading and answer"}, nil
	})
	if applied, err := rt.applySteering(ctx, nil); err != nil || applied || len(sess.Entries()) != before {
		t.Fatal("redelivery duplicated input", applied, err)
	}
}

func TestSteeringDoesNotAcknowledgePersistenceFailure(t *testing.T) {
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "k"); err != nil {
		t.Fatal(err)
	}
	sess, err := store.LoadExclusive("a", "k")
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	ack := false
	rt := &Runtime{Session: sess}
	ctx := WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) {
		return &SteeringMessage{ID: "x", Text: "correction", Acknowledge: func() error { ack = true; return nil }}, nil
	})
	if _, err := rt.applySteering(ctx, nil); err == nil || ack {
		t.Fatal("failed write acknowledged", err, ack)
	}
}

func TestSteeringSourceErrorAndInvalidInputDoNotWrite(t *testing.T) {
	rt := &Runtime{Session: session.NewSession("a", "k")}
	sentinel := errors.New("queue unavailable")
	ctx := WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) { return nil, sentinel })
	if _, err := rt.applySteering(ctx, nil); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	ctx = WithSteering(context.Background(), func(context.Context) (*SteeringMessage, error) { return &SteeringMessage{ID: "x", Text: " "}, nil })
	if _, err := rt.applySteering(ctx, nil); err == nil {
		t.Fatal("invalid correction accepted")
	}
	if len(rt.Session.Entries()) != 0 {
		t.Fatal("failed steering changed history")
	}
}
