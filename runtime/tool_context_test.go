package runtime

import (
	"context"
	"errors"
	"github.com/sausheong/harness/llm"
	"strings"
	"testing"
)

func TestToolContextProjectionPreservesProtectedRequest(t *testing.T) {
	original := []llm.Message{{Role: "system", Content: "policy"}, {Role: "user", Content: "instruction"}, {Role: "user", ToolCallID: "tool-1", Content: " original ", IsError: true}}
	req := llm.ChatRequest{Messages: original}
	r := &Runtime{}
	r.AgentLoop.Hooks.TransformToolContext = func(ctx context.Context, items []ToolContextText) ([]ToolContextText, error) {
		if len(items) != 1 || items[0].Index != 2 {
			t.Fatal(items)
		}
		items[0].Text = "transformed"
		return items, nil
	}
	called := false
	stream, err := r.observeChat(context.Background(), req, llm.CallGeneration, func(ctx context.Context, got llm.ChatRequest) (<-chan llm.ChatEvent, error) {
		called = true
		if got.Messages[0].Content != "policy" || got.Messages[1].Content != "instruction" || got.Messages[2].Content != "transformed" || got.Messages[2].ToolCallID != "tool-1" || !got.Messages[2].IsError {
			t.Fatal(got.Messages)
		}
		ch := make(chan llm.ChatEvent)
		close(ch)
		return ch, nil
	})
	if stream != nil {
		for range stream {
		}
	}
	if err != nil || !called || original[2].Content != " original " {
		t.Fatal("request or source mutation", err, original)
	}
}
func TestToolContextRejectsInvalidBatchBeforeProvider(t *testing.T) {
	for _, mode := range []string{"protected-index", "missing", "large", "error", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := &Runtime{}
			r.AgentLoop.Hooks.TransformToolContext = func(ctx context.Context, items []ToolContextText) ([]ToolContextText, error) {
				switch mode {
				case "protected-index":
					items[0].Index = 0
				case "missing":
					return nil, nil
				case "large":
					items[0].Text = strings.Repeat("x", MaxTransformedToolTextBytes+1)
				case "error":
					return nil, errors.New("mandatory transform failed")
				case "cancel":
					cancel()
				}
				return items, nil
			}
			_, err := r.observeChat(ctx, llm.ChatRequest{Messages: []llm.Message{{Role: "user", Content: "protected"}, {Role: "user", ToolCallID: "id", Content: "original"}}}, llm.CallGeneration, func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error) {
				t.Fatal("provider called after invalid transform")
				return nil, nil
			})
			if err == nil {
				t.Fatal("invalid transform succeeded")
			}
		})
	}
}
