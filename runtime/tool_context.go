package runtime

import (
	"context"
	"errors"
	"unicode/utf8"

	"github.com/sausheong/harness/llm"
)

// ToolContextText is an editable projection of a tool-result message. Index is
// its host-owned request position; role, identity and error metadata are absent.
type ToolContextText struct {
	Index int
	Text  string
}

const MaxTransformedToolTextBytes = 256 << 10

func (r *Runtime) observeChat(ctx context.Context, req llm.ChatRequest, category llm.CallCategory, call func(context.Context, llm.ChatRequest) (<-chan llm.ChatEvent, error)) (<-chan llm.ChatEvent, error) {
	hook := r.AgentLoop.Hooks.TransformToolContext
	if hook != nil {
		input := []ToolContextText{}
		for i, message := range req.Messages {
			if (message.Role == "user" || message.Role == "tool") && message.ToolCallID != "" {
				input = append(input, ToolContextText{Index: i, Text: message.Content})
			}
		}
		if len(input) > 0 {
			// Preserve an independent list of host-owned indices even if the hook edits
			// its input slice in place. Strings cannot mutate recorded message bytes.
			expected := append([]ToolContextText(nil), input...)
			output, err := hook(ctx, input)
			if err != nil {
				return nil, err
			}
			if err = ctx.Err(); err != nil {
				return nil, err
			}
			if len(output) != len(expected) {
				return nil, errors.New("tool context transform changed message count")
			}
			for i, item := range output {
				if item.Index != expected[i].Index || len(item.Text) > MaxTransformedToolTextBytes || !utf8.ValidString(item.Text) {
					return nil, errors.New("invalid transformed tool context")
				}
			}
			req.Messages = append([]llm.Message(nil), req.Messages...)
			for _, item := range output {
				req.Messages[item.Index].Content = item.Text
			}
		}
	}
	return llm.ObserveChat(ctx, req, category, call)
}
