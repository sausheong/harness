package runtime

import "github.com/sausheong/harness/llm"

// addUsage returns the field-wise sum of two usage snapshots. A nil operand is
// treated as zero. It returns a fresh pointer and never mutates its arguments,
// so callers may safely pass a pointer that aliases a provider stream event.
// Returns nil only when BOTH operands are nil, preserving the contract that
// EventDone.Usage stays nil for providers that never report usage.
func addUsage(acc, turn *llm.Usage) *llm.Usage {
	if acc == nil && turn == nil {
		return nil
	}
	var out llm.Usage
	if acc != nil {
		out = *acc
	}
	if turn != nil {
		out.InputTokens += turn.InputTokens
		out.OutputTokens += turn.OutputTokens
		out.CacheCreationInputTokens += turn.CacheCreationInputTokens
		out.CacheReadInputTokens += turn.CacheReadInputTokens
	}
	return &out
}
