package compaction

import "github.com/sausheong/harness/session"

// Split divides history into (toCompact, toPreserve) at a clean turn boundary.
//
// Algorithm: walk backwards from the leaf, count user messages. After we have
// seen K of them, the next encountered user message is the cutoff. Everything
// from that user message forward is preserved verbatim; everything before is
// the to-be-compacted range.
//
// ok is false when the path contains <= K user messages — there is no cutoff
// that preserves K turns. Caller should refuse to compact rather than over-
// compacting.
//
// A user message is always a clean boundary by construction in the
// harness runtime (user msg → assistant text → tool_call → tool_result
// → next user msg). Splitting before a user message therefore never
// orphans a tool pair.
func Split(history []session.SessionEntry, K int) (toCompact, toPreserve []session.SessionEntry, ok bool) {
	if K <= 0 || len(history) == 0 {
		return nil, nil, false
	}

	// Walk backwards counting user messages. cutoffIdx will land on the
	// (K+1)-th user message from the end (i.e. the first user message that
	// belongs to the to-be-compacted range — preserved range starts at the
	// next user message we already counted).
	userCount := 0
	cutoffIdx := -1
	for i := len(history) - 1; i >= 0; i-- {
		e := history[i]
		if e.Type != session.EntryTypeMessage || e.Role != "user" {
			continue
		}
		userCount++
		if userCount > K {
			cutoffIdx = i
			break
		}
	}
	if cutoffIdx < 0 {
		return nil, nil, false
	}

	// Find the next user message AFTER cutoffIdx — that is the start of the
	// preserved range. Everything strictly before it is compacted.
	preserveStart := -1
	for i := cutoffIdx + 1; i < len(history); i++ {
		e := history[i]
		if e.Type == session.EntryTypeMessage && e.Role == "user" {
			preserveStart = i
			break
		}
	}
	if preserveStart < 0 {
		// Shouldn't happen given the count, but guard against it.
		return nil, nil, false
	}

	toCompact = append(toCompact, history[:preserveStart]...)
	toPreserve = append(toPreserve, history[preserveStart:]...)
	return toCompact, toPreserve, true
}
