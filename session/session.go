package session

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// EntryType describes the kind of session entry.
type EntryType string

const (
	EntryTypeMessage    EntryType = "message"
	EntryTypeToolCall   EntryType = "tool_call"
	EntryTypeToolResult EntryType = "tool_result"
	EntryTypeMeta       EntryType = "meta"
	EntryTypeCompaction EntryType = "compaction"
)

// SessionEntry is a single node in the session DAG.
type SessionEntry struct {
	ID        string          `json:"id"`
	ParentID  string          `json:"parentId,omitempty"`
	Type      EntryType       `json:"type"`
	Role      string          `json:"role,omitempty"` // user, assistant, system
	Timestamp int64           `json:"timestamp"`
	Data      json.RawMessage `json:"data"`
}

// ImageData holds a base64-encoded image for session persistence.
type ImageData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"` // base64-encoded
}

// ThinkingBlockData stores a thinking block from a model response.
// It must be echoed back verbatim in subsequent turns.
type ThinkingBlockData struct {
	Thinking  string `json:"thinking"`
	Signature string `json:"signature,omitempty"`
}

// MessageData holds text message content.
type MessageData struct {
	Text           string              `json:"text"`
	Images         []ImageData         `json:"images,omitempty"`
	ThinkingBlocks []ThinkingBlockData `json:"thinking_blocks,omitempty"`
}

// ToolCallData holds a tool call's details.
type ToolCallData struct {
	Tool  string          `json:"tool"`
	ID    string          `json:"id"`
	Input json.RawMessage `json:"input"`
}

// ToolResultData holds the result of a tool call.
type ToolResultData struct {
	ToolCallID string      `json:"tool_call_id"`
	Output     string      `json:"output"`
	Error      string      `json:"error,omitempty"`
	IsError    bool        `json:"is_error,omitempty"`
	Aborted    bool        `json:"aborted,omitempty"` // true when the user cancelled mid-dispatch
	Images     []ImageData `json:"images,omitempty"`
}

// CompactionData holds an append-only summary of an older portion of the
// session. The session view assembles messages by reading entries after the
// most recent CompactionData entry, prepending the summary as a leading
// synthetic user message. The raw entries before the compaction stay on
// disk in JSONL — they are skipped at view assembly only.
type CompactionData struct {
	Summary              string `json:"summary"`
	RangeStartID         string `json:"range_start_id,omitempty"` // first entry covered by the summary
	RangeEndID           string `json:"range_end_id,omitempty"`   // last entry covered by the summary
	Model                string `json:"model"`                    // e.g. "local/qwen2.5:3b-instruct"
	TokensBefore         int    `json:"tokens_before"`
	TokensEstimatedAfter int    `json:"tokens_estimated_after"`
	TurnsCompacted       int    `json:"turns_compacted"`
}

// Session holds a conversation session with DAG-structured entries.
type Session struct {
	ID      string
	AgentID string
	Key     string // channel + peer derived key

	mu       sync.RWMutex // guards entries / entryMap / leafID
	entries  []SessionEntry
	entryMap map[string]*SessionEntry
	leafID   string // current leaf for history traversal
	store    *Store

	// writeMu serializes store writes and preserves their order. Acquired in
	// Append while s.mu is still held (lock-coupling), then s.mu is released
	// before the disk write — so disk latency stays off s.mu while on-disk
	// order still matches in-memory append order under concurrent callers. (P5)
	writeMu sync.Mutex
}

// NewSession creates a new empty session.
func NewSession(agentID, key string) *Session {
	return &Session{
		ID:       generateID("ses"),
		AgentID:  agentID,
		Key:      key,
		entryMap: make(map[string]*SessionEntry),
	}
}

// Append adds an entry to the session.
func (s *Session) Append(entry SessionEntry) {
	s.mu.Lock()
	if entry.ID == "" {
		entry.ID = generateID("e")
	}
	if entry.Timestamp == 0 {
		entry.Timestamp = time.Now().Unix()
	}
	if s.leafID != "" && entry.ParentID == "" {
		entry.ParentID = s.leafID
	}

	s.entries = append(s.entries, entry)
	s.entryMap[entry.ID] = &s.entries[len(s.entries)-1]
	s.leafID = entry.ID

	finalized := entry // value copy with all fields set
	store := s.store

	// Lock-coupling: take writeMu BEFORE releasing s.mu so the disk-write
	// order matches the in-memory append order even under concurrent callers,
	// then release s.mu so disk latency doesn't block concurrent View()/append.
	if store != nil {
		s.writeMu.Lock()
		s.mu.Unlock()
		store.AppendEntry(s, finalized)
		s.writeMu.Unlock()
	} else {
		s.mu.Unlock()
	}
}

// History walks the DAG from root to current leaf and returns the path.
func (s *Session) History() []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.entries) == 0 {
		return nil
	}

	// Build path from leaf back to root
	var path []SessionEntry
	current := s.leafID
	for current != "" {
		entry, ok := s.entryMap[current]
		if !ok {
			break
		}
		path = append(path, *entry)
		current = entry.ParentID
	}

	// Reverse to get root→leaf order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return path
}

// View returns the post-compaction message view for the LLM. Walks the
// current branch from leaf back to root via ParentID; if a compaction entry
// is encountered it becomes the first emitted entry and everything before
// it is dropped. Without any compaction entries, View() is identical to
// History().
//
// Multiple compaction entries stack naturally — only the most recent one
// matters for assembly. Older compactions remain on disk in JSONL.
func (s *Session) View() []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.entries) == 0 {
		return nil
	}

	var path []SessionEntry
	current := s.leafID
	for current != "" {
		entry, ok := s.entryMap[current]
		if !ok {
			break
		}
		path = append(path, *entry)
		if entry.Type == EntryTypeCompaction {
			break // most recent compaction terminates the walk-back
		}
		current = entry.ParentID
	}

	// Reverse to root→leaf order.
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

// Entries returns all entries in append order.
func (s *Session) Entries() []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]SessionEntry, len(s.entries))
	copy(out, s.entries)
	return out
}

// LeafID returns the current leaf entry ID.
func (s *Session) LeafID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.leafID
}

// Branch moves the leaf pointer to the specified entry ID, creating a branch.
// New entries appended after this will have the branch point as their parent.
//
// NOTE: not lock-guarded — see Phase B spec, runs during /resume or DAG branching, never from parallel dispatch goroutines.
func (s *Session) Branch(entryID string) error {
	if _, ok := s.entryMap[entryID]; !ok {
		return fmt.Errorf("entry %q not found in session", entryID)
	}
	s.leafID = entryID
	return nil
}

// EstimateTokens returns a rough token estimate for the current history.
// Uses a simple heuristic of ~4 characters per token.
func (s *Session) EstimateTokens() int {
	history := s.History()
	totalChars := 0
	for _, entry := range history {
		totalChars += len(entry.Data)
		totalChars += len(entry.Role)
	}
	return totalChars / 4
}

// Compact replaces older history entries with a summary entry.
// It keeps the most recent keepEntries entries and replaces everything
// before them with a single summary meta entry.
// The summary text should be generated by the caller (typically by asking
// the LLM to summarize).
//
// NOTE: not lock-guarded — runs between turns only. Calls History() and
// Entries() (via store.Rewrite) internally; do NOT wrap in s.mu.Lock()
// without also restructuring those calls, or you will deadlock on the
// non-recursive sync.RWMutex.
func (s *Session) Compact(summary string, keepEntries int) {
	history := s.History()
	if len(history) <= keepEntries {
		return // nothing to compact
	}

	// Split into old (to compact) and recent (to keep)
	cutoff := len(history) - keepEntries
	recentEntries := history[cutoff:]

	// Create summary meta entry
	summaryData, _ := json.Marshal(MessageData{Text: summary})
	summaryEntry := SessionEntry{
		ID:        generateID("compact"),
		Type:      EntryTypeMeta,
		Role:      "system",
		Timestamp: time.Now().Unix(),
		Data:      summaryData,
	}

	// Rebuild the session with summary + recent entries
	s.entries = nil
	s.entryMap = make(map[string]*SessionEntry)
	s.leafID = ""

	// Add summary entry
	s.entries = append(s.entries, summaryEntry)
	s.entryMap[summaryEntry.ID] = &s.entries[0]
	s.leafID = summaryEntry.ID

	// Add recent entries, re-parenting the first one to the summary
	for i, entry := range recentEntries {
		if i == 0 {
			entry.ParentID = summaryEntry.ID
		} else {
			entry.ParentID = recentEntries[i-1].ID
		}
		s.entries = append(s.entries, entry)
		s.entryMap[entry.ID] = &s.entries[len(s.entries)-1]
		s.leafID = entry.ID
	}

	// Rewrite the session file if store is set
	if s.store != nil {
		s.store.Rewrite(s)
	}
}

// SetStore associates a Store for automatic persistence.
//
// NOTE: not lock-guarded — must be called once during session construction,
// before the session is shared with any goroutine.
func (s *Session) SetStore(store *Store) {
	s.store = store
}

// Helper constructors for common entry types

// UserMessageEntry creates a user message entry.
func UserMessageEntry(text string) SessionEntry {
	data, _ := json.Marshal(MessageData{Text: text})
	return SessionEntry{
		Type: EntryTypeMessage,
		Role: "user",
		Data: data,
	}
}

// UserMessageWithImagesEntry creates a user message entry with image attachments.
func UserMessageWithImagesEntry(text string, images []ImageData) SessionEntry {
	data, _ := json.Marshal(MessageData{Text: text, Images: images})
	return SessionEntry{
		Type: EntryTypeMessage,
		Role: "user",
		Data: data,
	}
}

// AssistantMessageEntry creates an assistant message entry.
func AssistantMessageEntry(text string) SessionEntry {
	data, _ := json.Marshal(MessageData{Text: text})
	return SessionEntry{
		Type: EntryTypeMessage,
		Role: "assistant",
		Data: data,
	}
}

// AssistantMessageEntryWithThinking creates an assistant message entry that
// includes thinking blocks which must be echoed back in subsequent turns.
func AssistantMessageEntryWithThinking(text string, blocks []ThinkingBlockData) SessionEntry {
	data, _ := json.Marshal(MessageData{Text: text, ThinkingBlocks: blocks})
	return SessionEntry{
		Type: EntryTypeMessage,
		Role: "assistant",
		Data: data,
	}
}

// ToolCallEntry creates a tool call entry.
//
// input is sanitised to "{}" when empty or not valid JSON. The
// marshal step is fragile against malformed RawMessage: an empty-
// but-non-nil json.RawMessage (length 0 but allocated, which is
// what happens when the LLM emits a tool_use whose arguments
// stream produced zero bytes) makes json.Marshal return an error.
// The previous `data, _ := json.Marshal(...)` swallowed that error
// and persisted Data: nil, which serialises on disk as `"data":null`.
// On reload, assembleMessages would then build a ToolCall with an
// empty ID, breaking the tool_use ↔ tool_result pairing in the next
// LLM request and producing a 400 from Anthropic of the form
// `messages.N.content.0: unexpected tool_use_id ... Each tool_result
// block must have a corresponding tool_use block in the previous
// message.` Substituting "{}" produces a valid empty-args tool_use,
// which the model can interpret correctly on the next turn.
func ToolCallEntry(toolCallID, toolName string, input json.RawMessage) SessionEntry {
	if len(input) == 0 || !json.Valid(input) {
		input = json.RawMessage(`{}`)
	}
	data, _ := json.Marshal(ToolCallData{
		Tool:  toolName,
		ID:    toolCallID,
		Input: input,
	})
	return SessionEntry{
		Type: EntryTypeToolCall,
		Data: data,
	}
}

// ToolResultEntry creates a tool result entry.
func ToolResultEntry(toolCallID, output, errMsg string, images []ImageData) SessionEntry {
	data, _ := json.Marshal(ToolResultData{
		ToolCallID: toolCallID,
		Output:     output,
		Error:      errMsg,
		IsError:    errMsg != "",
		Images:     images,
	})
	return SessionEntry{
		Type: EntryTypeToolResult,
		Data: data,
	}
}

// AbortedToolResultEntry creates a synthetic tool result for a tool call that
// was cancelled before completion. Pairs with a previously-appended ToolCallEntry
// to satisfy the API invariant that every tool_use has a matching tool_result.
func AbortedToolResultEntry(toolCallID string) SessionEntry {
	data, _ := json.Marshal(ToolResultData{
		ToolCallID: toolCallID,
		Error:      "aborted by user",
		IsError:    true,
		Aborted:    true,
	})
	return SessionEntry{
		Type: EntryTypeToolResult,
		Data: data,
	}
}

// CompactionEntry creates a new compaction entry summarizing an older
// range of session history. The append-only path: callers Append() this
// entry, the JSONL on disk is never rewritten.
func CompactionEntry(summary, rangeStartID, rangeEndID, model string, tokensBefore, tokensEstimatedAfter, turnsCompacted int) SessionEntry {
	data, _ := json.Marshal(CompactionData{
		Summary:              summary,
		RangeStartID:         rangeStartID,
		RangeEndID:           rangeEndID,
		Model:                model,
		TokensBefore:         tokensBefore,
		TokensEstimatedAfter: tokensEstimatedAfter,
		TurnsCompacted:       turnsCompacted,
	})
	return SessionEntry{
		Type: EntryTypeCompaction,
		Role: "system",
		Data: data,
	}
}

func generateID(prefix string) string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + hex.EncodeToString(b)
}
