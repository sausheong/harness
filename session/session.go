package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/sausheong/harness/attachment"
	"github.com/sausheong/harness/process"
	"sync"
	"sync/atomic"
	"time"
)

// EntryType describes the kind of session entry.
type EntryType string

const (
	EntryTypeMessage    EntryType = "message"
	EntryTypeHeader     EntryType = "session_header"
	EntryTypeToolCall   EntryType = "tool_call"
	EntryTypeToolResult EntryType = "tool_result"
	EntryTypeMeta       EntryType = "meta"
	EntryTypeCompaction EntryType = "compaction"
	EntryTypeAnnotation EntryType = "annotation" // durable metadata, never a conversation node
	EntryTypeSelection  EntryType = "selection"  // versioned control record; never part of conversation history
)

// SessionEntry is a single node in the session DAG.
type SessionEntry struct {
	SchemaVersion int             `json:"schemaVersion,omitempty"`
	ID            string          `json:"id"`
	ParentID      string          `json:"parentId,omitempty"`
	Type          EntryType       `json:"type"`
	Role          string          `json:"role,omitempty"` // user, assistant, system
	Timestamp     int64           `json:"timestamp"`
	Data          json.RawMessage `json:"data"`
}

// ImageData holds a legacy inline image or a durable attachment reference.
type ImageData struct {
	MimeType  string          `json:"mime_type"`
	Data      string          `json:"data,omitempty"` // base64-encoded legacy representation
	Reference *attachment.Ref `json:"attachment,omitempty"`
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
	Artifacts  map[string]process.ArtifactInfo `json:"artifacts,omitempty"`
	ToolCallID string                          `json:"tool_call_id"`
	Output     string                          `json:"output"`
	Error      string                          `json:"error,omitempty"`
	IsError    bool                            `json:"is_error,omitempty"`
	Aborted    bool                            `json:"aborted,omitempty"` // true when the user cancelled mid-dispatch
	Images     []ImageData                     `json:"images,omitempty"`
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
	leaseMu            sync.Mutex
	lease              *writerLease
	writerClosed       bool
	persistenceFailure atomic.Pointer[persistenceFailure]
	ID                 string
	AgentID            string
	Key                string // channel + peer derived key

	mu       sync.RWMutex // guards entries / entryMap / leafID
	entries  []SessionEntry
	entryMap map[string]*SessionEntry
	header   SessionEntry // immutable after construction/load
	leafID   string       // current leaf for history traversal
	store    *Store

	// writeMu serializes store writes and preserves their order. Acquired in
	// Append while s.mu is still held (lock-coupling), then s.mu is released
	// before the disk write — so disk latency stays off s.mu while on-disk
	// order still matches in-memory append order under concurrent callers. (P5)
	writeMu sync.Mutex
}

// NewSession creates a new empty session.
func NewSession(agentID, key string) *Session {
	id := generateID("ses")
	return &Session{ID: id, AgentID: agentID, Key: key, entryMap: make(map[string]*SessionEntry), header: SessionEntry{ID: id, Type: EntryTypeHeader, SchemaVersion: 1, Timestamp: time.Now().Unix()}}
}

// Append adds an entry to the session.
func (s *Session) Append(entry SessionEntry) { _ = s.AppendContext(context.Background(), entry) }

// AppendContext persists image blobs before publishing their referencing entry.
// Cancellation during blob I/O is exposed as a sticky persistence failure.
func (s *Session) AppendContext(ctx context.Context, entry SessionEntry) error {
	s.mu.Lock()
	fail := func(err error) error {
		s.persistenceFailure.CompareAndSwap(nil, &persistenceFailure{err})
		s.mu.Unlock()
		return err
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	if err := s.PersistenceError(); err != nil {
		return fail(err)
	}
	if s.store != nil {
		s.writeMu.Lock()
		s.leaseMu.Lock()
		closed := s.writerClosed
		s.leaseMu.Unlock()
		if closed {
			s.writeMu.Unlock()
			return fail(ErrSessionClosed)
		}
		var err error
		entry, err = s.externalizeImages(ctx, entry)
		s.writeMu.Unlock()
		if err != nil {
			return fail(err)
		}
	}

	if entry.ID == "" {
		entry.ID = generateID("e")
	}
	if entry.Timestamp == 0 {
		entry.Timestamp = time.Now().Unix()
	}
	if s.leafID != "" && entry.ParentID == "" {
		entry.ParentID = s.leafID
	}

	if err := validateGraphNode(entry, func(id string) (EntryType, bool) {
		if id == s.header.ID {
			return EntryTypeHeader, true
		}
		node, ok := s.entryMap[id]
		if !ok {
			return "", false
		}
		return node.Type, true
	}); err != nil {
		return fail(err)
	}
	// Selection control records must use Branch so their durability and leaf
	// semantics cannot be bypassed through the generic append API.
	if entry.Type == EntryTypeSelection || entry.Type == EntryTypeHeader || entry.Type == EntryTypeAnnotation {
		return fail(fmt.Errorf("control records require their dedicated API"))
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
	return s.PersistenceError()
}

// History walks the DAG from root to current leaf and returns the path.
func (s *Session) History() []SessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.historyLocked()
}

func (s *Session) historyLocked() []SessionEntry {
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

// Branch durably selects an existing conversation node. Selection is an
// append-only control record, so copying/renaming/exporting the JSONL also
// preserves the selected branch. It is excluded from History and View.
func (s *Session) Branch(entryID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	target, ok := s.entryMap[entryID]
	if !ok || target.Type == EntryTypeSelection || target.Type == EntryTypeHeader || target.Type == EntryTypeAnnotation {
		return fmt.Errorf("entry %q is not a conversation node", entryID)
	}
	if err := s.PersistenceError(); err != nil {
		return err
	}
	marker := SessionEntry{ID: generateID("select"), ParentID: entryID, Type: EntryTypeSelection, Timestamp: time.Now().Unix(), Data: json.RawMessage(`{"version":1}`)}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.store != nil {
		s.store.AppendEntry(s, marker)
		if err := s.Flush(); err != nil {
			return err
		}
	}
	s.entries = append(s.entries, marker)
	s.entryMap[marker.ID] = &s.entries[len(s.entries)-1]
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

// Compact creates a new summarised branch while retaining all original
// records and branches. Kept messages receive new IDs and parents; the original
// nodes are immutable and remain available for branching and export. The
// replacement is atomic on disk and keeps appends ordered behind the snapshot.
func (s *Session) Compact(summary string, keepEntries int) {
	if keepEntries < 0 {
		keepEntries = 0
	}
	s.mu.Lock()
	history := s.historyLocked()
	if len(history) <= keepEntries {
		s.mu.Unlock()
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

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	// Add summary entry
	s.entries = append(s.entries, summaryEntry)
	s.entryMap[summaryEntry.ID] = &s.entries[len(s.entries)-1]
	s.leafID = summaryEntry.ID

	// Clone recent entries onto the new summary branch. Never rewrite the
	// identities or parent links of records in the original graph.
	for _, entry := range recentEntries {
		entry.ID = generateID("e")
		entry.ParentID = s.leafID
		s.entries = append(s.entries, entry)
		s.entryMap[entry.ID] = &s.entries[len(s.entries)-1]
		s.leafID = entry.ID
	}

	entries := append([]SessionEntry(nil), s.entries...)
	store := s.store
	s.mu.Unlock()
	// Keep appends ordered behind the snapshot until its replacement commits.
	if store != nil {
		store.rewriteEntries(s, entries)
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
	return ToolResultWithArtifactsEntry(toolCallID, output, errMsg, images, nil)
}

// ToolResultWithArtifactsEntry records immutable captured-output references.
func ToolResultWithArtifactsEntry(toolCallID, output, errMsg string, images []ImageData, artifacts map[string]process.ArtifactInfo) SessionEntry {
	data, _ := json.Marshal(ToolResultData{
		Artifacts:  artifacts,
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
	return AbortedToolResultWithReasonEntry(toolCallID, "aborted by user")
}

// AbortedToolResultWithReasonEntry retains the cause of an interrupted tool.
func AbortedToolResultWithReasonEntry(toolCallID, reason string) SessionEntry {
	data, _ := json.Marshal(ToolResultData{
		ToolCallID: toolCallID,
		Error:      reason,
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
