// Package memory defines the writable memory subsystem for harness:
// the MemoryStore interface that any backend implements, the canonical
// Entry record type, and the OriginKey context value for write
// provenance.
//
// The MemoryTool in tool.go wraps any MemoryStore implementation as a
// tool.Tool the agent can call. The default disk-backed implementation
// lives in tool/memory/jsonl/ — that package's *Store value satisfies
// MemoryStore directly, and satisfies runtime.MemoryProvider via
// Store.AsMemoryProvider() which returns a small wrapper that adapts
// the read signature. One underlying *Store can therefore serve as the
// system-prompt index source, the foreground memory tool's backend,
// AND the review-pass tool's backend.
package memory

import (
	"context"
	"errors"
	"time"
)

// Entry is one durable memory record. ID is stable across reads/writes;
// the store generates one when the caller doesn't supply it (Save with
// Entry.ID == ""). Content is opaque to harness.
type Entry struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Tags      []string  `json:"tags,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Origin    string    `json:"origin,omitempty"` // "user", "agent", "review", or any caller-defined label
}

// MemoryStore is the writable backend the MemoryTool dispatches against.
// Implementations should be safe for concurrent use within a single
// process. Cross-process safety is implementation-specific (the JSONL
// default uses flock).
//
// Implementations may also satisfy runtime.MemoryProvider (FormatIndex
// + Get(string) (string, bool)) so the same instance powers both the
// system-prompt index and the write tool.
type MemoryStore interface {
	// Save appends a new entry. If e.ID is empty, the store assigns one.
	// CreatedAt and UpdatedAt are set by the store; caller-supplied
	// values are ignored. Returns the persisted entry with all fields
	// populated.
	Save(ctx context.Context, e Entry) (Entry, error)

	// Update replaces the content of an existing entry. The new entry
	// gets a fresh ID and supersedes the old one in projection; the
	// old ID becomes invalid for subsequent reads. Returns the new
	// entry. Returns ErrNotFound if id is unknown.
	Update(ctx context.Context, id string, content string) (Entry, error)

	// Remove tombstones an entry. Idempotent — removing an unknown id
	// returns nil (the store may log a warning).
	Remove(ctx context.Context, id string) error

	// List returns all live entries in CreatedAt order. tag == "" returns
	// all; tag != "" filters to entries whose Tags contains the tag.
	List(ctx context.Context, tag string) ([]Entry, error)

	// Get returns one entry by id. The bool is false (no error) when the
	// id is unknown or has been tombstoned.
	Get(ctx context.Context, id string) (Entry, bool, error)
}

// OriginKey is the context.Value key the MemoryTool reads to tag the
// Origin field of saved entries. runtime.Review sets this to "review";
// foreground calls inherit the default "agent". Callers that want a
// different label set the value before invoking the tool.
type contextKey struct{ name string }

var OriginKey = &contextKey{"memory.origin"}

// Sentinel errors returned by MemoryStore implementations.
var (
	// ErrNotFound is returned by Update and Get when the id is unknown
	// or has been tombstoned.
	ErrNotFound = errors.New("memory: entry not found")

	// ErrInvalidContent is returned by Save and Update when content
	// fails server-side validation (empty, exceeds size cap).
	ErrInvalidContent = errors.New("memory: invalid content")

	// ErrInvalidID is returned by Save when a caller-supplied ID fails
	// pattern validation.
	ErrInvalidID = errors.New("memory: invalid id")
)
