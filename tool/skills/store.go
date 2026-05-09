// Package skills defines the writable skills subsystem for harness:
// the SkillStore interface that any backend implements, the canonical
// Skill record type, and the OriginKey context value for write
// provenance.
//
// The SkillTool in tool.go wraps any SkillStore implementation as a
// tool.Tool the agent can call. The default disk-backed implementation
// lives in tool/skills/disk/ — that package's *Store value satisfies
// SkillStore directly, and satisfies runtime.SkillProvider via
// Store.AsSkillProvider() which returns a small wrapper that adapts
// the Get signature ((Skill, bool, error) → (string, bool)). One
// underlying *Store can therefore serve as the system-prompt index
// source, the foreground skill tool's backend, AND the review-pass
// tool's backend.
//
// Skill names are kebab-case and validated by ValidName: lowercase
// letters and digits, hyphen-separated, 1–64 chars, no leading hyphen,
// no slashes, no dots, no underscores, no double-underscore prefix
// (reserved for runtime use, e.g. "__review__").
package skills

import (
	"context"
	"errors"
	"regexp"
	"time"
)

// Skill is one durable procedural-knowledge record. Name is the stable
// identifier (and primary key); Body is the full SKILL.md content
// including any frontmatter. Description is taken from the
// caller-supplied field when set; otherwise the store derives it from
// the body's frontmatter `description:` line.
type Skill struct {
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Body        string    `json:"body,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Origin      string    `json:"origin,omitempty"`
}

// SkillStore is the writable backend the SkillTool dispatches against.
// Implementations should be safe for concurrent use within a single
// process. Cross-process safety is implementation-specific and not
// guaranteed by the default disk implementation.
//
// Implementations may also satisfy runtime.SkillProvider (FormatIndex
// + Get(string) (string, bool)) so the same instance powers both the
// system-prompt index and the write tool.
type SkillStore interface {
	// Create writes a new skill. Rejects with ErrAlreadyExists when a
	// skill of the same name already exists; rejects with ErrInvalidName
	// when s.Name fails ValidName. CreatedAt and UpdatedAt are set by
	// the store; caller-supplied values are ignored. Description is
	// parsed from frontmatter on the way in. Returns the persisted
	// skill with all fields populated.
	Create(ctx context.Context, s Skill) (Skill, error)

	// Patch replaces the first occurrence of oldContent with newContent
	// inside the existing skill's Body. Errors:
	//   - ErrNotFound       — no skill of that name
	//   - ErrPatchNoMatch   — oldContent not found in the body
	//   - ErrPatchAmbiguous — oldContent matched more than once
	//     (error wraps the count for the model to read)
	//   - ErrPatchIdentical — oldContent == newContent
	// On success returns the updated skill (UpdatedAt refreshed).
	Patch(ctx context.Context, name, oldContent, newContent string) (Skill, error)

	// Replace overwrites the entire body of an existing skill. Returns
	// ErrNotFound when name is unknown. Preserves CreatedAt; refreshes
	// UpdatedAt; re-derives Description from the new frontmatter.
	Replace(ctx context.Context, name, body string) (Skill, error)

	// Remove deletes a skill. Idempotent — removing an unknown name
	// returns nil (the store may log a warning).
	Remove(ctx context.Context, name string) error

	// List returns all skills in Name order.
	List(ctx context.Context) ([]Skill, error)

	// Get returns one skill by name. The bool is false (no error) when
	// the name is unknown.
	Get(ctx context.Context, name string) (Skill, bool, error)
}

// OriginKey is the context.Value key the SkillTool reads to tag the
// Origin field of created skills. runtime.Review sets this to "review";
// foreground calls inherit the default "agent". Callers that want a
// different label set the value before invoking the tool.
//
// Note: tool/memory defines its own memory.OriginKey. The two are
// independent context values; runtime.Review sets both when it forks
// a reviewer Runtime. Per-package keys keep tool/skills and tool/memory
// import-independent.
type contextKey struct{ name string }

var OriginKey = &contextKey{"skills.origin"}

// nameRE is the on-disk name pattern. Kebab-case, lowercase, 1–64 chars,
// no leading hyphen, no underscores, no slashes, no dots.
var nameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

// ValidName reports whether name is a legal skill name. Used by stores
// before any filesystem touch — keeps invalid names out of the store
// and out of error logs.
func ValidName(name string) bool {
	if !nameRE.MatchString(name) {
		return false
	}
	// Reject the "__"-prefix range (reserved for runtime-internal names
	// like "__review__"). The base regex already excludes underscores
	// entirely so this is currently belt-and-braces, but kept explicit
	// in case the regex relaxes later.
	if len(name) >= 2 && name[0] == '_' && name[1] == '_' {
		return false
	}
	return true
}

// Sentinel errors returned by SkillStore implementations.
var (
	// ErrNotFound is returned by Patch, Replace, and Get when the name
	// is unknown.
	ErrNotFound = errors.New("skills: skill not found")

	// ErrAlreadyExists is returned by Create when the name is taken.
	ErrAlreadyExists = errors.New("skills: skill already exists")

	// ErrInvalidName is returned by Create when the name fails ValidName.
	ErrInvalidName = errors.New("skills: invalid name")

	// ErrInvalidContent is returned by Create/Replace when the body
	// fails server-side validation (empty, exceeds size cap).
	ErrInvalidContent = errors.New("skills: invalid content")

	// ErrPatchNoMatch is returned by Patch when oldContent is not
	// present in the body.
	ErrPatchNoMatch = errors.New("skills: patch old_string not found")

	// ErrPatchAmbiguous is returned by Patch when oldContent matches
	// more than once. The error message includes the match count.
	ErrPatchAmbiguous = errors.New("skills: patch old_string matched multiple times")

	// ErrPatchIdentical is returned by Patch when oldContent == newContent
	// (no-op rejected as likely model mistake).
	ErrPatchIdentical = errors.New("skills: patch old_string and new_string are identical")
)
