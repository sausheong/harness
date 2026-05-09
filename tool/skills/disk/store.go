// Package disk is the default disk-backed SkillStore implementation.
// It uses one directory per skill: <root>/<name>/SKILL.md holds the
// full body, including a YAML-ish frontmatter block at the top with
// description, tags, origin, and created_at. The directory shape leaves
// room for future supporting files (references/, templates/, scripts/)
// without changing the on-disk schema.
//
// Concurrent writes within one process are protected by a per-skill
// mutex (Task 8). Cross-process safety is best-effort: Create/Replace
// write to a temp file then atomically rename over SKILL.md, but two
// processes racing on the same skill name may still produce
// last-writer-wins behavior. Single-process per store is the
// assumption.
package disk

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sausheong/harness/tool/skills"
)

// Store is a directory-on-disk SkillStore.
type Store struct {
	root string

	mu    sync.Mutex             // guards locks map only
	locks map[string]*sync.Mutex // skill name → per-skill lock
}

// NewStore returns a Store rooted at dir. The directory is created
// lazily on first write; reads of a non-existent root return an empty
// result. Callers should pass an absolute path.
func NewStore(root string) *Store {
	return &Store{root: root}
}

// lockFor returns the per-skill mutex, allocating it on first use.
// Callers must call mu.Lock() before any read-modify-write on the
// skill's directory.
func (s *Store) lockFor(name string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.locks == nil {
		s.locks = make(map[string]*sync.Mutex)
	}
	if l, ok := s.locks[name]; ok {
		return l
	}
	l := &sync.Mutex{}
	s.locks[name] = l
	return l
}

// skillDir returns the per-skill directory path. Caller must have
// validated name via skills.ValidName first.
func (s *Store) skillDir(name string) string {
	return filepath.Join(s.root, name)
}

// skillFile returns the SKILL.md path for the given name.
func (s *Store) skillFile(name string) string {
	return filepath.Join(s.skillDir(name), "SKILL.md")
}

// writeBodyAtomic writes body to the skill's SKILL.md via temp+rename,
// creating the parent directory if needed. Caller holds the per-skill
// lock.
func (s *Store) writeBodyAtomic(name, body string) error {
	dir := s.skillDir(name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".SKILL.md.*")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	if _, err := tmp.WriteString(body); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return fmt.Errorf("write: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return fmt.Errorf("fsync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return fmt.Errorf("close: %w", err)
	}
	if err := os.Rename(tmp.Name(), s.skillFile(name)); err != nil {
		os.Remove(tmp.Name())
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

// composeBody assembles a SKILL.md body from a Skill struct: emits a
// frontmatter block with description, tags, origin, created_at, then
// the body sans any pre-existing frontmatter.
func composeBody(s skills.Skill, body string) string {
	var b strings.Builder
	b.WriteString("---\n")
	if s.Description != "" {
		b.WriteString("description: ")
		b.WriteString(s.Description)
		b.WriteString("\n")
	}
	if len(s.Tags) > 0 {
		b.WriteString("tags: [")
		b.WriteString(strings.Join(s.Tags, ", "))
		b.WriteString("]\n")
	}
	if s.Origin != "" {
		b.WriteString("origin: ")
		b.WriteString(s.Origin)
		b.WriteString("\n")
	}
	if !s.CreatedAt.IsZero() {
		b.WriteString("created_at: ")
		b.WriteString(s.CreatedAt.UTC().Format(time.RFC3339Nano))
		b.WriteString("\n")
	}
	b.WriteString("---\n\n")
	// Strip any leading frontmatter from the caller-supplied body so we
	// don't double-emit. parseFrontmatter (added in Task 3) returns the
	// body without the FM block; for now the simple "if it starts with
	// ---\n, drop until the next ---\n" trick suffices.
	body = stripFrontmatter(body)
	b.WriteString(body)
	return b.String()
}

// stripFrontmatter removes a leading YAML-ish frontmatter block from
// body if present. Conservative: only strips when body starts with
// "---\n" and a closing "\n---\n" or "\n---" is found.
func stripFrontmatter(body string) string {
	if !strings.HasPrefix(body, "---\n") {
		return body
	}
	rest := body[4:]
	if _, after, ok := strings.Cut(rest, "\n---\n"); ok {
		// Eat one optional leading blank line after the closing fence.
		return strings.TrimPrefix(after, "\n")
	}
	if i := strings.Index(rest, "\n---"); i >= 0 && i == len(rest)-4 {
		return ""
	}
	return body
}

// Create implements skills.SkillStore.
func (s *Store) Create(_ context.Context, sk skills.Skill) (skills.Skill, error) {
	if !skills.ValidName(sk.Name) {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrInvalidName, sk.Name)
	}

	lock := s.lockFor(sk.Name)
	lock.Lock()
	defer lock.Unlock()

	// Reject duplicates.
	if _, err := os.Stat(s.skillFile(sk.Name)); err == nil {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrAlreadyExists, sk.Name)
	} else if !errors.Is(err, os.ErrNotExist) {
		return skills.Skill{}, fmt.Errorf("stat: %w", err)
	}

	// Derive metadata from body frontmatter when caller didn't set it.
	if sk.Description == "" || len(sk.Tags) == 0 || sk.Origin == "" {
		fm, _ := parseFrontmatter(sk.Body)
		if sk.Description == "" {
			sk.Description = fm.Description
		}
		if len(sk.Tags) == 0 {
			sk.Tags = fm.Tags
		}
		if sk.Origin == "" {
			sk.Origin = fm.Origin
		}
	}

	now := time.Now().UTC()
	sk.CreatedAt = now
	sk.UpdatedAt = now

	if err := s.writeBodyAtomic(sk.Name, composeBody(sk, sk.Body)); err != nil {
		return skills.Skill{}, err
	}
	return sk, nil
}

// frontmatter is the parsed shape of the YAML-ish header block.
type frontmatter struct {
	Description string
	Tags        []string
	Origin      string
	CreatedAt   time.Time
}

// parseFrontmatter extracts the leading frontmatter block from body.
// Returns the parsed metadata and the body sans frontmatter. If body
// does not begin with "---\n" or the closing fence is missing, returns
// (zero, body) untouched.
//
// Recognized keys: description, tags, origin, created_at. Unknown keys
// are silently ignored. Tags parse as either "[a, b, c]" or a single
// scalar treated as one tag.
func parseFrontmatter(body string) (frontmatter, string) {
	if !strings.HasPrefix(body, "---\n") {
		return frontmatter{}, body
	}
	rest := body[4:]
	var fmText, afterFM string
	if before, after, ok := strings.Cut(rest, "\n---\n"); ok {
		fmText = before
		afterFM = after
	} else if strings.HasSuffix(rest, "\n---") {
		fmText = rest[:len(rest)-4]
		afterFM = ""
	} else {
		return frontmatter{}, body
	}
	afterFM = strings.TrimPrefix(afterFM, "\n")

	var fm frontmatter
	for line := range strings.SplitSeq(fmText, "\n") {
		line = strings.TrimRight(line, "\r")
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "description":
			fm.Description = val
		case "origin":
			fm.Origin = val
		case "created_at":
			if t, err := time.Parse(time.RFC3339Nano, val); err == nil {
				fm.CreatedAt = t
			}
		case "tags":
			fm.Tags = parseTags(val)
		}
	}
	return fm, afterFM
}

// parseTags accepts "[a, b, c]" (returns ["a","b","c"]) or "a"
// (returns ["a"]). Empty input returns nil.
func parseTags(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		s = s[1 : len(s)-1]
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// Patch implements skills.SkillStore. Replaces the unique occurrence
// of oldContent with newContent. Returns one of three sentinel errors
// when the patch can't apply cleanly so the model can recover:
//
//   - ErrPatchNoMatch:   re-read with Get and try again
//   - ErrPatchAmbiguous: include more surrounding context in oldContent
//   - ErrPatchIdentical: no change requested
//
// Error message includes the match count for the ambiguous case.
func (s *Store) Patch(ctx context.Context, name, oldContent, newContent string) (skills.Skill, error) {
	if !skills.ValidName(name) {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrInvalidName, name)
	}

	lock := s.lockFor(name)
	lock.Lock()
	defer lock.Unlock()

	existing, ok, err := s.Get(ctx, name)
	if err != nil {
		return skills.Skill{}, err
	}
	if !ok {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrNotFound, name)
	}
	if oldContent == newContent {
		return skills.Skill{}, skills.ErrPatchIdentical
	}

	count := strings.Count(existing.Body, oldContent)
	switch count {
	case 0:
		return skills.Skill{}, skills.ErrPatchNoMatch
	case 1:
		// happy path
	default:
		return skills.Skill{}, fmt.Errorf("%w (matched %d times)", skills.ErrPatchAmbiguous, count)
	}

	newBody := strings.Replace(existing.Body, oldContent, newContent, 1)
	if err := s.writeBodyAtomic(name, newBody); err != nil {
		return skills.Skill{}, err
	}

	got, ok, err := s.Get(ctx, name)
	if err != nil || !ok {
		return skills.Skill{}, fmt.Errorf("post-write read: %w", err)
	}
	return got, nil
}
// Replace implements skills.SkillStore.
func (s *Store) Replace(ctx context.Context, name, body string) (skills.Skill, error) {
	if !skills.ValidName(name) {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrInvalidName, name)
	}

	lock := s.lockFor(name)
	lock.Lock()
	defer lock.Unlock()

	// Confirm existence and read existing CreatedAt.
	existing, ok, err := s.Get(ctx, name)
	if err != nil {
		return skills.Skill{}, err
	}
	if !ok {
		return skills.Skill{}, fmt.Errorf("%w: %q", skills.ErrNotFound, name)
	}

	// Compose with preserved CreatedAt; refresh UpdatedAt to now.
	sk := skills.Skill{
		Name:      name,
		Body:      body,
		Origin:    existing.Origin, // preserved across Replace; caller can include `origin:` in new frontmatter to override
		CreatedAt: existing.CreatedAt,
	}
	fm, _ := parseFrontmatter(body)
	if fm.Description != "" {
		sk.Description = fm.Description
	}
	if fm.Origin != "" {
		sk.Origin = fm.Origin
	}
	if len(fm.Tags) > 0 {
		sk.Tags = fm.Tags
	} else {
		sk.Tags = existing.Tags
	}

	if err := s.writeBodyAtomic(name, composeBody(sk, body)); err != nil {
		return skills.Skill{}, err
	}

	// Re-read to get true mtime. CreatedAt round-trips with full
	// nanosecond precision via RFC3339Nano in compose+parse.
	got, ok, err := s.Get(ctx, name)
	if err != nil || !ok {
		return skills.Skill{}, fmt.Errorf("post-write read: %w", err)
	}
	return got, nil
}
// Remove implements skills.SkillStore. Idempotent — removing an unknown
// name returns nil. Removes the entire per-skill directory (so any
// supporting files added in future, like references/, are cleaned up
// too).
func (s *Store) Remove(_ context.Context, name string) error {
	if !skills.ValidName(name) {
		return fmt.Errorf("%w: %q", skills.ErrInvalidName, name)
	}

	lock := s.lockFor(name)
	lock.Lock()
	defer lock.Unlock()

	if err := os.RemoveAll(s.skillDir(name)); err != nil {
		return fmt.Errorf("remove: %w", err)
	}
	return nil
}

// List implements skills.SkillStore.
func (s *Store) List(ctx context.Context) ([]skills.Skill, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []skills.Skill{}, nil
		}
		return nil, fmt.Errorf("readdir: %w", err)
	}
	out := make([]skills.Skill, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !skills.ValidName(name) {
			continue
		}
		// Confirm SKILL.md exists; otherwise skip.
		if _, err := os.Stat(filepath.Join(s.root, name, "SKILL.md")); err != nil {
			continue
		}
		sk, ok, err := s.Get(ctx, name)
		if err != nil || !ok {
			continue
		}
		out = append(out, sk)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Get implements skills.SkillStore.
func (s *Store) Get(_ context.Context, name string) (skills.Skill, bool, error) {
	if !skills.ValidName(name) {
		return skills.Skill{}, false, fmt.Errorf("%w: %q", skills.ErrInvalidName, name)
	}
	data, err := os.ReadFile(s.skillFile(name))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return skills.Skill{}, false, nil
		}
		return skills.Skill{}, false, fmt.Errorf("read: %w", err)
	}
	stat, err := os.Stat(s.skillFile(name))
	if err != nil {
		return skills.Skill{}, false, fmt.Errorf("stat: %w", err)
	}
	body := string(data)
	fm, _ := parseFrontmatter(body)
	return skills.Skill{
		Name:        name,
		Description: fm.Description,
		Body:        body, // full body including frontmatter; callers see what's on disk
		Tags:        fm.Tags,
		Origin:      fm.Origin,
		CreatedAt:   fm.CreatedAt,
		UpdatedAt:   stat.ModTime().UTC(),
	}, true, nil
}

// SkillProviderAdapter wraps a *Store with the read-only signature
// runtime.SkillProvider expects (FormatIndex + Get(string)(string,bool)).
// Lets one *Store value serve both writes (via the SkillStore interface)
// and reads (via runtime.SkillProvider).
type SkillProviderAdapter struct{ s *Store }

// AsSkillProvider returns a wrapper that satisfies
// runtime.SkillProvider — pass it as RuntimeInputs.Skills while still
// using the same *Store as the SkillTool's backend.
func (s *Store) AsSkillProvider() *SkillProviderAdapter {
	return &SkillProviderAdapter{s: s}
}

// FormatIndex satisfies runtime.SkillProvider. Returns a markdown block
// for inclusion in the static system prompt; empty when the store has
// no skills. Format: one bullet per skill, "name: description". Bodies
// are NOT included — that's what load_skill is for.
func (a *SkillProviderAdapter) FormatIndex() string {
	all, err := a.s.List(context.Background())
	if err != nil || len(all) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("## Skills\n\n")
	for _, sk := range all {
		b.WriteString("- ")
		b.WriteString(sk.Name)
		if sk.Description != "" {
			b.WriteString(": ")
			b.WriteString(sk.Description)
		}
		b.WriteString("\n")
	}
	return b.String()
}

// Get satisfies runtime.SkillProvider. Returns the full SKILL.md body
// (frontmatter included).
func (a *SkillProviderAdapter) Get(name string) (string, bool) {
	sk, ok, err := a.s.Get(context.Background(), name)
	if err != nil || !ok {
		return "", false
	}
	return sk.Body, true
}
