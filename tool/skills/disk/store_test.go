package disk

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sausheong/harness/tool/skills"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStore_CreateWritesSkillMd(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	created, err := s.Create(context.Background(), skills.Skill{
		Name: "my-skill",
		Body: "---\ndescription: A test skill.\n---\n\nDo the thing.\n",
	})
	require.NoError(t, err)
	assert.Equal(t, "my-skill", created.Name)
	assert.Equal(t, "A test skill.", created.Description)
	assert.False(t, created.CreatedAt.IsZero())
	assert.Equal(t, created.CreatedAt, created.UpdatedAt)

	// File on disk has expected content.
	path := filepath.Join(dir, "my-skill", "SKILL.md")
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	body := string(data)
	assert.True(t, strings.HasPrefix(body, "---\n"), "should have frontmatter")
	assert.Contains(t, body, "description: A test skill.")
	assert.Contains(t, body, "Do the thing.")
}

func TestStore_CreateRejectsInvalidName(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{
		Name: "Bad Name With Spaces",
		Body: "anything",
	})
	require.ErrorIs(t, err, skills.ErrInvalidName)
}

func TestStore_CreateRejectsDuplicate(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{Name: "dup", Body: "first"})
	require.NoError(t, err)

	_, err = s.Create(context.Background(), skills.Skill{Name: "dup", Body: "second"})
	require.ErrorIs(t, err, skills.ErrAlreadyExists)
}

func TestStore_CreateRejectsPathTraversal(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	// Names with ".." or "/" are caught by ValidName (regex rejects them).
	// Confirm the store relays the rejection.
	for _, bad := range []string{"../escape", "sub/skill", ".."} {
		_, err := s.Create(context.Background(), skills.Skill{Name: bad, Body: "x"})
		require.ErrorIs(t, err, skills.ErrInvalidName, "name %q should be rejected", bad)
	}

	// And the parent dir contains nothing the store didn't put there.
	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	for _, e := range entries {
		assert.NotEqual(t, "..", e.Name())
	}
}

func TestStore_GetReturnsCreatedSkill(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	body := "---\ndescription: My test skill.\ntags: [a, b]\n---\n\nBody content.\n"
	created, err := s.Create(context.Background(), skills.Skill{Name: "round-trip", Body: body})
	require.NoError(t, err)

	got, ok, err := s.Get(context.Background(), "round-trip")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, "round-trip", got.Name)
	assert.Equal(t, "My test skill.", got.Description)
	assert.Equal(t, []string{"a", "b"}, got.Tags)
	assert.Contains(t, got.Body, "Body content.")
	assert.False(t, got.CreatedAt.IsZero())
	assert.Equal(t, created.CreatedAt.Truncate(time.Second), got.CreatedAt.Truncate(time.Second))
}

func TestStore_GetUnknownReturnsFalse(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	got, ok, err := s.Get(context.Background(), "does-not-exist")
	require.NoError(t, err)
	assert.False(t, ok)
	assert.Equal(t, skills.Skill{}, got)
}

func TestStore_GetRejectsInvalidName(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, _, err := s.Get(context.Background(), "../escape")
	require.ErrorIs(t, err, skills.ErrInvalidName)
}

func TestStore_GetReadsFrontmatterTagsAndOrigin(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{
		Name:   "tagged",
		Body:   "Body.",
		Tags:   []string{"workflow", "git"},
		Origin: "review",
	})
	require.NoError(t, err)

	got, ok, err := s.Get(context.Background(), "tagged")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, []string{"workflow", "git"}, got.Tags)
	assert.Equal(t, "review", got.Origin)
}

func TestStore_ListReturnsAllInNameOrder(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	for _, name := range []string{"charlie", "alpha", "bravo"} {
		_, err := s.Create(context.Background(), skills.Skill{
			Name: name,
			Body: "---\ndescription: " + name + " skill.\n---\n\nBody.\n",
		})
		require.NoError(t, err)
	}

	all, err := s.List(context.Background())
	require.NoError(t, err)
	require.Len(t, all, 3)
	assert.Equal(t, "alpha", all[0].Name)
	assert.Equal(t, "bravo", all[1].Name)
	assert.Equal(t, "charlie", all[2].Name)
	assert.Equal(t, "alpha skill.", all[0].Description)
}

func TestStore_ListEmptyRoot(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	all, err := s.List(context.Background())
	require.NoError(t, err)
	assert.Empty(t, all)
}

func TestStore_ListNonexistentRoot(t *testing.T) {
	// Pointing at a non-existent root should yield empty, not error.
	s := NewStore("/tmp/does-not-exist-skills-xyz-" + t.Name())
	all, err := s.List(context.Background())
	require.NoError(t, err)
	assert.Empty(t, all)
}

func TestStore_ListSkipsBogusDirs(t *testing.T) {
	// A directory in root that doesn't contain SKILL.md, or whose name
	// fails ValidName, must be silently skipped (it's user data we
	// don't own).
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{Name: "real", Body: "ok"})
	require.NoError(t, err)

	// Empty subdir.
	require.NoError(t, os.Mkdir(filepath.Join(dir, "empty"), 0o755))

	// Subdir with bogus name (uppercase) but containing SKILL.md.
	require.NoError(t, os.Mkdir(filepath.Join(dir, "BAD"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "BAD", "SKILL.md"), []byte("x"), 0o644))

	// File at root level (not a directory).
	require.NoError(t, os.WriteFile(filepath.Join(dir, "stray.txt"), []byte("x"), 0o644))

	all, err := s.List(context.Background())
	require.NoError(t, err)
	require.Len(t, all, 1)
	assert.Equal(t, "real", all[0].Name)
}

func TestStore_ReplaceOverwritesBody(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	original, err := s.Create(context.Background(), skills.Skill{
		Name: "evolving",
		Body: "---\ndescription: v1.\n---\n\nv1 body.\n",
	})
	require.NoError(t, err)
	originalCreated := original.CreatedAt

	// Sleep 1s so mtime tick is observable.
	time.Sleep(1100 * time.Millisecond)

	updated, err := s.Replace(context.Background(), "evolving",
		"---\ndescription: v2.\n---\n\nv2 body.\n")
	require.NoError(t, err)

	assert.Equal(t, "evolving", updated.Name)
	assert.Equal(t, "v2.", updated.Description)
	assert.Contains(t, updated.Body, "v2 body.")
	assert.NotContains(t, updated.Body, "v1 body.")
	assert.True(t, originalCreated.Equal(updated.CreatedAt) || updated.CreatedAt.IsZero(),
		"CreatedAt should survive (got %v, want %v)", updated.CreatedAt, originalCreated)
	assert.True(t, updated.UpdatedAt.After(originalCreated),
		"UpdatedAt must advance past original CreatedAt")
}

func TestStore_ReplaceRejectsUnknown(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Replace(context.Background(), "ghost", "x")
	require.ErrorIs(t, err, skills.ErrNotFound)
}

func TestStore_ReplaceRejectsInvalidName(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Replace(context.Background(), "Bad Name", "x")
	require.ErrorIs(t, err, skills.ErrInvalidName)
}

func TestStore_ReplacePreservesCreatedAtAcrossDays(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	original, err := s.Create(context.Background(), skills.Skill{
		Name: "stable", Body: "---\ndescription: x.\n---\n\nbody",
	})
	require.NoError(t, err)

	_, err = s.Replace(context.Background(), "stable", "---\ndescription: y.\n---\n\nbody")
	require.NoError(t, err)

	got, ok, err := s.Get(context.Background(), "stable")
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t,
		original.CreatedAt.Truncate(time.Second),
		got.CreatedAt.Truncate(time.Second),
		"CreatedAt must survive Replace",
	)
}

func TestStore_RemoveDeletesSkill(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{Name: "doomed", Body: "x"})
	require.NoError(t, err)

	require.NoError(t, s.Remove(context.Background(), "doomed"))

	_, ok, err := s.Get(context.Background(), "doomed")
	require.NoError(t, err)
	assert.False(t, ok)

	// Directory is gone.
	_, err = os.Stat(filepath.Join(dir, "doomed"))
	assert.True(t, os.IsNotExist(err))
}

func TestStore_RemoveUnknownIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	require.NoError(t, s.Remove(context.Background(), "ghost"))
	require.NoError(t, s.Remove(context.Background(), "ghost"))
}

func TestStore_RemoveRejectsInvalidName(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	err := s.Remove(context.Background(), "../escape")
	require.ErrorIs(t, err, skills.ErrInvalidName)
}

func TestStore_PatchReplacesUniqueOccurrence(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{
		Name: "patchable",
		Body: "---\ndescription: x.\n---\n\nThe quick brown fox.\n",
	})
	require.NoError(t, err)

	updated, err := s.Patch(context.Background(), "patchable", "brown fox", "lazy dog")
	require.NoError(t, err)
	assert.Contains(t, updated.Body, "The quick lazy dog.")
	assert.NotContains(t, updated.Body, "brown fox")
}

func TestStore_PatchNoMatchReturnsErrPatchNoMatch(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Create(context.Background(), skills.Skill{Name: "p1", Body: "Hello world."})
	require.NoError(t, err)

	_, err = s.Patch(context.Background(), "p1", "missing", "x")
	require.ErrorIs(t, err, skills.ErrPatchNoMatch)
}

func TestStore_PatchMultipleMatchesReturnsAmbiguous(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Create(context.Background(), skills.Skill{Name: "p2", Body: "foo bar foo bar foo"})
	require.NoError(t, err)

	_, err = s.Patch(context.Background(), "p2", "foo", "x")
	require.ErrorIs(t, err, skills.ErrPatchAmbiguous)
	assert.Contains(t, err.Error(), "3", "error message should include match count")
}

func TestStore_PatchIdenticalReturnsErrPatchIdentical(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Create(context.Background(), skills.Skill{Name: "p3", Body: "hello"})
	require.NoError(t, err)

	_, err = s.Patch(context.Background(), "p3", "hello", "hello")
	require.ErrorIs(t, err, skills.ErrPatchIdentical)
}

func TestStore_PatchUnknownReturnsErrNotFound(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Patch(context.Background(), "ghost", "a", "b")
	require.ErrorIs(t, err, skills.ErrNotFound)
}

func TestStore_PatchUnknownTakesPrecedenceOverIdentical(t *testing.T) {
	// Patch on a missing skill with old==new must return ErrNotFound,
	// not ErrPatchIdentical — so the model gets the actionable error.
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Patch(context.Background(), "ghost", "x", "x")
	require.ErrorIs(t, err, skills.ErrNotFound)
	require.NotErrorIs(t, err, skills.ErrPatchIdentical)
}

func TestStore_PatchRejectsInvalidName(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	_, err := s.Patch(context.Background(), "Bad Name", "a", "b")
	require.ErrorIs(t, err, skills.ErrInvalidName)
}

func TestStore_PatchPreservesCreatedAtRefreshesUpdatedAt(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	original, err := s.Create(context.Background(), skills.Skill{
		Name: "p4",
		Body: "---\ndescription: orig.\n---\n\nfind me.\n",
	})
	require.NoError(t, err)

	time.Sleep(1100 * time.Millisecond)

	updated, err := s.Patch(context.Background(), "p4", "find me", "FOUND")
	require.NoError(t, err)

	assert.Equal(t,
		original.CreatedAt.Truncate(time.Second),
		updated.CreatedAt.Truncate(time.Second),
		"CreatedAt must survive Patch")
	assert.True(t, updated.UpdatedAt.After(original.CreatedAt),
		"UpdatedAt must advance")
}

func TestStore_ConcurrentPatchesDoNotInterleave(t *testing.T) {
	// 20 goroutines each Patch a unique substring within the same skill.
	// Without per-skill serialization, two concurrent read-modify-write
	// cycles would race and lose updates. With it, every patch must
	// land.
	dir := t.TempDir()
	s := NewStore(dir)

	// Body has 20 distinct sentinels: M00, M01, ..., M19.
	var b strings.Builder
	b.WriteString("---\ndescription: race test.\n---\n\n")
	for i := range 20 {
		fmt.Fprintf(&b, "M%02d\n", i)
	}
	_, err := s.Create(context.Background(), skills.Skill{Name: "race", Body: b.String()})
	require.NoError(t, err)

	var wg sync.WaitGroup
	wg.Add(20)
	errs := make(chan error, 20)
	for i := range 20 {
		go func(i int) {
			defer wg.Done()
			old := fmt.Sprintf("M%02d", i)
			replacement := fmt.Sprintf("X%02d", i)
			if _, err := s.Patch(context.Background(), "race", old, replacement); err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("patch error: %v", err)
	}

	got, ok, err := s.Get(context.Background(), "race")
	require.NoError(t, err)
	require.True(t, ok)
	for i := range 20 {
		assert.NotContains(t, got.Body, fmt.Sprintf("M%02d", i),
			"sentinel M%02d should have been patched", i)
		assert.Contains(t, got.Body, fmt.Sprintf("X%02d", i),
			"replacement X%02d should be present", i)
	}
}

func TestStore_ConcurrentCreatesDoNotDoubleWrite(t *testing.T) {
	// 10 goroutines try to Create the same name. Exactly one should
	// succeed; the rest get ErrAlreadyExists.
	dir := t.TempDir()
	s := NewStore(dir)

	var wg sync.WaitGroup
	wg.Add(10)
	successes := make(chan struct{}, 10)
	conflicts := make(chan struct{}, 10)
	for range 10 {
		go func() {
			defer wg.Done()
			_, err := s.Create(context.Background(), skills.Skill{
				Name: "race-create", Body: "x",
			})
			switch {
			case err == nil:
				successes <- struct{}{}
			case errors.Is(err, skills.ErrAlreadyExists):
				conflicts <- struct{}{}
			default:
				t.Errorf("unexpected error: %v", err)
			}
		}()
	}
	wg.Wait()
	close(successes)
	close(conflicts)
	assert.Len(t, drain(successes), 1, "exactly one Create should succeed")
	assert.Len(t, drain(conflicts), 9, "the other nine should see ErrAlreadyExists")
}

func drain(ch <-chan struct{}) []struct{} {
	out := []struct{}{}
	for v := range ch {
		out = append(out, v)
	}
	return out
}

func TestStore_AsSkillProvider_FormatIndexEmpty(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	provider := s.AsSkillProvider()
	assert.Empty(t, provider.FormatIndex())
}

func TestStore_AsSkillProvider_FormatIndexLists(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{
		Name: "writing-clearly",
		Body: "---\ndescription: Tighten prose.\n---\n\nbody",
	})
	require.NoError(t, err)
	_, err = s.Create(context.Background(), skills.Skill{
		Name: "git-bisect",
		Body: "---\ndescription: Find the bad commit.\n---\n\nbody",
	})
	require.NoError(t, err)

	provider := s.AsSkillProvider()
	idx := provider.FormatIndex()

	assert.Contains(t, idx, "## Skills")
	assert.Contains(t, idx, "git-bisect")
	assert.Contains(t, idx, "Find the bad commit.")
	assert.Contains(t, idx, "writing-clearly")
	assert.Contains(t, idx, "Tighten prose.")
	// Body content must not leak — only one-liners.
	assert.NotContains(t, idx, "\nbody")
}

func TestStore_AsSkillProvider_GetReturnsBody(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	_, err := s.Create(context.Background(), skills.Skill{
		Name: "lookup",
		Body: "---\ndescription: x.\n---\n\nThe full body.\n",
	})
	require.NoError(t, err)

	provider := s.AsSkillProvider()
	body, ok := provider.Get("lookup")
	assert.True(t, ok)
	assert.Contains(t, body, "The full body.")

	_, ok = provider.Get("does-not-exist")
	assert.False(t, ok)
}
