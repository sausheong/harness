package skills

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSkillTool_NameDescriptionSchema(t *testing.T) {
	tool := &SkillTool{}

	assert.Equal(t, "skill_manage", tool.Name())
	assert.NotEmpty(t, tool.Description())

	var schema struct {
		Type       string                 `json:"type"`
		Properties map[string]interface{} `json:"properties"`
		Required   []string               `json:"required"`
	}
	require.NoError(t, json.Unmarshal(tool.Parameters(), &schema))
	assert.Equal(t, "object", schema.Type)
	for _, key := range []string{"action", "name", "body", "old_string", "new_string"} {
		assert.Contains(t, schema.Properties, key, "missing %q in schema", key)
	}
	assert.Equal(t, []string{"action"}, schema.Required)
}

func TestSkillTool_IsConcurrencySafeFalse(t *testing.T) {
	tool := &SkillTool{}
	assert.False(t, tool.IsConcurrencySafe(nil))
}

// fakeStore is an in-memory SkillStore for tool tests.
type fakeStore struct {
	skills map[string]Skill
}

func newFakeStore() *fakeStore { return &fakeStore{skills: map[string]Skill{}} }

func (f *fakeStore) Create(_ context.Context, s Skill) (Skill, error) {
	if !ValidName(s.Name) {
		return Skill{}, ErrInvalidName
	}
	if _, ok := f.skills[s.Name]; ok {
		return Skill{}, ErrAlreadyExists
	}
	now := time.Now()
	s.CreatedAt = now
	s.UpdatedAt = now
	f.skills[s.Name] = s
	return s, nil
}

func (f *fakeStore) Patch(_ context.Context, name, oldC, newC string) (Skill, error) {
	if oldC == newC {
		return Skill{}, ErrPatchIdentical
	}
	s, ok := f.skills[name]
	if !ok {
		return Skill{}, ErrNotFound
	}
	if !contains(s.Body, oldC) {
		return Skill{}, ErrPatchNoMatch
	}
	s.Body = replaceFirst(s.Body, oldC, newC)
	s.UpdatedAt = time.Now()
	f.skills[name] = s
	return s, nil
}

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

func replaceFirst(haystack, old, new string) string {
	for i := 0; i+len(old) <= len(haystack); i++ {
		if haystack[i:i+len(old)] == old {
			return haystack[:i] + new + haystack[i+len(old):]
		}
	}
	return haystack
}

func (f *fakeStore) Replace(_ context.Context, name, body string) (Skill, error) {
	s, ok := f.skills[name]
	if !ok {
		return Skill{}, ErrNotFound
	}
	s.Body = body
	s.UpdatedAt = time.Now()
	f.skills[name] = s
	return s, nil
}

func (f *fakeStore) Remove(_ context.Context, name string) error {
	delete(f.skills, name)
	return nil
}

func (f *fakeStore) List(_ context.Context) ([]Skill, error) {
	out := make([]Skill, 0, len(f.skills))
	for _, s := range f.skills {
		out = append(out, s)
	}
	return out, nil
}

func (f *fakeStore) Get(_ context.Context, name string) (Skill, bool, error) {
	s, ok := f.skills[name]
	return s, ok, nil
}

func TestSkillTool_CreateHappyPath(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{
		Action: "create",
		Name:   "my-skill",
		Body:   "---\ndescription: A skill.\n---\n\nbody",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, "skill", ar.Target)
	assert.Equal(t, "my-skill", ar.Name)
	assert.Contains(t, ar.Message, "my-skill")

	require.Len(t, store.skills, 1)
}

func TestSkillTool_CreateRejectsMissingName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "create", Body: "x"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "name")
}

func TestSkillTool_CreateRejectsInvalidName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "create", Name: "Bad Name", Body: "x"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "invalid")
}

func TestSkillTool_CreateRejectsEmptyBody(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "create", Name: "x", Body: ""})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "body")
}

func TestSkillTool_CreateRejectsOversizedBody(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	big := make([]byte, 32001)
	for i := range big {
		big[i] = 'x'
	}
	in, _ := json.Marshal(skillInput{Action: "create", Name: "big", Body: string(big)})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "32000")
}

func TestSkillTool_CreateOriginFromContext(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	ctx := context.WithValue(context.Background(), OriginKey, "review")
	in, _ := json.Marshal(skillInput{Action: "create", Name: "from-review", Body: "x"})
	_, err := tool.Execute(ctx, in)
	require.NoError(t, err)

	require.Len(t, store.skills, 1)
	assert.Equal(t, "review", store.skills["from-review"].Origin)
}

func TestSkillTool_CreateOriginDefault(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "create", Name: "from-fg", Body: "x"})
	_, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	require.Len(t, store.skills, 1)
	assert.Equal(t, "agent", store.skills["from-fg"].Origin)
}

func TestSkillTool_CreateAlreadyExistsRetainsOriginal(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{Action: "create", Name: "dup", Body: "first"})
	_, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	in, _ = json.Marshal(skillInput{Action: "create", Name: "dup", Body: "second"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "exists")

	// Original still intact.
	assert.Equal(t, "first", store.skills["dup"].Body)
}

func TestSkillTool_PatchHappyPath(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "p", Body: "find me here"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{
		Action: "patch", Name: "p", OldString: "find me", NewString: "FOUND",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, "p", ar.Name)
	assert.Contains(t, ar.Message, "patched")
	assert.Equal(t, "FOUND here", store.skills["p"].Body)
}

func TestSkillTool_PatchRejectsMissingName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "patch", OldString: "a", NewString: "b"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "name")
}

func TestSkillTool_PatchRejectsMissingOldString(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "p", Body: "x"})
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "patch", Name: "p", NewString: "b"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "old_string")
}

func TestSkillTool_PatchSurfacesNoMatchError(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "p", Body: "hello world"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{
		Action: "patch", Name: "p", OldString: "missing", NewString: "x",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestSkillTool_PatchSurfacesIdenticalError(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "p", Body: "x"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{
		Action: "patch", Name: "p", OldString: "same", NewString: "same",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "identical")
}

func TestSkillTool_PatchUnknownNameReturnsErrNotFound(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{
		Action: "patch", Name: "ghost", OldString: "a", NewString: "b",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestSkillTool_ReplaceHappyPath(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "r", Body: "v1"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{Action: "replace", Name: "r", Body: "v2"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Contains(t, ar.Message, "replaced")
	assert.Equal(t, "v2", store.skills["r"].Body)
}

func TestSkillTool_ReplaceRejectsMissingName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "replace", Body: "v2"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "name")
}

func TestSkillTool_ReplaceRejectsEmptyBody(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "r", Body: "v1"})
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "replace", Name: "r", Body: ""})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
}

func TestSkillTool_ReplaceRejectsOversizedBody(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "r", Body: "v1"})
	tool := &SkillTool{Store: store}
	big := make([]byte, 32001)
	for i := range big {
		big[i] = 'x'
	}
	in, _ := json.Marshal(skillInput{Action: "replace", Name: "r", Body: string(big)})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "32000")
}

func TestSkillTool_ReplaceUnknownReturnsErrNotFound(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "replace", Name: "ghost", Body: "x"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestSkillTool_RemoveHappyPath(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "doomed", Body: "x"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{Action: "remove", Name: "doomed"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
	assert.Equal(t, "doomed", ar.Name)
	assert.Contains(t, ar.Message, "removed")
	assert.Empty(t, store.skills)
}

func TestSkillTool_RemoveRejectsMissingName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "remove"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "name")
}

func TestSkillTool_RemoveUnknownIsSuccessful(t *testing.T) {
	// Idempotent.
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "remove", Name: "ghost"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)
}

func TestSkillTool_ListReturnsAllAsJSON(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "alpha", Body: "alpha body", Description: "first"})
	_, _ = store.Create(context.Background(), Skill{Name: "bravo", Body: "bravo body", Description: "second"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{Action: "list"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	assert.Empty(t, res.Error)

	// List output is the raw skills array (without bodies — this is an
	// index, not a content dump).
	var entries []Skill
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entries))
	require.Len(t, entries, 2)
	for _, e := range entries {
		assert.Empty(t, e.Body, "list output must omit Body to keep the index small")
	}
}

func TestSkillTool_ListEmpty(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "list"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	assert.Equal(t, "[]", res.Output)
}

func TestSkillTool_GetHappyPath(t *testing.T) {
	store := newFakeStore()
	_, _ = store.Create(context.Background(), Skill{Name: "g", Body: "the body"})
	tool := &SkillTool{Store: store}

	in, _ := json.Marshal(skillInput{Action: "get", Name: "g"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var sk Skill
	require.NoError(t, json.Unmarshal([]byte(res.Output), &sk))
	assert.Equal(t, "g", sk.Name)
	assert.Equal(t, "the body", sk.Body)
}

func TestSkillTool_GetUnknownReturnsError(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "get", Name: "ghost"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "not found")
}

func TestSkillTool_GetRejectsMissingName(t *testing.T) {
	store := newFakeStore()
	tool := &SkillTool{Store: store}
	in, _ := json.Marshal(skillInput{Action: "get"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var ar actionResult
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.False(t, ar.Success)
	assert.Contains(t, ar.Error, "name")
}
