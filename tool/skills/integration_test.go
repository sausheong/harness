package skills_test

// Separate _test package to verify the public surface wires together
// without internal fakes.

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sausheong/harness/tool/skills"
	"github.com/sausheong/harness/tool/skills/disk"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEndToEnd_CreatePatchReplaceRemoveRoundTrip(t *testing.T) {
	dir := t.TempDir()
	store := disk.NewStore(filepath.Join(dir, "skills"))
	tool := &skills.SkillTool{Store: store}

	// Create two skills.
	for _, name := range []string{"writing-clearly", "git-bisect"} {
		body := "---\ndescription: " + name + " skill.\n---\n\nThe " + name + " body.\n"
		in, _ := json.Marshal(map[string]any{
			"action": "create",
			"name":   name,
			"body":   body,
		})
		res, err := tool.Execute(context.Background(), in)
		require.NoError(t, err)
		assert.Empty(t, res.Error)
	}

	// List both back (index-only — bodies stripped).
	in, _ := json.Marshal(map[string]any{"action": "list"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	var entries []skills.Skill
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entries))
	require.Len(t, entries, 2)
	for _, e := range entries {
		assert.Empty(t, e.Body, "list output must strip body")
		assert.NotEmpty(t, e.Description, "list output must include description")
	}

	// Patch one.
	in, _ = json.Marshal(map[string]any{
		"action":     "patch",
		"name":       "writing-clearly",
		"old_string": "writing-clearly body.",
		"new_string": "writing-clearly body, revised.",
	})
	res, err = tool.Execute(context.Background(), in)
	require.NoError(t, err)
	var ar struct {
		Success bool `json:"success"`
	}
	require.NoError(t, json.Unmarshal([]byte(res.Output), &ar))
	assert.True(t, ar.Success)

	// Verify via the provider adapter (the system-prompt path).
	provider := store.AsSkillProvider()
	body, ok := provider.Get("writing-clearly")
	assert.True(t, ok)
	assert.Contains(t, body, "writing-clearly body, revised.")

	// FormatIndex includes both skills with descriptions, no body content.
	idx := provider.FormatIndex()
	assert.Contains(t, idx, "writing-clearly")
	assert.Contains(t, idx, "git-bisect")
	assert.Contains(t, idx, "writing-clearly skill.")
	assert.NotContains(t, idx, "writing-clearly body, revised.",
		"FormatIndex must not leak body content")

	// Replace the other.
	newBody := "---\ndescription: New body.\n---\n\nFresh content.\n"
	in, _ = json.Marshal(map[string]any{
		"action": "replace",
		"name":   "git-bisect",
		"body":   newBody,
	})
	res, err = tool.Execute(context.Background(), in)
	require.NoError(t, err)
	var rr struct {
		Success bool `json:"success"`
	}
	require.NoError(t, json.Unmarshal([]byte(res.Output), &rr))
	assert.True(t, rr.Success)

	got, ok := provider.Get("git-bisect")
	assert.True(t, ok)
	assert.Contains(t, got, "Fresh content.")

	// Remove one.
	in, _ = json.Marshal(map[string]any{"action": "remove", "name": "writing-clearly"})
	_, err = tool.Execute(context.Background(), in)
	require.NoError(t, err)

	in, _ = json.Marshal(map[string]any{"action": "list"})
	res, err = tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal([]byte(res.Output), &entries))
	require.Len(t, entries, 1)
	assert.Equal(t, "git-bisect", entries[0].Name)
}

func TestEndToEnd_GetReturnsFullBody(t *testing.T) {
	dir := t.TempDir()
	store := disk.NewStore(filepath.Join(dir, "skills"))
	tool := &skills.SkillTool{Store: store}

	body := "---\ndescription: A test.\n---\n\nFull body content here.\n"
	in, _ := json.Marshal(map[string]any{
		"action": "create",
		"name":   "fetch-me",
		"body":   body,
	})
	_, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	in, _ = json.Marshal(map[string]any{"action": "get", "name": "fetch-me"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)

	var sk skills.Skill
	require.NoError(t, json.Unmarshal([]byte(res.Output), &sk))
	assert.Equal(t, "fetch-me", sk.Name)
	assert.Equal(t, "A test.", sk.Description)
	assert.True(t, strings.Contains(sk.Body, "Full body content here."))
}
