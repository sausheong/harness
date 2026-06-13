package runtime

import (
	jsonpkg "encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sausheong/harness/session"
	"github.com/stretchr/testify/require"
)

func TestBuildStaticSystemPromptWithIdentityFile(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(dir, "IDENTITY.md"),
		[]byte("CUSTOM IDENTITY"),
		0o644,
	))

	got := BuildStaticSystemPrompt(
		dir, "", "alpha", "Alpha",
		[]string{"read_file"},
		"Configured channels: cli",
		"\n\n## Skills Index\n\n- foo",
		"",
		"",
	)
	require.Contains(t, got, "CUSTOM IDENTITY")
	require.Contains(t, got, `"Alpha" agent (id: alpha)`)
	require.Contains(t, got, "Configured channels: cli")
	require.Contains(t, got, "## Skills Index")
}

func TestBuildStaticSystemPromptConfigOverride(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "IDENTITY.md"), []byte("FROM_IDENTITY_FILE"), 0o644))

	got := BuildStaticSystemPrompt(dir, "FROM CONFIG", "id", "Name", nil, "", "", "", "")
	require.Contains(t, got, "FROM CONFIG")
	require.NotContains(t, got, "FROM_IDENTITY_FILE")
}

func TestBuildStaticSystemPromptDefaultIdentity(t *testing.T) {
	dir := t.TempDir() // no IDENTITY.md
	got := BuildStaticSystemPrompt(dir, "", "id", "Name", []string{"read_file", "bash"}, "", "", "", "")
	require.Contains(t, got, defaultIdentityBase)
	require.Contains(t, got, "read files")
	require.Contains(t, got, "bash commands")
}

func TestBuildStaticSystemPromptByteStableAcrossCalls(t *testing.T) {
	dir := t.TempDir()
	a := BuildStaticSystemPrompt(dir, "", "id", "Name", []string{"read_file"}, "summary", "index", "memidx", "")
	b := BuildStaticSystemPrompt(dir, "", "id", "Name", []string{"read_file"}, "summary", "index", "memidx", "")
	require.Equal(t, a, b, "BuildStaticSystemPrompt must be deterministic")
}

func TestBuildStaticSystemPromptIncludesMemoryIndex(t *testing.T) {
	dir := t.TempDir()
	got := BuildStaticSystemPrompt(
		dir, "", "id", "Name",
		[]string{"read_file"},
		"", "",
		"\n\n## Memory Index\n\n- **note1** — A note: one-line desc",
		"",
	)
	require.Contains(t, got, "## Memory Index")
	require.Contains(t, got, "**note1**")
	require.Contains(t, got, "one-line desc")
}

func TestBuildStaticSystemPromptIncludesMemoryFiles(t *testing.T) {
	dir := t.TempDir()
	got := BuildStaticSystemPrompt(
		dir, "", "id", "Name",
		[]string{"read_file"},
		"", "", "",
		"\n\n## Project memory: /tmp/x\n\nUNIQUE_MEM_FILES_SENTINEL",
	)
	require.Contains(t, got, "UNIQUE_MEM_FILES_SENTINEL")
	require.Contains(t, got, "## Project memory:")
}

func TestBuildDynamicSystemPromptSuffixEmpty(t *testing.T) {
	got := buildDynamicSystemPromptSuffix("", "")
	require.Equal(t, "", got)
}

func TestBuildDynamicSystemPromptSuffixHintOnly(t *testing.T) {
	got := buildDynamicSystemPromptSuffix("", "\n\nHINT")
	require.Equal(t, "\n\nHINT", got)
}

func TestBuildDynamicSystemPromptSuffixDateAndHint(t *testing.T) {
	got := buildDynamicSystemPromptSuffix("Today's date is 2026-05-01.", "\n\nKG_HINT")
	dateIdx := strings.Index(got, "Today's date is")
	hintIdx := strings.Index(got, "KG_HINT")
	require.True(t, dateIdx >= 0 && hintIdx > dateIdx,
		"order must be date < hint; got %d %d", dateIdx, hintIdx)
}

func TestBuildDynamicSystemPromptSuffixIncludesDate(t *testing.T) {
	got := buildDynamicSystemPromptSuffix("Today's date is 2026-05-01.", "")
	require.True(t, strings.HasPrefix(got, "Today's date is 2026-05-01."),
		"date line must appear at the top of the dynamic suffix")
}

func TestFormatDateLine(t *testing.T) {
	cases := []struct {
		name string
		in   time.Time
		want string
	}{
		{"may day", time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC), "Today's date is 2026-05-01."},
		{"new year", time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC), "Today's date is 2027-01-01."},
		{"single digit month", time.Date(2026, 3, 9, 23, 59, 59, 0, time.UTC), "Today's date is 2026-03-09."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := FormatDateLine(tc.in)
			require.Equal(t, tc.want, got)
		})
	}
}

func mustJSON(t *testing.T, v any) jsonpkg.RawMessage {
	t.Helper()
	b, err := jsonpkg.Marshal(v)
	require.NoError(t, err)
	return b
}

func TestAssembleMessagesInjectsOrphanedToolResult(t *testing.T) {
	history := []session.SessionEntry{
		{Type: session.EntryTypeMessage, Role: "user", Data: mustJSON(t, session.MessageData{Text: "hi"})},
		{Type: session.EntryTypeToolCall, Role: "assistant", Data: mustJSON(t, session.ToolCallData{
			ID: "call_1", Tool: "bash", Input: jsonpkg.RawMessage(`{}`),
		})},
		{Type: session.EntryTypeMessage, Role: "user", Data: mustJSON(t, session.MessageData{Text: "still there?"})},
	}
	msgs := assembleMessages(history)
	var found bool
	for _, m := range msgs {
		if m.ToolCallID == "call_1" {
			found = true
		}
	}
	require.True(t, found, "orphaned tool call must get a synthetic tool result from the terminal pass")
}
