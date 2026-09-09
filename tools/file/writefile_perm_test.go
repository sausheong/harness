package file

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWriteFileIsRestrictive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "secret.txt")

	tool := &WriteFileTool{WorkDir: dir}
	in, _ := json.Marshal(map[string]string{"path": path, "content": "data"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Empty(t, res.Error)

	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o600), info.Mode().Perm())
}

func TestWriteFilePreservesExecutableMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "script.sh")
	require.NoError(t, os.WriteFile(path, []byte("old"), 0751))
	require.NoError(t, os.Chmod(path, 0751))
	writer := &WriteFileTool{WorkDir: dir}
	input, _ := json.Marshal(map[string]string{"path": path, "content": "#!/bin/sh\nexit 0\n"})
	result, err := writer.Execute(context.Background(), input)
	require.NoError(t, err)
	require.Empty(t, result.Error)
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0751), info.Mode().Perm())
}

func TestFailedWritePreservesExistingContentAndMode(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Fatal("permission failure fixture requires an unprivileged user")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "script.sh")
	require.NoError(t, os.WriteFile(path, []byte("original"), 0751))
	require.NoError(t, os.Chmod(path, 0751))
	require.NoError(t, os.Chmod(dir, 0500))
	defer os.Chmod(dir, 0700)
	writer := &WriteFileTool{WorkDir: dir}
	input, _ := json.Marshal(map[string]string{"path": path, "content": "replacement"})
	result, err := writer.Execute(context.Background(), input)
	require.NoError(t, err)
	require.NotEmpty(t, result.Error)
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, "original", string(data))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0751), info.Mode().Perm())
	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	require.Len(t, entries, 1)
}
