package file

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func writeSized(t *testing.T, dir, name string, n int) string {
	t.Helper()
	p := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(p, make([]byte, n), 0o600))
	return p
}

func TestReadFile_RejectsOversizedText(t *testing.T) {
	dir := t.TempDir()
	p := writeSized(t, dir, "big.txt", maxTextFileSize+1)
	tool := &ReadFileTool{}
	in, _ := json.Marshal(map[string]string{"path": p})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Contains(t, res.Error, "too large")
}

func TestReadFile_AllowsNormalText(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "ok.txt")
	require.NoError(t, os.WriteFile(p, []byte("hello"), 0o600))
	tool := &ReadFileTool{}
	in, _ := json.Marshal(map[string]string{"path": p})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Empty(t, res.Error)
	require.Equal(t, "hello", res.Output)
}

func TestReadFile_RejectsOversizedImage(t *testing.T) {
	dir := t.TempDir()
	p := writeSized(t, dir, "big.png", maxImageFileSize+1)
	tool := &ReadFileTool{}
	in, _ := json.Marshal(map[string]string{"path": p})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Contains(t, res.Error, "too large")
}

func TestEditFile_RejectsOversized(t *testing.T) {
	dir := t.TempDir()
	p := writeSized(t, dir, "big.txt", maxTextFileSize+1)
	tool := &EditFileTool{}
	in, _ := json.Marshal(map[string]string{"path": p, "old_string": "a", "new_string": "b"})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Contains(t, res.Error, "too large")
}
