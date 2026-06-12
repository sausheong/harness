package tool

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.txt")

	require.NoError(t, WriteFileAtomic(path, []byte("hello"), 0o600))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, "hello", string(data))

	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o600), info.Mode().Perm())

	require.NoError(t, WriteFileAtomic(path, []byte("world"), 0o644))
	data, _ = os.ReadFile(path)
	require.Equal(t, "world", string(data))

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	require.Len(t, entries, 1)
	require.Equal(t, "out.txt", entries[0].Name())
}
