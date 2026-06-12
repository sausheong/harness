package file

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEditFilePreservesMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "code.txt")
	require.NoError(t, os.WriteFile(path, []byte("alpha beta"), 0o640))

	tool := &EditFileTool{WorkDir: dir}
	in, _ := json.Marshal(map[string]string{
		"path":       path,
		"old_string": "alpha",
		"new_string": "gamma",
	})
	res, err := tool.Execute(context.Background(), in)
	require.NoError(t, err)
	require.Empty(t, res.Error)

	data, _ := os.ReadFile(path)
	require.Equal(t, "gamma beta", string(data))

	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o640), info.Mode().Perm())
}
