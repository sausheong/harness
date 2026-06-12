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
