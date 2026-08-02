package file

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadFileTool(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	require.NoError(t, os.WriteFile(path, []byte("hello world"), 0o644))

	tool := &ReadFileTool{}
	input, _ := json.Marshal(readFileInput{Path: path})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, "hello world", result.Output)
	assert.Empty(t, result.Error)
}

func TestReadFileToolMissing(t *testing.T) {
	tool := &ReadFileTool{}
	input, _ := json.Marshal(readFileInput{Path: "/nonexistent/file"})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.NotEmpty(t, result.Error)
}

func TestReadFileToolWorkDirClamps(t *testing.T) {
	workspace := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	require.NoError(t, os.WriteFile(outside, []byte("nope"), 0o644))

	tool := &ReadFileTool{WorkDir: workspace}
	input, _ := json.Marshal(readFileInput{Path: outside})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "outside workspace",
		"reads outside the workspace must be rejected when WorkDir is set")
}

func TestReadFileToolWorkDirRejectsEscapingFileSymlink(t *testing.T) {
	workspace := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	require.NoError(t, os.WriteFile(outside, []byte("nope"), 0o644))
	link := filepath.Join(workspace, "secret-link")
	require.NoError(t, os.Symlink(outside, link))

	readTool := &ReadFileTool{WorkDir: workspace}
	input, _ := json.Marshal(readFileInput{Path: link})
	result, err := readTool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "outside workspace")
	assert.Empty(t, result.Output)

	writeTool := &WriteFileTool{WorkDir: workspace}
	writeInput, _ := json.Marshal(writeFileInput{Path: link, Content: "overwritten"})
	result, err = writeTool.Execute(context.Background(), writeInput)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "outside workspace")
	data, readErr := os.ReadFile(outside)
	require.NoError(t, readErr)
	assert.Equal(t, "nope", string(data))
}

func TestWriteFileTool(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "subdir", "output.txt")

	tool := &WriteFileTool{}
	input, _ := json.Marshal(writeFileInput{Path: path, Content: "test content"})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Empty(t, result.Error)
	assert.Contains(t, result.Output, "Successfully wrote")

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, "test content", string(data))
}

func TestEditFileTool(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	require.NoError(t, os.WriteFile(path, []byte("hello world"), 0o644))

	tool := &EditFileTool{}
	input, _ := json.Marshal(editFileInput{
		Path:      path,
		OldString: "world",
		NewString: "Go",
	})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Empty(t, result.Error)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, "hello Go", string(data))
}

func TestEditFileToolNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	require.NoError(t, os.WriteFile(path, []byte("hello world"), 0o644))

	tool := &EditFileTool{}
	input, _ := json.Marshal(editFileInput{
		Path:      path,
		OldString: "missing",
		NewString: "replacement",
	})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "not found")
}
