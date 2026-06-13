package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sausheong/harness/tool"
)

// EditFileTool performs a string-replace edit on a file.
type EditFileTool struct {
	WorkDir string // if set, restricts edits to this directory
}

type editFileInput struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func (t *EditFileTool) Name() string { return "edit_file" }

func (t *EditFileTool) Description() string {
	return "Edit a file by replacing an exact string match. The old_string must match exactly one occurrence in the file. Use this for targeted edits rather than rewriting entire files."
}

func (t *EditFileTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "The path to the file to edit"
			},
			"old_string": {
				"type": "string",
				"description": "The exact text to find and replace"
			},
			"new_string": {
				"type": "string",
				"description": "The replacement text"
			}
		},
		"required": ["path", "old_string", "new_string"]
	}`)
}

// IsConcurrencySafe returns false — edit_file mutates the filesystem.
func (t *EditFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

func (t *EditFileTool) Execute(_ context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in editFileInput
	if err := json.Unmarshal(input, &in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
	}

	if in.Path == "" {
		return tool.ToolResult{Error: "path is required"}, nil
	}

	in.Path = tool.ExpandHome(in.Path)
	if t.WorkDir != "" && !filepath.IsAbs(in.Path) {
		in.Path = filepath.Join(t.WorkDir, in.Path)
	}
	in.Path = tool.ResolveExistingPath(in.Path)

	if t.WorkDir != "" {
		if err := tool.ValidatePathInWorkDir(in.Path, t.WorkDir); err != nil {
			return tool.ToolResult{Error: err.Error()}, nil
		}
	}

	if msg := checkFileSize(in.Path, int64(maxTextFileSize), "text"); msg != "" {
		return tool.ToolResult{Error: msg}, nil
	}

	data, err := os.ReadFile(in.Path)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to read file: %v", err)}, nil
	}

	content := string(data)
	count := strings.Count(content, in.OldString)

	if count == 0 {
		return tool.ToolResult{Error: "old_string not found in file"}, nil
	}
	if count > 1 {
		return tool.ToolResult{Error: fmt.Sprintf("old_string found %d times in file, must be unique", count)}, nil
	}

	newContent := strings.Replace(content, in.OldString, in.NewString, 1)

	mode := os.FileMode(0o600)
	if info, statErr := os.Stat(in.Path); statErr == nil {
		mode = info.Mode().Perm()
	}
	if err := tool.WriteFileAtomic(in.Path, []byte(newContent), mode); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to write file: %v", err)}, nil
	}

	return tool.ToolResult{Output: "Successfully edited file"}, nil
}
