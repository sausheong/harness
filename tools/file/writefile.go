package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"github.com/sausheong/harness/tool"
)

// WriteFileTool creates or overwrites a file.
type WriteFileTool struct {
	WorkDir string // if set, restricts writes to this directory
}

type writeFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

func (t *WriteFileTool) Name() string { return "write_file" }

func (t *WriteFileTool) Description() string {
	return "Write content to a file at the given path. Creates the file and any parent directories if they don't exist. Overwrites existing files."
}

func (t *WriteFileTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "The absolute or relative path to the file to write"
			},
			"content": {
				"type": "string",
				"description": "The content to write to the file"
			}
		},
		"required": ["path", "content"]
	}`)
}

// IsConcurrencySafe returns false — write_file mutates the filesystem.
func (t *WriteFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

func (t *WriteFileTool) Execute(_ context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in writeFileInput
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

	if t.WorkDir != "" {
		if err := tool.ValidatePathInWorkDir(in.Path, t.WorkDir); err != nil {
			return tool.ToolResult{Error: err.Error()}, nil
		}
	}

	// Create parent directories
	dir := filepath.Dir(in.Path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to create directory: %v", err)}, nil
	}

	if err := tool.WriteFileAtomic(in.Path, []byte(in.Content), 0o600); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to write file: %v", err)}, nil
	}

	return tool.ToolResult{Output: fmt.Sprintf("Successfully wrote %d bytes to %s", len(in.Content), in.Path)}, nil
}
