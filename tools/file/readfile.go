package file

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
)

// ReadFileTool reads the contents of a file.
type ReadFileTool struct {
	WorkDir string // if set, restricts reads to this directory
}

type readFileInput struct {
	Path string `json:"path"`
}

func (t *ReadFileTool) Name() string { return "read_file" }

func (t *ReadFileTool) Description() string {
	return "Read the contents of a file at the given path. Returns the file contents as text. For image files (jpg, png, gif, webp, bmp), returns the image for visual inspection."
}

func (t *ReadFileTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "The absolute or relative path to the file to read"
			}
		},
		"required": ["path"]
	}`)
}

// imageExtMap maps file extensions to MIME types for image files.
var imageExtMap = map[string]string{
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".png":  "image/png",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
}

// detectImageMIMEFromBytes inspects magic bytes to determine the actual image
// format, falling back to hint (typically derived from file extension).
func detectImageMIMEFromBytes(data []byte, hint string) string {
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg"
	}
	if len(data) >= 4 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' {
		return "image/png"
	}
	if len(data) >= 4 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F' && data[3] == '8' {
		return "image/gif"
	}
	if len(data) >= 4 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' {
		return "image/webp"
	}
	return hint
}

// IsConcurrencySafe returns true — read_file is a pure read.
func (t *ReadFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }

func (t *ReadFileTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in readFileInput
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

	if err := ctx.Err(); err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}

	ext := strings.ToLower(filepath.Ext(in.Path))
	limit := int64(maxTextFileSize)
	kind := "text"
	if _, isImg := imageExtMap[ext]; isImg {
		limit = int64(maxImageFileSize)
		kind = "image"
	}
	if msg := checkFileSize(in.Path, limit, kind); msg != "" {
		return tool.ToolResult{Error: msg}, nil
	}

	f, err := os.Open(in.Path)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to read file: %v", err)}, nil
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("failed to read file: %v", err)}, nil
	}
	if int64(len(data)) > limit {
		return tool.ToolResult{Error: fmt.Sprintf("file too large: exceeds the %d byte limit for %s files; read a specific range or use a different tool", limit, kind)}, nil
	}

	// Check if this is an image file
	if mimeType, ok := imageExtMap[ext]; ok {
		mimeType = detectImageMIMEFromBytes(data, mimeType)
		return tool.ToolResult{
			Output: fmt.Sprintf("Image file: %s (%d bytes)", filepath.Base(in.Path), len(data)),
			Images: []llm.ImageContent{
				{MimeType: mimeType, Data: data},
			},
		}, nil
	}

	return tool.ToolResult{Output: string(data)}, nil
}
