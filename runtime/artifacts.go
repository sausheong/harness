package runtime

import (
	"github.com/sausheong/harness/process"
	"github.com/sausheong/harness/tool"
)

// Only typed, completed built-in capture metadata becomes an artifact reference.
// MCP JSON maps or paths mentioned in tool output are never interpreted as files.
func resultArtifacts(result tool.ToolResult) map[string]process.ArtifactInfo {
	var artifacts map[string]process.ArtifactInfo
	for _, stream := range []string{"stdout", "stderr"} {
		info, ok := result.Metadata[stream+"_artifact"].(process.ArtifactInfo)
		if !ok || info.Path == "" || len(info.Path) > 4096 || len(info.SHA256) != 64 || info.Bytes < 0 || info.Bytes > process.ArtifactFileLimit || info.Error != "" {
			continue
		}
		if artifacts == nil {
			artifacts = make(map[string]process.ArtifactInfo)
		}
		artifacts[stream] = info
	}
	return artifacts
}
