package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
)

// mcpToolAdapter implements tool.Tool by delegating Execute to an MCP
// session's CallTool. Constructed by Client.Connect via discoverTools.
type mcpToolAdapter struct {
	serverName string
	sdkTool    *sdk.Tool
	session    *sdk.ClientSession
}

// Name returns "mcp__<server>__<tool>". The double-underscore
// namespacing prevents collisions with built-in harness tools and
// is the convention most agent harnesses use for MCP-sourced tools.
func (a *mcpToolAdapter) Name() string {
	return "mcp__" + a.serverName + "__" + a.sdkTool.Name
}

func (a *mcpToolAdapter) Description() string { return a.sdkTool.Description }

// Parameters returns the server-supplied JSON Schema. SDK leaves
// InputSchema as `any`; we marshal it. If the schema fails to marshal
// (vanishingly unlikely — server-supplied JSON), we return an empty
// object schema so the LLM still sees the tool as callable.
func (a *mcpToolAdapter) Parameters() json.RawMessage {
	if a.sdkTool.InputSchema == nil {
		return json.RawMessage(`{"type":"object"}`)
	}
	b, err := json.Marshal(a.sdkTool.InputSchema)
	if err != nil {
		return json.RawMessage(`{"type":"object"}`)
	}
	return b
}

// IsConcurrencySafe reports false — MCP doesn't expose a safety hint
// per tool, and the only safe default for arbitrary external code is
// "no". Callers who know a specific MCP tool is read-only can wrap the
// adapter to override this.
func (a *mcpToolAdapter) IsConcurrencySafe(_ json.RawMessage) bool { return false }

// Execute marshals input into the MCP CallTool payload, invokes the
// server, and translates the response back into a tool.ToolResult.
//
// MCP's tool-error model differs from harness's:
//   - Protocol errors (transport, schema validation) → returned as Go
//     error, surfaced to the runtime as a tool result with Error set.
//   - Tool errors (CallToolResult.IsError=true) → also surfaced as
//     ToolResult.Error so the agent sees them the same way it sees a
//     harness-native tool error.
func (a *mcpToolAdapter) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var args any
	if len(input) > 0 {
		if err := json.Unmarshal(input, &args); err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
		}
	}
	res, err := a.session.CallTool(ctx, &sdk.CallToolParams{
		Name:      a.sdkTool.Name,
		Arguments: args,
	})
	if err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	return mcpResultToToolResult(res), nil
}

// mcpResultToToolResult flattens MCP content blocks into a harness
// ToolResult: TextContent concatenated into Output (or Error if
// IsError), ImageContent collected into Images. Other content types
// (audio, embedded resources) are described in Output as a placeholder.
func mcpResultToToolResult(res *sdk.CallToolResult) tool.ToolResult {
	var text strings.Builder
	var images []llm.ImageContent
	for _, c := range res.Content {
		switch v := c.(type) {
		case *sdk.TextContent:
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(v.Text)
		case *sdk.ImageContent:
			images = append(images, llm.ImageContent{
				MimeType: v.MIMEType,
				Data:     v.Data,
			})
		default:
			// Unknown content type — note it so the agent isn't silently
			// missing data.
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			fmt.Fprintf(&text, "[unsupported MCP content: %T]", c)
		}
	}
	out := tool.ToolResult{Images: images}
	if res.IsError {
		out.Error = text.String()
	} else {
		out.Output = text.String()
	}
	return out
}
