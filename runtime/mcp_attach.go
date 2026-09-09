package runtime

import (
	"errors"
	"fmt"

	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

var ErrMCPAttachBusy = errors.New("runtime is active; defer MCP attachment until idle")

// AttachMCP transfers ownership only on success. The caller retains ownership
// on all errors, including busy, and may retry or close the client. Connection
// happens outside the runtime, so local runs can continue during discovery.
func (r *Runtime) AttachMCP(client *mcp.Client, optional bool) error {
	if client == nil {
		return errors.New("nil MCP client")
	}
	if !r.runMu.TryLock() {
		return ErrMCPAttachBusy
	}
	defer r.runMu.Unlock()
	r.mcpMu.Lock()
	defer r.mcpMu.Unlock()
	if r.mcpClosed {
		return errors.New("runtime is closed")
	}
	reg, ok := r.Tools.(*tool.Registry)
	if !ok || r.refreshToolPrompt == nil {
		return errors.New("runtime does not support dynamic MCP attachment")
	}
	name := client.Name()
	for _, s := range r.mcpStatus {
		if s.Name == name && s.State == "connected" {
			return fmt.Errorf("MCP server %q already connected", name)
		}
	}
	if err := reg.RegisterUnique(client.Tools()); err != nil {
		return err
	}
	r.StaticSystemPrompt = r.refreshToolPrompt(reg.Names())
	r.mcpClients = append(r.mcpClients, client)
	status := MCPServerStatus{Name: name, Optional: optional, State: "connected", Tools: len(client.Tools())}
	for i, s := range r.mcpStatus {
		if s.Name == name {
			r.mcpStatus[i] = status
			return nil
		}
	}
	r.mcpStatus = append(r.mcpStatus, status)
	return nil
}
