package runtime

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

func TestMCPAttachAtIdleUpdatesCatalogueAndOwnsShutdown(t *testing.T) {
	config, marker := constructionServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	client, err := mcp.Connect(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	reg := tool.NewRegistry()
	rt, err := BuildRuntime(RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{ID: "test"})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	rt.runMu.Lock()
	err = rt.AttachMCP(client, true)
	rt.runMu.Unlock()
	if !errors.Is(err, ErrMCPAttachBusy) || len(reg.Names()) != 0 {
		t.Fatalf("active attachment: %v", err)
	}
	if _, err = os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("rejected attachment closed caller's client")
	}
	if err = rt.AttachMCP(client, true); err != nil {
		t.Fatal(err)
	}
	if len(reg.Names()) != 1 || rt.Tools.ToolDefs()[0].Name != "mcp__healthy__echo" || rt.StaticSystemPrompt != BuildStaticSystemPrompt("", "", "test", "", reg.Names(), "", "", "", "") {
		t.Fatal("tool catalogue and model context diverged")
	}
	if status := rt.MCPStatus(); len(status) != 1 || status[0].State != "connected" {
		t.Fatalf("status %+v", status)
	}
	if err = rt.AttachMCP(client, true); err == nil {
		t.Fatal("duplicate attachment accepted")
	}
	if err = rt.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(marker); err != nil {
		t.Fatal("attached client not joined on close")
	}
	if err = rt.AttachMCP(client, true); err == nil {
		t.Fatal("attachment after close")
	}
}
