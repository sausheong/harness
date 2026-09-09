package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

func TestConstructionServerFixture(t *testing.T) {
	marker := os.Getenv("HARNESS_TEST_MCP_SHUTDOWN_MARKER")
	if marker == "" {
		return
	}
	server := sdk.NewServer(&sdk.Implementation{Name: "construction-fixture", Version: "1"}, nil)
	server.AddTool(&sdk.Tool{Name: "echo", InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		return &sdk.CallToolResult{}, nil
	})
	err := server.Run(context.Background(), &sdk.StdioTransport{})
	if err != nil {
		os.Exit(1)
	}
	if err := os.WriteFile(marker, []byte("closed"), 0600); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func constructionServer(t *testing.T) (mcp.ServerConfig, string) {
	t.Helper()
	marker := filepath.Join(t.TempDir(), "closed")
	return mcp.ServerConfig{Name: "healthy", Command: os.Args[0], Args: []string{"-test.run=^TestConstructionServerFixture$"}, Env: map[string]string{"HARNESS_TEST_MCP_SHUTDOWN_MARKER": marker, "GORACE": "atexit_sleep_ms=0"}}, marker
}

func TestBuildRuntimeContextCancellationClosesPartialConnections(t *testing.T) {
	healthy, marker := constructionServer(t)
	reg := tool.NewRegistry()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	start := time.Now()
	rt, err := BuildRuntimeContext(ctx, RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{MCPServers: []mcp.ServerConfig{healthy, {Name: "hung", Command: "sleep", Args: []string{"30"}}}})
	if rt != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("build = %v, %v", rt, err)
	}
	if len(reg.Names()) != 0 {
		t.Fatalf("partial tools leaked into registry: %v", reg.Names())
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("healthy connection not closed before return: %v", err)
	}
	if time.Since(start) > 3*time.Second {
		t.Fatal("construction shutdown exceeded bound")
	}
}

func TestBuildRuntimeContextAlreadyCancelledDoesNotMutateRegistry(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reg := tool.NewRegistry()
	rt, err := BuildRuntimeContext(ctx, RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{})
	if rt != nil || !errors.Is(err, context.Canceled) || len(reg.Names()) != 0 {
		t.Fatalf("build = %v, %v, tools %v", rt, err, reg.Names())
	}
}

func TestBuildRuntimeContextDoesNotOwnSuccessfulSessionLifetime(t *testing.T) {
	healthy, marker := constructionServer(t)
	reg := tool.NewRegistry()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	rt, err := BuildRuntimeContext(ctx, RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{MCPServers: []mcp.ServerConfig{healthy}})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	cancel()
	pingCtx, pingCancel := context.WithTimeout(context.Background(), time.Second)
	defer pingCancel()
	if err := rt.mcpClients[0].Ping(pingCtx); err != nil {
		t.Fatalf("construction context killed established session: %v", err)
	}
	if len(reg.Names()) != 1 || reg.Names()[0] != "mcp__healthy__echo" {
		t.Fatalf("tools %v", reg.Names())
	}
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal("runtime close did not join child")
	}
}

func TestOptionalMCPFailurePreservesHealthyCatalogue(t *testing.T) {
	healthy, _ := constructionServer(t)
	reg := tool.NewRegistry()
	rt, err := BuildRuntimeContext(context.Background(), RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{MCPServers: []mcp.ServerConfig{
		{Name: "optional", Optional: true, Command: "sleep", Args: []string{"30"}, ConnectTimeout: 50 * time.Millisecond}, healthy,
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if names := reg.Names(); len(names) != 1 || names[0] != "mcp__healthy__echo" {
		t.Fatalf("catalogue %v", names)
	}
	status := rt.MCPStatus()
	if len(status) != 2 || status[0].State != "unavailable" || !status[0].Optional || status[1].State != "connected" || status[1].Tools != 1 {
		t.Fatalf("status %+v", status)
	}
	status[0].State = "mutated"
	if rt.MCPStatus()[0].State != "unavailable" {
		t.Fatal("mutable status escaped")
	}
}
func TestOptionalMCPDoesNotSwallowOwnerCancellation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	rt, err := BuildRuntimeContext(ctx, RuntimeDeps{}, RuntimeInputs{Tools: tool.NewRegistry()}, AgentSpec{MCPServers: []mcp.ServerConfig{{Name: "optional", Optional: true, Command: "sleep", Args: []string{"30"}}}})
	if rt != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("owner cancellation: %v %v", rt, err)
	}
}
func TestMCPDuplicateNamesRejectedBeforeConstruction(t *testing.T) {
	reg := tool.NewRegistry()
	rt, err := BuildRuntimeContext(context.Background(), RuntimeDeps{}, RuntimeInputs{Tools: reg}, AgentSpec{MCPServers: []mcp.ServerConfig{{Name: "same"}, {Name: "same"}}})
	if rt != nil || err == nil || len(reg.Names()) != 0 {
		t.Fatalf("duplicate names: %v %v", rt, err)
	}
}
