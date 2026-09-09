package runtime

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/mcp"
)

type startupTransport func(*http.Request) (*http.Response, error)

func (f startupTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestMCPConnectionsIndependentAndBounded(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	release := make(chan struct{})
	entered := make(chan struct{}, 32)
	var active, peak atomic.Int32
	client := &http.Client{Transport: startupTransport(func(r *http.Request) (*http.Response, error) {
		n := active.Add(1)
		defer active.Add(-1)
		for old := peak.Load(); n > old; old = peak.Load() {
			if peak.CompareAndSwap(old, n) {
				break
			}
		}
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
		return nil, errors.New("fixture unavailable")
	})}
	servers := make([]mcp.ServerConfig, 8)
	for i := range servers {
		servers[i] = mcp.ServerConfig{Name: fmt.Sprintf("server%d", i), Optional: true, URL: "http://fixture.invalid/mcp", HTTPClient: client}
	}
	type result struct {
		rt  *Runtime
		err error
	}
	done := make(chan result, 1)
	go func() {
		rt, err := BuildRuntimeContext(ctx, RuntimeDeps{}, RuntimeInputs{Tools: tool.NewRegistry()}, AgentSpec{MCPServers: servers})
		done <- result{rt, err}
	}()
	for i := 0; i < MCPConnectConcurrency; i++ {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("connections did not start independently")
		}
	}
	if n := active.Load(); n != MCPConnectConcurrency {
		t.Fatalf("active connections %d", n)
	}
	close(release)
	r := <-done
	if r.err != nil {
		t.Fatal(r.err)
	}
	defer r.rt.Close()
	if n := peak.Load(); n > MCPConnectConcurrency {
		t.Fatalf("concurrency exceeded: %d", n)
	}
	status := r.rt.MCPStatus()
	if len(status) != len(servers) {
		t.Fatalf("status count %d", len(status))
	}
	for i, s := range status {
		if s.Name != servers[i].Name || s.State != "unavailable" {
			t.Fatalf("status order %+v", status)
		}
	}
}
