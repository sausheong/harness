package runtime

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sausheong/harness/tools/mcp"
)

const MCPConnectConcurrency = 4

// connectMCPServers joins every worker before returning. Results retain config
// order, independent of completion order; publication remains transactional in
// BuildRuntimeContext. The caller owns and must close returned clients on error.
func connectMCPServers(parent context.Context, servers []mcp.ServerConfig) ([]*mcp.Client, []MCPServerStatus, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	clients := make([]*mcp.Client, len(servers))
	status := make([]MCPServerStatus, len(servers))
	var firstErr error
	var fail sync.Once
	jobs := make(chan int)
	var workers sync.WaitGroup
	for w := 0; w < min(MCPConnectConcurrency, len(servers)); w++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for i := range jobs {
				if ctx.Err() != nil {
					continue
				}
				srv := servers[i]
				serverCtx := ctx
				stop := func() {}
				timeout := srv.ConnectTimeout
				if timeout == 0 && srv.Optional {
					timeout = 5 * time.Second
				}
				if timeout > 0 {
					serverCtx, stop = context.WithTimeout(ctx, timeout)
				}
				cli, err := mcp.Connect(serverCtx, srv)
				stop()
				status[i] = MCPServerStatus{Name: srv.Name, Optional: srv.Optional, State: "unavailable"}
				if err != nil {
					if !srv.Optional {
						fail.Do(func() { firstErr = fmt.Errorf("mcp server %q: %w", srv.Name, err); cancel() })
					}
					continue
				}
				clients[i] = cli
				status[i].State = "connected"
				status[i].Tools = len(cli.Tools())
			}
		}()
	}
dispatch:
	for i := range servers {
		select {
		case jobs <- i:
		case <-ctx.Done():
			break dispatch
		}
	}
	close(jobs)
	workers.Wait()
	connected := make([]*mcp.Client, 0, len(clients))
	for _, cli := range clients {
		if cli != nil {
			connected = append(connected, cli)
		}
	}
	if parent.Err() != nil {
		return connected, nil, parent.Err()
	}
	if firstErr != nil {
		return connected, nil, firstErr
	}
	return connected, status, nil
}
