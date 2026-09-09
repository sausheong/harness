package mcp

import (
	"context"
	"os/exec"
	"sync"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sausheong/harness/process"
)

type managedCommandTransport struct{ sdk.CommandTransport }

func (t *managedCommandTransport) Connect(ctx context.Context) (sdk.Connection, error) {
	conn, err := t.CommandTransport.Connect(ctx)
	if err != nil {
		return nil, err
	}
	return &managedCommandConnection{Connection: conn, command: t.Command}, nil
}

type managedCommandConnection struct {
	sdk.Connection
	command  *exec.Cmd
	once     sync.Once
	closeErr error
}

func (c *managedCommandConnection) Close() error {
	c.once.Do(func() {
		// Let the SDK close stdin and join the server first, then remove any
		// descendants it left behind. WaitDelay bounds inherited-pipe draining.
		c.closeErr = c.Connection.Close()
		if err := process.KillGroup(c.command); c.closeErr == nil {
			c.closeErr = err
		}
	})
	return c.closeErr
}
