// Package process owns bounded subprocess shutdown. On Unix each command has
// its own process group. This is lifecycle management, not a security sandbox.
package process

import (
	"context"
	"os/exec"
	"time"
)

// Command creates a command whose cancellation kills its owned process group.
// Run joins it and removes descendants left after normal exit. Callers using
// Start/Wait directly must call KillGroup when their protocol closes.
func Command(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	configureGroup(cmd)
	cmd.Cancel = func() error { return KillGroup(cmd) }
	cmd.WaitDelay = 250 * time.Millisecond
	return cmd
}
func Run(cmd *exec.Cmd) error {
	err := cmd.Run()
	cleanupErr := KillGroup(cmd)
	if err != nil {
		return err
	}
	return cleanupErr
}
