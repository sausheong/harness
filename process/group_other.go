//go:build !unix

package process

import (
	"errors"
	"os"
	"os/exec"
)

func configureGroup(cmd *exec.Cmd) {}

// Other platforms stop the direct child only. Process-tree guarantees apply
// to the advertised Linux/macOS backends; isolation is a separate capability.
func KillGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	err := cmd.Process.Kill()
	if errors.Is(err, os.ErrProcessDone) {
		return nil
	}
	return err
}
