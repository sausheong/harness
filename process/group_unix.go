//go:build unix

package process

import (
	"errors"
	"os/exec"
	"syscall"
)

func configureGroup(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

// KillGroup only signals the group explicitly created for this command.
func KillGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	var err error
	if cmd.SysProcAttr != nil && cmd.SysProcAttr.Setpgid {
		err = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	} else {
		err = cmd.Process.Kill()
	}
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}
