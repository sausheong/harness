//go:build unix

package process

import (
	"errors"
	"os"
	"syscall"
)

func lockArtifact(f *os.File, nonblocking bool) error {
	flags := syscall.LOCK_EX
	if nonblocking {
		flags |= syscall.LOCK_NB
	}
	return syscall.Flock(int(f.Fd()), flags)
}
func artifactBusy(err error) bool { return errors.Is(err, syscall.EWOULDBLOCK) }

func artifactReadFlags() int { return os.O_RDONLY | syscall.O_NONBLOCK | syscall.O_NOFOLLOW }
