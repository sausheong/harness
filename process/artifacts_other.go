//go:build !unix

package process

import (
	"errors"
	"os"
)

func lockArtifact(_ *os.File, _ bool) error {
	return errors.New("output artifact locking is unsupported on this platform")
}
func artifactBusy(_ error) bool { return false }

func artifactReadFlags() int { return os.O_RDONLY }
