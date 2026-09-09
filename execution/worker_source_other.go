//go:build !unix

package execution

import (
	"errors"
	"os"
)

func openWorkerSource(string) (*os.File, error) {
	return nil, errors.New("worker snapshot source validation is unavailable on this platform")
}
