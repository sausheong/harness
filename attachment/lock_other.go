//go:build !darwin && !linux

package attachment

import (
	"context"
	"errors"
	"os"
)

func lock(context.Context, *os.File) error {
	return errors.New("attachment writer locking unsupported on this platform")
}
func unlock(*os.File) {}
