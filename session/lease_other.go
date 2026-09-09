//go:build !linux && !darwin

package session

import (
	"errors"
	"os"
)

const supportsWriterLease = false

func openLeaseFile(string) (*os.File, error) {
	return nil, errors.New("session writer leases are unsupported on this platform")
}
func lockLeaseFile(*os.File) error {
	return errors.New("session writer leases are unsupported on this platform")
}
func leaseWouldBlock(error) bool { return false }
