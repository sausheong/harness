package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
)

const MaxWorkerBytes int64 = 128 << 20

// snapshotWorker freezes verified bytes in a private directory outside the
// workspace. The daemon mounts this copy, never the mutable configured path.
func snapshotWorker(ctx context.Context, source, expected, directory string) (string, error) {
	digest, err := hex.DecodeString(expected)
	if err != nil || len(digest) != sha256.Size {
		return "", errors.New("worker requires a SHA-256 digest")
	}
	canonical, err := filepath.EvalSymlinks(source)
	if err != nil {
		return "", err
	}
	input, err := openWorkerSource(canonical)
	if err != nil {
		return "", err
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Mode().Perm()&0111 == 0 || info.Size() > MaxWorkerBytes {
		return "", errors.New("worker must be a regular executable within 128 MiB")
	}
	destination := filepath.Join(directory, "verified-worker")
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0500)
	if err != nil {
		return "", err
	}
	complete := false
	defer func() {
		output.Close()
		if !complete {
			os.Remove(destination)
		}
	}()
	hash := sha256.New()
	reader := io.LimitReader(input, MaxWorkerBytes+1)
	buffer := make([]byte, 32<<10)
	var total int64
	for {
		if err = ctx.Err(); err != nil {
			return "", err
		}
		n, readErr := reader.Read(buffer)
		if n > 0 {
			total += int64(n)
			if total > MaxWorkerBytes {
				return "", errors.New("worker exceeds 128 MiB")
			}
			if _, err = output.Write(buffer[:n]); err != nil {
				return "", err
			}
			hash.Write(buffer[:n])
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	if hex.EncodeToString(hash.Sum(nil)) != expected {
		return "", errors.New("worker SHA-256 mismatch; refusing execution")
	}
	if err = output.Chmod(0555); err != nil {
		return "", err
	}
	if err = output.Sync(); err != nil {
		return "", err
	}
	if err = output.Close(); err != nil {
		return "", err
	}
	complete = true
	return destination, nil
}
