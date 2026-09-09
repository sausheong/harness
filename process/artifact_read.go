package process

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Read verifies a completed artifact within this store. It never follows paths
// supplied by output prose. Missing/expired, active, corrupt or legacy unverified
// artifacts fail explicitly. Truncated describes capture completeness separately
// from integrity: the verified captured prefix may still be read.
func (s *ArtifactStore) Read(ctx context.Context, info ArtifactInfo) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	digest, err := hex.DecodeString(info.SHA256)
	if err != nil || len(digest) != sha256.Size || info.Bytes < 0 || info.Bytes > ArtifactFileLimit || info.Error != "" {
		return nil, errors.New("artifact has no valid completed integrity record")
	}
	rel, err := filepath.Rel(s.dir, info.Path)
	if err != nil || filepath.Base(rel) != rel || !strings.HasPrefix(rel, "output-") || !strings.HasSuffix(rel, ".log") {
		return nil, errors.New("artifact is outside this store")
	}
	root, err := os.OpenRoot(s.dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	before, err := root.Lstat(rel)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() {
		return nil, errors.New("artifact is not a regular file")
	}
	f, err := root.OpenFile(rel, artifactReadFlags(), 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := lockArtifact(f, true); err != nil {
		return nil, fmt.Errorf("artifact is active or unavailable: %w", err)
	}
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !stat.Mode().IsRegular() || stat.Mode().Perm()&0077 != 0 || stat.Size() != info.Bytes {
		return nil, errors.New("artifact size or permissions changed")
	}
	data := make([]byte, 0, int(info.Bytes))
	buf := make([]byte, 32<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, readErr := f.Read(buf)
		if int64(len(data)+n) > info.Bytes {
			return nil, errors.New("artifact grew during read")
		}
		data = append(data, buf[:n]...)
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	actual := sha256.Sum256(data)
	if int64(len(data)) != info.Bytes || hex.EncodeToString(actual[:]) != strings.ToLower(info.SHA256) {
		return nil, errors.New("artifact integrity mismatch")
	}
	return data, nil
}
