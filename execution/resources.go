package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ReadOnlyResource is an explicitly pinned file exposed under
// /harness-resources/<Relative>. Only a verified private copy is mounted.
type ReadOnlyResource struct {
	Path       string
	Relative   string
	SHA256     string
	Executable bool
}

func snapshotResources(ctx context.Context, files []ReadOnlyResource, directory string) (string, error) {
	if len(files) == 0 || len(files) > 65 {
		return "", errors.New("resource bundle requires 1 to 65 files")
	}
	seen := map[string]bool{}
	for _, f := range files {
		digest, err := hex.DecodeString(f.SHA256)
		if !filepath.IsAbs(f.Path) || !fs.ValidPath(f.Relative) || f.Relative == "." || strings.ContainsAny(f.Relative, "\\,:\r\n\x00") || err != nil || len(digest) != 32 || hex.EncodeToString(digest) != f.SHA256 {
			return "", errors.New("invalid pinned resource")
		}
		key := strings.ToLower(f.Relative)
		if seen[key] {
			return "", errors.New("duplicate resource path")
		}
		seen[key] = true
	}
	for key := range seen {
		for parent := filepath.ToSlash(filepath.Dir(key)); parent != "."; parent = filepath.ToSlash(filepath.Dir(parent)) {
			if seen[parent] {
				return "", errors.New("resource path overlaps a directory")
			}
		}
	}
	root, err := os.MkdirTemp(directory, "verified-resources-")
	if err != nil {
		return "", err
	}
	complete := false
	defer func() {
		if !complete {
			os.RemoveAll(root)
		}
	}()
	var total int64
	for _, f := range files {
		if err = ctx.Err(); err != nil {
			return "", err
		}
		n, e := copyResource(ctx, f, filepath.Join(root, filepath.FromSlash(f.Relative)), 160<<20-total)
		if e != nil {
			return "", e
		}
		total += n
	}
	// Files are immutable to the container UID and the entire bind mount is read-only.
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() {
			return os.Chmod(path, 0755)
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	complete = true
	return root, nil
}
func copyResource(ctx context.Context, f ReadOnlyResource, target string, remaining int64) (int64, error) {
	input, err := openWorkerSource(f.Path)
	if err != nil {
		return 0, err
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil {
		return 0, err
	}
	if !info.Mode().IsRegular() || info.Size() > MaxWorkerBytes || info.Size() > remaining {
		return 0, errors.New("resource size or file type invalid")
	}
	if err = os.MkdirAll(filepath.Dir(target), 0700); err != nil {
		return 0, err
	}
	output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return 0, err
	}
	defer output.Close()
	hash := sha256.New()
	buffer := make([]byte, 32<<10)
	reader := io.LimitReader(input, info.Size()+1)
	var total int64
	for {
		if err = ctx.Err(); err != nil {
			return 0, err
		}
		n, e := reader.Read(buffer)
		if n > 0 {
			total += int64(n)
			if total > info.Size() {
				return 0, errors.New("resource changed size")
			}
			if _, err = output.Write(buffer[:n]); err != nil {
				return 0, err
			}
			hash.Write(buffer[:n])
		}
		if e == io.EOF {
			break
		}
		if e != nil {
			return 0, e
		}
	}
	if total != info.Size() || hex.EncodeToString(hash.Sum(nil)) != f.SHA256 {
		return 0, errors.New("resource digest mismatch")
	}
	mode := os.FileMode(0444)
	if f.Executable {
		mode = 0555
	}
	if err = output.Chmod(mode); err != nil {
		return 0, err
	}
	if err = output.Sync(); err != nil {
		return 0, err
	}
	return total, output.Close()
}
