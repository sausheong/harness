// Package attachment stores immutable session attachments by their content hash.
package attachment

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const MaxBlobBytes int64 = 32 << 20
const DefaultStoreBytes int64 = 1 << 30
const maxEntries = 8192

// Ref has no path chosen by a caller. MIME type belongs to the referring entry.
type Ref struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

func (r Ref) Validate() error {
	if len(r.SHA256) != 64 || r.Size < 0 || r.Size > MaxBlobBytes {
		return errors.New("invalid attachment reference")
	}
	for _, c := range r.SHA256 {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return errors.New("invalid attachment digest")
		}
	}
	return nil
}

type Store struct {
	root  *os.Root
	limit int64
}

// Open opens a private directory. Close after all operations join. A full store
// rejects new content; existing session attachments are never evicted.
func Open(dir string, limit int64) (*Store, error) {
	if limit <= 0 {
		return nil, errors.New("attachment quota must be positive")
	}
	if err := os.Mkdir(dir, 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return nil, errors.New("attachment directory must be private (0700)")
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	// A durable blob directory also needs its name synced in the parent before
	// any session record can refer to a blob inside it. The parent must exist.
	parent, err := os.Open(filepath.Dir(dir))
	if err != nil {
		root.Close()
		return nil, err
	}
	if err = errors.Join(parent.Sync(), parent.Close()); err != nil {
		root.Close()
		return nil, err
	}
	return &Store{root: root, limit: limit}, nil
}
func (s *Store) Close() error { return s.root.Close() }

func (s *Store) Put(ctx context.Context, data []byte) (Ref, error) {
	if err := ctx.Err(); err != nil {
		return Ref{}, err
	}
	if int64(len(data)) > MaxBlobBytes {
		return Ref{}, errors.New("attachment exceeds blob limit")
	}
	digest := sha256.Sum256(data)
	ref := Ref{SHA256: hex.EncodeToString(digest[:]), Size: int64(len(data))}
	guard, err := s.root.OpenFile(".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return Ref{}, err
	}
	defer guard.Close()
	if err := lock(ctx, guard); err != nil {
		return Ref{}, err
	}
	defer unlock(guard)
	if _, err := s.root.Lstat(ref.SHA256); err == nil {
		if _, err := s.Read(ctx, ref); err != nil {
			return Ref{}, err
		}
		// Reestablish a durability fence when reusing a prior publication.
		f, err := s.root.Open(ref.SHA256)
		if err != nil {
			return Ref{}, err
		}
		err = f.Sync()
		closeErr := f.Close()
		if err = errors.Join(err, closeErr, s.sync()); err != nil {
			return Ref{}, err
		}
		return ref, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return Ref{}, err
	}
	used, err := s.usage(ctx)
	if err != nil {
		return Ref{}, err
	}
	if ref.Size > s.limit-used {
		return Ref{}, errors.New("attachment store quota exceeded")
	}
	name := ".pending-" + rand.Text()
	f, err := s.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return Ref{}, err
	}
	defer s.root.Remove(name)
	for offset := 0; offset < len(data); {
		if err = ctx.Err(); err != nil {
			f.Close()
			return Ref{}, err
		}
		end := min(offset+64*1024, len(data))
		n, writeErr := f.Write(data[offset:end])
		offset += n
		if writeErr != nil {
			f.Close()
			return Ref{}, writeErr
		}
		if n == 0 {
			f.Close()
			return Ref{}, io.ErrShortWrite
		}
	}
	err = f.Sync()
	err = errors.Join(err, f.Close())
	if err != nil {
		return Ref{}, err
	}
	if err = ctx.Err(); err != nil {
		return Ref{}, err
	}
	if err = s.root.Link(name, ref.SHA256); err != nil {
		return Ref{}, err
	}
	if err = s.sync(); err != nil {
		return Ref{}, fmt.Errorf("attachment published but directory sync failed: %w", err)
	}
	return ref, nil
}

func (s *Store) Read(ctx context.Context, ref Ref) ([]byte, error) {
	if err := ref.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := s.root.Lstat(ref.SHA256)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() != ref.Size {
		return nil, errors.New("attachment is not a regular file of the declared size")
	}
	f, err := s.root.Open(ref.SHA256)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	actual, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !actual.Mode().IsRegular() || !os.SameFile(info, actual) || actual.Size() != ref.Size {
		return nil, errors.New("attachment changed while opening")
	}
	data := make([]byte, 0, int(ref.Size))
	buffer := make([]byte, 64*1024)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := f.Read(buffer)
		if int64(len(data)+n) > ref.Size {
			return nil, errors.New("attachment grew past declared size")
		}
		data = append(data, buffer[:n]...)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	digest := sha256.Sum256(data)
	if int64(len(data)) != ref.Size || hex.EncodeToString(digest[:]) != ref.SHA256 {
		return nil, errors.New("attachment integrity check failed")
	}
	return data, nil
}

func (s *Store) sync() error {
	dir, err := s.root.Open(".")
	if err != nil {
		return err
	}
	return errors.Join(dir.Sync(), dir.Close())
}

// Called under the cross-process writer lock. Pending files are leftovers of
// interrupted puts; published blobs are never removed by this cleanup.
func (s *Store) usage(ctx context.Context) (int64, error) {
	dir, err := s.root.Open(".")
	if err != nil {
		return 0, err
	}
	defer dir.Close()
	entries, err := dir.ReadDir(maxEntries + 1)
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, err
	}
	if len(entries) > maxEntries {
		return 0, errors.New("attachment directory entry limit exceeded")
	}
	var used int64
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		name := entry.Name()
		if name == ".lock" {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return 0, err
		}
		if !info.Mode().IsRegular() {
			return 0, errors.New("non-regular attachment store entry")
		}
		if strings.HasPrefix(name, ".pending-") {
			if err := s.root.Remove(name); err != nil {
				return 0, err
			}
			continue
		}
		if err := (Ref{SHA256: name, Size: info.Size()}).Validate(); err != nil {
			return 0, err
		}
		if info.Size() > s.limit-used {
			return 0, errors.New("attachment store exceeds quota")
		}
		used += info.Size()
	}
	return used, nil
}
