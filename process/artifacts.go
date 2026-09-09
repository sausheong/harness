package process

import (
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"hash"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const ArtifactFileLimit int64 = 8 << 20
const ArtifactStoreLimit int64 = 64 << 20
const ArtifactRetention = 24 * time.Hour

// ArtifactStore owns a private directory used only for command output. Each
// active capture reserves one full file allowance. Advisory file locks protect
// active files across cooperating processes; old completed files are evicted
// oldest first. Retention is enforced when a new artifact is allocated.
type ArtifactStore struct{ dir string }

func NewArtifactStore(dir string) (*ArtifactStore, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(abs, 0700); err != nil {
		return nil, err
	}
	info, err := os.Lstat(abs)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return nil, errors.New("artifact directory must be a private directory (0700)")
	}
	return &ArtifactStore{dir: abs}, nil
}

func (s *ArtifactStore) allocate() (*os.File, string, error) {
	root, err := os.OpenRoot(s.dir)
	if err != nil {
		return nil, "", err
	}
	defer root.Close()
	guard, err := root.OpenFile(".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, "", err
	}
	defer guard.Close()
	if err = lockArtifact(guard, false); err != nil {
		return nil, "", err
	}
	names, err := fs.ReadDir(root.FS(), ".")
	if err != nil {
		return nil, "", err
	}
	type candidate struct {
		name     string
		modified time.Time
	}
	var files []candidate
	for _, e := range names {
		if !strings.HasPrefix(e.Name(), "output-") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return nil, "", err
		}
		if !info.Mode().IsRegular() || info.Size() > ArtifactFileLimit {
			return nil, "", errors.New("invalid output artifact in store")
		}
		files = append(files, candidate{e.Name(), info.ModTime()})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modified.Before(files[j].modified) })
	count := len(files)
	for _, f := range files {
		expired := time.Since(f.modified) > ArtifactRetention
		if !expired && int64(count+1)*ArtifactFileLimit <= ArtifactStoreLimit {
			continue
		}
		file, err := root.OpenFile(f.name, os.O_RDWR, 0)
		if err != nil {
			return nil, "", err
		}
		err = lockArtifact(file, true)
		if artifactBusy(err) {
			file.Close()
			continue
		}
		if err != nil {
			file.Close()
			return nil, "", err
		}
		err = root.Remove(f.name)
		file.Close()
		if err != nil {
			return nil, "", err
		}
		count--
	}
	if int64(count+1)*ArtifactFileLimit > ArtifactStoreLimit {
		return nil, "", errors.New("output artifact quota occupied by active captures")
	}
	name := "output-" + rand.Text() + ".log"
	file, err := root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		return nil, "", err
	}
	if err = lockArtifact(file, false); err != nil {
		file.Close()
		root.Remove(name)
		return nil, "", err
	}
	return file, filepath.Join(s.dir, name), nil
}

// OutputCapture drains every byte, retains a bounded memory prefix, and spills
// output exceeding that prefix into an artifact with an independent disk cap.
// Call Close after the subprocess joins, before consuming Info. Write errors
// are recorded instead of stopping the drain and changing subprocess behaviour.
type OutputCapture struct {
	mu     sync.Mutex
	prefix *Capture
	store  *ArtifactStore
	file   *os.File
	info   ArtifactInfo
	closed bool
	digest hash.Hash
}

type ArtifactInfo struct {
	SHA256    string `json:"sha256,omitempty"`
	Path      string `json:"path,omitempty"`
	Bytes     int64  `json:"bytes"`
	Truncated bool   `json:"truncated"`
	Error     string `json:"error,omitempty"`
}

func (s *ArtifactStore) Capture(memoryLimit int) *OutputCapture {
	return &OutputCapture{store: s, prefix: NewCapture(memoryLimit), digest: sha256.New()}
}

func (c *OutputCapture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, os.ErrClosed
	}
	var before string
	if c.file == nil && c.info.Error == "" {
		_, retained, limit := c.prefix.stats()
		if len(p) > limit-retained {
			before, _, _ = c.prefix.Snapshot()
		}
	}
	c.prefix.Write(p)
	total, retained, _ := c.prefix.stats()
	truncated := total > int64(retained)
	if truncated && c.file == nil && c.info.Error == "" {
		f, path, err := c.store.allocate()
		if err != nil {
			c.info.Error = err.Error()
		} else {
			c.file = f
			c.info.Path = path
			c.writeDisk([]byte(before))
		}
	}
	if c.file != nil {
		c.writeDisk(p)
	}
	c.info.Truncated = truncated && (c.info.Bytes < total)
	return len(p), nil
}
func (c *OutputCapture) writeDisk(p []byte) {
	if c.info.Error != "" {
		return
	}
	keep := min(int64(len(p)), ArtifactFileLimit-c.info.Bytes)
	n, err := c.file.Write(p[:int(keep)])
	c.info.Bytes += int64(n)
	c.digest.Write(p[:n])
	if err == nil && int64(n) != keep {
		err = errors.New("short artifact write")
	}
	if err != nil {
		c.info.Error = fmt.Sprintf("capture output: %v", err)
	}
}
func (c *OutputCapture) Snapshot() (string, int64, bool) { return c.prefix.Snapshot() }
func (c *OutputCapture) Info() ArtifactInfo              { c.mu.Lock(); defer c.mu.Unlock(); return c.info }
func (c *OutputCapture) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.file == nil {
		return nil
	}
	err := errors.Join(c.file.Sync(), c.file.Close())
	c.info.SHA256 = fmt.Sprintf("%x", c.digest.Sum(nil))
	if err != nil {
		c.info.Error = err.Error()
	}
	return err
}
