package process

import "sync"

// Capture retains a bounded prefix while draining all writes. It deliberately
// does not embed a buffer: io.Copy must not bypass Write through ReaderFrom.
// The zero value drains without retaining bytes.
type Capture struct {
	mu    sync.Mutex
	limit int
	data  []byte
	total int64
}

func NewCapture(limit int) *Capture {
	if limit < 0 {
		limit = 0
	}
	return &Capture{limit: limit}
}

func (c *Capture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := len(p)
	c.total += int64(n)
	keep := min(n, c.limit-len(c.data))
	c.data = append(c.data, p[:keep]...)
	return n, nil
}

// Snapshot returns an independent prefix, observed byte count and truncation.
func (c *Capture) Snapshot() (string, int64, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return string(c.data), c.total, c.total > int64(len(c.data))
}

// stats inspects accounting without allocating a copy of the retained bytes.
func (c *Capture) stats() (total int64, retained, limit int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.total, len(c.data), c.limit
}
