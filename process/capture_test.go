package process

import (
	"io"
	"strings"
	"sync"
	"testing"
)

func TestCaptureCopyBound(t *testing.T) {
	c := NewCapture(17)
	n, err := io.Copy(c, strings.NewReader(strings.Repeat("x", 1<<20)))
	prefix, total, truncated := c.Snapshot()
	if err != nil || n != 1<<20 || total != n || prefix != strings.Repeat("x", 17) || !truncated {
		t.Fatalf("copy=%d err=%v prefix=%q total=%d truncated=%v", n, err, prefix, total, truncated)
	}
}

func TestCaptureConcurrentAndEmpty(t *testing.T) {
	var c Capture
	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() {
			for range 100 {
				_, _ = c.Write([]byte("abc"))
				c.Snapshot()
			}
		})
	}
	wg.Wait()
	prefix, total, truncated := c.Snapshot()
	if prefix != "" || total != 3000 || !truncated {
		t.Fatalf("%q %d %v", prefix, total, truncated)
	}
	empty := NewCapture(10)
	_, total, truncated = empty.Snapshot()
	if total != 0 || truncated {
		t.Fatal("empty capture marked truncated")
	}
}
