package process

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestOutputCaptureFloodAllocationsPlateau(t *testing.T) {
	store, err := NewArtifactStore(filepath.Join(t.TempDir(), "output"))
	if err != nil {
		t.Fatal(err)
	}
	capture := store.Capture(64 << 10)
	defer capture.Close()
	block := bytes.Repeat([]byte("x"), 32<<10)
	for i := 0; i < 300; i++ {
		if _, err = capture.Write(block); err != nil {
			t.Fatal(err)
		}
	}
	if info := capture.Info(); info.Bytes != ArtifactFileLimit || !info.Truncated {
		t.Fatalf("disk not saturated: %+v", info)
	}
	allocations := testing.AllocsPerRun(100, func() {
		if _, err := capture.Write(block); err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("saturated capture allocates per write: %f", allocations)
	}
	prefix, total, truncated := capture.Snapshot()
	if len(prefix) != 64<<10 || total <= 300*int64(len(block)) || !truncated {
		t.Fatal("capture accounting changed")
	}
	t.Logf("saturated_write_allocations=%f retained_prefix_bytes=%d backing_capacity_bytes=%d", allocations, len(prefix), cap(capture.prefix.data))
	if cap(capture.prefix.data) > 2*(64<<10)+8192 {
		t.Fatal("capture backing storage exceeded allowance")
	}
}
