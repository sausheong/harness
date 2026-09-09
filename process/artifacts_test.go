//go:build unix

package process

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testStore(t *testing.T) *ArtifactStore {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "output")
	s, err := NewArtifactStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	return s
}
func TestArtifactSpillExactAndCap(t *testing.T) {
	s := testStore(t)
	c := s.Capture(7)
	for _, p := range []string{"abc", "def", "ghij", "klmnop"} {
		c.Write([]byte(p))
	}
	c.Close()
	info := c.Info()
	raw, err := os.ReadFile(info.Path)
	if err != nil || string(raw) != "abcdefghijklmnop" || info.Bytes != 16 || info.Truncated || info.Error != "" {
		t.Fatalf("%q %+v %v", raw, info, err)
	}
	stat, _ := os.Stat(info.Path)
	if stat.Mode().Perm() != 0600 {
		t.Fatal(stat.Mode())
	}
	capped := s.Capture(4)
	block := bytes.Repeat([]byte("z"), 1<<20)
	for range 10 {
		capped.Write(block)
	}
	capped.Close()
	info = capped.Info()
	stat, err = os.Stat(info.Path)
	if err != nil || stat.Size() != ArtifactFileLimit || info.Bytes != ArtifactFileLimit || !info.Truncated {
		t.Fatalf("%+v %v", info, err)
	}
	prefix, total, truncated := capped.Snapshot()
	if prefix != "zzzz" || total != 10<<20 || !truncated {
		t.Fatalf("%q %d %v", prefix, total, truncated)
	}
}
func TestArtifactQuotaProtectsActiveAndEvictsCompleted(t *testing.T) {
	s := testStore(t)
	var active []*OutputCapture
	for range 8 {
		c := s.Capture(1)
		c.Write([]byte("abc"))
		if c.Info().Error != "" {
			t.Fatal(c.Info())
		}
		active = append(active, c)
		defer c.Close()
	}
	// A separate store instance must obey the same active-file reservations.
	other, err := NewArtifactStore(s.dir)
	if err != nil {
		t.Fatal(err)
	}
	full := other.Capture(1)
	full.Write([]byte("abc"))
	full.Close()
	if full.Info().Error == "" || !full.Info().Truncated || full.Info().Path != "" {
		t.Fatal(full.Info())
	}
	old := active[0].Info().Path
	active[0].Close()
	next := other.Capture(1)
	next.Write([]byte("abc"))
	defer next.Close()
	if next.Info().Error != "" {
		t.Fatal(next.Info())
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("completed artifact not evicted: %v", err)
	}
	for _, c := range active[1:] {
		if _, err := os.Stat(c.Info().Path); err != nil {
			t.Fatal("active artifact evicted", err)
		}
	}
}
func TestArtifactRetentionAndPrivateDirectory(t *testing.T) {
	s := testStore(t)
	c := s.Capture(1)
	c.Write([]byte("abc"))
	c.Close()
	old := c.Info().Path
	expired := time.Now().Add(-ArtifactRetention - time.Hour)
	if err := os.Chtimes(old, expired, expired); err != nil {
		t.Fatal(err)
	}
	next := s.Capture(1)
	next.Write([]byte("abc"))
	next.Close()
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("expired artifact remains: %v", err)
	}
	small := s.Capture(10)
	small.Write([]byte("abc"))
	small.Close()
	if small.Info().Path != "" || small.Info().Truncated {
		t.Fatal(small.Info())
	}
	if _, err := small.Write([]byte("x")); err == nil {
		t.Fatal("write after close succeeded")
	}
	if err := os.Chmod(s.dir, 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := NewArtifactStore(s.dir); err == nil {
		t.Fatal("public directory accepted")
	}
}

func TestArtifactCrossProcessReservation(t *testing.T) {
	if dir := os.Getenv("HARNESS_ARTIFACT_QUOTA_FIXTURE"); dir != "" {
		s, err := NewArtifactStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		c := s.Capture(1)
		c.Write([]byte("abc"))
		c.Close()
		if c.Info().Error == "" || c.Info().Path != "" {
			t.Fatal("child bypassed active reservations", c.Info())
		}
		return
	}
	s := testStore(t)
	for range 8 {
		c := s.Capture(1)
		c.Write([]byte("abc"))
		defer c.Close()
		if c.Info().Error != "" {
			t.Fatal(c.Info())
		}
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := Command(ctx, exe, "-test.run=^TestArtifactCrossProcessReservation$")
	cmd.Env = append(os.Environ(), "HARNESS_ARTIFACT_QUOTA_FIXTURE="+s.dir, "GORACE=atexit_sleep_ms=0")
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := Run(cmd); err != nil {
		t.Fatalf("child quota check: %v %s", err, output.String())
	}
}

func TestArtifactAllocationFailureStillDrains(t *testing.T) {
	s := testStore(t)
	if err := os.Mkdir(filepath.Join(s.dir, ".lock"), 0700); err != nil {
		t.Fatal(err)
	}
	c := s.Capture(2)
	n, err := c.Write([]byte("abcdef"))
	c.Close()
	if n != 6 || err != nil {
		t.Fatalf("drain: %d %v", n, err)
	}
	prefix, total, truncated := c.Snapshot()
	if prefix != "ab" || total != 6 || !truncated || c.Info().Error == "" || !c.Info().Truncated {
		t.Fatalf("%q %d %v %+v", prefix, total, truncated, c.Info())
	}
}
