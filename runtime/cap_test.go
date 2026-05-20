package runtime

import (
	"testing"
)

// TestMaxToolResultLen_Precedence locks in the config > env > default
// precedence so a regression here doesn't silently swap one source for
// another.
func TestMaxToolResultLen_Precedence(t *testing.T) {
	t.Run("default when nothing set", func(t *testing.T) {
		t.Setenv("HARNESS_MAX_TOOL_RESULT_LEN", "")
		r := &Runtime{}
		if got := r.maxToolResultLen(); got != defaultMaxToolResultLen {
			t.Fatalf("want %d, got %d", defaultMaxToolResultLen, got)
		}
	})

	t.Run("env overrides default", func(t *testing.T) {
		t.Setenv("HARNESS_MAX_TOOL_RESULT_LEN", "12345")
		r := &Runtime{}
		if got := r.maxToolResultLen(); got != 12345 {
			t.Fatalf("want 12345, got %d", got)
		}
	})

	t.Run("config overrides env", func(t *testing.T) {
		t.Setenv("HARNESS_MAX_TOOL_RESULT_LEN", "12345")
		r := &Runtime{AgentLoop: LoopConfig{MaxToolResultLen: 25000}}
		if got := r.maxToolResultLen(); got != 25000 {
			t.Fatalf("want 25000, got %d", got)
		}
	})

	t.Run("zero env is ignored", func(t *testing.T) {
		t.Setenv("HARNESS_MAX_TOOL_RESULT_LEN", "0")
		r := &Runtime{}
		if got := r.maxToolResultLen(); got != defaultMaxToolResultLen {
			t.Fatalf("want %d, got %d", defaultMaxToolResultLen, got)
		}
	})

	t.Run("garbage env is ignored", func(t *testing.T) {
		t.Setenv("HARNESS_MAX_TOOL_RESULT_LEN", "not-a-number")
		r := &Runtime{}
		if got := r.maxToolResultLen(); got != defaultMaxToolResultLen {
			t.Fatalf("want %d, got %d", defaultMaxToolResultLen, got)
		}
	})
}
