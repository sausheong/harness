package runtime

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// SpillDirForSession returns the absolute path of the spill subdirectory
// owned by (workspace, sessionKey). Returns "" if either input is empty,
// because we never want to accidentally walk a non-spill location.
//
// Mirrors the path layout used by spillToolResult in context.go:
//
//	<workspace>/.harness/spill/<sessionKey>/
func SpillDirForSession(workspace, sessionKey string) string {
	if workspace == "" || sessionKey == "" {
		return ""
	}
	return filepath.Join(workspace, ".harness", "spill", sessionKey)
}

// RemoveSessionSpill removes the per-session spill directory if it exists.
// Safe to call when the directory was never created (no spillover happened).
// Logs at warn on unexpected I/O errors but never returns one — spill cleanup
// is best-effort and must not block session deletion.
func RemoveSessionSpill(workspace, sessionKey string) {
	dir := SpillDirForSession(workspace, sessionKey)
	if dir == "" {
		return
	}
	if err := os.RemoveAll(dir); err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.Warn("spill cleanup failed", "dir", dir, "error", err)
	}
}

// SpillRoot returns the parent directory under which all per-session spill
// directories live for the given workspace. Used by the startup janitor.
func SpillRoot(workspace string) string {
	if workspace == "" {
		return ""
	}
	return filepath.Join(workspace, ".harness", "spill")
}

// LiveSessionKeysFn returns the set of currently-existing session keys for
// the agent that owns the given workspace. The startup janitor uses this
// callback to decide which spill dirs are orphans (have no matching
// session JSONL on disk).
type LiveSessionKeysFn func() (map[string]bool, error)

// CleanupOrphanedSpills walks <workspace>/.harness/spill/ and removes any
// per-session subdirectory whose key is not in the set returned by liveKeys.
// Returns the number of directories removed and the first error encountered
// (if any). Missing root is not an error — it just means nothing to clean.
//
// Designed to be called once at startup per agent workspace.
func CleanupOrphanedSpills(workspace string, liveKeys LiveSessionKeysFn) (int, error) {
	root := SpillRoot(workspace)
	if root == "" {
		return 0, nil
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	live, err := liveKeys()
	if err != nil {
		return 0, err
	}
	removed := 0
	var firstErr error
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		// Defensive: skip anything that doesn't look like a session-key dir
		// (e.g., a stray dotfile someone dropped here manually).
		name := e.Name()
		if name == "" || strings.HasPrefix(name, ".") {
			continue
		}
		if live[name] {
			continue
		}
		full := filepath.Join(root, name)
		if rerr := os.RemoveAll(full); rerr != nil {
			if firstErr == nil {
				firstErr = rerr
			}
			slog.Warn("orphan spill cleanup failed", "dir", full, "error", rerr)
			continue
		}
		removed++
	}
	if removed > 0 {
		slog.Info("removed orphan spill directories", "workspace", workspace, "count", removed)
	}
	return removed, firstErr
}
