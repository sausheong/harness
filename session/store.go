package session

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SessionInfo describes a session without loading its full contents.
type SessionInfo struct {
	Key          string    `json:"key"`
	CreatedAt    time.Time `json:"createdAt"`
	LastActivity time.Time `json:"lastActivity"`
	EntryCount   int       `json:"entryCount"`
}

// Store handles JSONL file I/O for sessions.
type Store struct {
	baseDir  string
	mu       sync.Mutex
	degraded atomic.Bool
}

// markDegraded emits a single warning the first time session persistence
// fails, so a user notices that in-memory state may not survive a restart.
// Subsequent failures stay at their existing Error level to avoid log spam.
func (s *Store) markDegraded(reason string, err error) {
	if s.degraded.CompareAndSwap(false, true) {
		slog.Warn("session persistence degraded; in-memory state may not survive restart",
			"reason", reason, "error", err)
	}
}

// NewStore creates a new session store.
func NewStore(baseDir string) *Store {
	return &Store{baseDir: baseDir}
}

// sessionDir returns the directory for a given agent's sessions.
func (s *Store) sessionDir(agentID string) string {
	return filepath.Join(s.baseDir, agentID)
}

// sessionPath returns the file path for a session.
func (s *Store) sessionPath(agentID, key string) string {
	return filepath.Join(s.sessionDir(agentID), key+".jsonl")
}

// Load reads a session from its JSONL file.
func (s *Store) Load(agentID, key string) (*Session, error) {
	path := s.sessionPath(agentID, key)

	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			sess := NewSession(agentID, key)
			sess.SetStore(s)
			return sess, nil
		}
		return nil, fmt.Errorf("open session file: %w", err)
	}
	defer f.Close()

	sess := NewSession(agentID, key)
	sess.SetStore(s)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024) // 10MB max line

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var entry SessionEntry
		if err := json.Unmarshal(line, &entry); err != nil {
			slog.Warn("skipping malformed session entry", "error", err)
			continue
		}

		// Add to session without re-persisting
		sess.entries = append(sess.entries, entry)
		sess.entryMap[entry.ID] = &sess.entries[len(sess.entries)-1]
		sess.leafID = entry.ID
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read session file: %w", err)
	}

	return sess, nil
}

// AppendEntry writes a single entry to the session's JSONL file.
func (s *Store) AppendEntry(sess *Session, entry SessionEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := s.sessionDir(sess.AgentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		s.markDegraded("create session dir", err)
		slog.Error("failed to create session dir", "error", err)
		return
	}

	path := s.sessionPath(sess.AgentID, sess.Key)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		s.markDegraded("open session file", err)
		slog.Error("failed to open session file", "error", err)
		return
	}
	defer f.Close()

	data, err := json.Marshal(entry)
	if err != nil {
		slog.Error("failed to marshal session entry", "error", err)
		return
	}

	data = append(data, '\n')
	if _, err := f.Write(data); err != nil {
		s.markDegraded("write session entry", err)
		slog.Error("failed to write session entry", "error", err)
	}
}

// Create creates an empty session file on disk so it shows up in List.
func (s *Store) Create(agentID, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := s.sessionDir(agentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create session dir: %w", err)
	}

	path := s.sessionPath(agentID, key)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("create session file: %w", err)
	}
	return f.Close()
}

// List returns metadata for all sessions belonging to the given agent.
func (s *Store) List(agentID string) ([]SessionInfo, error) {
	dir := s.sessionDir(agentID)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read session dir: %w", err)
	}

	var sessions []SessionInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
			continue
		}

		key := strings.TrimSuffix(entry.Name(), ".jsonl")
		path := filepath.Join(dir, entry.Name())

		info := SessionInfo{Key: key}

		// Count lines and extract timestamps from first/last entries
		f, err := os.Open(path)
		if err != nil {
			continue
		}

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)
		var firstTS, lastTS int64
		lineCount := 0
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			lineCount++
			var partial struct {
				Timestamp int64 `json:"timestamp"`
			}
			if json.Unmarshal(line, &partial) == nil && partial.Timestamp > 0 {
				if firstTS == 0 {
					firstTS = partial.Timestamp
				}
				lastTS = partial.Timestamp
			}
		}
		f.Close()

		info.EntryCount = lineCount
		if firstTS > 0 {
			info.CreatedAt = time.Unix(firstTS, 0)
		} else {
			// Fall back to file modification time
			if fi, err := entry.Info(); err == nil {
				info.CreatedAt = fi.ModTime()
			}
		}
		if lastTS > 0 {
			info.LastActivity = time.Unix(lastTS, 0)
		} else {
			info.LastActivity = info.CreatedAt
		}

		sessions = append(sessions, info)
	}

	return sessions, nil
}

// Exists checks whether a session file exists for the given agent and key.
func (s *Store) Exists(agentID, key string) bool {
	path := s.sessionPath(agentID, key)
	_, err := os.Stat(path)
	return err == nil
}

// Rename renames a session file from oldKey to newKey.
func (s *Store) Rename(agentID, oldKey, newKey string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	oldPath := s.sessionPath(agentID, oldKey)
	newPath := s.sessionPath(agentID, newKey)

	if _, err := os.Stat(oldPath); os.IsNotExist(err) {
		return fmt.Errorf("session %q does not exist", oldKey)
	}
	if _, err := os.Stat(newPath); err == nil {
		return fmt.Errorf("session %q already exists", newKey)
	}

	return os.Rename(oldPath, newPath)
}

// Delete removes a session's JSONL file.
func (s *Store) Delete(agentID, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.sessionPath(agentID, key)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove session file: %w", err)
	}
	return nil
}

// Rewrite replaces the entire session JSONL file with the current entries.
// Used after compaction to replace the old file.
func (s *Store) Rewrite(sess *Session) {
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := s.sessionDir(sess.AgentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Error("failed to create session dir", "error", err)
		return
	}

	path := s.sessionPath(sess.AgentID, sess.Key)

	f, err := os.Create(path)
	if err != nil {
		slog.Error("failed to create session file for rewrite", "error", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, entry := range sess.Entries() {
		data, err := json.Marshal(entry)
		if err != nil {
			slog.Error("failed to marshal session entry", "error", err)
			continue
		}
		w.Write(data)
		w.WriteByte('\n')
	}

	if err := w.Flush(); err != nil {
		slog.Error("failed to flush session file", "error", err)
	}
}
