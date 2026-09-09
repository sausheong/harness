package session

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SessionInfo describes a session without loading its full contents.
type SessionInfo struct {
	ID           string    `json:"id,omitempty"`
	Key          string    `json:"key"`
	CreatedAt    time.Time `json:"createdAt"`
	LastActivity time.Time `json:"lastActivity"`
	EntryCount   int       `json:"entryCount"`
	// MetadataError means counts, identity and record timestamps are unavailable.
	// The key remains discoverable for explicit recovery. Listing is not graph validation.
	MetadataError string `json:"metadataError,omitempty"`
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

// validateStoreComponent keeps caller-provided identifiers as one filesystem
// path component. Empty values remain supported for backward compatibility,
// but separators, volume names, and dot traversal are rejected.
func validateStoreComponent(kind, value string) error {
	if value == "." || value == ".." || filepath.VolumeName(value) != "" ||
		strings.ContainsAny(value, `/\`) {
		return fmt.Errorf("invalid %s %q: must be a single path component", kind, value)
	}
	return nil
}

func validateStoreIDs(agentID, key string) error {
	if err := validateStoreComponent("agent ID", agentID); err != nil {
		return err
	}
	return validateStoreComponent("session key", key)
}

// Load reads a session from its JSONL file.
func (s *Store) Load(agentID, key string) (*Session, error) {
	return s.loadContext(context.Background(), agentID, key)
}

func (s *Store) loadContext(ctx context.Context, agentID, key string) (*Session, error) {
	if err := validateStoreIDs(agentID, key); err != nil {
		return nil, err
	}
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

	entries, err := decodeSessionRecords(&recoveryReader{ctx: ctx, reader: f})
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.Type == EntryTypeHeader {
			sess.ID = entry.ID
			sess.header = entry
			continue
		}
		sess.entries = append(sess.entries, entry)
		sess.entryMap[entry.ID] = &sess.entries[len(sess.entries)-1]
		if entry.Type == EntryTypeSelection {
			sess.leafID = entry.ParentID
		} else if entry.Type != EntryTypeAnnotation {
			sess.leafID = entry.ID
		}
	}

	return sess, nil
}

// AppendEntry writes a single entry to the session's JSONL file.
func (s *Store) AppendEntry(sess *Session, entry SessionEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	release, err := s.writeLease(sess)
	if err != nil {
		s.recordPersistenceFailure(sess, "acquire writer lease", err)
		return
	}
	defer release()

	if err := validateStoreIDs(sess.AgentID, sess.Key); err != nil {
		s.recordPersistenceFailure(sess, "invalid session path", err)
		slog.Error("refusing to persist session outside store", "error", err)
		return
	}

	dir := s.sessionDir(sess.AgentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		s.recordPersistenceFailure(sess, "create session dir", err)
		slog.Error("failed to create session dir", "error", err)
		return
	}

	path := s.sessionPath(sess.AgentID, sess.Key)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		s.recordPersistenceFailure(sess, "open session file", err)
		slog.Error("failed to open session file", "error", err)
		return
	}
	defer func() {
		if err := f.Close(); err != nil {
			s.recordPersistenceFailure(sess, "close session file", err)
		}
	}()

	data, err := json.Marshal(entry)
	if err != nil {
		s.recordPersistenceFailure(sess, "marshal session entry", err)
		slog.Error("failed to marshal session entry", "error", err)
		return
	}

	// A valid last record may lack its final newline. Delimit it before
	// appending, otherwise two valid JSON objects become one corrupt record.
	info, err := f.Stat()
	if err != nil {
		s.recordPersistenceFailure(sess, "stat before append", err)
		return
	}
	if info.Size() == 0 {
		header, err := json.Marshal(sess.header)
		if err != nil {
			s.recordPersistenceFailure(sess, "marshal header", err)
			return
		}
		data = append(append(header, '\n'), data...)
	}
	if info.Size() > 0 {
		var last [1]byte
		if _, err := f.ReadAt(last[:], info.Size()-1); err != nil {
			s.recordPersistenceFailure(sess, "read append boundary", err)
			return
		}
		if last[0] != '\n' {
			data = append([]byte{'\n'}, data...)
		}
	}
	data = append(data, '\n')
	if n, err := f.Write(data); err != nil || n != len(data) {
		if err == nil {
			err = io.ErrShortWrite
		}
		s.recordPersistenceFailure(sess, "write session entry", err)
		slog.Error("failed to write session entry", "error", err)
	}
}

// Create creates an empty session file on disk so it shows up in List.
func (s *Store) Create(agentID, key string) error {
	if err := validateStoreIDs(agentID, key); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lease, err := s.acquireOperationLease(agentID, key)
	if err != nil {
		return err
	}
	defer lease.close()

	dir := s.sessionDir(agentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create session dir: %w", err)
	}

	path := s.sessionPath(agentID, key)
	var encoded bytes.Buffer
	if err := json.NewEncoder(&encoded).Encode(NewSession(agentID, key).header); err != nil {
		return err
	}
	temporary, err := writeMigrationFile(context.Background(), dir, "."+key+".create-*", encoded.Bytes())
	if err != nil {
		return err
	}
	defer os.Remove(temporary)
	// Link commits a complete synced file without overwriting an existing key.
	if err := os.Link(temporary, path); err != nil {
		return fmt.Errorf("create session file: %w", err)
	}
	return syncRecoveryDirectory(dir)

}

// List returns metadata for all sessions belonging to the given agent.
func (s *Store) List(agentID string) ([]SessionInfo, error) {
	if err := validateStoreComponent("agent ID", agentID); err != nil {
		return nil, err
	}
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
		fi, statErr := entry.Info()
		if statErr != nil {
			info.MetadataError = statErr.Error()
		} else if !fi.Mode().IsRegular() {
			info.MetadataError = "session file is not regular"
		} else {
			info.CreatedAt = fi.ModTime()
			info.LastActivity = fi.ModTime()
			f, openErr := os.Open(path)
			if openErr != nil {
				info.MetadataError = openErr.Error()
			} else {
				metadata, scanErr := scanSessionMetadata(f, MaxLegacyImageRecordBytes)
				closeErr := f.Close()
				if scanErr == nil {
					scanErr = closeErr
				}
				if scanErr != nil {
					info.MetadataError = scanErr.Error()
				} else {
					info.ID, info.EntryCount = metadata.ID, metadata.EntryCount
					if !metadata.CreatedAt.IsZero() {
						info.CreatedAt = metadata.CreatedAt
						info.LastActivity = metadata.LastActivity
					}
				}
			}
		}

		sessions = append(sessions, info)
	}

	return sessions, nil
}

// scanSessionMetadata extracts complete syntactic metadata with bounded per-record
// memory. Legacy graph conversion and full format validation belong to loading.
func scanSessionMetadata(reader io.Reader, limit int) (SessionInfo, error) {
	var info SessionInfo
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, min(64*1024, limit)), limit+1)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if len(scanner.Bytes()) > limit {
			return SessionInfo{}, ErrSessionRecordTooLarge
		}
		var partial struct {
			Timestamp int64     `json:"timestamp"`
			Type      EntryType `json:"type"`
			ID        string    `json:"id"`
		}
		if err := json.Unmarshal(line, &partial); err != nil {
			return SessionInfo{}, fmt.Errorf("metadata line %d: %w", lineNumber, err)
		}
		if partial.Type == EntryTypeHeader {
			info.ID = partial.ID
		} else {
			info.EntryCount++
		}
		if partial.Timestamp > 0 {
			if info.CreatedAt.IsZero() {
				info.CreatedAt = time.Unix(partial.Timestamp, 0)
			}
			info.LastActivity = time.Unix(partial.Timestamp, 0)
		}
	}
	if err := scanner.Err(); err != nil {
		return SessionInfo{}, fmt.Errorf("scan session metadata: %w", err)
	}
	return info, nil
}

// Exists checks whether a session file exists for the given agent and key.
func (s *Store) Exists(agentID, key string) bool {
	if validateStoreIDs(agentID, key) != nil {
		return false
	}
	path := s.sessionPath(agentID, key)
	_, err := os.Stat(path)
	return err == nil
}

// Rename renames a session file from oldKey to newKey.
func (s *Store) Rename(agentID, oldKey, newKey string) error {
	if err := validateStoreIDs(agentID, oldKey); err != nil {
		return err
	}
	if err := validateStoreComponent("session key", newKey); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if oldKey == newKey {
		return fmt.Errorf("rename requires a different session key")
	}
	keys := []string{oldKey, newKey}
	slices.Sort(keys)
	first, err := s.acquireOperationLease(agentID, keys[0])
	if err != nil {
		return err
	}
	defer first.close()
	second, err := s.acquireOperationLease(agentID, keys[1])
	if err != nil {
		return err
	}
	defer second.close()

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
	if err := validateStoreIDs(agentID, key); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	lease, err := s.acquireOperationLease(agentID, key)
	if err != nil {
		return err
	}
	defer lease.close()

	path := s.sessionPath(agentID, key)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove session file: %w", err)
	}
	return nil
}

// Rewrite replaces the entire session JSONL file with the current entries.
// Used after compaction to replace the old file.
func (s *Store) Rewrite(sess *Session) {
	// Match Append's lock order, and take the snapshot before store.mu. Taking
	// a session read lock while holding store.mu can deadlock a queued append.
	sess.mu.Lock()
	sess.writeMu.Lock()
	entries := append([]SessionEntry(nil), sess.entries...)
	sess.mu.Unlock()
	defer sess.writeMu.Unlock()
	s.rewriteEntries(sess, entries)
}

func (s *Store) rewriteEntries(sess *Session, entries []SessionEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	release, err := s.writeLease(sess)
	if err != nil {
		s.recordPersistenceFailure(sess, "acquire writer lease", err)
		return
	}
	defer release()

	fail := func(reason string, err error) { s.recordPersistenceFailure(sess, reason, err) }
	if err := validateStoreIDs(sess.AgentID, sess.Key); err != nil {
		fail("invalid rewrite path", err)
		return
	}
	dir := s.sessionDir(sess.AgentID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		fail("create rewrite directory", err)
		return
	}
	f, err := os.CreateTemp(dir, "."+sess.Key+".rewrite-*")
	if err != nil {
		fail("create temporary rewrite", err)
		return
	}
	temporary := f.Name()
	defer os.Remove(temporary)
	defer f.Close()
	w := bufio.NewWriter(f)
	encoder := json.NewEncoder(w)
	if err := encoder.Encode(sess.header); err != nil {
		fail("encode session header", err)
		return
	}
	for _, entry := range entries {
		if err := encoder.Encode(entry); err != nil {
			fail("encode rewrite entry", err)
			return
		}
	}
	if err := w.Flush(); err != nil {
		fail("flush rewrite", err)
		return
	}
	if err := f.Sync(); err != nil {
		fail("sync rewrite", err)
		return
	}
	if err := f.Close(); err != nil {
		fail("close rewrite", err)
		return
	}
	if err := os.Rename(temporary, s.sessionPath(sess.AgentID, sess.Key)); err != nil {
		fail("commit rewrite", err)
		return
	}
	directory, err := os.Open(dir)
	if err != nil {
		fail("open rewrite directory", err)
		return
	}
	syncErr := directory.Sync()
	closeErr := directory.Close()
	if syncErr != nil {
		fail("sync rewrite directory", syncErr)
		return
	}
	if closeErr != nil {
		fail("close rewrite directory", closeErr)
	}
}
