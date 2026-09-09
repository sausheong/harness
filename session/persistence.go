package session

import (
	"fmt"
	"os"
)

type persistenceFailure struct{ err error }

// PersistenceError returns the first failure for this in-memory session. It is
// sticky: a later successful write cannot restore entries already lost. Recovery
// requires reconciling durable state and opening a new session instance.
func (s *Session) PersistenceError() error {
	if failure := s.persistenceFailure.Load(); failure != nil {
		return failure.err
	}
	return nil
}
func (store *Store) recordPersistenceFailure(s *Session, reason string, err error) {
	if err == nil {
		return
	}
	s.persistenceFailure.CompareAndSwap(nil, &persistenceFailure{fmt.Errorf("session persistence %s: %w", reason, err)})
	store.markDegraded(reason, err)
}

// Flush synchronizes the session file and its containing directory, surfacing
// write/close/sync failures. Call after joining append/compaction workers; Flush
// does not claim to join concurrent producers. In-memory sessions need no I/O.
// This does not implement writer leases, branch recovery or migration.
func (s *Session) Flush() error {
	if err := s.PersistenceError(); err != nil {
		return err
	}
	if s.store == nil {
		return nil
	}
	store := s.store
	store.mu.Lock()
	defer store.mu.Unlock()
	release, err := store.writeLease(s)
	if err != nil {
		store.recordPersistenceFailure(s, "acquire sync lease", err)
		return s.PersistenceError()
	}
	defer release()
	fail := func(reason string, err error) error {
		store.recordPersistenceFailure(s, reason, err)
		return s.PersistenceError()
	}
	if err := validateStoreIDs(s.AgentID, s.Key); err != nil {
		return fail("invalid session path", err)
	}
	f, err := os.OpenFile(store.sessionPath(s.AgentID, s.Key), os.O_WRONLY, 0)
	if err != nil {
		return fail("open for sync", err)
	}
	syncErr := f.Sync()
	closeErr := f.Close()
	if syncErr != nil {
		return fail("sync session file", syncErr)
	}
	if closeErr != nil {
		return fail("close synced session", closeErr)
	}
	dir, err := os.Open(store.sessionDir(s.AgentID))
	if err != nil {
		return fail("open session directory", err)
	}
	syncErr = dir.Sync()
	closeErr = dir.Close()
	if syncErr != nil {
		return fail("sync session directory", syncErr)
	}
	if closeErr != nil {
		return fail("close session directory", closeErr)
	}
	return s.PersistenceError()
}
