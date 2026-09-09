package session

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

var ErrSessionBusy = errors.New("session already has an active writer")
var ErrSessionClosed = errors.New("session writer is closed")

type writerLease struct {
	file         *os.File
	store        *Store
	agentID, key string
}

func (l *writerLease) close() error {
	if l.file == nil {
		return nil
	}
	return l.file.Close()
}

// LoadExclusive acquires a kernel-managed writer lease before reading. The
// lease lasts until Session.Close or process exit. It never steals an active
// lease based on a PID/timestamp; a dead process's lock is released by the OS.
func (s *Store) LoadExclusive(agentID, key string) (*Session, error) {
	lease, err := s.acquireLease(agentID, key)
	if err != nil {
		return nil, err
	}
	sess, err := s.Load(agentID, key)
	if err != nil {
		lease.close()
		return nil, err
	}
	sess.lease = lease
	return sess, nil
}

func (s *Store) acquireLease(agentID, key string) (*writerLease, error) {
	if err := validateStoreIDs(agentID, key); err != nil {
		return nil, err
	}
	dir := filepath.Join(s.sessionDir(agentID), ".leases")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	f, err := openLeaseFile(filepath.Join(dir, key+".lock"))
	if err != nil {
		return nil, err
	}
	if err := lockLeaseFile(f); err != nil {
		f.Close()
		if leaseWouldBlock(err) {
			return nil, fmt.Errorf("%w: %s/%s", ErrSessionBusy, agentID, key)
		}
		return nil, err
	}
	// Metadata is advisory only; stale contents never override the kernel lock.
	if err := f.Truncate(0); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := fmt.Fprintf(f, "pid=%d\n", os.Getpid()); err != nil {
		f.Close()
		return nil, err
	}
	return &writerLease{file: f, store: s, agentID: agentID, key: key}, nil
}

// writeLease is called under Store.mu. Legacy callers without a lifetime lease
// acquire a temporary operation lease, so they cannot bypass an exclusive owner.
// Two legacy callers are not a safe DAG writer protocol: use LoadExclusive.
func (s *Store) writeLease(sess *Session) (func(), error) {
	sess.leaseMu.Lock()
	defer sess.leaseMu.Unlock()
	if sess.writerClosed {
		return nil, ErrSessionClosed
	}
	if sess.lease != nil {
		if sess.lease.store != s || sess.lease.agentID != sess.AgentID || sess.lease.key != sess.Key {
			return nil, errors.New("session identity no longer matches its writer lease")
		}
		return func() {}, nil
	}
	lease, err := s.acquireOperationLease(sess.AgentID, sess.Key)
	if err != nil {
		return nil, err
	}
	return func() { lease.close() }, nil
}

// Close joins synchronous writes before releasing the lease. Further persistent
// writes through this instance fail. Callers must also stop their own producers.
func (s *Session) Close() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.store != nil {
		s.store.mu.Lock()
		defer s.store.mu.Unlock()
	}
	s.leaseMu.Lock()
	defer s.leaseMu.Unlock()
	if s.writerClosed {
		return nil
	}
	s.writerClosed = true
	if s.lease != nil {
		err := s.lease.close()
		s.lease = nil
		return err
	}
	return nil
}

// Legacy operations retain their pre-lease behaviour on unsupported platforms.
// LoadExclusive always fails there rather than promising an unavailable lease.
func (s *Store) acquireOperationLease(agentID, key string) (*writerLease, error) {
	if !supportsWriterLease {
		return &writerLease{}, nil
	}
	return s.acquireLease(agentID, key)
}
