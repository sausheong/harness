package session

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// RecoveryReport describes a repair. BackupPath is populated as soon as the
// complete original is durably backed up, including when a later step fails.
// Repaired means the atomic replacement occurred; a later sync error still
// returns an error and must not be treated as successful durable recovery.
type RecoveryReport struct {
	BackupPath   string
	DroppedBytes int64
	Repaired     bool
}

// LoadRecovering holds an exclusive writer lease while validating and, only
// for an incomplete final JSON object, repairing the session. Valid records
// are preserved byte for byte, including unknown fields. The original is
// synced to a private backup before replacement. The caller owns the returned
// session's lease and must Close it. Ordinary Load never performs repairs.
func (s *Store) LoadRecovering(ctx context.Context, agentID, key string) (*Session, RecoveryReport, error) {
	return s.loadRecovering(ctx, agentID, key, syncRecoveryDirectory)
}

func (s *Store) loadRecovering(ctx context.Context, agentID, key string, syncDirectory func(string) error) (*Session, RecoveryReport, error) {
	var report RecoveryReport
	if err := ctx.Err(); err != nil {
		return nil, report, err
	}
	lease, err := s.acquireLease(agentID, key)
	if err != nil {
		return nil, report, err
	}
	keep := false
	defer func() {
		if !keep {
			lease.close()
		}
	}()
	path := s.sessionPath(agentID, key)
	info, err := os.Lstat(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, report, err
	}
	if err == nil && !info.Mode().IsRegular() {
		return nil, report, errors.New("recovery requires a regular session file")
	}
	sess, err := s.loadContext(ctx, agentID, key)
	if err == nil {
		if err = ctx.Err(); err != nil {
			return nil, report, err
		}
		sess.lease = lease
		keep = true
		return sess, report, nil
	}
	var record *RecordError
	if !errors.As(err, &record) || !record.RecoverableTail {
		return nil, report, err
	}
	source, err := os.Open(path)
	if err != nil {
		return nil, report, err
	}
	defer source.Close()
	// Validate again on the descriptor which will be backed up and copied.
	_, err = decodeSessionRecords(&recoveryReader{ctx: ctx, reader: source})
	if !errors.As(err, &record) || !record.RecoverableTail {
		if err == nil {
			err = errors.New("source is no longer an incomplete tail")
		}
		return nil, report, fmt.Errorf("revalidate recovery source: %w", err)
	}
	size, err := source.Stat()
	if err != nil {
		return nil, report, err
	}
	report.DroppedBytes = size.Size() - record.Offset
	if report.DroppedBytes <= 0 {
		return nil, report, errors.New("recovery source changed during validation")
	}
	dir := filepath.Dir(path)
	backup, err := os.CreateTemp(dir, "."+key+".recovery-backup-*")
	if err != nil {
		return nil, report, err
	}
	backupPath := backup.Name()
	backupComplete := false
	defer func() {
		backup.Close()
		if !backupComplete {
			os.Remove(backupPath)
		}
	}()
	if _, err = source.Seek(0, io.SeekStart); err != nil {
		return nil, report, err
	}
	if _, err = io.Copy(backup, &recoveryReader{ctx: ctx, reader: source}); err != nil {
		return nil, report, err
	}
	if err = backup.Sync(); err != nil {
		return nil, report, err
	}
	if err = backup.Close(); err != nil {
		return nil, report, err
	}
	if err = syncDirectory(dir); err != nil {
		return nil, report, err
	}
	backupComplete = true
	report.BackupPath = backupPath
	replacement, err := os.CreateTemp(dir, "."+key+".recovery-replace-*")
	if err != nil {
		return nil, report, err
	}
	defer func() { replacement.Close(); os.Remove(replacement.Name()) }()
	if _, err = source.Seek(0, io.SeekStart); err != nil {
		return nil, report, err
	}
	if _, err = io.CopyN(replacement, &recoveryReader{ctx: ctx, reader: source}, record.Offset); err != nil {
		return nil, report, err
	}
	if err = replacement.Sync(); err != nil {
		return nil, report, err
	}
	if err = replacement.Close(); err != nil {
		return nil, report, err
	}
	if err = ctx.Err(); err != nil {
		return nil, report, err
	}
	if err = os.Rename(replacement.Name(), path); err != nil {
		return nil, report, err
	}
	report.Repaired = true
	if err = syncDirectory(dir); err != nil {
		return nil, report, err
	}
	sess, err = s.Load(agentID, key)
	if err != nil {
		return nil, report, err
	}
	sess.lease = lease
	keep = true
	return sess, report, nil
}

func syncRecoveryDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	err = dir.Sync()
	return errors.Join(err, dir.Close())
}

type recoveryReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r *recoveryReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}
