package session

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
)

type MigrationReport struct {
	LegacyConversion
	TailRecovered      bool
	DiscardedTailBytes int64
	ExternalizedImages int
	FormatUpgraded     bool
	BackupPath         string
	Migrated           bool // replacement occurred; a later sync error still requires reconciliation
}

// MigrateLegacy converts historical duplicate-ID compaction logs in place,
// holding the lifetime writer lease. It preserves a private, synced original
// backup before atomic replacement. Legacy logs receive a versioned identity header. Already-versioned valid logs without inline images remain byte-identical.
// The returned session owns the lease until Close. This is a log-format
// migration primitive, not a workspace/session catalogue importer.
func (s *Store) MigrateLegacy(ctx context.Context, agentID, key string) (*Session, MigrationReport, error) {
	return s.migrateLegacy(ctx, agentID, key, syncRecoveryDirectory)
}

// MigrateRecovering explicitly permits removing a syntactically unfinished
// final record. The complete original is backed up and the retained prefix
// must pass migration and graph validation under one lifetime writer lease.
func (s *Store) MigrateRecovering(ctx context.Context, agentID, key string) (*Session, MigrationReport, error) {
	return s.migrateLegacyWithRecovery(ctx, agentID, key, syncRecoveryDirectory, true)
}

func (s *Store) migrateLegacy(ctx context.Context, agentID, key string, syncDirectory func(string) error) (*Session, MigrationReport, error) {
	return s.migrateLegacyWithRecovery(ctx, agentID, key, syncDirectory, false)
}

func (s *Store) migrateLegacyWithRecovery(ctx context.Context, agentID, key string, syncDirectory func(string) error, recoverTail bool) (*Session, MigrationReport, error) {
	var report MigrationReport
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
	if err != nil {
		return nil, report, err
	}
	if !info.Mode().IsRegular() {
		return nil, report, errors.New("migration requires a regular session file")
	}
	source, err := os.Open(path)
	if err != nil {
		return nil, report, err
	}
	raw, err := io.ReadAll(io.LimitReader(&recoveryReader{ctx: ctx, reader: source}, MaxLegacyImportBytes+1))
	err = errors.Join(err, source.Close())
	if err != nil {
		return nil, report, err
	}
	if int64(len(raw)) > MaxLegacyImportBytes {
		return nil, report, errors.New("legacy session exceeds import size limit")
	}
	normalized, imageCount, err := s.externalizeLegacyImages(ctx, agentID, raw)
	report.ExternalizedImages = imageCount
	var recoveredTailBytes int64
	if err != nil && recoverTail {
		var record *RecordError
		if errors.As(err, &record) && record.RecoverableTail && record.Offset >= 0 && record.Offset < int64(len(raw)) {
			recoveredTailBytes = int64(len(raw)) - record.Offset
			err = nil
		}
	}
	if err != nil {
		return nil, report, err
	}
	converted, conversion, err := ConvertLegacySession(&recoveryReader{ctx: ctx, reader: bytes.NewReader(normalized)})
	originalHash := sha256.Sum256(raw)
	conversion.SourceSHA256 = hex.EncodeToString(originalHash[:])
	report.LegacyConversion = conversion
	if err != nil {
		return nil, report, err
	}
	if err := ctx.Err(); err != nil {
		return nil, report, err
	}
	entries, err := decodeSessionRecords(bytes.NewReader(converted))
	if err != nil {
		return nil, report, err
	}
	if len(entries) == 0 || entries[0].Type != EntryTypeHeader {
		header := NewSession(agentID, key).header
		if len(entries) > 0 && entries[0].Timestamp > 0 {
			header.Timestamp = entries[0].Timestamp
		} else {
			header.Timestamp = info.ModTime().Unix()
		}
		encoded, err := json.Marshal(header)
		if err != nil {
			return nil, report, err
		}
		converted = append(append(encoded, '\n'), converted...)
		if _, err := decodeSessionRecords(bytes.NewReader(converted)); err != nil {
			return nil, report, err
		}
		report.FormatUpgraded = true
	}
	if conversion.RemappedIDs > 0 || report.FormatUpgraded || report.ExternalizedImages > 0 || recoveredTailBytes > 0 {
		dir := filepath.Dir(path)
		backup, err := writeMigrationFile(ctx, dir, "."+key+".migration-backup-*", raw)
		if err != nil {
			return nil, report, err
		}
		if err := syncDirectory(dir); err != nil {
			os.Remove(backup)
			return nil, report, err
		}
		report.BackupPath = backup
		replacement, err := writeMigrationFile(ctx, dir, "."+key+".migration-replace-*", converted)
		if err != nil {
			return nil, report, err
		}
		defer os.Remove(replacement)
		if err := ctx.Err(); err != nil {
			return nil, report, err
		}
		if err := os.Rename(replacement, path); err != nil {
			return nil, report, err
		}
		report.Migrated = true
		report.TailRecovered = recoveredTailBytes > 0
		report.DiscardedTailBytes = recoveredTailBytes
		if err := syncDirectory(dir); err != nil {
			return nil, report, err
		}
	}
	sess, err := s.loadContext(ctx, agentID, key)
	if err != nil {
		return nil, report, err
	}
	sess.lease = lease
	keep = true
	return sess, report, nil
}

func writeMigrationFile(ctx context.Context, dir, pattern string, raw []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return "", err
	}
	path := file.Name()
	complete := false
	defer func() {
		file.Close()
		if !complete {
			os.Remove(path)
		}
	}()
	if _, err := io.Copy(file, &recoveryReader{ctx: ctx, reader: bytes.NewReader(raw)}); err != nil {
		return "", err
	}
	if err := file.Sync(); err != nil {
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	complete = true
	return path, nil
}
