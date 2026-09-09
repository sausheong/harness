package session

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

const MaxSessionRecordBytes = 10 << 20

var ErrSessionRecordTooLarge = errors.New("session record exceeds size limit")

// RecordError locates corruption without including session content in the
// message. Only an unfinished JSON object at physical EOF is a candidate for
// tail recovery; newline-terminated or interior corruption must be rejected.
type RecordError struct {
	Line            int
	Offset          int64
	RecoverableTail bool
	Cause           error
}

func (e *RecordError) Error() string {
	return fmt.Sprintf("invalid session record at line %d, byte %d (recoverable tail: %t): %v", e.Line, e.Offset, e.RecoverableTail, e.Cause)
}
func (e *RecordError) Unwrap() error { return e.Cause }

func readSessionRecord(r *bufio.Reader) ([]byte, bool, error) {
	var record []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(record)+len(part) > MaxSessionRecordBytes {
			return nil, false, ErrSessionRecordTooLarge
		}
		record = append(record, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF {
			return record, true, nil
		}
		return record, false, err
	}
}

func decodeSessionRecords(reader io.Reader) ([]SessionEntry, error) {
	r := bufio.NewReaderSize(reader, 64<<10)
	var entries []SessionEntry
	seen := make(map[string]EntryType)
	lookup := func(id string) (EntryType, bool) { kind, ok := seen[id]; return kind, ok }
	var offset int64
	for number := 1; ; number++ {
		line, final, err := readSessionRecord(r)
		if err != nil {
			return nil, &RecordError{Line: number, Offset: offset, Cause: err}
		}
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) > 0 {
			entry, err := decodeSessionRecord(trimmed, final, number, offset)
			if err != nil {
				return nil, err
			}
			if entry.Type == EntryTypeHeader && len(entries) != 0 {
				return nil, &RecordError{Line: number, Offset: offset, Cause: errors.New("session header must be the first record")}
			}
			if err := validateGraphNode(entry, lookup); err != nil {
				return nil, &RecordError{Line: number, Offset: offset, Cause: err}
			}
			seen[entry.ID] = entry.Type
			entries = append(entries, entry)
		}
		offset += int64(len(line))
		if final {
			return entries, nil
		}
	}
}

// decodeSessionRecord validates one nonblank record without graph assumptions.
func decodeSessionRecord(trimmed []byte, final bool, number int, offset int64) (SessionEntry, error) {
	entry := SessionEntry{}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	err := decoder.Decode(&entry)
	if !utf8.Valid(trimmed) {
		err = errors.New("record is not valid UTF-8")
	}
	if err != nil {
		return SessionEntry{}, &RecordError{Line: number, Offset: offset, RecoverableTail: final && trimmed[0] == '{' && errors.Is(err, io.ErrUnexpectedEOF), Cause: err}
	}
	if trimmed[0] != '{' {
		return SessionEntry{}, &RecordError{Line: number, Offset: offset, Cause: errors.New("record must be a JSON object")}
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return SessionEntry{}, &RecordError{Line: number, Offset: offset, Cause: errors.New("record contains trailing JSON data")}
	}
	if err := validateRecordFields(trimmed); err != nil {
		return SessionEntry{}, &RecordError{Line: number, Offset: offset, Cause: err}
	}

	return entry, nil
}

// Unknown extension fields remain available to the legacy converter. Known
// graph fields must use their canonical names, and no field may be duplicated.
// Nested data belongs to its own schema (and may contain opaque tool input).
func validateRecordFields(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	if _, err := d.Token(); err != nil {
		return err
	}
	seen := make(map[string]bool)
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok || seen[key] {
			return errors.New("duplicate or invalid session record field")
		}
		seen[key] = true
		for _, canonical := range []string{"schemaVersion", "id", "parentId", "type", "role", "timestamp", "data"} {
			if key != canonical && strings.EqualFold(key, canonical) {
				return errors.New("noncanonical session record field")
			}
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return err
		}
	}
	return nil
}
