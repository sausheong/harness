package session

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
)

// MaxLegacyImportBytes bounds a single legacy conversion. Larger sessions need
// an explicit migration policy instead of unbounded allocation.
const MaxLegacyImportBytes int64 = 128 << 20

const MaxLegacyImportRecords = 100000

// LegacyConversion describes a deterministic conversion, not a durable import.
// The caller must back up the original before installing the returned bytes.
type LegacyConversion struct {
	SourceSHA256 string
	Records      int
	RemappedIDs  int
}

// ConvertLegacySession validates a complete legacy log and converts the old
// compactor's repeated nodes to unique graph IDs. It never edits the source.
// Duplicate nodes are accepted only when content is identical and the duplicate
// continues a compaction clone chain. Other duplicate IDs remain corruption.
// Output preserves unknown JSON fields and all original node versions. Input
// without duplicates is returned byte-for-byte. A truncated tail is an error;
// recovery and migration must remain separately reported operations.
func ConvertLegacySession(reader io.Reader) ([]byte, LegacyConversion, error) {
	var report LegacyConversion
	raw, err := io.ReadAll(io.LimitReader(reader, MaxLegacyImportBytes+1))
	if err != nil {
		return nil, report, err
	}
	if int64(len(raw)) > MaxLegacyImportBytes {
		return nil, report, errors.New("legacy session exceeds import size limit")
	}
	digest := sha256.Sum256(raw)
	report.SourceSHA256 = hex.EncodeToString(digest[:])
	type record struct {
		entry  SessionEntry
		fields map[string]json.RawMessage
	}
	var records []record
	originalIDs := make(map[string]bool)
	r := bufio.NewReaderSize(bytes.NewReader(raw), 64<<10)
	var offset int64
	for lineNumber := 1; ; lineNumber++ {
		line, final, err := readSessionRecord(r)
		if err != nil {
			return nil, report, &RecordError{Line: lineNumber, Offset: offset, Cause: err}
		}
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) > 0 {
			entry, err := decodeSessionRecord(trimmed, final, lineNumber, offset)
			if err != nil {
				return nil, report, err
			}
			if entry.ID == "" {
				return nil, report, errors.New("legacy entry ID is empty")
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(trimmed, &fields); err != nil {
				return nil, report, err
			}
			if len(records) >= MaxLegacyImportRecords {
				return nil, report, errors.New("legacy session exceeds import record limit")
			}
			records = append(records, record{entry, fields})
			originalIDs[entry.ID] = true
		}
		offset += int64(len(line))
		if final {
			break
		}
	}
	report.Records = len(records)
	latest := make(map[string]string)
	first := make(map[string]record)
	seen := make(map[string]EntryType)
	lookup := func(id string) (EntryType, bool) { kind, ok := seen[id]; return kind, ok }
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	cloneChain := false
	previousOriginal := ""
	previousConverted := ""
	for index, rec := range records {
		entry := rec.entry
		if entry.Type == EntryTypeHeader && index != 0 {
			return nil, report, errors.New("session header must be the first record")
		}
		oldID := entry.ID
		if entry.ParentID != "" {
			mapped, ok := latest[entry.ParentID]
			if !ok {
				return nil, report, errors.New("legacy parent does not precede child")
			}
			entry.ParentID = mapped
		}
		previous, duplicate := first[oldID]
		if duplicate {
			// Only the historical compactor's identical-node replay is supported.
			if !legacyContentEqual(previous.fields, rec.fields) || entry.Type == EntryTypeCompaction || entry.Type == EntryTypeSelection || entry.Type == EntryTypeAnnotation {
				return nil, report, errors.New("legacy duplicate ID has conflicting content")
			}
			// The first clone hangs from a newly appended compaction; subsequent
			// clones must follow the immediately preceding original node's ID.
			if !cloneChain || entry.ParentID != previousConverted || (seen[previousConverted] != EntryTypeCompaction && rec.entry.ParentID != previousOriginal) {
				return nil, report, errors.New("legacy duplicate is outside a compaction clone chain")
			}
			sum := sha256.Sum256([]byte(fmt.Sprintf("harness-legacy-node-v1\x00%s\x00%d\x00%s", report.SourceSHA256, index, oldID)))
			entry.ID = "import_" + hex.EncodeToString(sum[:])
			if originalIDs[entry.ID] {
				return nil, report, errors.New("legacy generated ID collides with source")
			}
			report.RemappedIDs++
		} else {
			first[oldID] = rec
			cloneChain = entry.Type == EntryTypeCompaction
		}
		if err := validateGraphNode(entry, lookup); err != nil {
			return nil, report, err
		}
		latest[oldID] = entry.ID
		seen[entry.ID] = entry.Type
		previousOriginal = oldID
		previousConverted = entry.ID
		if entry.Type == EntryTypeCompaction && len(entry.Data) > 0 && string(entry.Data) != "null" {
			var data map[string]json.RawMessage
			if err := json.Unmarshal(entry.Data, &data); err != nil {
				return nil, report, err
			}
			for _, key := range []string{"range_start_id", "range_end_id"} {
				value, exists := data[key]
				if !exists {
					continue
				}
				var id string
				if err := json.Unmarshal(value, &id); err != nil {
					return nil, report, err
				}
				if id != "" {
					mapped, ok := latest[id]
					if !ok || id == oldID {
						return nil, report, errors.New("legacy compaction range references a missing or current node")
					}
					data[key], _ = json.Marshal(mapped)
				}
			}
			rec.fields["data"], _ = json.Marshal(data)
		}
		rec.fields["id"], _ = json.Marshal(entry.ID)
		if entry.ParentID != "" {
			rec.fields["parentId"], _ = json.Marshal(entry.ParentID)
		}
		if err := encoder.Encode(rec.fields); err != nil {
			return nil, report, err
		}
		if int64(output.Len()) > 2*MaxLegacyImportBytes {
			return nil, report, errors.New("converted session exceeds output size limit")
		}
	}
	if report.RemappedIDs == 0 {
		return raw, report, nil
	}
	// Verify the result with the strict production loader before returning it.
	converted := output.Bytes()
	if _, err := decodeSessionRecords(bytes.NewReader(converted)); err != nil {
		return nil, report, fmt.Errorf("converted graph: %w", err)
	}
	return converted, report, nil
}

func legacyContentEqual(a, b map[string]json.RawMessage) bool {
	// Parent links and graph IDs are the only fields historical compaction
	// changed. Compare decoded values so whitespace in JSON is immaterial.
	normalize := func(fields map[string]json.RawMessage) (map[string]any, error) {
		out := make(map[string]any)
		for key, raw := range fields {
			if key == "id" || key == "parentId" {
				continue
			}
			var value any
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.UseNumber()
			if err := dec.Decode(&value); err != nil {
				return nil, err
			}
			out[key] = value
		}
		return out, nil
	}
	x, err := normalize(a)
	if err != nil {
		return false
	}
	y, err := normalize(b)
	return err == nil && reflect.DeepEqual(x, y)
}
