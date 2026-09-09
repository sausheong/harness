package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
	"unicode/utf8"
)

const MaxAnnotationBytes = 64 << 10

// AnnotationData is versioned application metadata. It is retained in Entries
// and rewrites/exports, but never added to History, View or the selected leaf.
type AnnotationData struct {
	Version int             `json:"version"`
	Kind    string          `json:"kind"`
	Payload json.RawMessage `json:"payload"`
}

func decodeAnnotation(data json.RawMessage) (AnnotationData, error) {
	var annotation AnnotationData
	if len(data) > MaxAnnotationBytes || !utf8.Valid(data) {
		return annotation, errors.New("annotation exceeds size limit or is not UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	opening, err := d.Token()
	if err != nil || opening != json.Delim('{') {
		return annotation, errors.New("annotation must be an object")
	}
	fields := make(map[string]json.RawMessage, 3)
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return annotation, err
		}
		key, ok := token.(string)
		if !ok || (key != "version" && key != "kind" && key != "payload") || fields[key] != nil {
			return annotation, errors.New("unknown or duplicate annotation field")
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return annotation, err
		}
		fields[key] = value
	}
	if closing, err := d.Token(); err != nil || closing != json.Delim('}') || len(fields) != 3 || d.Decode(new(any)) != io.EOF {
		return annotation, errors.New("incomplete annotation object")
	}
	if err := json.Unmarshal(fields["version"], &annotation.Version); err != nil {
		return annotation, err
	}
	if err := json.Unmarshal(fields["kind"], &annotation.Kind); err != nil {
		return annotation, err
	}
	annotation.Payload = fields["payload"]
	if annotation.Version != 1 {
		return annotation, fmt.Errorf("unsupported annotation version %d", annotation.Version)
	}
	if annotation.Kind == "" || len(annotation.Kind) > 128 || strings.IndexFunc(annotation.Kind, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-')
	}) >= 0 {
		return annotation, errors.New("invalid annotation kind")
	}
	if !json.Valid(annotation.Payload) {
		return annotation, errors.New("invalid annotation payload")
	}
	return annotation, nil
}

// Annotate durably appends metadata without changing the selected conversation
// leaf. Input is copied by encoding before the record is retained.
var ErrAnnotationConflict = errors.New("annotation revision changed")

// AnnotateIfCount appends only if kind still has expected records. This allows
// durable application ledgers to reserve atomically across concurrent owners.
func (s *Session) AnnotateIfCount(kind string, payload json.RawMessage, expected int) error {
	if expected < 0 {
		return errors.New("negative annotation revision")
	}
	return s.annotate(kind, payload, &expected)
}

func (s *Session) Annotate(kind string, payload json.RawMessage) error {
	return s.annotate(kind, payload, nil)
}
func (s *Session) annotate(kind string, payload json.RawMessage, expected *int) error {
	if len(payload) > MaxAnnotationBytes {
		return errors.New("annotation payload exceeds size limit")
	}
	data, err := json.Marshal(AnnotationData{Version: 1, Kind: kind, Payload: payload})
	if err != nil {
		return err
	}
	if _, err := decodeAnnotation(data); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if expected != nil {
		count := 0
		for _, entry := range s.entries {
			if entry.Type != EntryTypeAnnotation {
				continue
			}
			annotation, err := decodeAnnotation(entry.Data)
			if err == nil && annotation.Kind == kind {
				count++
			}
		}
		if count != *expected {
			return ErrAnnotationConflict
		}
	}

	if err := s.PersistenceError(); err != nil {
		return err
	}
	entry := SessionEntry{ID: generateID("annotation"), ParentID: s.leafID, Type: EntryTypeAnnotation, Timestamp: time.Now().Unix(), Data: data}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.store != nil {
		s.store.AppendEntry(s, entry)
		if err := s.Flush(); err != nil {
			return err
		}
	}
	s.entries = append(s.entries, entry)
	s.entryMap[entry.ID] = &s.entries[len(s.entries)-1]
	return nil
}

// Annotations returns independent payload copies in append order across all
// branches. Applications decide whether a record belongs to current-branch or
// whole-session accounting; ParentID in Entries provides its attachment node.
func (s *Session) Annotations(kind string) []AnnotationData {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []AnnotationData
	for _, entry := range s.entries {
		if entry.Type != EntryTypeAnnotation {
			continue
		}
		annotation, err := decodeAnnotation(entry.Data)
		if err == nil && (kind == "" || annotation.Kind == kind) {
			out = append(out, annotation)
		}
	}
	return out
}
