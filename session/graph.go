package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// validateGraphNode requires parents to precede children in the append log.
// This rejects missing links, duplicate IDs and cycles before graph traversal.
func validateGraphNode(entry SessionEntry, lookup func(string) (EntryType, bool)) error {
	if entry.Type == EntryTypeHeader {
		if entry.SchemaVersion != 1 {
			return fmt.Errorf("unsupported session schema version %d", entry.SchemaVersion)
		}
		if entry.ParentID != "" {
			return errors.New("session header cannot have a parent")
		}
	} else if entry.SchemaVersion != 0 {
		return errors.New("schema version belongs only on the session header")
	}
	if entry.ID == "" {
		return errors.New("session entry ID is empty")
	}
	if _, exists := lookup(entry.ID); exists {
		return fmt.Errorf("duplicate session entry ID %q", entry.ID)
	}
	if entry.ParentID != "" {
		parent, exists := lookup(entry.ParentID)
		if !exists || parent == EntryTypeSelection || parent == EntryTypeHeader || parent == EntryTypeAnnotation {
			return errors.New("session parent is missing or is a selection control record")
		}
	}
	if entry.Type == EntryTypeAnnotation {
		if _, err := decodeAnnotation(entry.Data); err != nil {
			return err
		}
	}
	if entry.Type == EntryTypeSelection {
		if entry.ParentID == "" {
			return errors.New("selection record has no target")
		}
		d := json.NewDecoder(bytes.NewReader(entry.Data))
		opening, err := d.Token()
		if err != nil || opening != json.Delim('{') {
			return errors.New("selection data must be an object")
		}
		key, err := d.Token()
		if err != nil || key != "version" {
			return errors.New("selection data requires its version field")
		}
		var version int
		if err := d.Decode(&version); err != nil {
			return fmt.Errorf("invalid selection record: %w", err)
		}
		if closing, err := d.Token(); err != nil || closing != json.Delim('}') || d.Decode(new(any)) != io.EOF {
			return errors.New("selection data has unexpected or duplicate fields")
		}
		if version != 1 {
			return fmt.Errorf("unsupported selection record version %d", version)
		}
	}
	return nil
}

var ErrSessionChanged = errors.New("session branch changed while preparing compaction")

// CommitCompaction installs a summary and preserved messages as one graph
// mutation. The caller supplies the leaf of the view it summarised. Concurrent
// appends or branch selections invalidate that snapshot instead of losing work.
// Original nodes remain immutable; cloned messages have fresh graph IDs while
// their tool-call IDs and content remain unchanged.
func (s *Session) CommitCompaction(expectedLeaf string, summary SessionEntry, preserved []SessionEntry) error {
	s.mu.Lock()
	if s.leafID != expectedLeaf {
		s.mu.Unlock()
		return ErrSessionChanged
	}
	if err := s.PersistenceError(); err != nil {
		s.mu.Unlock()
		return err
	}
	if summary.Type != EntryTypeCompaction {
		s.mu.Unlock()
		return errors.New("expected compaction summary")
	}
	summary.ID = generateID("compact")
	if err := validateGraphNode(summary, func(id string) (EntryType, bool) {
		if id == s.header.ID {
			return EntryTypeHeader, true
		}
		node, ok := s.entryMap[id]
		if !ok {
			return "", false
		}
		return node.Type, true
	}); err != nil {
		s.mu.Unlock()
		return err
	}
	for _, node := range preserved {
		if node.Type == EntryTypeSelection || node.Type == EntryTypeHeader || node.Type == EntryTypeAnnotation {
			s.mu.Unlock()
			return errors.New("selection record cannot be preserved as a message")
		}
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.entries = append(s.entries, summary)
	s.entryMap[summary.ID] = &s.entries[len(s.entries)-1]
	s.leafID = summary.ID
	for _, node := range preserved {
		node.ID = generateID("e")
		node.ParentID = s.leafID
		s.entries = append(s.entries, node)
		s.entryMap[node.ID] = &s.entries[len(s.entries)-1]
		s.leafID = node.ID
	}
	entries := append([]SessionEntry(nil), s.entries...)
	store := s.store
	s.mu.Unlock()
	if store != nil {
		store.rewriteEntries(s, entries)
	}
	return s.PersistenceError()
}
