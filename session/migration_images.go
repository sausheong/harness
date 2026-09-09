package session

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
)

// Only migration may read larger legacy image records. Every converted record
// must still pass the normal 10 MiB parser before the original is replaced.
const MaxLegacyImageRecordBytes = 64 << 20

func (s *Store) externalizeLegacyImages(ctx context.Context, agentID string, raw []byte) ([]byte, int, error) {
	var output bytes.Buffer
	images := 0
	temporary := &Session{store: s, AgentID: agentID}
	for offset, lineNumber := 0, 1; offset < len(raw); lineNumber++ {
		if err := ctx.Err(); err != nil {
			return nil, images, err
		}
		if lineNumber > MaxLegacyImportRecords {
			return nil, images, errors.New("legacy session exceeds import record limit")
		}
		length := bytes.IndexByte(raw[offset:], '\n')
		final := length < 0
		if final {
			length = len(raw) - offset
		} else {
			length++
		}
		line := raw[offset : offset+length]
		trimmed := bytes.TrimSpace(line)
		if len(line) > MaxLegacyImageRecordBytes {
			return nil, images, &RecordError{Line: lineNumber, Offset: int64(offset), Cause: ErrSessionRecordTooLarge}
		}
		next := line
		if len(trimmed) > 0 {
			entry, err := decodeSessionRecord(trimmed, final, lineNumber, int64(offset))
			if err != nil {
				return output.Bytes(), images, err
			}
			inline := 0
			_, err = mapImages(entry, func(img ImageData) (ImageData, error) {
				if img.Reference == nil {
					inline++
				}
				return img, nil
			})
			if err != nil {
				return nil, images, err
			}
			if inline > 0 {
				converted, err := temporary.externalizeImages(ctx, entry)
				if err != nil {
					return nil, images, err
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(trimmed, &fields); err != nil {
					return nil, images, err
				}
				fields["data"] = converted.Data
				next, err = json.Marshal(fields)
				if err != nil {
					return nil, images, err
				}
				if !final {
					next = append(next, '\n')
				}
				if len(next) > MaxSessionRecordBytes {
					return nil, images, ErrSessionRecordTooLarge
				}
				images += inline
			}
		}
		if int64(output.Len()+len(next)) > MaxLegacyImportBytes {
			return nil, images, errors.New("normalised legacy session exceeds import size limit")
		}
		output.Write(next)
		offset += length
	}
	return output.Bytes(), images, nil
}
