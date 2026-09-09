package session

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"path/filepath"

	"github.com/sausheong/harness/attachment"
)

func (s *Store) OpenAttachments(agentID string) (*attachment.Store, error) {
	if err := validateStoreComponent("agent ID", agentID); err != nil {
		return nil, err
	}
	return attachment.Open(filepath.Join(s.sessionDir(agentID), ".attachments"), attachment.DefaultStoreBytes)
}

func mapImages(entry SessionEntry, transform func(ImageData) (ImageData, error)) (SessionEntry, error) {
	if len(entry.Data) == 0 || (entry.Type != EntryTypeMessage && entry.Type != EntryTypeToolResult) {
		return entry, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(entry.Data, &fields); err != nil {
		return entry, err
	}
	raw, ok := fields["images"]
	if !ok {
		return entry, nil
	}
	var images []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &images); err != nil {
		return entry, err
	}
	for _, fields := range images {
		encoded, err := json.Marshal(fields)
		if err != nil {
			return entry, err
		}
		var img ImageData
		if err := json.Unmarshal(encoded, &img); err != nil {
			return entry, err
		}
		img, err = transform(img)
		if err != nil {
			return entry, err
		}
		delete(fields, "data")
		delete(fields, "attachment")
		if img.Reference != nil {
			fields["attachment"], err = json.Marshal(img.Reference)
		} else {
			fields["data"], err = json.Marshal(img.Data)
		}
		if err != nil {
			return entry, err
		}
	}
	encoded, err := json.Marshal(images)
	if err != nil {
		return entry, err
	}
	fields["images"] = encoded
	entry.Data, err = json.Marshal(fields)
	return entry, err
}

// externalizeImages runs while the caller holds the session write fence.
func (s *Session) externalizeImages(ctx context.Context, entry SessionEntry) (SessionEntry, error) {
	var blobs *attachment.Store
	defer func() {
		if blobs != nil {
			blobs.Close()
		}
	}()
	return mapImages(entry, func(img ImageData) (ImageData, error) {
		if img.Reference != nil && img.Data != "" {
			return img, errors.New("image has both inline data and attachment reference")
		}
		if s.store == nil {
			if img.Reference != nil {
				return img, errors.New("attachment reference requires a persistent session store")
			}
			return img, nil
		}
		if blobs == nil {
			var err error
			blobs, err = s.store.OpenAttachments(s.AgentID)
			if err != nil {
				return img, err
			}
		}
		if img.Reference != nil {
			if _, err := blobs.Read(ctx, *img.Reference); err != nil {
				return img, err
			}
			return img, nil
		}
		if len(img.Data) > base64.StdEncoding.EncodedLen(int(attachment.MaxBlobBytes)) {
			return img, errors.New("inline image exceeds attachment limit")
		}
		data, err := base64.StdEncoding.DecodeString(img.Data)
		if err != nil {
			return img, err
		}
		ref, err := blobs.Put(ctx, data)
		if err != nil {
			return img, err
		}
		img.Data = ""
		img.Reference = &ref
		return img, nil
	})
}

// ResolveImages returns independent replay entries. Stored records retain
// references. Missing/corrupt blobs are errors, never silently omitted images.
func (s *Session) ResolveImages(ctx context.Context, entries []SessionEntry) ([]SessionEntry, error) {
	var blobs *attachment.Store
	defer func() {
		if blobs != nil {
			blobs.Close()
		}
	}()
	resolved := make([]SessionEntry, 0, len(entries))
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		next, err := mapImages(entry, func(img ImageData) (ImageData, error) {
			if img.Reference == nil {
				return img, nil
			}
			if img.Data != "" {
				return img, errors.New("ambiguous image representation")
			}
			if s.store == nil {
				return img, errors.New("attachment store unavailable")
			}
			if blobs == nil {
				var err error
				blobs, err = s.store.OpenAttachments(s.AgentID)
				if err != nil {
					return img, err
				}
			}
			data, err := blobs.Read(ctx, *img.Reference)
			if err != nil {
				return img, err
			}
			img.Data = base64.StdEncoding.EncodeToString(data)
			img.Reference = nil
			return img, nil
		})
		if err != nil {
			return nil, err
		}
		resolved = append(resolved, next)
	}
	return resolved, nil
}

// AttachmentReferences returns unique validated references across all supplied
// branches. Inline legacy images are not references and remain in their record.
func AttachmentReferences(entries []SessionEntry) ([]attachment.Ref, error) {
	var refs []attachment.Ref
	seen := map[string]int64{}
	for _, entry := range entries {
		_, err := mapImages(entry, func(img ImageData) (ImageData, error) {
			if img.Reference == nil {
				return img, nil
			}
			if img.Data != "" {
				return img, errors.New("ambiguous image representation")
			}
			ref := *img.Reference
			if err := ref.Validate(); err != nil {
				return img, err
			}
			if size, ok := seen[ref.SHA256]; ok {
				if size != ref.Size {
					return img, errors.New("conflicting attachment sizes")
				}
				return img, nil
			}
			seen[ref.SHA256] = ref.Size
			refs = append(refs, ref)
			return img, nil
		})
		if err != nil {
			return nil, err
		}
	}
	return refs, nil
}

// CopyAttachments copies referenced blobs before publishing a copied session.
// At most one blob is materialised at a time; existing content is verified and
// reused. Unreferenced source-store content is never copied.
func (s *Store) CopyAttachments(ctx context.Context, agentID string, target *Store, targetAgent string, entries []SessionEntry) error {
	refs, err := AttachmentReferences(entries)
	if err != nil {
		return err
	}
	if len(refs) == 0 {
		return ctx.Err()
	}
	from, err := s.OpenAttachments(agentID)
	if err != nil {
		return err
	}
	defer from.Close()
	to, err := target.OpenAttachments(targetAgent)
	if err != nil {
		return err
	}
	defer to.Close()
	for _, ref := range refs {
		data, err := from.Read(ctx, ref)
		if err != nil {
			return err
		}
		got, err := to.Put(ctx, data)
		if err != nil {
			return err
		}
		if got != ref {
			return errors.New("copied attachment reference changed")
		}
	}
	return nil
}
