package session

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestListMetadataLargeLegacyAndDamage(t *testing.T) {
	dir := t.TempDir()
	large := `{"id":"a","type":"message","timestamp":100,"data":{"content":"` + strings.Repeat("x", 2<<20) + `"}}` + "\n" + `{"id":"b","type":"message","timestamp":200}` + "\n"
	for name, raw := range map[string]string{"large": large, "damaged": large + `{"unfinished":`} {
		if err := os.WriteFile(filepath.Join(dir, name+".jsonl"), []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(filepath.Join(dir, "large.jsonl"), filepath.Join(dir, "link.jsonl")); err != nil {
		t.Fatal(err)
	}
	infos, err := NewStore(dir).List("")
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 3 {
		t.Fatalf("lost discoverable files: %+v", infos)
	}
	for _, info := range infos {
		if info.Key == "large" {
			if info.MetadataError != "" || info.EntryCount != 2 || info.CreatedAt.Unix() != 100 || info.LastActivity.Unix() != 200 {
				t.Fatalf("incomplete metadata: %+v", info)
			}
		} else if info.MetadataError == "" || info.EntryCount != 0 || info.ID != "" {
			t.Fatalf("damage presented as complete: %+v", info)
		}
	}
}

type metadataFailReader struct{}

func (metadataFailReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func TestMetadataScanFailsWithoutPartialCounts(t *testing.T) {
	valid := `{"id":"a","type":"message","timestamp":100}` + "\n"
	_, err := scanSessionMetadata(io.MultiReader(strings.NewReader(valid), metadataFailReader{}), 100)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("lost read error: %v", err)
	}
	for _, raw := range []string{valid + strings.Repeat("x", 101), valid + "{\n"} {
		info, err := scanSessionMetadata(strings.NewReader(raw), 100)
		if err == nil || info.EntryCount != 0 {
			t.Fatalf("partial success: %+v %v", info, err)
		}
	}
}
