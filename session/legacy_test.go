package session

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func legacyCompactionFixture() string {
	return strings.Join([]string{
		`{"id":"a","type":"message","data":{"text":"root"}}`,
		`{"id":"b","parentId":"a","type":"message","data":{"text":"kept"},"future":{"value":9007199254740993}}`,
		`{"id":"c","parentId":"b","type":"message","data":{"text":"last"}}`,
		`{"id":"summary","parentId":"a","type":"compaction","data":{"summary":"root summary"}}`,
		`{"id":"b","parentId":"summary","type":"message","data":{"text":"kept"},"future":{"value":9007199254740993}}`,
		`{"id":"c","parentId":"b","type":"message","data":{"text":"last"}}`,
	}, "\n") + "\n"
}

func TestConvertLegacyCompactionRetainsEveryVersion(t *testing.T) {
	raw := legacyCompactionFixture()
	converted, report, err := ConvertLegacySession(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if report.Records != 6 || report.RemappedIDs != 2 || len(report.SourceSHA256) != 64 {
		t.Fatal(report)
	}
	entries, err := decodeSessionRecords(bytes.NewReader(converted))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 6 || entries[1].ID != "b" || entries[1].ParentID != "a" || entries[2].ID != "c" || entries[2].ParentID != "b" {
		t.Fatal("original graph lost", entries)
	}
	if entries[4].ID == "b" || entries[4].ParentID != "summary" || entries[5].ID == "c" || entries[5].ParentID != entries[4].ID {
		t.Fatal("clone chain not restored", entries)
	}
	if bytes.Count(converted, []byte(`"value":9007199254740993`)) != 2 {
		t.Fatal("unknown fields or integer precision lost")
	}
	again, second, err := ConvertLegacySession(bytes.NewReader(converted))
	if err != nil || !bytes.Equal(converted, again) || second.RemappedIDs != 0 {
		t.Fatal("conversion not idempotent", err, second)
	}
	repeat, _, err := ConvertLegacySession(strings.NewReader(raw))
	if err != nil || !bytes.Equal(converted, repeat) {
		t.Fatal("conversion not deterministic", err)
	}
}

func TestConvertLegacyRejectsUnrelatedCorruption(t *testing.T) {
	for name, raw := range map[string]string{
		"self range":            `{"id":"s","type":"compaction","data":{"range_end_id":"s"}}`,
		"conflicting duplicate": strings.Replace(legacyCompactionFixture(), `"parentId":"summary","type":"message","data":{"text":"kept"}`, `"parentId":"summary","type":"message","data":{"text":"changed"}`, 1),
		"unrelated duplicate":   `{"id":"a","type":"message"}` + "\n" + `{"id":"a","type":"message"}`,
		"missing parent":        `{"id":"a","parentId":"missing"}`,
		"tail":                  legacyCompactionFixture() + `{"id":`,
		"interior":              "not json\n" + legacyCompactionFixture(),
		"future selection":      `{"id":"a"}` + "\n" + `{"id":"s","parentId":"a","type":"selection","data":{"version":2}}`,
	} {
		t.Run(name, func(t *testing.T) {
			out, _, err := ConvertLegacySession(strings.NewReader(raw))
			if err == nil || out != nil {
				t.Fatal("corrupt legacy log accepted")
			}
		})
	}
}

func TestConvertValidLegacyPreservesExactBytes(t *testing.T) {
	raw := "\n  " + `{"id":"a", "future": true}` + " \n"
	out, report, err := ConvertLegacySession(strings.NewReader(raw))
	if err != nil || string(out) != raw || report.RemappedIDs != 0 {
		t.Fatal("valid log changed", err)
	}
}

func FuzzLegacyConversion(f *testing.F) {
	for _, seed := range []string{legacyCompactionFixture(), `{"id":"a"}`, `{"id":`, "null\n"} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		out, report, err := ConvertLegacySession(bytes.NewReader(raw))
		if err != nil {
			if out != nil {
				t.Fatal("failed conversion returned output")
			}
			return
		}
		entries, err := decodeSessionRecords(bytes.NewReader(out))
		if err != nil || len(entries) != report.Records {
			t.Fatal("converter produced invalid graph", err)
		}
		second, r, err := ConvertLegacySession(bytes.NewReader(out))
		if err != nil || !bytes.Equal(second, out) || r.RemappedIDs != 0 {
			t.Fatal("conversion not idempotent", err)
		}
		for _, entry := range entries {
			if len(entry.Data) > 0 && !json.Valid(entry.Data) {
				t.Fatal("invalid data")
			}
		}
	})
}

func TestLegacyConversionRemapsCompactionRanges(t *testing.T) {
	raw := legacyCompactionFixture() + `{"id":"summary2","parentId":"c","type":"compaction","data":{"summary":"second","range_start_id":"b","range_end_id":"c","unknown":true}}` + "\n"
	out, _, err := ConvertLegacySession(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	entries, err := decodeSessionRecords(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	var data map[string]any
	if err := json.Unmarshal(entries[6].Data, &data); err != nil {
		t.Fatal(err)
	}
	if data["range_start_id"] != entries[4].ID || data["range_end_id"] != entries[5].ID || data["unknown"] != true {
		t.Fatal("compaction provenance references wrong versions", data)
	}
}

func TestLegacyConversionRecordLimit(t *testing.T) {
	out, _, err := ConvertLegacySession(strings.NewReader(strings.Repeat("{\"id\":\"a\"}\n", MaxLegacyImportRecords+1)))
	if err == nil || out != nil || !strings.Contains(err.Error(), "record limit") {
		t.Fatal("record cap not enforced", err)
	}
}
