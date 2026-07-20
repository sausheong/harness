package llm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStripFieldsTopLevel(t *testing.T) {
	in := json.RawMessage(`{
		"type": "object",
		"properties": {"x": {"type": "string"}},
		"$ref": "#/defs/foo"
	}`)
	out, diags := StripFields("mytool", in, []string{"$ref"})

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(out, &parsed))
	_, hasRef := parsed["$ref"]
	assert.False(t, hasRef, "$ref must be stripped")
	require.Len(t, diags, 1)
	assert.Equal(t, "mytool", diags[0].ToolName)
	assert.Equal(t, "$ref", diags[0].Field)
	assert.Equal(t, "stripped", diags[0].Action)
}

func TestStripFieldsNested(t *testing.T) {
	in := json.RawMessage(`{
		"type": "object",
		"properties": {
			"url": {"type": "string", "format": "uri"},
			"items": {
				"type": "array",
				"items": {"type": "string", "format": "email"}
			}
		}
	}`)
	out, diags := StripFields("read", in, []string{"format"})

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(out, &parsed))
	props := parsed["properties"].(map[string]any)
	url := props["url"].(map[string]any)
	_, hasFormat := url["format"]
	assert.False(t, hasFormat, "nested format under properties.url must be stripped")
	items := props["items"].(map[string]any)
	itemsItems := items["items"].(map[string]any)
	_, hasNestedFormat := itemsItems["format"]
	assert.False(t, hasNestedFormat, "format under properties.items.items must be stripped")
	require.Len(t, diags, 2)
	// Keys are visited in sorted order: "items" < "url" alphabetically,
	// so the items.items.format diagnostic comes first.
	assert.Equal(t, "properties.items.items.format", diags[0].Field)
	assert.Equal(t, "properties.url.format", diags[1].Field)
}

func TestStripFieldsAdditionalProperties(t *testing.T) {
	in := json.RawMessage(`{
		"type": "object",
		"additionalProperties": {"type": "string", "$ref": "#/x"}
	}`)
	out, diags := StripFields("t", in, []string{"$ref"})
	var parsed map[string]any
	require.NoError(t, json.Unmarshal(out, &parsed))
	addl := parsed["additionalProperties"].(map[string]any)
	_, has := addl["$ref"]
	assert.False(t, has)
	require.Len(t, diags, 1)
	assert.Equal(t, "additionalProperties.$ref", diags[0].Field)
}

func TestStripFieldsNoOp(t *testing.T) {
	in := json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`)
	out, diags := StripFields("t", in, []string{"$ref", "format"})
	// Output bytes may differ from input due to re-marshal, but parsing
	// both must produce the same structure and no fields are removed.
	var inDoc, outDoc any
	require.NoError(t, json.Unmarshal(in, &inDoc))
	require.NoError(t, json.Unmarshal(out, &outDoc))
	assert.Equal(t, inDoc, outDoc, "no fields to strip → structurally unchanged")
	assert.Empty(t, diags)
}

func TestStripFieldsDeterministic(t *testing.T) {
	// Same input must produce the same output and diagnostic order
	// across repeated calls. Required for cache stability.
	in := json.RawMessage(`{
		"properties": {
			"a": {"format": "uri"},
			"b": {"format": "email"},
			"c": {"format": "date"}
		}
	}`)
	out1, diags1 := StripFields("t", in, []string{"format"})
	out2, diags2 := StripFields("t", in, []string{"format"})
	assert.Equal(t, string(out1), string(out2))
	assert.Equal(t, diags1, diags2)
}

func TestStripFieldsEmptyInputs(t *testing.T) {
	// Empty fields list → no-op, returns input unchanged.
	out, diags := StripFields("t", json.RawMessage(`{"x":1}`), nil)
	assert.Equal(t, `{"x":1}`, string(out))
	assert.Nil(t, diags)
	// Empty schema → no-op.
	out2, diags2 := StripFields("t", nil, []string{"$ref"})
	assert.Nil(t, out2)
	assert.Nil(t, diags2)
}

func TestStripFieldsMalformedSchema(t *testing.T) {
	// Malformed JSON → return as-is, no diagnostic. Provider SDK will
	// reject with its own error.
	bad := json.RawMessage(`{"unclosed`)
	out, diags := StripFields("t", bad, []string{"$ref"})
	assert.Equal(t, bad, out)
	assert.Empty(t, diags)
}

func TestStripCacheKeyIncludesCacheControl(t *testing.T) {
	base := ToolDef{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}
	fiveMinutes := base
	fiveMinutes.CacheControl = &CacheControl{Type: "ephemeral", TTL: "5m"}
	oneHour := base
	oneHour.CacheControl = &CacheControl{Type: "ephemeral", TTL: "1h"}
	assert.NotEqual(t, stripCacheKey([]ToolDef{fiveMinutes}, nil), stripCacheKey([]ToolDef{oneHour}, nil))

	cloned := cloneToolDefs([]ToolDef{oneHour})
	cloned[0].CacheControl.TTL = "5m"
	assert.Equal(t, "1h", oneHour.CacheControl.TTL, "cached normalization results must not alias cache metadata")
}
