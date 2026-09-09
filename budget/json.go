package budget

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"time"
	"unicode/utf8"
)

// All version-1 budget schema names are lowercase. Validate the complete JSON
// structure before typed decoding can accept aliases, duplicates or null scalar
// values. The existing typed decoder still enforces the per-object field set.
func decodeBudgetJSON(raw json.RawMessage, destination any) error {
	trimmed := bytes.TrimSpace(raw)
	if !utf8.Valid(raw) || len(trimmed) == 0 || trimmed[0] != '{' {
		return errors.New("budget record requires a UTF-8 object")
	}
	scanner := json.NewDecoder(bytes.NewReader(raw))
	if err := scanBudgetJSON(scanner, "", 0); err != nil {
		return err
	}
	if scanner.Decode(new(any)) != io.EOF {
		return errors.New("trailing budget JSON")
	}
	if err := requireBudgetFields(raw, reflect.TypeOf(destination)); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	return d.Decode(destination)
}

func scanBudgetJSON(d *json.Decoder, field string, depth int) error {
	if depth > 16 {
		return errors.New("budget JSON nesting exceeds schema limit")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	if token == nil && field != "snapshot" && field != "attempt" {
		return errors.New("budget scalar field cannot be null")
	}
	delimiter, compound := token.(json.Delim)
	if !compound {
		return nil
	}
	switch delimiter {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			token, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := token.(string)
			if !ok || key != strings.ToLower(key) || seen[key] {
				return errors.New("duplicate or noncanonical budget field")
			}
			seen[key] = true
			if err := scanBudgetJSON(d, key, depth+1); err != nil {
				return err
			}
		}
		if token, err := d.Token(); err != nil || token != json.Delim('}') {
			return errors.New("invalid budget object")
		}
	case '[':
		for d.More() {
			if err := scanBudgetJSON(d, "", depth+1); err != nil {
				return err
			}
		}
		if token, err := d.Token(); err != nil || token != json.Delim(']') {
			return errors.New("invalid budget array")
		}
	default:
		return errors.New("unexpected budget delimiter")
	}
	return nil
}

// The on-disk contract matches the writer: fields without omitempty must be
// present even when their value is zero. This keeps omitted settlement charges
// or price counters from being silently manufactured by typed unmarshalling.
func requireBudgetFields(raw json.RawMessage, typ reflect.Type) error {
	if typ == nil {
		return errors.New("budget destination type missing")
	}
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.Kind() != reflect.Struct || typ == reflect.TypeOf(time.Time{}) || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if field.PkgPath != "" {
			continue
		}
		tag := strings.Split(field.Tag.Get("json"), ",")
		name := tag[0]
		if name == "-" {
			continue
		}
		if name == "" {
			name = field.Name
		}
		optional := false
		for _, option := range tag[1:] {
			if option == "omitempty" {
				optional = true
			}
		}
		value, exists := fields[name]
		if !exists {
			if optional {
				continue
			}
			return errors.New("missing required budget field: " + name)
		}
		if err := requireBudgetFields(value, field.Type); err != nil {
			return err
		}
	}
	return nil
}
