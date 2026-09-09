package runtime

import "fmt"

// Provider IDs identify result ownership within a single response. Never
// overwrite a running result channel or dispatch an ambiguous second call.
func acceptToolCallID(seen map[string]bool, id string) error {
	if id == "" {
		return fmt.Errorf("provider returned an empty tool call ID")
	}
	if seen[id] {
		return fmt.Errorf("provider returned a duplicate tool call ID")
	}
	seen[id] = true
	return nil
}
