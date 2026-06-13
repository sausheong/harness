package file

import (
	"fmt"
	"os"
)

const (
	maxTextFileSize  = 10 * 1024 * 1024 // 10 MB
	maxImageFileSize = 5 * 1024 * 1024  // 5 MB
)

// checkFileSize stats path and returns a non-empty error string if the file
// exceeds limit. kind is "text" or "image" for the message. Empty return means
// within bounds (or stat failed, in which case the caller's read surfaces the
// real error).
func checkFileSize(path string, limit int64, kind string) string {
	fi, err := os.Stat(path)
	if err != nil {
		return ""
	}
	if fi.Size() > limit {
		return fmt.Sprintf("file too large: %d bytes exceeds the %d byte limit for %s files; read a specific range or use a different tool",
			fi.Size(), limit, kind)
	}
	return ""
}
