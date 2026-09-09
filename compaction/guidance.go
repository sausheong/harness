package compaction

import (
	"context"
	"errors"
	"strings"
	"unicode/utf8"
)

type guidanceKey struct{}

// WithGuidance binds immutable caller guidance to synchronous and asynchronous
// compaction derived from this context. It does not modify shared managers.
func WithGuidance(ctx context.Context, guidance string) (context.Context, error) {
	if len(guidance) > 48<<10 || !utf8.ValidString(guidance) || strings.ContainsRune(guidance, 0) {
		return ctx, errors.New("compaction guidance exceeds 48 KiB or is invalid UTF-8")
	}
	return context.WithValue(ctx, guidanceKey{}, guidance), nil
}
func guidanceFor(ctx context.Context, focus string) (string, error) {
	guidance, _ := ctx.Value(guidanceKey{}).(string)
	if len(focus) > 16<<10 || !utf8.ValidString(focus) || strings.ContainsRune(focus, 0) {
		return "", errors.New("compaction focus exceeds 16 KiB or is invalid UTF-8")
	}
	if guidance == "" {
		return focus, nil
	}
	return guidance + "\n\nCurrent compaction focus:\n" + focus, nil
}
