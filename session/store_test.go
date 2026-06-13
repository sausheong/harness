package session

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStore_DegradedPersistenceWarnsOnce(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	// Make a path where the agent dir cannot be created: a regular FILE used
	// as baseDir means sessionDir(agentID) = <file>/<agentID> and MkdirAll fails.
	tmp := t.TempDir()
	blocker := filepath.Join(tmp, "blocker")
	require.NoError(t, os.WriteFile(blocker, []byte("x"), 0o600))
	s := NewStore(blocker)

	sess := &Session{AgentID: "a", Key: "k"}
	s.AppendEntry(sess, SessionEntry{})
	s.AppendEntry(sess, SessionEntry{})
	s.AppendEntry(sess, SessionEntry{})

	require.Equal(t, 1, strings.Count(buf.String(), "session persistence degraded"),
		"degraded warning must fire exactly once across multiple failures")
}
