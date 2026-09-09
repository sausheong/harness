//go:build darwin || linux

package bash

import (
	"context"
	"encoding/json"
	"github.com/sausheong/harness/process"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCancellationStopsShellDescendant(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "survived")
	input, _ := json.Marshal(bashInput{Command: "sh -c 'sleep 1; echo survived > \"$1\"' fixture " + shellSingleQuote(marker) + " & wait"})
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	result, err := (&BashTool{}).Execute(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Error == "" {
		t.Fatal("cancelled command succeeded")
	}
	// Waiting beyond the descendant's scheduled write distinguishes stopping it
	// from merely returning while it continues to run in the background.
	time.Sleep(1100 * time.Millisecond)
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("shell descendant survived cancellation and wrote its marker")
	}
}

func TestNormalExitStopsShellDescendant(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "survived")
	input, _ := json.Marshal(bashInput{Command: "sh -c 'sleep 1; echo survived > \"$1\"' fixture " + shellSingleQuote(marker) + " >/dev/null 2>&1 &"})
	result, err := (&BashTool{}).Execute(context.Background(), input)
	if err != nil || result.Error != "" {
		t.Fatalf("%+v %v", result, err)
	}
	time.Sleep(1100 * time.Millisecond)
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("descendant marker: %v", err)
	}
}

func TestOutputCaptureBoundsBothStreams(t *testing.T) {
	input := json.RawMessage(`{"command":"head -c 1048576 /dev/zero; head -c 1048576 /dev/zero >&2"}`)
	store, storeErr := process.NewArtifactStore(filepath.Join(t.TempDir(), "artifacts"))
	if storeErr != nil {
		t.Fatal(storeErr)
	}
	result, err := (&BashTool{OutputStore: store}).Execute(context.Background(), input)
	if err != nil || result.Error != "" {
		t.Fatalf("error: %v %s", err, result.Error)
	}
	if len(result.Output) > 2*(64<<10)+1000 {
		t.Fatalf("unbounded output: %d", len(result.Output))
	}
	for _, stream := range []string{"stdout", "stderr"} {
		artifact := result.Metadata[stream+"_artifact"].(process.ArtifactInfo)
		raw, err := os.ReadFile(artifact.Path)
		if err != nil || len(raw) != 1048576 || artifact.Truncated || artifact.Error != "" {
			t.Fatalf("artifact %+v: len=%d err=%v", artifact, len(raw), err)
		}
		for _, b := range raw {
			if b != 0 {
				t.Fatal("artifact content corrupted")
			}
		}
		if result.Metadata[stream+"_bytes"] != int64(1048576) || result.Metadata[stream+"_truncated"] != true {
			t.Fatalf("incorrect capture metadata: %+v", result.Metadata)
		}
	}
}
