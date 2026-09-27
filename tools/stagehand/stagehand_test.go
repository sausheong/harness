package stagehand

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sausheong/harness/llm/llmtest"
)

func newTestTool(t *testing.T) *StagehandTool {
	t.Helper()
	st, err := NewStagehandTool(Config{Provider: &llmtest.Stub{}, Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(st.Shutdown)
	return st
}

func TestNewStagehandToolRequiresProviderAndModel(t *testing.T) {
	if _, err := NewStagehandTool(Config{Model: "m"}); err == nil {
		t.Error("want error without provider")
	}
	if _, err := NewStagehandTool(Config{Provider: &llmtest.Stub{}}); err == nil {
		t.Error("want error without model")
	}
}

func TestStagehandParametersIsValidJSON(t *testing.T) {
	var schema map[string]any
	if err := json.Unmarshal(newTestTool(t).Parameters(), &schema); err != nil {
		t.Fatal(err)
	}
}

func TestStagehandValidationFailsBeforeLaunch(t *testing.T) {
	st := newTestTool(t)
	var launches atomic.Int32
	st.launch = func(context.Context) (*session, error) {
		launches.Add(1)
		return nil, errors.New("should not launch")
	}
	cases := map[string]struct {
		input string
		want  string
	}{
		"missing action":       {`{}`, "action is required"},
		"unknown action":       {`{"action":"fly","url":"https://example.com"}`, "unknown action"},
		"navigate without url": {`{"action":"navigate","session":"s"}`, "url is required"},
		"act without text":     {`{"action":"act","session":"s"}`, "instruction is required"},
		"extract without text": {`{"action":"extract","url":"https://example.com"}`, "instruction is required"},
		"bad scheme":           {`{"action":"navigate","url":"file:///etc/passwd"}`, "must start with http"},
		"internal url":         {`{"action":"navigate","url":"http://127.0.0.1:8080"}`, "navigate failed"},
		"no url or session":    {`{"action":"screenshot"}`, "without 'url' or 'session'"},
		"close without name":   {`{"action":"close"}`, "session is required"},
		"schema not object":    {`{"action":"extract","instruction":"x","session":"s","schema":[1]}`, "schema must be a JSON object"},
		"malformed json":       {`{`, "invalid input"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			res, err := st.Execute(context.Background(), json.RawMessage(tc.input))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(res.Error, tc.want) {
				t.Errorf("error = %q, want substring %q", res.Error, tc.want)
			}
		})
	}
	if n := launches.Load(); n != 0 {
		t.Errorf("browser launched %d times on invalid input", n)
	}
}

func TestStagehandLaunchErrorIsReported(t *testing.T) {
	st := newTestTool(t)
	st.launch = func(context.Context) (*session, error) { return nil, errors.New("no chrome") }
	res, err := st.Execute(context.Background(), json.RawMessage(`{"action":"navigate","url":"https://example.com","session":"s"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Error, "no chrome") {
		t.Errorf("error = %q", res.Error)
	}
	if len(st.sessions) != 0 {
		t.Error("failed launch left a session registered")
	}
}

func TestStagehandSessionReuseLimitAndClose(t *testing.T) {
	st := newTestTool(t)
	var launches atomic.Int32
	st.launch = func(context.Context) (*session, error) {
		launches.Add(1)
		return &session{lastUsed: time.Now()}, nil
	}
	ctx := context.Background()

	a1, release, err := st.acquire(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	release()
	a2, release, err := st.acquire(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	release()
	if a1 != a2 || launches.Load() != 1 {
		t.Fatalf("named session not reused (launches=%d)", launches.Load())
	}

	for _, name := range []string{"b", "c", "d", "e"} {
		_, release, err := st.acquire(ctx, name)
		if err != nil {
			t.Fatal(err)
		}
		release()
	}
	if _, _, err := st.acquire(ctx, "f"); err == nil || !strings.Contains(err.Error(), "too many open sessions") {
		t.Fatalf("want session limit error, got %v", err)
	}

	res, _ := st.Execute(ctx, json.RawMessage(`{"action":"close","session":"a"}`))
	if !strings.Contains(res.Output, `Closed session "a"`) {
		t.Errorf("close output = %q", res.Output)
	}
	res, _ = st.Execute(ctx, json.RawMessage(`{"action":"close","session":"a"}`))
	if !strings.Contains(res.Output, "No active session") {
		t.Errorf("second close output = %q", res.Output)
	}
	if _, release, err := st.acquire(ctx, "f"); err != nil {
		t.Fatalf("slot not freed after close: %v", err)
	} else {
		release()
	}
}

func TestStagehandEphemeralSessionIsNotRegistered(t *testing.T) {
	st := newTestTool(t)
	st.launch = func(context.Context) (*session, error) { return &session{}, nil }
	_, release, err := st.acquire(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	release()
	if len(st.sessions) != 0 {
		t.Error("ephemeral session was registered")
	}
}

func TestStagehandReapIdleSkipsBusySessions(t *testing.T) {
	st := newTestTool(t)
	st.launch = func(context.Context) (*session, error) { return &session{}, nil }
	ctx := context.Background()

	_, release, _ := st.acquire(ctx, "idle")
	release()
	_, releaseBusy, _ := st.acquire(ctx, "busy")
	defer releaseBusy()

	st.reapIdle(time.Now().Add(sessionIdleTimeout + time.Minute))

	st.mu.Lock()
	_, idleLeft := st.sessions["idle"]
	_, busyLeft := st.sessions["busy"]
	st.mu.Unlock()
	if idleLeft {
		t.Error("idle session not reaped")
	}
	if !busyLeft {
		t.Error("busy session was reaped mid-call")
	}
}

func TestTruncate(t *testing.T) {
	long := strings.Repeat("x", maxTextOutput+10)
	got := truncate(long)
	if !strings.Contains(got, "[truncated:") || len(got) < maxTextOutput {
		t.Errorf("truncate produced %d bytes", len(got))
	}
	if truncate("short") != "short" {
		t.Error("short string modified")
	}
}
