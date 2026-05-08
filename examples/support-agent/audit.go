package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// auditLog appends one JSON object per line to a file. Used by the
// LifecycleHooks (AfterToolUse + OnStop) to record every tool action
// and run outcome, the way a real compliance-sensitive support
// deployment would. Synchronous, mutex-guarded — fine for an
// interactive REPL with one Run at a time.
type auditLog struct {
	mu sync.Mutex
	f  *os.File
}

func newAuditLog(path string) (*auditLog, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &auditLog{f: f}, nil
}

// Log writes one JSON record. The record gets a "ts" field stamped
// at write time; callers should not set "ts" themselves.
func (a *auditLog) Log(record map[string]any) {
	a.mu.Lock()
	defer a.mu.Unlock()
	record["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	b, err := json.Marshal(record)
	if err != nil {
		fmt.Fprintln(os.Stderr, "audit marshal:", err)
		return
	}
	if _, err := a.f.Write(append(b, '\n')); err != nil {
		fmt.Fprintln(os.Stderr, "audit write:", err)
	}
}

func (a *auditLog) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.f.Close()
}
