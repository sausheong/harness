package browser

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBrowserToolName(t *testing.T) {
	tool := &BrowserTool{}
	assert.Equal(t, "browser", tool.Name())
}

func TestBrowserToolParameters(t *testing.T) {
	tool := &BrowserTool{}
	params := tool.Parameters()
	assert.True(t, json.Valid(params), "Parameters() should return valid JSON")
}

func TestBrowserToolMissingAction(t *testing.T) {
	tool := &BrowserTool{}
	input, _ := json.Marshal(browserInput{})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "action is required")
}

func TestBrowserToolUnknownAction(t *testing.T) {
	tool := &BrowserTool{}
	input, _ := json.Marshal(browserInput{Action: "fly"})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "unknown action")
	assert.Contains(t, result.Error, "fly")
}

func TestBrowserNavigateMissingURL(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.navigate(context.Background(), browserInput{Action: "navigate"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "url is required")
}

func TestBrowserNavigateInvalidURL(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.navigate(context.Background(), browserInput{Action: "navigate", URL: "ftp://example.com"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "url must start with http")
}

func TestBrowserClickMissingSelector(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.click(context.Background(), browserInput{Action: "click"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "selector is required")
}

func TestBrowserTypeMissingSelector(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.typeText(context.Background(), browserInput{Action: "type", Text: "hello"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "selector is required")
}

func TestBrowserTypeMissingText(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.typeText(context.Background(), browserInput{Action: "type", Selector: "#input"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "text is required")
}

func TestBrowserEvaluateMissingScript(t *testing.T) {
	tool := &BrowserTool{}
	result, err := tool.evaluate(context.Background(), browserInput{Action: "evaluate"})
	require.NoError(t, err)
	assert.Contains(t, result.Error, "script is required")
}

// Verify the about:blank trap detection: post-navigation actions called
// without url AND without session must fail fast with an actionable hint
// rather than silently producing blank output.
func TestBrowserAboutBlankTrap(t *testing.T) {
	tool := &BrowserTool{}
	for _, action := range []string{"screenshot", "get_text", "evaluate", "click", "type"} {
		in := browserInput{Action: action}
		switch action {
		case "click":
			in.Selector = "#x"
		case "type":
			in.Selector = "#x"
			in.Text = "hi"
		case "evaluate":
			in.Script = "1+1"
		}
		body, _ := json.Marshal(in)
		result, err := tool.Execute(context.Background(), body)
		require.NoError(t, err)
		assert.Contains(t, result.Error, action, "action name should appear in error")
		assert.Contains(t, result.Error, "url", "error should mention url")
		assert.Contains(t, result.Error, "session", "error should mention session")
	}
}

// With a session set, the trap detection does NOT fire — calls are valid
// because state from a prior navigate in the same session may exist.
// (This still launches Chrome via getOrCreateSession, so we only check
// that the blank-trap message is absent from any error returned.)
func TestBrowserAboutBlankTrapBypassedBySession(t *testing.T) {
	tool := &BrowserTool{}
	// Use a small per-call timeout so even if Chrome launches we abort fast.
	body, _ := json.Marshal(browserInput{Action: "screenshot", Session: "test-session", Timeout: 1})
	result, _ := tool.Execute(context.Background(), body)
	assert.NotContains(t, result.Error, "fresh browser starting on about:blank")
}

// Same: with a URL set, the trap detection does NOT fire — the URL is
// validated, not the about:blank case.
func TestBrowserAboutBlankTrapBypassedByURL(t *testing.T) {
	tool := &BrowserTool{}
	body, _ := json.Marshal(browserInput{Action: "screenshot", URL: "ftp://nope"})
	result, _ := tool.Execute(context.Background(), body)
	// Should hit the URL-prefix check, not the about:blank check.
	assert.Contains(t, result.Error, "http://")
	assert.NotContains(t, result.Error, "fresh browser")
}

func TestBrowserCloseRequiresSession(t *testing.T) {
	tool := &BrowserTool{}
	input, _ := json.Marshal(browserInput{Action: "close"})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Contains(t, result.Error, "session is required")
}

func TestBrowserCloseUnknownSession(t *testing.T) {
	tool := &BrowserTool{}
	input, _ := json.Marshal(browserInput{Action: "close", Session: "nope"})

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.Empty(t, result.Error)
	assert.Contains(t, result.Output, `No active session "nope"`)
}

// TestBrowserSessionLimit verifies the in-memory bookkeeping refuses a 6th
// session without ever launching Chrome by pre-populating sentinel entries.
func TestBrowserSessionLimit(t *testing.T) {
	tool := &BrowserTool{}
	tool.sessions = make(map[string]*browserSession)
	for i := 0; i < sessionMaxCount; i++ {
		tool.sessions[fmt.Sprintf("s%d", i)] = &browserSession{taskCancel: func() {}, allocCancel: func() {}}
	}
	_, err := tool.getOrCreateSession("overflow")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "session limit reached")
}

func TestBrowserCloseSessionCleansMap(t *testing.T) {
	tool := &BrowserTool{}
	tool.sessions = map[string]*browserSession{
		"a": {taskCancel: func() {}, allocCancel: func() {}},
	}
	require.True(t, tool.closeSession("a"))
	assert.NotContains(t, tool.sessions, "a")
	require.False(t, tool.closeSession("a"))
}
