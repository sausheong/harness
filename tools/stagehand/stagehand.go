// Package stagehand provides a browser tool built on Stagehand
// (github.com/browserbase/stagehand), which drives the browser with
// natural-language instructions — "click the sign in button", "extract every
// invoice in the table" — instead of CSS selectors. Stagehand's own inference
// runs through the agent's harness LLMProvider (see NewGenerateFunc).
//
// Two backends: a local Chrome launched on this machine (default), or a
// Browserbase cloud browser when Config.BrowserbaseAPIKey is set.
package stagehand

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	sh "github.com/browserbase/stagehand/packages/sdk-go/v4"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/web"
)

const (
	// callTimeout bounds one tool call. act/extract each make one or more
	// LLM round trips on top of browser work, so this is looser than a
	// plain CDP action budget.
	callTimeout = 180 * time.Second
	// launchTimeout bounds browser launch + Stagehand init.
	launchTimeout = 90 * time.Second
	// closeTimeout bounds teardown so a wedged browser can't hang shutdown.
	closeTimeout = 15 * time.Second

	defaultViewportW = 1440
	defaultViewportH = 900
	maxTextOutput    = 100_000

	sessionIdleTimeout = 10 * time.Minute
	sessionMaxCount    = 5
	janitorInterval    = 1 * time.Minute
)

// Config configures a StagehandTool.
type Config struct {
	// Provider and Model serve Stagehand's internal inference. Required.
	Provider llm.LLMProvider
	Model    string

	// BrowserbaseAPIKey selects the Browserbase cloud backend when non-empty;
	// otherwise a local Chrome is launched. The tool never reads the
	// environment itself — callers opt in explicitly, e.g.
	// os.Getenv("BROWSERBASE_API_KEY").
	BrowserbaseAPIKey string

	// Headful shows the local Chrome window. Ignored for Browserbase.
	Headful bool
	// ChromePath overrides the local Chrome executable. Empty = auto-detect.
	ChromePath string
}

// session is one browser + Stagehand client pair. Calls into the same
// session serialize via mu: a Stagehand client drives one active page and
// interleaved act/navigate calls would race on it.
type session struct {
	mu       sync.Mutex
	browser  *sh.Browser
	client   *sh.Stagehand
	lastUsed time.Time
	// proxy is the egress proxy a local Chrome is forced through; nil for
	// Browserbase and in tests.
	proxy *web.EgressProxy
}

// StagehandTool is an AI-driven browser tool.
//
// Like tools/browser it has two execution modes:
//   - Ephemeral (no "session"): each call launches a browser, runs the
//     action, and tears down. Requires "url".
//   - Persistent ("session"): the named browser persists across calls so
//     cookies, auth and page state carry over. Sessions auto-close after
//     sessionIdleTimeout, or on "action": "close".
type StagehandTool struct {
	cfg Config

	// launch is the browser+client factory; tests replace it.
	launch func(ctx context.Context) (*session, error)

	mu          sync.Mutex
	sessions    map[string]*session
	stopJanitor chan struct{}
}

type stagehandInput struct {
	Action      string          `json:"action"`
	Session     string          `json:"session,omitempty"`
	URL         string          `json:"url,omitempty"`
	Instruction string          `json:"instruction,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	FullPage    bool            `json:"full_page,omitempty"`
}

// NewStagehandTool validates cfg and returns a tool. No browser is launched
// until the first call.
func NewStagehandTool(cfg Config) (*StagehandTool, error) {
	if cfg.Provider == nil {
		return nil, errors.New("stagehand: Config.Provider is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("stagehand: Config.Model is required")
	}
	t := &StagehandTool{cfg: cfg}
	t.launch = t.launchSession
	return t, nil
}

func (t *StagehandTool) Name() string { return "stagehand" }

func (t *StagehandTool) Description() string {
	return `Control a web browser with natural-language instructions (Stagehand). Prefer this over the "browser" tool when you don't know the page's CSS selectors: describe WHAT to do and Stagehand finds the elements.

SESSIONS — each call without "session" uses a fresh browser that is closed afterwards, so it must include "url". For any multi-step flow (navigate, then act, then extract), pass the SAME "session": "<label>" on EVERY call. Sessions auto-close after 10 minutes idle (max 5).

ACTIONS:
- "navigate": Go to "url". Returns the page title and final URL.
- "act": Perform ONE atomic action described by "instruction", e.g. "click the Sign in button", "type 'laptop' into the search box", "select 'Newest' in the sort dropdown". Break multi-step tasks into several act calls.
- "observe": List actionable elements matching "instruction" (optional, e.g. "find the pagination links"). Returns descriptions and selectors without acting.
- "extract": Extract data described by "instruction", e.g. "the title and price of every product". Optional "schema" (JSON Schema object) describes the exact shape you want. Returns JSON.
- "get_text": Return the visible text of the current page.
- "screenshot": Screenshot the current page ("full_page": true for the whole page). Returns the image.
- "close": Close a persistent session. Requires "session".

Any action except "close" accepts "url" to navigate first.

EXAMPLE:
  1) {"action":"navigate","url":"https://news.ycombinator.com","session":"hn"}
  2) {"action":"act","instruction":"click the 'new' link in the top bar","session":"hn"}
  3) {"action":"extract","instruction":"the title and points of the first 5 stories","session":"hn"}
  4) {"action":"close","session":"hn"}

Never put passwords or secrets inside "instruction" — instructions are sent to an LLM.`
}

func (t *StagehandTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["navigate", "act", "observe", "extract", "get_text", "screenshot", "close"],
				"description": "The browser action to perform"
			},
			"session": {
				"type": "string",
				"description": "Optional session name. The browser persists across calls with the same name. Omit for one-shot calls. Required for close."
			},
			"url": {
				"type": "string",
				"description": "URL to navigate to (required for navigate; optional for other actions, which navigate first)"
			},
			"instruction": {
				"type": "string",
				"description": "Natural-language instruction (required for act and extract; optional for observe)"
			},
			"schema": {
				"type": "object",
				"description": "Optional JSON Schema describing the data shape for extract"
			},
			"full_page": {
				"type": "boolean",
				"description": "For screenshot: capture the full scrollable page instead of the viewport"
			}
		},
		"required": ["action"]
	}`)
}

// IsConcurrencySafe returns false — sessions are stateful.
func (t *StagehandTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

func (t *StagehandTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in stagehandInput
	if err := json.Unmarshal(input, &in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
	}
	if msg := validate(in); msg != "" {
		return tool.ToolResult{Error: msg}, nil
	}

	if in.Action == "close" {
		if t.closeSession(in.Session) {
			return tool.ToolResult{Output: fmt.Sprintf("Closed session %q", in.Session)}, nil
		}
		return tool.ToolResult{Output: fmt.Sprintf("No active session %q", in.Session)}, nil
	}

	callCtx, cancel := context.WithTimeout(ctx, callTimeout)
	defer cancel()

	sess, release, err := t.acquire(callCtx, in.Session)
	if err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	defer release()

	return t.dispatch(callCtx, sess, in)
}

// validate returns a user-facing error message, or "" when in is valid.
func validate(in stagehandInput) string {
	switch in.Action {
	case "":
		return "action is required"
	case "close":
		if in.Session == "" {
			return "session is required for close action"
		}
		return ""
	case "navigate":
		if in.URL == "" {
			return "url is required for navigate action"
		}
	case "act", "extract":
		if strings.TrimSpace(in.Instruction) == "" {
			return fmt.Sprintf("instruction is required for %s action", in.Action)
		}
	case "observe", "get_text", "screenshot":
	default:
		return fmt.Sprintf("unknown action: %q (valid: navigate, act, observe, extract, get_text, screenshot, close)", in.Action)
	}
	if in.URL != "" {
		if !strings.HasPrefix(in.URL, "http://") && !strings.HasPrefix(in.URL, "https://") {
			return "url must start with http:// or https://"
		}
		if err := web.ValidateURLNotInternal(in.URL); err != nil {
			return fmt.Sprintf("navigate failed: %v", err)
		}
	}
	if in.URL == "" && in.Session == "" {
		return fmt.Sprintf("%s called without 'url' or 'session'. Each call without a session uses a fresh browser on about:blank, so state does NOT carry over. Either pass 'url' to navigate first, or pass the same 'session' on every call in a multi-step flow.", in.Action)
	}
	if len(in.Schema) > 0 && in.Action == "extract" {
		var probe map[string]any
		if err := json.Unmarshal(in.Schema, &probe); err != nil {
			return "schema must be a JSON object"
		}
	}
	return ""
}

func (t *StagehandTool) dispatch(ctx context.Context, sess *session, in stagehandInput) (tool.ToolResult, error) {
	bctx, err := sess.browser.Context()
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("browser context: %v", err)}, nil
	}
	page, err := bctx.ActivePage(ctx)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("active page: %v", err)}, nil
	}
	if in.URL != "" {
		if _, err := page.Goto(ctx, in.URL, nil); err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
		}
	}

	switch in.Action {
	case "navigate":
		title, _ := page.Title(ctx)
		url, _ := page.URL(ctx)
		return tool.ToolResult{Output: fmt.Sprintf("Navigated to %s\nTitle: %s", url, title)}, nil

	case "act":
		res, err := sess.client.Act(ctx, sh.ActInstruction(in.Instruction), &sh.StagehandClientActOptions{Page: page})
		if err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("act failed: %v", err)}, nil
		}
		out := jsonOutput(res.Data)
		if !res.Data.Success {
			return tool.ToolResult{Error: "act did not succeed: " + out}, nil
		}
		return tool.ToolResult{Output: out}, nil

	case "observe":
		var instruction *string
		if in.Instruction != "" {
			instruction = &in.Instruction
		}
		res, err := sess.client.Observe(ctx, instruction, &sh.StagehandClientObserveOptions{Page: page})
		if err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("observe failed: %v", err)}, nil
		}
		if len(res.Data) == 0 {
			return tool.ToolResult{Output: "No matching elements found."}, nil
		}
		return tool.ToolResult{Output: jsonOutput(res.Data)}, nil

	case "extract":
		// Extract derives its schema from a compile-time Go type, so a
		// caller-supplied schema can't be passed through directly. It is
		// appended to the instruction instead, and the result is decoded
		// generically.
		instruction := in.Instruction
		if len(in.Schema) > 0 {
			instruction += "\n\nReturn the data in exactly this JSON Schema shape:\n" + string(in.Schema)
		}
		res, err := sh.Extract[map[string]any](ctx, sess.client, instruction, &sh.StagehandClientExtractOptions{Page: page})
		if err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("extract failed: %v", err)}, nil
		}
		return tool.ToolResult{Output: truncate(jsonOutput(res.Data))}, nil

	case "get_text":
		text, err := sh.EvaluateAs[string](ctx, page, "document.body ? document.body.innerText : ''")
		if err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("get_text failed: %v", err)}, nil
		}
		return tool.ToolResult{Output: truncate(text)}, nil

	case "screenshot":
		opts := &sh.ScreenshotOptions{}
		if in.FullPage {
			full := true
			opts.FullPage = &full
		}
		png, err := page.Screenshot(ctx, opts)
		if err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("screenshot failed: %v", err)}, nil
		}
		url, _ := page.URL(ctx)
		return tool.ToolResult{
			Output: fmt.Sprintf("Screenshot of %s (%d bytes)", url, len(png)),
			Images: []llm.ImageContent{{MimeType: "image/png", Data: png}},
		}, nil
	}
	return tool.ToolResult{Error: fmt.Sprintf("unknown action: %q", in.Action)}, nil
}

// acquire returns a locked session and a release func. Named sessions are
// created on first use and reused; unnamed calls get a throwaway session
// that release tears down.
func (t *StagehandTool) acquire(ctx context.Context, name string) (*session, func(), error) {
	if name == "" {
		sess, err := t.launch(ctx)
		if err != nil {
			return nil, nil, err
		}
		return sess, func() { sess.close() }, nil
	}

	t.mu.Lock()
	sess, ok := t.sessions[name]
	if !ok {
		if len(t.sessions) >= sessionMaxCount {
			t.mu.Unlock()
			return nil, nil, fmt.Errorf("too many open sessions (max %d); close one with {\"action\":\"close\",\"session\":\"<name>\"}", sessionMaxCount)
		}
		// Launch under t.mu so two concurrent first calls for the same name
		// don't each start a browser. Launches are rare and bounded by
		// launchTimeout.
		var err error
		sess, err = t.launch(ctx)
		if err != nil {
			t.mu.Unlock()
			return nil, nil, err
		}
		if t.sessions == nil {
			t.sessions = make(map[string]*session)
		}
		t.sessions[name] = sess
		t.startJanitorLocked()
	}
	t.mu.Unlock()

	sess.mu.Lock()
	sess.lastUsed = time.Now()
	return sess, func() {
		sess.lastUsed = time.Now()
		sess.mu.Unlock()
	}, nil
}

// launchSession starts a browser on the configured backend and attaches a
// Stagehand client whose inference runs through the harness provider.
//
// A local Chrome is forced through an in-process web.EgressProxy that refuses
// private/internal destinations at dial time (loopback, RFC 1918, link-local/
// metadata, names resolving to them, DNS rebinding), covering sub-resource and
// script-initiated requests that the Go-side ValidateURLNotInternal check
// never sees. Chrome's --host-resolver-rules is not used: it only glob-matches
// host strings, so CIDR entries never match. Browserbase browsers run remotely
// and can't reach this machine's network anyway.
//
// The implicit loopback bypass is removed (<-loopback), so localhost and
// 127.0.0.1 are refused too; the one exception is Chrome's own debugging port,
// which Stagehand's extension needs (a page could still reach that port).
func (t *StagehandTool) launchSession(ctx context.Context) (*session, error) {
	// The browser outlives this call for persistent sessions, so detach it
	// from the caller's cancellation and bound only the launch itself.
	launchCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), launchTimeout)
	defer cancel()

	var browser *sh.Browser
	var proxy *web.EgressProxy
	var err error
	if t.cfg.BrowserbaseAPIKey != "" {
		browser, err = sh.LaunchBrowserbase(launchCtx, sh.BrowserbaseLaunchOptions{APIKey: t.cfg.BrowserbaseAPIKey})
	} else {
		proxy, err = web.StartEgressProxy()
		if err != nil {
			return nil, fmt.Errorf("start egress proxy: %w", err)
		}
		// Stagehand's in-browser extension connects to Chrome's own debugging
		// port over loopback, so pick that port up front and let exactly it
		// through the proxy.
		var debugPort int
		debugPort, err = freeLoopbackPort()
		if err != nil {
			_ = proxy.Close()
			return nil, fmt.Errorf("pick debugging port: %w", err)
		}
		proxy.AllowLoopbackPort(debugPort)
		browser, err = sh.LaunchLocalBrowser(launchCtx, &sh.LocalBrowserLaunchOptions{
			Headless:       !t.cfg.Headful,
			ExecutablePath: t.cfg.ChromePath,
			Port:           debugPort,
			Args: []string{
				"--proxy-server=" + proxy.URL(),
				// Drop Chrome's implicit loopback/link-local bypass so those
				// destinations hit the proxy (and are refused).
				"--proxy-bypass-list=<-loopback>",
				// Keep WebRTC from sending UDP around the proxy.
				"--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
				"--webrtc-ip-handling-policy=disable_non_proxied_udp",
			},
			Viewport: &sh.LocalViewport{Width: defaultViewportW, Height: defaultViewportH},
		})
	}
	if err != nil {
		if proxy != nil {
			_ = proxy.Close()
		}
		return nil, fmt.Errorf("launch browser: %w", err)
	}

	client, err := sh.Create(launchCtx, sh.CreateOptions{
		Browser:  browser,
		Generate: NewGenerateFunc(t.cfg.Provider, t.cfg.Model),
		Logging:  &sh.StagehandClientLoggingConfig{Level: sh.StagehandClientLogLevelOff},
	})
	if err != nil {
		closeCtx, cancelClose := context.WithTimeout(context.Background(), closeTimeout)
		defer cancelClose()
		cerr := browser.Close(closeCtx)
		if proxy != nil {
			_ = proxy.Close()
		}
		return nil, errors.Join(fmt.Errorf("start stagehand: %w", err), cerr)
	}
	return &session{browser: browser, client: client, lastUsed: time.Now(), proxy: proxy}, nil
}

// close tears down the client then the browser. Safe on a partially
// constructed session (tests use sessions with nil fields).
func (s *session) close() {
	ctx, cancel := context.WithTimeout(context.Background(), closeTimeout)
	defer cancel()
	if s.client != nil {
		_ = s.client.Close(ctx)
	}
	if s.browser != nil {
		_ = s.browser.Close(ctx)
	}
	if s.proxy != nil {
		_ = s.proxy.Close()
	}
}

func (t *StagehandTool) closeSession(name string) bool {
	t.mu.Lock()
	sess, ok := t.sessions[name]
	if ok {
		delete(t.sessions, name)
	}
	t.mu.Unlock()
	if !ok {
		return false
	}
	// Wait for any in-flight call on this session before tearing it down.
	sess.mu.Lock()
	defer sess.mu.Unlock()
	sess.close()
	return true
}

// Shutdown closes every session and stops the idle janitor. Safe to call
// multiple times.
func (t *StagehandTool) Shutdown() {
	t.mu.Lock()
	all := t.sessions
	t.sessions = nil
	if t.stopJanitor != nil {
		close(t.stopJanitor)
		t.stopJanitor = nil
	}
	t.mu.Unlock()
	for _, sess := range all {
		sess.close()
	}
}

// startJanitorLocked starts the idle reaper if it isn't running. Caller
// holds t.mu.
func (t *StagehandTool) startJanitorLocked() {
	if t.stopJanitor != nil {
		return
	}
	stop := make(chan struct{})
	t.stopJanitor = stop
	go func() {
		ticker := time.NewTicker(janitorInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				t.reapIdle(time.Now())
			}
		}
	}()
}

// reapIdle closes sessions idle longer than sessionIdleTimeout. A session
// whose lock is held is mid-call and therefore not idle.
func (t *StagehandTool) reapIdle(now time.Time) {
	t.mu.Lock()
	var idle []*session
	for name, sess := range t.sessions {
		if !sess.mu.TryLock() {
			continue
		}
		if now.Sub(sess.lastUsed) > sessionIdleTimeout {
			delete(t.sessions, name)
			idle = append(idle, sess)
		} else {
			sess.mu.Unlock()
		}
	}
	t.mu.Unlock()
	for _, sess := range idle {
		sess.close()
		sess.mu.Unlock()
	}
}

func jsonOutput(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func truncate(s string) string {
	if len(s) <= maxTextOutput {
		return s
	}
	return s[:maxTextOutput] + fmt.Sprintf("\n\n[truncated: %d of %d bytes shown]", maxTextOutput, len(s))
}

// freeLoopbackPort returns a currently unused 127.0.0.1 TCP port.
func freeLoopbackPort() (int, error) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}
