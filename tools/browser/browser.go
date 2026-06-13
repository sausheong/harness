package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/chromedp"
	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/tool"
	"github.com/sausheong/harness/tools/web"
)

const (
	browserTimeout       = 120 * time.Second
	defaultViewportW     = 1440
	defaultViewportH     = 900
	networkIdleBudget    = 5 * time.Second
	waitForBudget        = 15 * time.Second
	defaultSettleMs      = 1000
	primaryNavigateLimit = 30 * time.Second
	// elementWaitBudget bounds per-action WaitVisible/WaitReady calls so a
	// single bad selector or a stuck readyState can't burn the entire
	// browserTimeout (120s). With this cap, the agent gets a fast failure and
	// has time left to retry with a different selector.
	elementWaitBudget = 25 * time.Second

	sessionIdleTimeout = 10 * time.Minute
	sessionMaxCount    = 5
	janitorInterval    = 1 * time.Minute
)

// hostResolverRules forces Chrome to fail DNS resolution for private/internal
// IP ranges and localhost, so in-page fetch() and evaluate()'d JS cannot reach
// cloud-metadata endpoints or localhost services — the Go-side
// ValidateURLNotInternal check only sees the top-level URL, so this covers
// sub-resource and script-initiated requests. (S3)
//
// Residual gap: name-based rules do not stop a page that dials a *literal*
// private IP URL in some Chrome builds; the complete fix is an allowlisting
// proxy (deferred). See the security-deeper spec's limitations section.
const hostResolverRules = "MAP localhost ~NOTFOUND," +
	"MAP *.localhost ~NOTFOUND," +
	"MAP 127.0.0.0/8 ~NOTFOUND," +
	"MAP 10.0.0.0/8 ~NOTFOUND," +
	"MAP 172.16.0.0/12 ~NOTFOUND," +
	"MAP 169.254.0.0/16 ~NOTFOUND," +
	"MAP 192.168.0.0/16 ~NOTFOUND," +
	"MAP [::1] ~NOTFOUND"

// stealthScript runs before any page script on every new document. It hides the
// most common headless/automation tells that JS-heavy sites use to refuse to
// render content (Cloudflare, Akamai, anti-bot scripts, "please enable JS"
// shells). Without this many SPAs return blank or interstitial pages.
const stealthScript = `(function(){
  try { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); } catch (e) {}
  try {
    if (!navigator.languages || navigator.languages.length === 0) {
      Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    }
  } catch (e) {}
  try { Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] }); } catch (e) {}
  if (!window.chrome) { window.chrome = { runtime: {} }; }
})();`

// browserSession is a long-lived Chrome target keyed by a caller-supplied name.
// One session = one tab. Calls into the same session serialize via mu so that
// chromedp actions don't interleave on the same target (chromedp targets are
// not safe under concurrent Run calls).
type browserSession struct {
	mu          sync.Mutex
	allocCancel context.CancelFunc
	taskCtx     context.Context
	taskCancel  context.CancelFunc
	lastUsed    time.Time
}

// BrowserTool provides headless browser automation via Chrome DevTools Protocol.
//
// Two execution modes:
//   - Ephemeral (no "session" field): each call launches a fresh browser, runs
//     the action, and tears down. Cookies/state do not survive.
//   - Persistent ("session" field): the named browser persists across calls so
//     cookies, auth, scroll position, and SPA state carry over. Sessions
//     auto-close after sessionIdleTimeout, or when "action": "close" is called.
type BrowserTool struct {
	mu          sync.Mutex
	sessions    map[string]*browserSession
	stopJanitor chan struct{}
}

type browserInput struct {
	Action   string `json:"action"`             // navigate, click, type, screenshot, get_text, evaluate, close
	Session  string `json:"session,omitempty"`  // optional session name for cross-call persistence
	URL      string `json:"url,omitempty"`      // for navigate
	Selector string `json:"selector,omitempty"` // CSS selector for click, type, get_text
	Text     string `json:"text,omitempty"`     // for type action
	Script   string `json:"script,omitempty"`   // for evaluate action
	Timeout  int    `json:"timeout,omitempty"`  // seconds, optional
	WaitFor  string `json:"wait_for,omitempty"` // CSS selector to wait visible after navigation
	WaitMs   int    `json:"wait_ms,omitempty"`  // extra settle sleep ms after the page is ready
}

func (t *BrowserTool) Name() string { return "browser" }

func (t *BrowserTool) Description() string {
	return `Control a headless Chrome browser.

CRITICAL — RULE OF SESSIONS:
Each tool call uses an isolated, fresh browser by default. State (current URL, cookies, scroll, SPA hydration) does NOT carry over between calls.

If you intend to call this tool MORE THAN ONCE for the same page or flow, you MUST pass "session": "<any-label>" on EVERY call. Pick a label like "hormuz" or "github-login" and reuse it. Otherwise the second call starts on about:blank and your screenshot is blank, your get_text is empty, your click finds nothing.

WORKED EXAMPLE — visit a page and screenshot it:
  WRONG (two ephemeral browsers):
    1) {"action":"navigate","url":"https://example.com"}
    2) {"action":"screenshot"}                              ← blank, screenshots about:blank
  RIGHT (one browser, two calls):
    1) {"action":"navigate","url":"https://example.com","session":"ex"}
    2) {"action":"screenshot","session":"ex"}
    3) {"action":"close","session":"ex"}                    ← optional; auto-closes after 10 min idle

ONE-SHOT pattern (no session) — pass "url" on every action so each call navigates first inside its own ephemeral browser:
    {"action":"screenshot","url":"https://example.com"}    ← navigates THEN screenshots, single call

ACTIONS:
- "navigate": Go to a URL. Requires "url".
- "click": Click an element. Requires "selector". Optional "url" to navigate first.
- "type": Type text into an input field. Requires "selector" and "text". Optional "url".
- "get_text": Get the inner HTML of an element or the full page. Optional "selector" (defaults to body). Optional "url".
- "screenshot": Take a screenshot of the current page. Optional "url". Returns the image.
- "evaluate": Execute JavaScript in the page. Requires "script". Optional "url".
- "close": Close a persistent session. Requires "session".

JS-HEAVY pages — the tool already waits for body-ready and network-idle automatically, but for SPAs that render content asynchronously the most reliable signal is "wait_for" (CSS selector that appears once content is rendered, e.g. "main article" or "#root .loaded"). "wait_ms" adds extra settle time in milliseconds after network-idle (default 1000). Sessions auto-close after 10 minutes of inactivity (max 5 concurrent).

SELECTORS — standard CSS only. Playwright/jQuery extensions are NOT supported: ":has-text(...)", "text=...", ">>" chains, "/...//xpath" all fail. To match by text, use attribute selectors ([aria-label="…"], [title="…"]), structural selectors (:nth-of-type, :nth-child), or "evaluate" with document.querySelector / XPath via document.evaluate.`
}

func (t *BrowserTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["navigate", "click", "type", "get_text", "screenshot", "evaluate", "close"],
				"description": "The browser action to perform"
			},
			"session": {
				"type": "string",
				"description": "Optional session name. If provided, the browser persists across calls with the same name (cookies, auth, SPA state survive). Omit for one-shot calls. Required for the close action."
			},
			"url": {
				"type": "string",
				"description": "URL to navigate to (required for navigate; optional for other actions)"
			},
			"selector": {
				"type": "string",
				"description": "CSS selector for the target element (required for click, type; optional for get_text)"
			},
			"text": {
				"type": "string",
				"description": "Text to type (required for type action)"
			},
			"script": {
				"type": "string",
				"description": "JavaScript code to evaluate (required for evaluate action)"
			},
			"timeout": {
				"type": "integer",
				"description": "Timeout in seconds (default: 120)"
			},
			"wait_for": {
				"type": "string",
				"description": "Optional CSS selector to wait visible after navigation, before performing the action. Use this for SPAs that render content asynchronously."
			},
			"wait_ms": {
				"type": "integer",
				"description": "Optional extra settle time in milliseconds after the page reaches network-idle (default: 1000)."
			}
		},
		"required": ["action"]
	}`)
}

// browserRegistry tracks every BrowserTool created via NewBrowserTool so the
// process can close all live Chrome subprocesses on shutdown. Tests that
// construct BrowserTool directly (&BrowserTool{}) are intentionally not
// tracked — they never spawn real browsers.
var browserRegistry struct {
	mu    sync.Mutex
	tools []*BrowserTool
}

// NewBrowserTool constructs a BrowserTool and registers it for global
// shutdown. Used by RegisterCoreTools.
func NewBrowserTool() *BrowserTool {
	t := &BrowserTool{}
	browserRegistry.mu.Lock()
	browserRegistry.tools = append(browserRegistry.tools, t)
	browserRegistry.mu.Unlock()
	return t
}

// ShutdownBrowsers closes all live browser sessions across every BrowserTool
// instance the process registered. Safe to call from a signal handler at
// process exit; safe to call when no sessions were ever opened.
func ShutdownBrowsers() {
	browserRegistry.mu.Lock()
	all := browserRegistry.tools
	browserRegistry.tools = nil
	browserRegistry.mu.Unlock()
	for _, t := range all {
		t.Shutdown()
	}
}

// Shutdown closes all live sessions and stops the idle janitor. Safe to call
// multiple times; safe to call before any session was ever created.
func (t *BrowserTool) Shutdown() {
	t.mu.Lock()
	defer t.mu.Unlock()
	for name, sess := range t.sessions {
		sess.taskCancel()
		sess.allocCancel()
		delete(t.sessions, name)
	}
	if t.stopJanitor != nil {
		close(t.stopJanitor)
		t.stopJanitor = nil
	}
}

// IsConcurrencySafe returns false — browser sessions are stateful (navigation,
// cookies, scroll position) and not safe to drive concurrently.
func (t *BrowserTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

func (t *BrowserTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	var in browserInput
	if err := json.Unmarshal(input, &in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("invalid input: %v", err)}, nil
	}
	if in.Action == "" {
		return tool.ToolResult{Error: "action is required"}, nil
	}

	// "close" is a session-management action; handled before any browser launch.
	if in.Action == "close" {
		if in.Session == "" {
			return tool.ToolResult{Error: "session is required for close action"}, nil
		}
		if t.closeSession(in.Session) {
			return tool.ToolResult{Output: fmt.Sprintf("Closed session %q", in.Session)}, nil
		}
		return tool.ToolResult{Output: fmt.Sprintf("No active session %q", in.Session)}, nil
	}

	// Validate URL early so a bad URL never launches Chrome.
	if in.URL != "" {
		if !strings.HasPrefix(in.URL, "http://") && !strings.HasPrefix(in.URL, "https://") {
			return tool.ToolResult{Error: "url must start with http:// or https://"}, nil
		}
		if err := web.ValidateURLNotInternal(in.URL); err != nil {
			return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
		}
	}
	if in.Action == "navigate" && in.URL == "" {
		return tool.ToolResult{Error: "url is required for navigate action"}, nil
	}
	switch in.Action {
	case "click":
		if in.Selector == "" {
			return tool.ToolResult{Error: "selector is required for click action"}, nil
		}
	case "type":
		if in.Selector == "" {
			return tool.ToolResult{Error: "selector is required for type action"}, nil
		}
		if in.Text == "" {
			return tool.ToolResult{Error: "text is required for type action"}, nil
		}
	case "evaluate":
		if in.Script == "" {
			return tool.ToolResult{Error: "script is required for evaluate action"}, nil
		}
	}

	// Catch the about:blank trap: any post-navigation action (screenshot,
	// get_text, evaluate, click, type) called WITHOUT a URL AND WITHOUT a
	// session will land in a fresh browser on about:blank — producing a
	// blank screenshot, empty text, or a no-op script. Each tool call uses
	// its own browser; state does not survive between calls unless "session"
	// is supplied. Fail fast with an actionable hint instead of silently
	// returning useless output.
	if in.URL == "" && in.Session == "" {
		switch in.Action {
		case "screenshot", "get_text", "evaluate", "click", "type":
			return tool.ToolResult{Error: fmt.Sprintf(
				"%s called without 'url' or 'session'. Each browser call uses a fresh browser starting on about:blank, so state does NOT carry over between calls. Either: (a) pass 'url' to navigate first, or (b) pass the same 'session: \"<name>\"' on every call in a multi-step flow (navigate, then act, then close).",
				in.Action,
			)}, nil
		}
	}

	timeout := browserTimeout
	if in.Timeout > 0 {
		timeout = time.Duration(in.Timeout) * time.Second
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	taskCtx, release, err := t.acquireBrowser(callCtx, in.Session, timeout)
	if err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	defer release()

	return t.dispatch(taskCtx, in)
}

// acquireBrowser returns a chromedp task context the caller can pass to
// chromedp.Run, plus a release function that MUST be deferred. For ephemeral
// calls (session == ""), release tears down the browser. For persistent
// sessions, release just unlocks the session and updates lastUsed; the browser
// keeps running until close or idle timeout.
func (t *BrowserTool) acquireBrowser(callCtx context.Context, session string, timeout time.Duration) (context.Context, func(), error) {
	if session == "" {
		taskCtx, cleanup, err := launchBrowser(callCtx)
		if err != nil {
			return nil, nil, fmt.Errorf("browser launch failed: %w", err)
		}
		return taskCtx, cleanup, nil
	}

	sess, err := t.getOrCreateSession(session)
	if err != nil {
		return nil, nil, err
	}
	sess.mu.Lock()

	// If the session's underlying browser died (canceled, crashed), discard and recreate.
	if sess.taskCtx.Err() != nil {
		sess.taskCancel()
		sess.allocCancel()
		sess.mu.Unlock()
		t.removeSession(session)
		// Recurse once to retry with a fresh session under the same name.
		return t.acquireBrowser(callCtx, session, timeout)
	}

	// Per-call deadline derived from the session's task ctx — child contexts
	// share the chromedp target, so chromedp.Run drives the same browser tab.
	callTaskCtx, callTaskCancel := context.WithTimeout(sess.taskCtx, timeout)
	release := func() {
		callTaskCancel()
		sess.lastUsed = time.Now()
		sess.mu.Unlock()
	}
	return callTaskCtx, release, nil
}

func (t *BrowserTool) getOrCreateSession(name string) (*browserSession, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.sessions == nil {
		t.sessions = make(map[string]*browserSession)
	}
	if sess, ok := t.sessions[name]; ok {
		return sess, nil
	}
	if len(t.sessions) >= sessionMaxCount {
		return nil, fmt.Errorf("session limit reached (%d). Close an existing session before opening %q", sessionMaxCount, name)
	}

	// Persistent browsers are rooted on context.Background so they outlive any
	// single call's deadline. They die only on explicit close, Shutdown, idle
	// timeout, or process exit.
	taskCtx, cleanup, err := launchBrowser(context.Background())
	if err != nil {
		return nil, fmt.Errorf("browser launch failed: %w", err)
	}
	// We need separate cancels for taskCtx vs allocCtx so closeSession can tear
	// down deterministically. launchBrowser returns a single cleanup that does
	// both — wrap it.
	sess := &browserSession{
		taskCtx:     taskCtx,
		taskCancel:  cleanup, // cleanup tears down both task + alloc
		allocCancel: func() {}, // no-op; cleanup already covers alloc
		lastUsed:    time.Now(),
	}
	t.sessions[name] = sess
	t.startJanitorOnce()
	return sess, nil
}

func (t *BrowserTool) closeSession(name string) bool {
	t.mu.Lock()
	sess, ok := t.sessions[name]
	if ok {
		delete(t.sessions, name)
	}
	t.mu.Unlock()
	if !ok {
		return false
	}
	sess.taskCancel()
	sess.allocCancel()
	return true
}

// removeSession deletes a session from the map without touching its cancels —
// caller has already canceled. Used by acquireBrowser's stale-session path.
func (t *BrowserTool) removeSession(name string) {
	t.mu.Lock()
	delete(t.sessions, name)
	t.mu.Unlock()
}

func (t *BrowserTool) startJanitorOnce() {
	if t.stopJanitor != nil {
		return
	}
	t.stopJanitor = make(chan struct{})
	stop := t.stopJanitor
	go func() {
		ticker := time.NewTicker(janitorInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				t.reapIdleSessions()
			}
		}
	}()
}

func (t *BrowserTool) reapIdleSessions() {
	now := time.Now()
	type victim struct {
		name string
		sess *browserSession
	}
	var victims []victim
	t.mu.Lock()
	for name, sess := range t.sessions {
		// Read lastUsed under the session's mutex; if a call is in flight we
		// skip this round (the call holds the lock).
		if !sess.mu.TryLock() {
			continue
		}
		idle := now.Sub(sess.lastUsed)
		sess.mu.Unlock()
		if idle >= sessionIdleTimeout {
			victims = append(victims, victim{name, sess})
			delete(t.sessions, name)
		}
	}
	t.mu.Unlock()
	for _, v := range victims {
		v.sess.taskCancel()
		v.sess.allocCancel()
	}
}

// launchBrowser starts a fresh Chrome and returns its task context plus a
// teardown function that closes both the task and the underlying allocator.
//
// Each launch gets its own user-data-dir created up-front so the path is
// known to both Chrome (it appears in argv as --user-data-dir=…) and the
// reaper watchdog. If the parent dies hard before cleanup runs, the
// watchdog SIGKILLs any process whose argv contains that path and rm -rfs
// the dir.
func launchBrowser(parent context.Context) (context.Context, context.CancelFunc, error) {
	udd, err := os.MkdirTemp("", "harness-browser-")
	if err != nil {
		return nil, nil, fmt.Errorf("create user-data-dir: %w", err)
	}
	trackDirForReaping(udd)

	allocCtx, allocCancel := chromedp.NewExecAllocator(parent,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.NoSandbox,
			chromedp.WindowSize(defaultViewportW, defaultViewportH),
			chromedp.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
			chromedp.Flag("disable-blink-features", "AutomationControlled"),
			chromedp.Flag("lang", "en-US"),
			chromedp.Flag("host-resolver-rules", hostResolverRules),
			chromedp.UserDataDir(udd),
		)...,
	)
	taskCtx, taskCancel := chromedp.NewContext(allocCtx)
	if err := chromedp.Run(taskCtx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			if _, err := page.AddScriptToEvaluateOnNewDocument(stealthScript).Do(ctx); err != nil {
				return err
			}
			return page.SetLifecycleEventsEnabled(true).Do(ctx)
		}),
	); err != nil {
		taskCancel()
		allocCancel()
		untrackDirForReaping(udd)
		_ = os.RemoveAll(udd)
		return nil, nil, err
	}
	cleanup := func() {
		taskCancel()
		allocCancel()
		untrackDirForReaping(udd)
		_ = os.RemoveAll(udd)
	}
	return taskCtx, cleanup, nil
}

func (t *BrowserTool) dispatch(taskCtx context.Context, in browserInput) (tool.ToolResult, error) {
	switch in.Action {
	case "navigate":
		return t.navigate(taskCtx, in)
	case "click":
		return t.click(taskCtx, in)
	case "type":
		return t.typeText(taskCtx, in)
	case "get_text":
		return t.getText(taskCtx, in)
	case "screenshot":
		return t.screenshot(taskCtx, in)
	case "evaluate":
		return t.evaluate(taskCtx, in)
	default:
		return tool.ToolResult{Error: fmt.Sprintf("unknown action: %q (valid: navigate, click, type, get_text, screenshot, evaluate, close)", in.Action)}, nil
	}
}

// navigateIfNeeded navigates to in.URL if non-empty, then waits for the page to
// be ready: body-visible, network-idle (soft 5s budget), an optional caller-
// supplied wait_for selector, and a small settle sleep. Returns nil if no URL
// was supplied (lets persistent sessions act on the already-loaded page).
//
// chromedp.Navigate waits for the load event, which is the most reliable
// signal but can hang on pages with stuck network calls (analytics scripts,
// never-resolving fetches). On load-wait timeout we retry with the low-level
// page.Navigate which returns as soon as navigation is committed.
func (t *BrowserTool) navigateIfNeeded(ctx context.Context, in browserInput) error {
	if in.URL == "" {
		return nil
	}
	return t.doNavigate(ctx, in)
}

func (t *BrowserTool) doNavigate(ctx context.Context, in browserInput) error {
	// Listener must be registered BEFORE navigation so we don't miss the
	// networkIdle event for fast pages.
	idleCh := make(chan struct{}, 1)
	chromedp.ListenTarget(ctx, func(ev any) {
		if e, ok := ev.(*page.EventLifecycleEvent); ok && e.Name == "networkIdle" {
			select {
			case idleCh <- struct{}{}:
			default:
			}
		}
	})

	primaryCtx, cancel := halfDeadlineContext(ctx, primaryNavigateLimit)
	defer cancel()
	err := chromedp.Run(primaryCtx,
		chromedp.Navigate(in.URL),
		chromedp.WaitReady("body"),
	)
	if err != nil {
		if !errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		// Fallback: low-level navigate that returns as soon as nav is committed.
		if err := chromedp.Run(ctx,
			chromedp.ActionFunc(func(ctx context.Context) error {
				_, _, _, _, err := page.Navigate(in.URL).Do(ctx)
				return err
			}),
			chromedp.WaitReady("body"),
		); err != nil {
			return err
		}
	}

	// Soft-wait for network to settle. WaitReady("body") fires before SPA
	// scripts have rendered content; the networkIdle lifecycle event is the
	// real "page is done loading" signal. Time out without erroring so pages
	// with persistent connections (websockets, long-poll) still proceed.
	select {
	case <-idleCh:
	case <-time.After(networkIdleBudget):
	case <-ctx.Done():
		return ctx.Err()
	}

	// Caller-supplied gating selector — strongest signal for SPAs.
	if in.WaitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, waitForBudget)
		defer waitCancel()
		if err := chromedp.Run(waitCtx, chromedp.WaitVisible(in.WaitFor)); err != nil {
			return fmt.Errorf("wait_for %q: %w", in.WaitFor, err)
		}
	}

	// Final settle sleep so post-render JS (animations, lazy hydration,
	// follow-up XHRs that don't reset networkIdle) gets a chance to run.
	settleMs := in.WaitMs
	if settleMs <= 0 {
		settleMs = defaultSettleMs
	}
	return chromedp.Run(ctx, chromedp.Sleep(time.Duration(settleMs)*time.Millisecond))
}

// waitForElement runs a wait action (WaitVisible / WaitReady / etc.) under a
// bounded sub-context so a single bad selector or stuck readyState can't burn
// the entire browser call budget. Returns a friendly error on timeout that
// names the most common causes — Playwright-style selectors and unrendered
// SPA content — instead of the opaque "context deadline exceeded".
func waitForElement(ctx context.Context, action, selector string, wait chromedp.Action) error {
	waitCtx, cancel := context.WithTimeout(ctx, elementWaitBudget)
	defer cancel()
	if err := chromedp.Run(waitCtx, wait); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("%s timed out waiting for selector %q after %ds. Common causes: invalid CSS selector (Playwright-style :has-text()/text=/>> are NOT supported — use attribute selectors or evaluate); element never appears (try a different selector or pass wait_for/wait_ms); page never finishes loading",
				action, selector, int(elementWaitBudget.Seconds()))
		}
		return fmt.Errorf("%s failed on %q: %w", action, selector, err)
	}
	return nil
}

// halfDeadlineContext returns a context whose deadline is half of the parent's
// remaining time, clamped to a minimum so it doesn't collapse to nothing on
// already-near-expiry contexts. If the parent has no deadline, the minimum is
// used as the absolute timeout.
func halfDeadlineContext(parent context.Context, minTimeout time.Duration) (context.Context, context.CancelFunc) {
	if dl, ok := parent.Deadline(); ok {
		remaining := time.Until(dl)
		half := max(remaining/2, minTimeout)
		return context.WithTimeout(parent, half)
	}
	return context.WithTimeout(parent, minTimeout)
}

func (t *BrowserTool) navigate(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	if in.URL == "" {
		return tool.ToolResult{Error: "url is required for navigate action"}, nil
	}
	if !strings.HasPrefix(in.URL, "http://") && !strings.HasPrefix(in.URL, "https://") {
		return tool.ToolResult{Error: "url must start with http:// or https://"}, nil
	}

	if err := t.doNavigate(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	// Title is read AFTER the readiness wait so SPAs that set <title> from JS
	// return their real title rather than the static placeholder.
	var title string
	if err := chromedp.Run(ctx, chromedp.Title(&title)); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("title failed: %v", err)}, nil
	}

	return tool.ToolResult{
		Output: fmt.Sprintf("Navigated to %s\nPage title: %s", in.URL, title),
		Metadata: map[string]any{
			"url":   in.URL,
			"title": title,
		},
	}, nil
}

func (t *BrowserTool) click(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	if in.Selector == "" {
		return tool.ToolResult{Error: "selector is required for click action"}, nil
	}

	if err := t.navigateIfNeeded(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	if err := waitForElement(ctx, "click", in.Selector, chromedp.WaitVisible(in.Selector)); err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	if err := chromedp.Run(ctx, chromedp.Click(in.Selector)); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("click failed on %q: %v", in.Selector, err)}, nil
	}

	return tool.ToolResult{Output: fmt.Sprintf("Clicked element: %s", in.Selector)}, nil
}

func (t *BrowserTool) typeText(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	if in.Selector == "" {
		return tool.ToolResult{Error: "selector is required for type action"}, nil
	}
	if in.Text == "" {
		return tool.ToolResult{Error: "text is required for type action"}, nil
	}

	if err := t.navigateIfNeeded(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	if err := waitForElement(ctx, "type", in.Selector, chromedp.WaitVisible(in.Selector)); err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	if err := chromedp.Run(ctx,
		chromedp.Clear(in.Selector),
		chromedp.SendKeys(in.Selector, in.Text),
	); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("type failed on %q: %v", in.Selector, err)}, nil
	}

	return tool.ToolResult{Output: fmt.Sprintf("Typed text into element: %s", in.Selector)}, nil
}

func (t *BrowserTool) getText(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	selector := in.Selector
	if selector == "" {
		selector = "body"
	}

	if err := t.navigateIfNeeded(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	if err := waitForElement(ctx, "get_text", selector, chromedp.WaitReady(selector)); err != nil {
		return tool.ToolResult{Error: err.Error()}, nil
	}
	var text string
	if err := chromedp.Run(ctx, chromedp.InnerHTML(selector, &text)); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("get_text failed on %q: %v", selector, err)}, nil
	}

	if len(text) > web.MaxOutputLength {
		text = text[:web.MaxOutputLength] + "\n\n[Content truncated]"
	}

	return tool.ToolResult{Output: text}, nil
}

func (t *BrowserTool) screenshot(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	if err := t.navigateIfNeeded(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	var buf []byte
	err := chromedp.Run(ctx,
		chromedp.FullScreenshot(&buf, 90),
	)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("screenshot failed: %v", err)}, nil
	}

	return tool.ToolResult{
		Output: fmt.Sprintf("Screenshot captured (%d bytes). The image is attached for visual inspection.", len(buf)),
		Images: []llm.ImageContent{
			{MimeType: "image/jpeg", Data: buf},
		},
		Metadata: map[string]any{
			"format": "jpeg",
			"size":   len(buf),
		},
	}, nil
}

func (t *BrowserTool) evaluate(ctx context.Context, in browserInput) (tool.ToolResult, error) {
	if in.Script == "" {
		return tool.ToolResult{Error: "script is required for evaluate action"}, nil
	}

	if err := t.navigateIfNeeded(ctx, in); err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("navigate failed: %v", err)}, nil
	}

	var result any
	err := chromedp.Run(ctx,
		chromedp.Evaluate(in.Script, &result),
	)
	if err != nil {
		return tool.ToolResult{Error: fmt.Sprintf("evaluate failed: %v", err)}, nil
	}

	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return tool.ToolResult{Output: fmt.Sprintf("%v", result)}, nil
	}

	return tool.ToolResult{Output: string(output)}, nil
}
