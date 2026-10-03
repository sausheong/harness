//go:build live

package browser

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/stretchr/testify/require"
)

// TestInPageFetchPrivateIPBlocked_Live verifies the egress proxy stops
// in-page fetches and navigations to loopback, private-resolving names and the
// metadata IP, without any request reaching a local server. Requires a real
// Chrome and network access. Run:
//
//	go test -tags live ./tools/browser/ -run TestInPageFetchPrivateIPBlocked_Live
func TestInPageFetchPrivateIPBlocked_Live(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		fmt.Fprint(w, "secret")
	}))
	defer srv.Close()
	port := srv.Listener.Addr().(interface{ String() string }).String()
	port = port[strings.LastIndex(port, ":")+1:]

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	taskCtx, ccancel, err := launchBrowser(ctx)
	require.NoError(t, err)
	defer ccancel()

	// Positive control: public sites work through the proxy.
	var title string
	require.NoError(t, chromedp.Run(taskCtx, chromedp.Navigate("https://example.com"), chromedp.Title(&title)))
	require.Contains(t, title, "Example")

	targets := []string{
		"http://127.0.0.1:" + port,
		"http://localhost:" + port,
		"http://127.0.0.1.nip.io:" + port,
		"http://169.254.169.254/latest/meta-data/",
	}
	for _, target := range targets {
		var result string
		script := fmt.Sprintf(`(async () => {
			const ac = new AbortController();
			const timer = setTimeout(() => ac.abort(), 10000);
			try {
				await fetch(%q, {signal: ac.signal});
				return 'REACHED';
			} catch (e) {
				return 'BLOCKED';
			} finally { clearTimeout(timer); }
		})()`, target)
		start := time.Now()
		err = chromedp.Run(taskCtx, chromedp.Evaluate(script, &result, func(p *runtime.EvaluateParams) *runtime.EvaluateParams {
			return p.WithAwaitPromise(true)
		}))
		require.NoError(t, err, target)
		require.Equal(t, "BLOCKED", result, "in-page fetch to %s must be blocked", target)
		require.Less(t, time.Since(start), 8*time.Second, "%s should be refused quickly, not time out", target)
	}

	navCtx, navCancel := context.WithTimeout(taskCtx, 20*time.Second)
	defer navCancel()
	// The proxy answers 403, which Chrome renders as an ordinary page, so
	// accept either a navigation error or the proxy's refusal text.
	var body string
	err = chromedp.Run(navCtx, chromedp.Navigate("http://127.0.0.1.nip.io:"+port), chromedp.Text("body", &body, chromedp.ByQuery))
	if err == nil {
		require.Contains(t, body, "blocked by egress policy", "direct navigation to a private-resolving name must be refused")
	}

	require.EqualValues(t, 0, hits.Load(), "local server must never be reached")
}
