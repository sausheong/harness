//go:build live

package browser

import (
	"context"
	"testing"
	"time"

	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/stretchr/testify/require"
)

// TestInPageFetchPrivateIPBlocked_Live verifies host-resolver-rules stops an
// in-page fetch to a metadata IP. Requires a real Chrome. Run:
//   go test -tags live ./tools/browser/ -run TestInPageFetchPrivateIPBlocked_Live
func TestInPageFetchPrivateIPBlocked_Live(t *testing.T) {
	tool := NewBrowserTool()
	defer tool.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	taskCtx, ccancel, err := launchBrowser(ctx)
	require.NoError(t, err)
	defer ccancel()

	require.NoError(t, chromedp.Run(taskCtx, chromedp.Navigate("https://example.com")))

	var result string
	script := `(async () => {
		try {
			await fetch('http://169.254.169.254/latest/meta-data/', {mode:'no-cors'});
			return 'REACHED';
		} catch (e) {
			return 'BLOCKED';
		}
	})()`
	err = chromedp.Run(taskCtx, chromedp.Evaluate(script, &result, func(p *runtime.EvaluateParams) *runtime.EvaluateParams {
		return p.WithAwaitPromise(true)
	}))
	require.NoError(t, err)
	require.Equal(t, "BLOCKED", result, "in-page fetch to metadata IP must be blocked by host-resolver-rules")
}
