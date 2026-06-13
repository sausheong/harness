//go:build live

package browser

import (
	"context"
	"testing"
	"time"

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
	res, err := tool.Execute(ctx, []byte(`{"action":"evaluate","url":"https://example.com","session":"ssrf","script":"fetch('http://169.254.169.254/').then(()=>'REACHED').catch(e=>'BLOCKED:'+e)"}`))
	require.NoError(t, err)
	require.NotContains(t, res.Output, "REACHED")
}
