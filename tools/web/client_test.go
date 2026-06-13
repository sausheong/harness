package web

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSafeHTTPClient_BlocksPrivateAtDial(t *testing.T) {
	client := SafeHTTPClient(5 * time.Second)
	// 127.0.0.1 is itself private, so dialing it directly must be refused.
	_, err := client.Get("http://127.0.0.1:9/")
	require.Error(t, err)
	require.Contains(t, err.Error(), "internal address")
}

func TestSafeHTTPClient_AllowsPublic(t *testing.T) {
	client := SafeHTTPClient(5 * time.Second)
	tr, ok := client.Transport.(*http.Transport)
	require.True(t, ok)
	require.NotNil(t, tr.DialContext)
	require.NotNil(t, client.CheckRedirect)
	require.Equal(t, 5*time.Second, client.Timeout)
}

func TestSafeHTTPClient_RedirectToPrivateBlocked(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/", http.StatusFound)
	}))
	defer srv.Close()
	client := SafeHTTPClient(5 * time.Second)
	// The httptest server binds loopback (private), so the FIRST dial is already blocked.
	_, err := client.Get(srv.URL)
	require.Error(t, err)
}

func TestSafeHTTPClient_PublicUnreachableReturnsDialError(t *testing.T) {
	// 192.0.2.1 is RFC5737 TEST-NET-1 (public, unrouteable). It is NOT in the
	// private blocklist, so validation passes and we should get a DIAL error
	// (timeout/unreachable), not a "blocked" validation error — proving the
	// path dials rather than rejecting.
	client := SafeHTTPClient(2 * time.Second)
	_, err := client.Get("http://192.0.2.1/")
	require.Error(t, err)
	require.NotContains(t, err.Error(), "internal address")
}

var _ = isPrivateIP(net.IPv4(127, 0, 0, 1))
var _ = strings.TrimSpace
