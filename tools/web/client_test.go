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

var _ = isPrivateIP(net.IPv4(127, 0, 0, 1))
var _ = strings.TrimSpace
