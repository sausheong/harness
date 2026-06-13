package web

import (
	"net"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateURLNotInternal_BlocksLoopback(t *testing.T) {
	cases := []string{
		"http://127.0.0.1/",
		"http://localhost/",
		"http://[::1]/",
		"http://10.0.0.1/",
		"http://172.16.0.1/",
		"http://192.168.1.1/",
		"http://169.254.169.254/",   // AWS/GCP/Azure metadata IP
		"http://metadata.google.internal/", // GCP metadata host
	}
	for _, u := range cases {
		t.Run(u, func(t *testing.T) {
			err := ValidateURLNotInternal(u)
			assert.Error(t, err, "url %s should be blocked as internal", u)
		})
	}
}

func TestValidateURLNotInternal_AllowsPublic(t *testing.T) {
	for _, u := range []string{
		"https://example.com/",
		"https://api.github.com/repos/foo/bar",
		"http://1.1.1.1/",
	} {
		err := ValidateURLNotInternal(u)
		assert.NoError(t, err, "public url %s should be allowed", u)
	}
}

func TestValidateURLNotInternal_RejectsBadScheme(t *testing.T) {
	// file:// URLs lack a host, so SSRF guard rejects on parse — either
	// "no host" or a scheme/http complaint is acceptable; what matters is
	// it does not silently allow the URL.
	err := ValidateURLNotInternal("file:///etc/passwd")
	assert.Error(t, err, "non-http schemes must be rejected")
	_ = strings.TrimSpace(err.Error())
}

func TestIsPrivateIP_BlocksReservedRanges(t *testing.T) {
	blocked := []string{
		"0.0.0.0",         // unspecified / routes to loopback on Linux
		"0.0.0.1",         // 0.0.0.0/8
		"100.64.0.0",      // CGNAT 100.64.0.0/10
		"100.127.255.255", // CGNAT upper
		"192.0.0.1",       // 192.0.0.0/24 IETF protocol assignments
		"::",              // IPv6 unspecified ::/128
	}
	for _, s := range blocked {
		ip := net.ParseIP(s)
		require.NotNil(t, ip, "parse %s", s)
		require.True(t, isPrivateIP(ip), "%s must be treated as private/blocked", s)
	}
}

func TestIsPrivateIP_AllowsPublic(t *testing.T) {
	pub := []string{"8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"}
	for _, s := range pub {
		ip := net.ParseIP(s)
		require.NotNil(t, ip, "parse %s", s)
		require.False(t, isPrivateIP(ip), "%s must be allowed (public)", s)
	}
}
