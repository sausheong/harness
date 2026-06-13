package web

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"
)

// SafeHTTPClient returns an *http.Client whose transport resolves and validates
// the destination host, then dials the exact validated IP — closing the
// TOCTOU/DNS-rebinding window where a stock http.Client would re-resolve the
// hostname independently of ValidateURLNotInternal. Redirects are re-validated.
func SafeHTTPClient(timeout time.Duration) *http.Client {
	dialer := &net.Dialer{Timeout: timeout}
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}
			ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
			if err != nil {
				return nil, fmt.Errorf("cannot resolve %q — blocking to prevent SSRF", host)
			}
			var dialIP net.IP
			for _, ip := range ips {
				if isPrivateIP(ip) {
					return nil, fmt.Errorf("access to internal address %s (%s) is blocked", host, ip)
				}
				if dialIP == nil {
					dialIP = ip
				}
			}
			if dialIP == nil {
				return nil, fmt.Errorf("no usable address for %q", host)
			}
			return dialer.DialContext(ctx, network, net.JoinHostPort(dialIP.String(), port))
		},
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects (max 10)")
			}
			if err := ValidateURLNotInternal(req.URL.String()); err != nil {
				return fmt.Errorf("redirect blocked: %w", err)
			}
			return nil
		},
	}
}
