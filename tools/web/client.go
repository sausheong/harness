package web

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
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
			return safeDialContext(ctx, dialer, isPrivateIP, network, addr)
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

// blockedError marks a destination refused by the SSRF policy (as opposed to
// an ordinary resolve/dial failure), so callers can tell policy from outage.
type blockedError struct{ msg string }

func (e *blockedError) Error() string { return e.msg }

// safeDialContext is the single SSRF-safe dial path shared by SafeHTTPClient
// and the browser egress proxy. It resolves addr's host, fails closed if
// blockIP reports ANY resolved IP (or the host is a known metadata name), then
// dials the exact validated IPs so nothing is re-resolved after validation.
func safeDialContext(ctx context.Context, dialer *net.Dialer, blockIP func(net.IP) bool, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	lower := strings.ToLower(strings.TrimSuffix(host, "."))
	if lower == "metadata.google.internal" || lower == "metadata" {
		return nil, &blockedError{"access to internal metadata endpoint is blocked"}
	}
	ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("cannot resolve %q — blocking to prevent SSRF", host)
	}
	// Fail closed: if ANY resolved IP is private, refuse — this defeats
	// DNS-rebinding where one A record is public and another is private.
	for _, ip := range ips {
		if blockIP(ip) {
			return nil, &blockedError{fmt.Sprintf("access to internal address %s (%s) is blocked", host, ip)}
		}
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("no usable address for %q", host)
	}
	return dialValidated(ctx, dialer, network, interleaveFamilies(ips), port)
}

// happyEyeballsDelay is how long dialValidated waits on one address before
// also starting the next (RFC 8305 uses 250ms).
const happyEyeballsDelay = 300 * time.Millisecond

// interleaveFamilies reorders ips to alternate address families, starting with
// the resolver's first, so a dead family can't stall the other.
func interleaveFamilies(ips []net.IP) []net.IP {
	var v4, v6 []net.IP
	for _, ip := range ips {
		if ip.To4() != nil {
			v4 = append(v4, ip)
		} else {
			v6 = append(v6, ip)
		}
	}
	first, second := v4, v6
	if len(ips) > 0 && ips[0].To4() == nil {
		first, second = v6, v4
	}
	out := make([]net.IP, 0, len(ips))
	for i := 0; i < len(first) || i < len(second); i++ {
		if i < len(first) {
			out = append(out, first[i])
		}
		if i < len(second) {
			out = append(out, second[i])
		}
	}
	return out
}

// dialValidated dials the already-validated literal IPs RFC 8305 style:
// attempts start staggered (or immediately after a failure) and the first to
// connect wins, so dual-stack hosts work on IPv4-only or IPv6-broken networks
// without waiting out a full dial timeout. Stock happy-eyeballs is bypassed
// because we dial literal IPs to pin the validated address.
func dialValidated(ctx context.Context, dialer *net.Dialer, network string, ips []net.IP, port string) (net.Conn, error) {
	if len(ips) == 1 {
		return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0].String(), port))
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type result struct {
		conn net.Conn
		err  error
	}
	ch := make(chan result, len(ips)) // buffered: attempts never block on send
	launched, pending := 0, 0
	launch := func() {
		addr := net.JoinHostPort(ips[launched].String(), port)
		launched++
		pending++
		go func() {
			c, err := dialer.DialContext(ctx, network, addr)
			ch <- result{c, err}
		}()
	}
	// Close any connections that complete after we've returned.
	drain := func() {
		n := pending
		go func() {
			for ; n > 0; n-- {
				if r := <-ch; r.conn != nil {
					_ = r.conn.Close()
				}
			}
		}()
	}
	timer := time.NewTimer(0)
	defer timer.Stop()
	var lastErr error
	for {
		select {
		case <-timer.C:
			if launched < len(ips) {
				launch()
				if launched < len(ips) {
					timer.Reset(happyEyeballsDelay)
				}
			}
		case r := <-ch:
			pending--
			if r.err == nil {
				drain()
				return r.conn, nil
			}
			lastErr = r.err
			if launched == len(ips) && pending == 0 {
				return nil, lastErr
			}
			if launched < len(ips) {
				timer.Reset(0) // failed fast: start the next address now
			}
		case <-ctx.Done():
			drain()
			return nil, ctx.Err()
		}
	}
}
