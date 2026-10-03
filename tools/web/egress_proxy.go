package web

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	egressDialTimeout = 30 * time.Second
	halfCloseGrace    = 60 * time.Second
)

// EgressProxy is a loopback HTTP proxy that only forwards to public
// destinations. Point a browser at it (--proxy-server) so every request it
// makes — top-level, sub-resource, in-page fetch, WebSocket — is validated at
// dial time against the same private-range policy as SafeHTTPClient. Names
// that resolve to private IPs (nip.io, DNS rebinding) are caught because the
// check runs on the resolved address, which is then dialed directly.
//
// It speaks CONNECT (https/wss) and absolute-URI HTTP (http/ws). It does not
// follow redirects or honor HTTP_PROXY: a 3xx goes back to the browser, whose
// next request re-enters the proxy and is validated again.
type EgressProxy struct {
	ln     net.Listener
	srv    *http.Server
	tr     *http.Transport
	dialer *net.Dialer
	// blockIP reports whether a resolved IP must be refused. Defaults to
	// isPrivateIP; tests override it to reach loopback httptest servers.
	blockIP func(net.IP) bool

	mu     sync.Mutex
	allow  map[string]struct{} // loopback ports reachable despite blockIP
	conns  map[net.Conn]struct{}
	closed bool
	wg     sync.WaitGroup
	once   sync.Once
}

// StartEgressProxy starts a proxy on 127.0.0.1 with an ephemeral port.
func StartEgressProxy() (*EgressProxy, error) {
	return startEgressProxy(isPrivateIP)
}

func startEgressProxy(blockIP func(net.IP) bool) (*EgressProxy, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("egress proxy listen: %w", err)
	}
	p := &EgressProxy{
		ln:      ln,
		dialer:  &net.Dialer{Timeout: egressDialTimeout},
		blockIP: blockIP,
		conns:   make(map[net.Conn]struct{}),
	}
	p.tr = &http.Transport{
		Proxy:              nil, // never chain through HTTP_PROXY
		DialContext:        p.dial,
		DisableCompression: true, // pass bytes through untouched
		MaxIdleConns:       32,
		IdleConnTimeout:    90 * time.Second,
	}
	rp := &httputil.ReverseProxy{
		Rewrite:      func(r *httputil.ProxyRequest) { r.Out.Header.Del("Proxy-Connection") },
		Transport:    p.tr,
		ErrorHandler: p.proxyError,
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodConnect:
			p.handleConnect(w, r)
		case r.URL.IsAbs() && r.URL.Host != "" && r.URL.Scheme == "http":
			rp.ServeHTTP(w, r)
		default:
			http.Error(w, "egress proxy: absolute http URL or CONNECT required", http.StatusBadRequest)
		}
	})
	p.srv = &http.Server{Handler: handler, ReadHeaderTimeout: 30 * time.Second, IdleTimeout: 120 * time.Second}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		_ = p.srv.Serve(ln)
	}()
	return p, nil
}

// AllowLoopbackPort lets the browser reach 127.0.0.1:port (and localhost:port)
// through the proxy despite the private-range policy, for a port the caller
// itself owns (e.g. the browser's own debugging port). Nothing else on
// loopback is opened.
func (p *EgressProxy) AllowLoopbackPort(port int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.allow == nil {
		p.allow = make(map[string]struct{})
	}
	p.allow[strconv.Itoa(port)] = struct{}{}
}

// allowedLoopback reports whether addr is an exact allowlisted loopback port.
func (p *EgressProxy) allowedLoopback(addr string) bool {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host != "127.0.0.1" && !strings.EqualFold(strings.TrimSuffix(host, "."), "localhost") {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.allow[port]
	return ok
}

// Addr returns the listen address as host:port.
func (p *EgressProxy) Addr() string { return p.ln.Addr().String() }

// URL returns the proxy URL suitable for --proxy-server.
func (p *EgressProxy) URL() string { return "http://" + p.Addr() }

// Close stops the listener and tears down every in-flight tunnel and upstream
// connection. Idempotent.
func (p *EgressProxy) Close() error {
	var err error
	p.once.Do(func() {
		p.mu.Lock()
		p.closed = true
		for c := range p.conns {
			_ = c.Close()
		}
		p.mu.Unlock()
		err = p.srv.Close()
		p.tr.CloseIdleConnections()
		p.wg.Wait()
	})
	return err
}

// track registers c so Close can sever it. Returns false (and closes c) if
// the proxy is already closed.
func (p *EgressProxy) track(c net.Conn) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		_ = c.Close()
		return false
	}
	p.conns[c] = struct{}{}
	return true
}

func (p *EgressProxy) untrack(c net.Conn) {
	p.mu.Lock()
	delete(p.conns, c)
	p.mu.Unlock()
}

type proxyConn struct {
	net.Conn
	p *EgressProxy
}

func (c *proxyConn) Close() error {
	c.p.untrack(c.Conn)
	return c.Conn.Close()
}

// dial is the only way the proxy reaches upstream: validated, pinned, tracked.
func (p *EgressProxy) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	var conn net.Conn
	var err error
	if p.allowedLoopback(addr) {
		_, port, _ := net.SplitHostPort(addr)
		conn, err = p.dialer.DialContext(ctx, network, net.JoinHostPort("127.0.0.1", port))
	} else {
		conn, err = safeDialContext(ctx, p.dialer, p.blockIP, network, addr)
	}
	if err != nil {
		return nil, err
	}
	if !p.track(conn) {
		return nil, errors.New("egress proxy closed")
	}
	return &proxyConn{Conn: conn, p: p}, nil
}

func (p *EgressProxy) proxyError(w http.ResponseWriter, r *http.Request, err error) {
	writeDialError(w, err)
}

func writeDialError(w http.ResponseWriter, err error) {
	var be *blockedError
	if errors.As(err, &be) {
		http.Error(w, "blocked by egress policy: "+be.msg, http.StatusForbidden)
		return
	}
	http.Error(w, "upstream unavailable", http.StatusBadGateway)
}

func (p *EgressProxy) handleConnect(w http.ResponseWriter, r *http.Request) {
	target := r.Host
	if h, port, err := net.SplitHostPort(target); err != nil || h == "" || port == "" {
		http.Error(w, "egress proxy: CONNECT target must be host:port", http.StatusBadRequest)
		return
	}
	// Validate and dial before hijacking so a refusal is a clean 403.
	up, err := p.dial(r.Context(), "tcp", target)
	if err != nil {
		writeDialError(w, err)
		return
	}
	hj, ok := w.(http.Hijacker)
	if !ok {
		_ = up.Close()
		http.Error(w, "hijack unsupported", http.StatusInternalServerError)
		return
	}
	client, buf, err := hj.Hijack()
	if err != nil {
		_ = up.Close()
		return
	}
	if !p.track(client) {
		_ = up.Close()
		return
	}
	defer func() {
		p.untrack(client)
		_ = client.Close()
		_ = up.Close()
	}()
	if _, err := client.Write([]byte("HTTP/1.1 200 Connection established\r\n\r\n")); err != nil {
		return
	}
	done := make(chan struct{}, 2)
	// Bytes the server already buffered from the client belong upstream.
	go func() {
		_, _ = io.Copy(up, buf)
		_, _ = io.Copy(up, client)
		closeWrite(up) // propagate the client's FIN, keep reading the reply
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(client, up)
		closeWrite(client)
		done <- struct{}{}
	}()
	// Wait for both directions (half-close is legal), but once one side has
	// finished give the other a bounded grace so a peer that vanished without
	// closing can't pin the tunnel. Close() also severs the conns.
	<-done
	grace := time.AfterFunc(halfCloseGrace, func() {
		_ = client.Close()
		_ = up.Close()
	})
	<-done
	grace.Stop()
}

// closeWrite half-closes c's write side when it supports it, else fully closes.
func closeWrite(c net.Conn) {
	if pc, ok := c.(*proxyConn); ok {
		c = pc.Conn
	}
	if cw, ok := c.(interface{ CloseWrite() error }); ok {
		_ = cw.CloseWrite()
		return
	}
	_ = c.Close()
}
