package web

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// allowLoopback lets tests reach httptest servers while still blocking every
// other private range, so the block path stays exercised.
func allowLoopback(ip net.IP) bool {
	if ip.IsLoopback() {
		return false
	}
	return isPrivateIP(ip)
}

func newTestProxy(t *testing.T, block func(net.IP) bool) *EgressProxy {
	t.Helper()
	p, err := startEgressProxy(block)
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// connect sends a CONNECT and returns the status code plus the live conn.
func connect(t *testing.T, p *EgressProxy, target string) (int, net.Conn, *bufio.Reader) {
	t.Helper()
	c, err := net.Dial("tcp", p.Addr())
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	fmt.Fprintf(c, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", target, target)
	br := bufio.NewReader(c)
	resp, err := http.ReadResponse(br, &http.Request{Method: http.MethodConnect})
	require.NoError(t, err)
	return resp.StatusCode, c, br
}

func countingServer(t *testing.T) (*httptest.Server, *atomic.Int64) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		fmt.Fprint(w, "secret")
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func proxyClient(p *EgressProxy) *http.Client {
	u, _ := url.Parse(p.URL())
	return &http.Client{
		Timeout:       10 * time.Second,
		Transport:     &http.Transport{Proxy: http.ProxyURL(u)},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

func TestEgressProxy_ConnectBlocksPrivateTargets(t *testing.T) {
	srv, hits := countingServer(t)
	p := newTestProxy(t, isPrivateIP)
	_, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	for _, target := range []string{
		"127.0.0.1:" + port,
		"localhost:" + port,
		"[::1]:" + port,
		"[::ffff:127.0.0.1]:" + port,
		"169.254.169.254:80",
		"10.0.0.1:443",
		"metadata.google.internal:80",
		"metadata:80",
	} {
		code, _, _ := connect(t, p, target)
		require.Equal(t, http.StatusForbidden, code, target)
	}
	require.EqualValues(t, 0, hits.Load())
}

func TestEgressProxy_PlainHTTPBlocksPrivate(t *testing.T) {
	srv, hits := countingServer(t)
	p := newTestProxy(t, isPrivateIP)
	resp, err := proxyClient(p).Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusForbidden, resp.StatusCode)
	require.EqualValues(t, 0, hits.Load())
}

func TestEgressProxy_PlainHTTPForwards(t *testing.T) {
	srv, hits := countingServer(t)
	p := newTestProxy(t, allowLoopback)
	resp, err := proxyClient(p).Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, "secret", string(body))
	require.EqualValues(t, 1, hits.Load())
}

func TestEgressProxy_RedirectNotFollowed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/", http.StatusFound)
	}))
	defer srv.Close()
	p := newTestProxy(t, allowLoopback)
	resp, err := proxyClient(p).Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusFound, resp.StatusCode)
}

func TestEgressProxy_ConnectTunnelForwards(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		_, _ = io.Copy(c, c) // echo
	}()
	p := newTestProxy(t, allowLoopback)
	code, c, br := connect(t, p, ln.Addr().String())
	require.Equal(t, http.StatusOK, code)
	_, err = c.Write([]byte("ping\n"))
	require.NoError(t, err)
	line, err := br.ReadString('\n')
	require.NoError(t, err)
	require.Equal(t, "ping\n", line)
}

func TestEgressProxy_OriginFormRejected(t *testing.T) {
	p := newTestProxy(t, isPrivateIP)
	c, err := net.Dial("tcp", p.Addr())
	require.NoError(t, err)
	defer c.Close()
	fmt.Fprint(c, "GET /foo HTTP/1.1\r\nHost: example.com\r\n\r\n")
	resp, err := http.ReadResponse(bufio.NewReader(c), nil)
	require.NoError(t, err)
	require.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func TestEgressProxy_CloseTearsDownTunnel(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		_, _ = io.Copy(io.Discard, c)
	}()
	p := newTestProxy(t, allowLoopback)
	code, c, _ := connect(t, p, ln.Addr().String())
	require.Equal(t, http.StatusOK, code)
	require.NoError(t, p.Close())
	require.NoError(t, p.Close(), "Close must be idempotent")
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	_, err = c.Read(make([]byte, 1))
	require.Equal(t, io.EOF, err, "tunnel must be closed, not timed out")
}

func TestEgressProxy_HalfClose(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		_, _ = io.Copy(io.Discard, c) // read to client EOF
		time.Sleep(100 * time.Millisecond)
		_, _ = c.Write([]byte("REPLY"))
	}()
	p := newTestProxy(t, allowLoopback)
	status, c, br := connect(t, p, ln.Addr().String())
	require.Equal(t, 200, status)
	_, _ = c.Write([]byte("hi"))
	require.NoError(t, c.(*net.TCPConn).CloseWrite())
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	got, _ := io.ReadAll(br)
	require.Equal(t, "REPLY", string(got))
}

func TestEgressProxy_AllowLoopbackPortIsExact(t *testing.T) {
	srv, hits := countingServer(t)
	other, otherHits := countingServer(t)
	p := newTestProxy(t, isPrivateIP) // default policy: loopback blocked
	_, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	var n int
	_, _ = fmt.Sscan(port, &n)
	p.AllowLoopbackPort(n)

	resp, err := proxyClient(p).Get(srv.URL)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, int64(1), hits.Load())

	resp, err = proxyClient(p).Get(other.URL)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, 403, resp.StatusCode)
	require.Zero(t, otherHits.Load())
}
