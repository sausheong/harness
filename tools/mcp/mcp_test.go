package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sausheong/harness/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// startInProcessServer spins up an in-process MCP server with the
// supplied tool configurations and returns a connected harness *Client
// pointing at it. The server is shut down via t.Cleanup.
//
// toolFns lets each test customise per-tool handler behaviour (return
// text, image, error, etc.).
func startInProcessServer(t *testing.T, name string, tools []sdkToolFixture) *Client {
	t.Helper()
	ctx := context.Background()

	server := sdk.NewServer(&sdk.Implementation{Name: "test-server", Version: "v0"}, nil)
	for _, tf := range tools {
		t := tf // capture
		server.AddTool(t.Tool, t.Handler)
	}
	clientTransport, serverTransport := sdk.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = serverSession.Close() })

	cli, err := connectWithTransport(ctx, name, clientTransport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = cli.Close() })
	return cli
}

type sdkToolFixture struct {
	Tool    *sdk.Tool
	Handler sdk.ToolHandler
}

// helper: build a JSON-schema RawMessage tool with a single string arg.
func stringArgTool(name, desc string, handler sdk.ToolHandler) sdkToolFixture {
	return sdkToolFixture{
		Tool: &sdk.Tool{
			Name:        name,
			Description: desc,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		},
		Handler: handler,
	}
}

func TestMCPAdapter_NameNamespacing(t *testing.T) {
	cli := startInProcessServer(t, "fs", []sdkToolFixture{
		stringArgTool("read", "read it", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "ok"}}}, nil
		}),
	})
	tools := cli.Tools()
	require.Len(t, tools, 1)
	assert.Equal(t, "mcp__fs__read", tools[0].Name())
}

func TestMCPAdapter_ParametersPassthrough(t *testing.T) {
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("echo", "echo back", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "x"}}}, nil
		}),
	})
	require.Len(t, cli.Tools(), 1)
	params := cli.Tools()[0].Parameters()
	var schema map[string]any
	require.NoError(t, json.Unmarshal(params, &schema))
	assert.Equal(t, "object", schema["type"])
	props, _ := schema["properties"].(map[string]any)
	require.NotNil(t, props["q"], "server-supplied schema property must round-trip")
}

func TestMCPAdapter_ExecuteCallsTool(t *testing.T) {
	var sawArgs json.RawMessage
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("echo", "echo", func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			sawArgs = req.Params.Arguments
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "got: " + string(req.Params.Arguments)}}}, nil
		}),
	})
	require.Len(t, cli.Tools(), 1)
	res, err := cli.Tools()[0].Execute(context.Background(), json.RawMessage(`{"q":"hello"}`))
	require.NoError(t, err)
	assert.Empty(t, res.Error)
	assert.Contains(t, res.Output, "hello")
	assert.JSONEq(t, `{"q":"hello"}`, string(sawArgs))
}

func TestMCPAdapter_ExecuteErrorPath(t *testing.T) {
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("boom", "fails", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return nil, errors.New("kaboom")
		}),
	})
	require.Len(t, cli.Tools(), 1)
	res, err := cli.Tools()[0].Execute(context.Background(), json.RawMessage(`{}`))
	require.NoError(t, err, "Execute itself returns nil; the tool error surfaces in result.Error")
	assert.NotEmpty(t, res.Error, "MCP tool error must surface as ToolResult.Error")
}

func TestMCPAdapter_ImageContentMapping(t *testing.T) {
	pngBytes := []byte{0x89, 'P', 'N', 'G', 0x0d}
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("snap", "take a snapshot", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{
				Content: []sdk.Content{
					&sdk.ImageContent{Data: pngBytes, MIMEType: "image/png"},
					&sdk.TextContent{Text: "caption"},
				},
			}, nil
		}),
	})
	require.Len(t, cli.Tools(), 1)
	res, err := cli.Tools()[0].Execute(context.Background(), json.RawMessage(`{}`))
	require.NoError(t, err)
	require.Len(t, res.Images, 1)
	assert.Equal(t, "image/png", res.Images[0].MimeType)
	assert.Equal(t, pngBytes, []byte(res.Images[0].Data))
	assert.Equal(t, "caption", res.Output)
	// Sanity: the harness ImageContent type is what tools expect.
	var _ llm.ImageContent = res.Images[0]
}

func TestMCPClient_ConnectDiscovers(t *testing.T) {
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("a", "first", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: ""}}}, nil
		}),
		stringArgTool("b", "second", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: ""}}}, nil
		}),
	})
	tools := cli.Tools()
	require.Len(t, tools, 2)
	got := []string{tools[0].Name(), tools[1].Name()}
	assert.ElementsMatch(t, []string{"mcp__srv__a", "mcp__srv__b"}, got)
}

func TestMCPClient_PingLiveAndDead(t *testing.T) {
	// Built inline (not via startInProcessServer) because the test needs the
	// server session to kill the connection mid-life.
	ctx := context.Background()
	server := sdk.NewServer(&sdk.Implementation{Name: "test-server", Version: "v0"}, nil)
	tf := stringArgTool("noop", "no-op", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: ""}}}, nil
	})
	server.AddTool(tf.Tool, tf.Handler)
	clientTransport, serverTransport := sdk.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	require.NoError(t, err)

	cli, err := connectWithTransport(ctx, "srv", clientTransport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = cli.Close() })

	require.NoError(t, cli.Ping(ctx), "ping against a live session must succeed")

	require.NoError(t, serverSession.Close())
	assert.Error(t, cli.Ping(ctx), "ping after the server side died must fail")
}

func TestMCPClient_CloseIsIdempotent(t *testing.T) {
	cli := startInProcessServer(t, "srv", []sdkToolFixture{
		stringArgTool("noop", "no-op", func(_ context.Context, _ *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: ""}}}, nil
		}),
	})
	require.NoError(t, cli.Close())
	require.NoError(t, cli.Close(), "second Close must be a no-op (no panic, no error)")
}

func TestConnect_ValidatesConfig(t *testing.T) {
	ctx := context.Background()
	_, err := Connect(ctx, ServerConfig{})
	require.Error(t, err, "empty config must fail")

	_, err = Connect(ctx, ServerConfig{Name: "x"})
	require.Error(t, err, "no transport set must fail")

	_, err = Connect(ctx, ServerConfig{Name: "x", Command: "true", URL: "http://x"})
	require.Error(t, err, "both transports set must fail")
}

func TestMergeEnv_OverrideWins(t *testing.T) {
	base := []string{"PATH=/usr/bin", "FOO=bar"}
	out := mergeEnv(base, map[string]string{"FOO": "baz", "NEW": "v"})
	// The PATH entry must be preserved verbatim; FOO must be replaced;
	// NEW must be appended.
	assert.Contains(t, out, "PATH=/usr/bin")
	assert.NotContains(t, out, "FOO=bar")
	assert.Contains(t, out, "FOO=baz")
	assert.Contains(t, out, "NEW=v")
}

// recordingRT captures every Request that hits its RoundTrip, then
// returns a canned 200 response. Used to verify the headerInjector
// passes headers through and doesn't mutate the caller's request.
type recordingRT struct {
	mu      sync.Mutex
	headers []http.Header
}

func (r *recordingRT) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.headers = append(r.headers, req.Header.Clone())
	r.mu.Unlock()
	return &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       http.NoBody,
		Request:    req,
	}, nil
}

func TestHeaderInjector_AddsHeadersAndDelegates(t *testing.T) {
	base := &recordingRT{}
	rt := &headerInjector{
		base:    base,
		headers: map[string]string{"Authorization": "Bearer t1", "X-Trace": "abc"},
	}
	req, err := http.NewRequest("POST", "https://example.com/mcp", nil)
	require.NoError(t, err)
	req.Header.Set("X-Existing", "kept")

	_, err = rt.RoundTrip(req)
	require.NoError(t, err)

	require.Len(t, base.headers, 1)
	got := base.headers[0]
	assert.Equal(t, "Bearer t1", got.Get("Authorization"))
	assert.Equal(t, "abc", got.Get("X-Trace"))
	assert.Equal(t, "kept", got.Get("X-Existing"), "pre-existing headers must survive")

	// The caller's req must NOT have been mutated.
	assert.Empty(t, req.Header.Get("Authorization"),
		"original request must remain unchanged so SDK retries see a clean req")
}

func TestWrapHTTPClient_NoHeadersReturnsSameClient(t *testing.T) {
	in := &http.Client{}
	out := wrapHTTPClient(in, nil)
	assert.Same(t, in, out, "no headers ⇒ no copy, no wrap")
}

func TestWrapHTTPClient_NilClientGetsDefaulted(t *testing.T) {
	out := wrapHTTPClient(nil, map[string]string{"X": "y"})
	require.NotNil(t, out)
	require.NotNil(t, out.Transport)
	_, ok := out.Transport.(*headerInjector)
	assert.True(t, ok, "headers ⇒ transport must be wrapped")
}

func TestWrapHTTPClient_DoesNotMutateCallerClient(t *testing.T) {
	original := &http.Client{Transport: http.DefaultTransport}
	out := wrapHTTPClient(original, map[string]string{"X": "y"})
	assert.NotSame(t, original, out, "wrapping must clone the client")
	assert.Equal(t, http.DefaultTransport, original.Transport,
		"caller's Transport must be untouched")
}

func TestBuildTransport_HTTPWithHeaders(t *testing.T) {
	tr, err := buildTransport(ServerConfig{
		Name: "x", URL: "https://example.com/mcp",
		Headers: map[string]string{"Authorization": "Bearer t1"},
	})
	require.NoError(t, err)
	st, ok := tr.(*sdk.StreamableClientTransport)
	require.True(t, ok)
	assert.Equal(t, "https://example.com/mcp", st.Endpoint)
	require.NotNil(t, st.HTTPClient)
	_, ok = st.HTTPClient.Transport.(*headerInjector)
	assert.True(t, ok, "HTTPClient.Transport must be a headerInjector when Headers set")
}

// TestStreamableHTTP_HeadersReachServer drives the whole transport via
// httptest.Server to confirm headers actually arrive. The handshake
// will fail (we don't speak MCP back), but the server captures the
// inbound request before that — proving header propagation end-to-end
// through the SDK.
func TestStreamableHTTP_HeadersReachServer(t *testing.T) {
	var (
		mu  sync.Mutex
		got http.Header
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		if got == nil {
			got = r.Header.Clone()
		}
		mu.Unlock()
		// Reply with something that won't satisfy the MCP handshake —
		// the transport will error, but only AFTER the server has seen
		// the request, which is what we're asserting.
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	ctx := context.Background()
	_, _ = Connect(ctx, ServerConfig{
		Name: "remote", URL: srv.URL,
		Headers: map[string]string{"Authorization": "Bearer real-token", "X-Tenant": "acme"},
	})
	// We don't assert err: depending on SDK retry behavior the call may
	// or may not return immediately. The header check is the contract.

	mu.Lock()
	defer mu.Unlock()
	require.NotNil(t, got, "server must have seen at least one request")
	assert.Equal(t, "Bearer real-token", got.Get("Authorization"),
		"Streamable HTTP transport must carry caller's Headers to the server")
	assert.Equal(t, "acme", got.Get("X-Tenant"))
}
