// Package mcp adapts an external Model Context Protocol (MCP) server's
// tools into harness tool.Tool implementations.
//
// Typical usage:
//
//	cli, err := mcp.Connect(ctx, mcp.ServerConfig{
//	    Name:    "fs",
//	    Command: "npx",
//	    Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
//	})
//	if err != nil { ... }
//	defer cli.Close()
//	for _, t := range cli.Tools() {
//	    reg.Register(t)
//	}
//
// For declarative wiring, set runtime.AgentSpec.MCPServers — the runtime
// builder will Connect each server, register its tools into the agent's
// registry, and tear them down via Runtime.Close.
package mcp

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sausheong/harness/tool"
)

// ServerConfig describes one MCP server to connect to. Exactly one
// transport must be specified: Command (stdio, the most common) or URL
// (Streamable HTTP via the SDK's StreamableClientTransport).
type ServerConfig struct {
	// Name namespaces the server's tools. Adapter tool names become
	// "mcp__<Name>__<tool>" — the conventional double-underscore
	// scheme that prevents collisions with built-in tools.
	Name string

	// Command + Args + Env wire the stdio transport. Env is merged into
	// the spawned process's environment (replacing duplicates from
	// os.Environ()).
	Command string
	Args    []string
	Env     map[string]string

	// URL wires the Streamable HTTP transport (StreamableClientTransport).
	// Mutually exclusive with Command — set exactly one.
	URL string

	// Headers attaches static HTTP headers to every request the
	// Streamable HTTP transport makes. Typical use: auth tokens
	// ("Authorization": "Bearer …", "X-API-Key": …). Ignored for
	// stdio. Wrapped around HTTPClient.Transport — your custom
	// RoundTripper still runs.
	Headers map[string]string

	// HTTPClient overrides the default *http.Client used by the
	// Streamable HTTP transport. Use this for mTLS, custom timeouts,
	// or full RoundTripper control. Ignored for stdio. nil ⇒ a fresh
	// http.Client{} with default transport.
	HTTPClient *http.Client

	// ClientImplementation is sent to the server during the initialize
	// handshake. Optional — defaults to {Name: "harness", Version: "v1"}.
	ClientImplementation *sdk.Implementation
}

// Client is a connected MCP server with its discovered tools adapted as
// harness tool.Tool implementations.
type Client struct {
	name    string
	session *sdk.ClientSession
	tools   []tool.Tool
	closed  bool
}

// Connect opens a session to the MCP server in cfg, lists its tools,
// and adapts each into a tool.Tool. Returned Client owns the session;
// call Close to release it.
func Connect(ctx context.Context, cfg ServerConfig) (*Client, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("mcp: ServerConfig.Name is required")
	}
	if cfg.Command == "" && cfg.URL == "" {
		return nil, fmt.Errorf("mcp: one of Command or URL must be set")
	}
	if cfg.Command != "" && cfg.URL != "" {
		return nil, fmt.Errorf("mcp: Command and URL are mutually exclusive")
	}

	transport, err := buildTransport(cfg)
	if err != nil {
		return nil, err
	}
	return connectWithTransport(ctx, cfg.Name, transport, cfg.ClientImplementation)
}

// connectWithTransport is the lower-level entrypoint shared by Connect
// and the test helpers. Exposed via mcp_test_helpers.go so tests can
// pair the client with an InMemoryTransport.
func connectWithTransport(ctx context.Context, name string, transport sdk.Transport, impl *sdk.Implementation) (*Client, error) {
	if impl == nil {
		impl = &sdk.Implementation{Name: "harness", Version: "v1"}
	}
	client := sdk.NewClient(impl, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp: connect %q: %w", name, err)
	}
	tools, err := discoverTools(ctx, name, session)
	if err != nil {
		_ = session.Close()
		return nil, err
	}
	return &Client{name: name, session: session, tools: tools}, nil
}

// Tools returns the adapter tool.Tool slice. Safe to call multiple
// times; the slice header is shared (callers should not mutate it).
func (c *Client) Tools() []tool.Tool { return c.tools }

// Name returns the configured server name.
func (c *Client) Name() string { return c.name }

// Ping verifies the session is alive by sending an MCP ping. A non-nil
// error means the session (and for stdio servers, likely the child
// process) is no longer usable.
func (c *Client) Ping(ctx context.Context) error {
	return c.session.Ping(ctx, nil)
}

// Close releases the underlying MCP session. Safe to call multiple
// times — subsequent calls are no-ops.
func (c *Client) Close() error {
	if c.closed {
		return nil
	}
	c.closed = true
	return c.session.Close()
}

// buildTransport selects a transport based on cfg. stdio uses
// CommandTransport with a process whose env is os.Environ() merged
// with cfg.Env (cfg.Env wins on conflict). Streamable HTTP uses
// StreamableClientTransport, with optional Headers + HTTPClient
// applied via wrapHTTPClient.
func buildTransport(cfg ServerConfig) (sdk.Transport, error) {
	if cfg.Command != "" {
		cmd := exec.Command(cfg.Command, cfg.Args...)
		cmd.Env = mergeEnv(os.Environ(), cfg.Env)
		return &sdk.CommandTransport{Command: cmd}, nil
	}
	return &sdk.StreamableClientTransport{
		Endpoint:   cfg.URL,
		HTTPClient: wrapHTTPClient(cfg.HTTPClient, cfg.Headers),
	}, nil
}

// wrapHTTPClient returns the *http.Client the Streamable HTTP transport
// should use. If headers is non-empty, the client's Transport is
// wrapped in a header-injecting RoundTripper so every outgoing request
// carries them. The caller's *http.Client is NOT mutated — a shallow
// copy is returned when wrapping is needed.
func wrapHTTPClient(client *http.Client, headers map[string]string) *http.Client {
	if client == nil {
		client = &http.Client{}
	}
	if len(headers) == 0 {
		return client
	}
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	out := *client // shallow copy — leave caller's struct alone
	out.Transport = &headerInjector{base: base, headers: headers}
	return &out
}

// headerInjector is an http.RoundTripper that adds static headers to
// every request before delegating. Request is cloned so the caller's
// *http.Request is never mutated (matters for retries inside the SDK).
type headerInjector struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerInjector) RoundTrip(req *http.Request) (*http.Response, error) {
	req2 := req.Clone(req.Context())
	for k, v := range h.headers {
		req2.Header.Set(k, v)
	}
	return h.base.RoundTrip(req2)
}

// mergeEnv returns base with overrides applied. Order is preserved for
// base; overrides are appended at the end after pruning duplicates so
// the spawned process sees override values win.
func mergeEnv(base []string, overrides map[string]string) []string {
	if len(overrides) == 0 {
		return base
	}
	out := make([]string, 0, len(base)+len(overrides))
	for _, kv := range base {
		// "K=V" — keep only entries whose key is NOT being overridden.
		key := envKey(kv)
		if _, ok := overrides[key]; ok {
			continue
		}
		out = append(out, kv)
	}
	for k, v := range overrides {
		out = append(out, k+"="+v)
	}
	return out
}

func envKey(kv string) string {
	for i := 0; i < len(kv); i++ {
		if kv[i] == '=' {
			return kv[:i]
		}
	}
	return kv
}

// discoverTools lists tools from the server (paginating if necessary)
// and wraps each in an adapter.
func discoverTools(ctx context.Context, serverName string, session *sdk.ClientSession) ([]tool.Tool, error) {
	var out []tool.Tool
	cursor := ""
	for {
		params := &sdk.ListToolsParams{Cursor: cursor}
		res, err := session.ListTools(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("mcp: list tools for %q: %w", serverName, err)
		}
		for _, t := range res.Tools {
			out = append(out, &mcpToolAdapter{
				serverName: serverName,
				sdkTool:    t,
				session:    session,
			})
		}
		if res.NextCursor == "" {
			break
		}
		cursor = res.NextCursor
	}
	return out, nil
}
