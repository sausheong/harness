package runtime

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/sausheong/harness/llm"
	"github.com/sausheong/harness/llm/llmtest"
	"github.com/sausheong/harness/session"
	"github.com/sausheong/harness/tool"
)

// recordingProvider captures every ChatRequest it receives and emits a
// canned text response. Used to inspect what the runtime sends to the
// LLM across turns.
type recordingProvider struct {
	llmtest.Base
	mu       sync.Mutex
	requests []llm.ChatRequest
	reply    string
}

func (r *recordingProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	r.mu.Lock()
	r.requests = append(r.requests, req)
	r.mu.Unlock()

	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: r.reply}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

// requestPrefixSignature renders the cache-relevant portion of a request:
// system prompt, tool definitions in their request-given order, and the
// message list excluding the final user turn. Two calls in the same
// session that differ only in the freshly-arrived user message must
// produce identical signatures.
//
// This intentionally does NOT sort tools — it renders them in the exact
// order the runtime hands them to the LLM, matching the byte sequence
// Anthropic/OpenAI prompt caches actually see. A separate test asserts
// that the tool registry returns tools in sorted order; this signature
// reflects what the runtime actually sends.
func requestPrefixSignature(t *testing.T, req llm.ChatRequest) string {
	t.Helper()
	var sb strings.Builder
	sb.WriteString("SYS:")
	sb.WriteString(req.SystemPrompt)
	sb.WriteString("\nTOOLS:")
	for _, td := range req.Tools {
		sb.WriteString("\n  ")
		sb.WriteString(td.Name)
		sb.WriteString("|")
		sb.WriteString(td.Description)
		sb.WriteString("|")
		sb.WriteString(string(td.Parameters))
	}
	sb.WriteString("\nMSGS_EXCL_LAST:")
	for i := 0; i < len(req.Messages)-1; i++ {
		sb.WriteString("\n  ")
		sb.WriteString(req.Messages[i].Role)
		sb.WriteString(":")
		sb.WriteString(req.Messages[i].Content)
	}
	return sb.String()
}

// TestRequestPrefixIsByteStableAcrossTurns runs two consecutive turns of the
// agent loop with identical inputs. The second request's prefix (system
// prompt + tool defs + all-but-last message) must be byte-identical to the
// first request's full content (system prompt + tool defs + all messages).
//
// This is the cache-stability invariant: turn N+1's prefix is turn N's full
// prompt. Anthropic and OpenAI prompt caches both depend on this. The
// compaction-View bug we shipped a fix for would have been caught by an
// earlier version of this test (it changed the prefix dramatically when
// compaction fired).
func TestRequestPrefixIsByteStableAcrossTurns(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}

	sess := session.NewSession("test-agent", "test-key")
	reg := tool.NewRegistry()
	reg.Register(&mockTool{name: "zebra", output: "z"})
	reg.Register(&mockTool{name: "alpha", output: "a"})
	reg.Register(&mockTool{name: "mango", output: "m"})

	rt := &Runtime{
		LLM:       rec,
		Tools:     reg,
		Session:   sess,
		Model:     "rec-model",
		Workspace: t.TempDir(),
		MaxTurns:  5,
	}

	// Turn 1
	events, err := rt.Run(context.Background(), "hello", nil)
	require.NoError(t, err)
	for range events {
	}

	// Turn 2 — same session, same agent, same tools.
	events, err = rt.Run(context.Background(), "world", nil)
	require.NoError(t, err)
	for range events {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 2,
		"expected at least 2 ChatStream calls, got %d", len(rec.requests))

	req1 := rec.requests[0]
	req2 := rec.requests[1]

	// turn 2's request layout: [...turn1.Messages, assistantN, userN+1]
	// turn 2's prefix = everything except the freshly-arrived user msg
	//                 = turn 1's full request + the assistant reply we
	//                   captured between turns.
	//
	// Build the expected prefix by appending the assistant reply onto turn 1's
	// message list, then render the resulting request the same way we render
	// turn 2 with its trailing user msg stripped. The two must be byte-equal.
	expected := req1
	expected.Messages = append(append([]llm.Message{}, req1.Messages...),
		llm.Message{Role: "assistant", Content: rec.reply})
	turn1FullPlusAssistant := fullSignature(t, expected)
	turn2Prefix := prefixWithoutLastMessage(t, req2)
	assert.Equal(t, turn1FullPlusAssistant, turn2Prefix,
		"turn 2 prefix must byte-match turn 1's full request plus the assistant reply")
}

// fullSignature renders the entire request the same way requestPrefixSignature
// does, but includes the final message too.
func fullSignature(t *testing.T, req llm.ChatRequest) string {
	t.Helper()
	sig := requestPrefixSignature(t, req)
	if len(req.Messages) > 0 {
		last := req.Messages[len(req.Messages)-1]
		sig += "\n  " + last.Role + ":" + last.Content
	}
	return sig
}

// prefixWithoutLastMessage is requestPrefixSignature but with the last
// message of the previous turn (the assistant reply that turn N's run
// appended) re-included, since that's part of turn N+1's prefix.
//
// turn N's request: [sys, tools, ...msgs, userN]
// after turn N: session also contains [...msgs, userN, assistantN]
// turn N+1's request: [sys, tools, ...msgs, userN, assistantN, userN+1]
//                                                  ^^^^^^^^^^^^^^^^^^^^^^
//                                                  the prefix at turn N+1
//                                                  excluding the new user msg
//
// So turn N+1's prefix = turn N's full request + assistantN.
// We test that subset by stripping the last message of turn N+1.
func prefixWithoutLastMessage(t *testing.T, req llm.ChatRequest) string {
	t.Helper()
	if len(req.Messages) == 0 {
		return requestPrefixSignature(t, req)
	}
	clone := req
	clone.Messages = req.Messages[:len(req.Messages)-1]
	return fullSignature(t, clone)
}

// strippingRecordingProvider is like recordingProvider but also strips
// the "format" field from tool schemas. Used to prove the runtime
// actually calls NormalizeToolSchema (rather than passing tools
// through unchanged).
type strippingRecordingProvider struct {
	llmtest.Base
	mu       sync.Mutex
	requests []llm.ChatRequest
	reply    string
}

func (p *strippingRecordingProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (<-chan llm.ChatEvent, error) {
	p.mu.Lock()
	p.requests = append(p.requests, req)
	p.mu.Unlock()
	ch := make(chan llm.ChatEvent, 4)
	go func() {
		defer close(ch)
		ch <- llm.ChatEvent{Type: llm.EventTextDelta, Text: p.reply}
		ch <- llm.ChatEvent{Type: llm.EventDone}
	}()
	return ch, nil
}

func (p *strippingRecordingProvider) NormalizeToolSchema(tools []llm.ToolDef) ([]llm.ToolDef, []llm.Diagnostic) {
	out := make([]llm.ToolDef, len(tools))
	var diags []llm.Diagnostic
	for i, t := range tools {
		newParams, d := llm.StripFields(t.Name, t.Parameters, []string{"format"})
		td := t
		td.Parameters = newParams
		out[i] = td
		diags = append(diags, d...)
	}
	return out, diags
}

// customMockTool is a local mock tool whose JSON Schema parameters can
// be supplied per-instance. We use it here (instead of the package-wide
// mockTool) so this test can register a tool whose schema contains the
// "format" field that strippingRecordingProvider strips, without
// touching agent_test.go's shared fixture.
type customMockTool struct {
	name   string
	schema []byte
	output string
}

func (t *customMockTool) Name() string                              { return t.name }
func (t *customMockTool) Description() string                       { return "custom mock tool" }
func (t *customMockTool) Parameters() json.RawMessage               { return json.RawMessage(t.schema) }
func (t *customMockTool) IsConcurrencySafe(_ json.RawMessage) bool  { return false }
func (t *customMockTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	return tool.ToolResult{Output: t.output}, nil
}

// TestRuntimeCallsNormalizeToolSchema asserts that the agent runtime
// invokes NormalizeToolSchema before constructing the ChatRequest
// (so the model sees the provider's normalized tool list, not the raw
// tool definitions). Also asserts byte-determinism across turns —
// required for prompt cache stability.
func TestRuntimeCallsNormalizeToolSchema(t *testing.T) {
	rec := &strippingRecordingProvider{reply: "ok"}

	sess := session.NewSession("test-agent", "test-key")
	reg := tool.NewRegistry()
	// Register a tool whose schema has "format" — the stripper above
	// will remove it. If the runtime doesn't call NormalizeToolSchema,
	// "format" survives and the assertion below fails.
	reg.Register(&customMockTool{
		name:   "fetch",
		schema: []byte(`{"type":"object","properties":{"url":{"type":"string","format":"uri"}}}`),
		output: "ok",
	})

	rt := &Runtime{
		LLM:       rec,
		Tools:     reg,
		Session:   sess,
		Model:     "rec-model",
		Workspace: t.TempDir(),
		MaxTurns:  3,
	}

	for i := 0; i < 3; i++ {
		events, err := rt.Run(context.Background(), "ping", nil)
		require.NoError(t, err)
		for range events {
		}
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 3, "expected at least 3 ChatStream calls")

	// Determinism: all turns must have byte-identical Tools.
	for i := 1; i < len(rec.requests); i++ {
		require.Equal(t, len(rec.requests[0].Tools), len(rec.requests[i].Tools))
		for j := range rec.requests[0].Tools {
			assert.Equal(t,
				string(rec.requests[0].Tools[j].Parameters),
				string(rec.requests[i].Tools[j].Parameters),
				"turn %d tool %d parameters must byte-match turn 0", i, j)
		}
	}

	// Normalization actually happened: the "format" field is gone.
	require.Greater(t, len(rec.requests[0].Tools), 0)
	var schema map[string]any
	require.NoError(t, json.Unmarshal(rec.requests[0].Tools[0].Parameters, &schema))
	props := schema["properties"].(map[string]any)
	url := props["url"].(map[string]any)
	_, hasFormat := url["format"]
	assert.False(t, hasFormat,
		"runtime must call NormalizeToolSchema; format should be stripped")
}

// TestReasoningIsInRequestPrefix asserts that the agent's Reasoning
// setting flows into ChatRequest and remains stable across turns.
// Required for prompt cache hits with reasoning enabled.
func TestReasoningIsInRequestPrefix(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	reg := tool.NewRegistry()

	rt := &Runtime{
		LLM:       rec,
		Tools:     reg,
		Session:   sess,
		Model:     "rec-model",
		Reasoning: llm.ReasoningHigh,
		Workspace: t.TempDir(),
		MaxTurns:  3,
	}

	for i := 0; i < 2; i++ {
		events, err := rt.Run(context.Background(), "ping", nil)
		require.NoError(t, err)
		for range events {
		}
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 2)
	assert.Equal(t, llm.ReasoningHigh, rec.requests[0].Reasoning)
	assert.Equal(t, llm.ReasoningHigh, rec.requests[1].Reasoning,
		"reasoning level must be stable across turns")
}

func TestToolDefsSortedByNameInRequest(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	reg := tool.NewRegistry()
	reg.Register(&mockTool{name: "zebra", output: "z"})
	reg.Register(&mockTool{name: "alpha", output: "a"})
	reg.Register(&mockTool{name: "mango", output: "m"})

	rt := &Runtime{
		LLM: rec, Tools: reg, Session: sess,
		Model: "rec-model", Workspace: t.TempDir(), MaxTurns: 5,
	}

	events, err := rt.Run(context.Background(), "hello", nil)
	require.NoError(t, err)
	for range events {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 1)
	names := make([]string, 0, len(rec.requests[0].Tools))
	for _, td := range rec.requests[0].Tools {
		names = append(names, td.Name)
	}
	require.Equal(t, []string{"alpha", "mango", "zebra"}, names)
}

func TestRuntimeSendsStructuredSystemPromptParts(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")

	spec := AgentSpec{ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5"}
	rt, err := BuildRuntime(
		RuntimeDeps{},
		RuntimeInputs{Provider: rec, Tools: tool.NewRegistry(), Session: sess},
		spec,
	)
	require.NoError(t, err)
	rt.MaxTurns = 5

	events, err := rt.Run(context.Background(), "hi", nil)
	require.NoError(t, err)
	for range events {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 1)
	req0 := rec.requests[0]
	require.NotEmpty(t, req0.SystemPromptParts, "expected SystemPromptParts to be populated")
	require.True(t, req0.SystemPromptParts[0].Cache, "first part must be cache-marked")
	require.Contains(t, req0.SystemPromptParts[0].Text, `"A" agent (id: a)`)
	require.True(t, req0.CacheLastMessage, "Anthropic provider must request CacheLastMessage")
}

func TestRuntimeStaticPromptByteStableAcrossTurns(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	spec := AgentSpec{ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5"}
	rt, err := BuildRuntime(
		RuntimeDeps{},
		RuntimeInputs{Provider: rec, Tools: tool.NewRegistry(), Session: sess},
		spec,
	)
	require.NoError(t, err)
	rt.MaxTurns = 5

	for _, msg := range []string{"hello", "world"} {
		ev, err := rt.Run(context.Background(), msg, nil)
		require.NoError(t, err)
		for range ev {
		}
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 2)
	require.Equal(t,
		rec.requests[0].SystemPromptParts[0].Text,
		rec.requests[1].SystemPromptParts[0].Text,
		"static system prompt must be byte-identical across turns",
	)
}

func TestRuntimeDynamicSuffixIncludesDate(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	spec := AgentSpec{ID: "a", Name: "A", Model: "anthropic/claude-sonnet-4-5"}
	rt, err := BuildRuntime(
		RuntimeDeps{},
		RuntimeInputs{Provider: rec, Tools: tool.NewRegistry(), Session: sess},
		spec,
	)
	require.NoError(t, err)
	rt.MaxTurns = 5

	ev, err := rt.Run(context.Background(), "hi", nil)
	require.NoError(t, err)
	for range ev {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 1)
	require.GreaterOrEqual(t, len(rec.requests[0].SystemPromptParts), 2,
		"expected static + dynamic parts")
	require.Contains(t, rec.requests[0].SystemPromptParts[1].Text, "Today's date is ",
		"dynamic suffix must contain the date line")
}

// TestRuntimeDateLineComputedOncePerRun is a pure-function check: calling
// buildDynamicSystemPromptSuffix twice with the same dateLine produces
// identical output for that portion. Combined with the runtime call site
// in runtime.go (which computes dateLine once before the for-turn loop),
// this proves the date is stable across turns of a single Run.
func TestRuntimeDateLineComputedOncePerRun(t *testing.T) {
	dl := "Today's date is 2026-05-01."
	a := buildDynamicSystemPromptSuffix("", dl, "")
	b := buildDynamicSystemPromptSuffix("", dl, "")
	require.Equal(t, a, b)
}

func TestRuntimeNonAnthropicHasCacheLastMessageFalse(t *testing.T) {
	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	spec := AgentSpec{ID: "a", Name: "A", Model: "openai/gpt-4o"}
	rt, err := BuildRuntime(
		RuntimeDeps{},
		RuntimeInputs{Provider: rec, Tools: tool.NewRegistry(), Session: sess},
		spec,
	)
	require.NoError(t, err)
	rt.MaxTurns = 5

	ev, err := rt.Run(context.Background(), "hi", nil)
	require.NoError(t, err)
	for range ev {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.False(t, rec.requests[0].CacheLastMessage)
}

func TestRuntimeStaticPromptIncludesMemoryFilesContent(t *testing.T) {
	workspace := t.TempDir()
	t.Setenv("HOME", t.TempDir())

	// Consumers compose MemoryFiles themselves and pass via RuntimeDeps —
	// the harness concatenates it verbatim into the static prompt.
	memBlock := "\n\n## Project memory: /tmp/x\n\nRUNTIME_MEMFILE_INTEGRATION_SENTINEL"

	rec := &recordingProvider{reply: "ok"}
	sess := session.NewSession("test-agent", "test-key")
	spec := AgentSpec{
		ID:        "a",
		Name:      "A",
		Workspace: workspace,
		Model:     "anthropic/claude-sonnet-4-5",
	}
	rt, err := BuildRuntime(
		RuntimeDeps{MemoryFiles: memBlock},
		RuntimeInputs{Provider: rec, Tools: tool.NewRegistry(), Session: sess},
		spec,
	)
	require.NoError(t, err)
	rt.MaxTurns = 5

	ev, err := rt.Run(context.Background(), "hi", nil)
	require.NoError(t, err)
	for range ev {
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.GreaterOrEqual(t, len(rec.requests), 1)
	staticPart := rec.requests[0].SystemPromptParts[0].Text
	require.Contains(t, staticPart, "RUNTIME_MEMFILE_INTEGRATION_SENTINEL")
	require.Contains(t, staticPart, "## Project memory:")
	require.True(t, rec.requests[0].SystemPromptParts[0].Cache,
		"static part with memory files must still be cache-marked")
}
