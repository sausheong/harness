package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sausheong/harness/tool"
)

// maxBodyBytes is the server-side cap on skill body length. Mirrors the
// schema's maxLength so the model and the server agree.
const maxBodyBytes = 32000

// SkillTool is the action-discriminated tool the agent calls to manage
// skills. Wraps any SkillStore; the default disk implementation is in
// tool/skills/disk/.
//
// Schema is one tool with an "action" enum (vs six separate tools) to
// keep the request prefix small (less schema bloat → better
// prompt-cache hits).
type SkillTool struct {
	Store SkillStore
}

// Name returns "skill_manage" (matches the Hermes naming for parity
// with the spec's reference implementation).
func (t *SkillTool) Name() string { return "skill_manage" }

// Description is shown to the model in the tool list.
func (t *SkillTool) Description() string {
	return "Create, patch, replace, remove, list, or get skills — durable " +
		"procedural knowledge that survives across sessions. Use to " +
		"capture workflows, conventions, or reusable techniques worth " +
		"keeping."
}

// Parameters returns the JSON-Schema for the tool input.
func (t *SkillTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["create", "patch", "replace", "remove", "list", "get"],
				"description": "Operation to perform."
			},
			"name": {
				"type": "string",
				"pattern": "^[a-z0-9][a-z0-9-]{0,63}$",
				"description": "Skill name (kebab-case, 1-64 chars). Required for create/patch/replace/remove/get."
			},
			"body": {
				"type": "string",
				"maxLength": 32000,
				"description": "Full SKILL.md content for create/replace. May include YAML-ish frontmatter (---\ndescription: ...\n---\n) at the top. Max 32000 characters."
			},
			"old_string": {
				"type": "string",
				"description": "Required for patch. The string to replace. Must occur exactly once in the existing body."
			},
			"new_string": {
				"type": "string",
				"description": "Required for patch. The replacement string."
			}
		},
		"required": ["action"]
	}`)
}

// IsConcurrencySafe returns false — every action that modifies the
// store mutates shared state. Even read-only actions (list, get) touch
// the same on-disk hierarchy and are best serialized.
func (t *SkillTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }

// skillInput is the parsed tool input.
type skillInput struct {
	Action    string `json:"action"`
	Name      string `json:"name"`
	Body      string `json:"body"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

// actionResult is the canonical envelope every successful or failed
// SkillTool call returns. runtime.Review reads this shape (target +
// success + message + name) to extract human-readable summaries.
//
// Same shape as memory.actionResult; we keep it duplicated rather than
// importing tool/memory for one struct so the two tool packages remain
// independent.
type actionResult struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
	Target  string `json:"target,omitempty"`
	Name    string `json:"name,omitempty"`
}

func (r actionResult) toToolResult() tool.ToolResult {
	b, _ := json.Marshal(r)
	return tool.ToolResult{Output: string(b)}
}

func successResult(message, name string) tool.ToolResult {
	return actionResult{
		Success: true,
		Message: message,
		Target:  "skill",
		Name:    name,
	}.toToolResult()
}

func errorResult(errMsg, name string) tool.ToolResult {
	return actionResult{
		Success: false,
		Error:   errMsg,
		Target:  "skill",
		Name:    name,
	}.toToolResult()
}

// truncatePreview returns content suitable for inclusion in a success
// message — truncated to ~60 chars with an ellipsis.
func truncatePreview(s string) string {
	const max = 60
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// originFromContext returns the OriginKey value or "agent" as default.
func originFromContext(ctx context.Context) string {
	if v := ctx.Value(OriginKey); v != nil {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	return "agent"
}

// Execute dispatches on the action enum.
func (t *SkillTool) Execute(ctx context.Context, input json.RawMessage) (tool.ToolResult, error) {
	if t.Store == nil {
		return errorResult("skill tool: no store configured", ""), nil
	}
	var in skillInput
	if err := json.Unmarshal(input, &in); err != nil {
		return errorResult(fmt.Sprintf("invalid input: %v", err), ""), nil
	}
	switch in.Action {
	case "create":
		return t.executeCreate(ctx, in), nil
	case "patch":
		return t.executePatch(ctx, in), nil
	case "replace":
		return t.executeReplace(ctx, in), nil
	case "remove":
		return t.executeRemove(ctx, in), nil
	case "list":
		return t.executeList(ctx, in), nil
	case "get":
		return t.executeGet(ctx, in), nil
	case "":
		return errorResult("action is required", ""), nil
	default:
		return errorResult(fmt.Sprintf("unknown action %q", in.Action), ""), nil
	}
}

func (t *SkillTool) executeCreate(ctx context.Context, in skillInput) tool.ToolResult {
	if in.Name == "" {
		return errorResult("name is required for create", "")
	}
	if !ValidName(in.Name) {
		return errorResult(fmt.Sprintf("invalid name %q (must be kebab-case, 1-64 chars)", in.Name), in.Name)
	}
	body := in.Body
	if strings.TrimSpace(body) == "" {
		return errorResult("body is required for create", in.Name)
	}
	if len(body) > maxBodyBytes {
		return errorResult(fmt.Sprintf("body exceeds max length (%d > %d)", len(body), maxBodyBytes), in.Name)
	}

	created, err := t.Store.Create(ctx, Skill{
		Name:   in.Name,
		Body:   body,
		Origin: originFromContext(ctx),
	})
	if err != nil {
		return errorResult(err.Error(), in.Name)
	}
	return successResult(fmt.Sprintf("created skill: %s", truncatePreview(created.Name)), created.Name)
}

func (t *SkillTool) executePatch(ctx context.Context, in skillInput) tool.ToolResult {
	if in.Name == "" {
		return errorResult("name is required for patch", "")
	}
	if in.OldString == "" {
		return errorResult("old_string is required for patch", in.Name)
	}
	updated, err := t.Store.Patch(ctx, in.Name, in.OldString, in.NewString)
	if err != nil {
		// All three Patch sentinels are returned via err.Error(); we
		// surface them verbatim so the model can recognize the category
		// from the message.
		return errorResult(err.Error(), in.Name)
	}
	return successResult(fmt.Sprintf("patched skill: %s", truncatePreview(updated.Name)), updated.Name)
}

func (t *SkillTool) executeReplace(ctx context.Context, in skillInput) tool.ToolResult {
	if in.Name == "" {
		return errorResult("name is required for replace", "")
	}
	body := in.Body
	if strings.TrimSpace(body) == "" {
		return errorResult("body is required for replace", in.Name)
	}
	if len(body) > maxBodyBytes {
		return errorResult(fmt.Sprintf("body exceeds max length (%d > %d)", len(body), maxBodyBytes), in.Name)
	}
	updated, err := t.Store.Replace(ctx, in.Name, body)
	if err != nil {
		return errorResult(err.Error(), in.Name)
	}
	return successResult(fmt.Sprintf("replaced skill: %s", truncatePreview(updated.Name)), updated.Name)
}

// executeList returns the skills as a raw JSON array (not the
// action-result envelope) so the model receives a list-shaped value it
// can iterate over directly. Body is stripped — the model can call
// action=get to fetch a specific body.
func (t *SkillTool) executeList(ctx context.Context, _ skillInput) tool.ToolResult {
	all, err := t.Store.List(ctx)
	if err != nil {
		return errorResult(err.Error(), "")
	}
	if all == nil {
		all = []Skill{}
	}
	stripped := make([]Skill, len(all))
	for i, sk := range all {
		sk.Body = "" // index only
		stripped[i] = sk
	}
	b, err := json.Marshal(stripped)
	if err != nil {
		return errorResult(err.Error(), "")
	}
	return tool.ToolResult{Output: string(b)}
}

// executeGet returns the raw Skill as JSON on success; the action-result
// envelope on failure.
func (t *SkillTool) executeGet(ctx context.Context, in skillInput) tool.ToolResult {
	if in.Name == "" {
		return errorResult("name is required for get", "")
	}
	sk, ok, err := t.Store.Get(ctx, in.Name)
	if err != nil {
		return errorResult(err.Error(), in.Name)
	}
	if !ok {
		return errorResult("not found", in.Name)
	}
	b, err := json.Marshal(sk)
	if err != nil {
		return errorResult(err.Error(), in.Name)
	}
	return tool.ToolResult{Output: string(b)}
}

func (t *SkillTool) executeRemove(ctx context.Context, in skillInput) tool.ToolResult {
	if in.Name == "" {
		return errorResult("name is required for remove", "")
	}
	if err := t.Store.Remove(ctx, in.Name); err != nil {
		return errorResult(err.Error(), in.Name)
	}
	return successResult(fmt.Sprintf("removed skill: %s", in.Name), in.Name)
}

// Compile-time assertion that *SkillTool satisfies tool.Tool.
var _ tool.Tool = (*SkillTool)(nil)
