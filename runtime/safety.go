package runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime/debug"

	"github.com/sausheong/harness/tool"
)

func recoveredPanicError(component string, recovered any) error {
	err := fmt.Errorf("%s panicked: %v", component, recovered)
	slog.Error("recovered extension panic", "component", component, "error", err, "stack", string(debug.Stack()))
	return err
}

func callToolExecute(ctx context.Context, ex tool.Executor, name string, input json.RawMessage) (result tool.ToolResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = recoveredPanicError("tool "+name, recovered)
		}
	}()
	return ex.Execute(ctx, name, input)
}

func callBeforeToolHook(ctx context.Context, hook func(context.Context, string, json.RawMessage) (HookDecision, error), name string, input json.RawMessage) (decision HookDecision, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = recoveredPanicError("BeforeToolUse hook", recovered)
		}
	}()
	return hook(ctx, name, input)
}

func callAfterToolHook(ctx context.Context, hook func(context.Context, string, json.RawMessage, tool.ToolResult), name string, input json.RawMessage, result tool.ToolResult) {
	defer func() {
		if recovered := recover(); recovered != nil {
			_ = recoveredPanicError("AfterToolUse hook", recovered)
		}
	}()
	hook(ctx, name, input, result)
}

func callOnStopHook(ctx context.Context, hook func(context.Context, string), reason string) {
	defer func() {
		if recovered := recover(); recovered != nil {
			_ = recoveredPanicError("OnStop hook", recovered)
		}
	}()
	hook(ctx, reason)
}

func callPermissionCheck(ctx context.Context, checker tool.PermissionChecker, agentID, name string, input json.RawMessage) (decision tool.Decision) {
	defer func() {
		if recovered := recover(); recovered != nil {
			decision = tool.Decision{Behavior: tool.DecisionDeny, Reason: recoveredPanicError("permission checker", recovered).Error()}
		}
	}()
	return checker.Check(ctx, agentID, name, input)
}
