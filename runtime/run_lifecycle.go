package runtime

import (
	"context"
	"time"
)

func (r *Runtime) startRunLifecycle(ctx context.Context) (err error) {
	if hook := r.AgentLoop.Hooks.OnRunStart; hook != nil {
		defer func() {
			if recovered := recover(); recovered != nil {
				err = recoveredPanicError("OnRunStart hook", recovered)
			}
		}()
		return hook(ctx)
	}
	return nil
}
func (r *Runtime) finishRunLifecycle(parent context.Context, reason string) {
	if hook := r.AgentLoop.Hooks.OnRunFinish; hook != nil {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 2*time.Second)
		defer cancel()
		defer func() {
			if recovered := recover(); recovered != nil {
				_ = recoveredPanicError("OnRunFinish hook", recovered)
			}
		}()
		hook(ctx, reason)
	}
}
