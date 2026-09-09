package execution

import (
	"context"
	"errors"

	"github.com/sausheong/harness/process"
)

// Starter is optional: backends that cannot own interactive processes must fail
// explicitly instead of routing those operations to the host.
type Starter interface {
	Start(context.Context, Request) (*Handle, error)
}

// Handle remains running until both the process and backend cleanup have joined.
type Handle struct {
	*process.Handle
	done chan struct{}
	err  error
}

func ownedHandle(handle *process.Handle, cleanup func() error) *Handle {
	h := &Handle{Handle: handle, done: make(chan struct{})}
	go func() { defer close(h.done); h.err = errors.Join(handle.Wait(), cleanup()) }()
	return h
}
func (h *Handle) Wait() error  { <-h.done; return h.err }
func (h *Handle) Close() error { h.Cancel(); return h.Wait() }
func (h *Handle) Snapshot() process.HandleSnapshot {
	s := h.Handle.Snapshot()
	select {
	case <-h.done:
		s.Running = false
		if h.err != nil {
			s.Error = h.err.Error()
		}
	default:
		s.Running = true
	}
	return s
}
func validateStart(r Request) error {
	if err := validate(r); err != nil {
		return err
	}
	if r.OutputStore == nil {
		return errors.New("asynchronous execution requires an output store")
	}
	if len(r.Stdin) != 0 {
		return errors.New("asynchronous input must use the owned handle")
	}
	if r.CaptureLimit != 0 {
		return errors.New("asynchronous execution uses standard bounded captures")
	}
	return nil
}
func (h Host) Start(ctx context.Context, r Request) (*Handle, error) {
	if err := validateStart(r); err != nil {
		return nil, err
	}
	handle, err := process.StartHandleWithEnv(ctx, r.OutputStore, environment(r.Env), h.Workspace, r.Argv[0], r.Argv[1:]...)
	if err != nil {
		return nil, err
	}
	return ownedHandle(handle, func() error { return nil }), nil
}
func (c Container) Start(ctx context.Context, r Request) (*Handle, error) {
	if err := validateStart(r); err != nil {
		return nil, err
	}
	plan, err := c.prepare(ctx, r)
	if err != nil {
		return nil, err
	}
	handle, err := process.StartHandleWithEnv(ctx, r.OutputStore, environment(nil), "", plan.docker, plan.args...)
	if err != nil {
		return nil, errors.Join(err, plan.cleanup())
	}
	return ownedHandle(handle, plan.cleanup), nil
}

func (h *Handle) Send(ctx context.Context, data []byte) error {
	err := h.Handle.Send(ctx, data)
	if ctx.Err() != nil {
		h.Cancel()
		<-h.done
	}
	return err
}
