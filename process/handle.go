package process

import (
	"context"
	"crypto/rand"
	"errors"
	"io"
	"os/exec"
)

const HandleInputLimit = 64 << 10

type inputWrite struct {
	data []byte
	done chan error
}

// Handle owns one deliberately asynchronous process group. It does not survive
// its owner context or application shutdown; callers must Close or Wait.
type Handle struct {
	ID              string
	ctx             context.Context
	cancel          context.CancelFunc
	command         *exec.Cmd
	stdin           io.WriteCloser
	stdout, stderr  *OutputCapture
	input           chan inputWrite
	inputDone, done chan struct{}
	err             error
}
type HandleSnapshot struct {
	ID                               string
	Running                          bool
	ExitCode                         int
	Error                            string
	Stdout, Stderr                   string
	StdoutBytes, StderrBytes         int64
	StdoutTruncated, StderrTruncated bool
	StdoutArtifact, StderrArtifact   ArtifactInfo
}

func StartHandle(parent context.Context, store *ArtifactStore, dir, name string, args ...string) (*Handle, error) {
	return startHandle(parent, store, nil, dir, name, args...)
}

// StartHandleWithEnv uses only the supplied environment, without ambient inheritance.
func StartHandleWithEnv(parent context.Context, store *ArtifactStore, env []string, dir, name string, args ...string) (*Handle, error) {
	return startHandle(parent, store, append([]string{}, env...), dir, name, args...)
}
func startHandle(parent context.Context, store *ArtifactStore, env []string, dir, name string, args ...string) (*Handle, error) {
	if store == nil {
		return nil, errors.New("background process requires an output store")
	}
	ctx, cancel := context.WithCancel(parent)
	cmd := Command(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = env
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	h := &Handle{ID: rand.Text(), ctx: ctx, cancel: cancel, command: cmd, stdin: stdin, stdout: store.Capture(64 << 10), stderr: store.Capture(64 << 10), input: make(chan inputWrite, 1), inputDone: make(chan struct{}), done: make(chan struct{})}
	cmd.Stdout = h.stdout
	cmd.Stderr = h.stderr
	if err := cmd.Start(); err != nil {
		cancel()
		stdin.Close()
		h.stdout.Close()
		h.stderr.Close()
		return nil, err
	}
	go h.writeInput()
	go func() {
		defer close(h.done)
		waitErr := cmd.Wait()
		cleanupErr := KillGroup(cmd)
		cancel()
		stdin.Close()
		<-h.inputDone
		h.err = errors.Join(waitErr, cleanupErr, h.stdout.Close(), h.stderr.Close())
	}()
	return h, nil
}
func (h *Handle) writeInput() {
	defer close(h.inputDone)
	for {
		select {
		case <-h.ctx.Done():
			return
		case request := <-h.input:
			if err := h.ctx.Err(); err != nil {
				request.done <- err
				continue
			}
			n, err := h.stdin.Write(request.data)
			if err == nil && n != len(request.data) {
				err = io.ErrShortWrite
			}
			request.done <- err
		}
	}
}

// Send waits for a bounded input write. If its context is cancelled after
// admission, the owned process is cancelled because a partial write cannot be
// withdrawn safely. Successful return means the pipe accepted all bytes.
func (h *Handle) Send(ctx context.Context, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(data) > HandleInputLimit {
		return errors.New("process input exceeds 64 KiB")
	}
	request := inputWrite{data: append([]byte(nil), data...), done: make(chan error, 1)}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-h.done:
		return errors.New("process already stopped")
	case <-h.ctx.Done():
		return h.ctx.Err()
	case h.input <- request:
	}
	select {
	case err := <-request.done:
		return err
	case <-h.done:
		select {
		case err := <-request.done:
			return err
		default:
			return errors.New("process stopped before input delivery completed")
		}
	case <-ctx.Done():
		h.cancel()
		<-h.done
		return ctx.Err()
	}
}
func (h *Handle) Cancel()      { h.cancel() }
func (h *Handle) Wait() error  { <-h.done; return h.err }
func (h *Handle) Close() error { h.cancel(); return h.Wait() }
func (h *Handle) Snapshot() HandleSnapshot {
	s := HandleSnapshot{ID: h.ID, Running: true, ExitCode: -1}
	select {
	case <-h.done:
		s.Running = false
		if h.command.ProcessState != nil {
			s.ExitCode = h.command.ProcessState.ExitCode()
		}
		if h.err != nil {
			s.Error = h.err.Error()
		}
	default:
	}
	s.Stdout, s.StdoutBytes, s.StdoutTruncated = h.stdout.Snapshot()
	s.Stderr, s.StderrBytes, s.StderrTruncated = h.stderr.Snapshot()
	s.StdoutArtifact, s.StderrArtifact = h.stdout.Info(), h.stderr.Info()
	return s
}
