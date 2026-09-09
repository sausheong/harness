package execution

import (
	"context"
	"errors"
	"io"
	"os"
	"sync"

	"github.com/sausheong/harness/process"
)

// StreamStarter opens a raw protocol stream inside the selected boundary.
// Consumers must bound framing and drain both output streams. No capture or
// implicit fallback is applied to protocol traffic.
type StreamStarter interface {
	OpenStream(context.Context, Request) (*Stream, error)
}

type Stream struct {
	Stdin     io.WriteCloser
	Stdout    io.ReadCloser
	Stderr    io.ReadCloser
	cancel    context.CancelFunc
	stop      func()
	done      chan struct{}
	err       error
	closeOnce sync.Once
}

// Wait joins the child and boundary cleanup. Readers remain usable until Close.
func (s *Stream) Wait() error { <-s.done; return s.err }

// Close stops the child, joins boundary cleanup and releases all owned pipes.
func (s *Stream) Close() error {
	s.closeOnce.Do(func() {
		s.cancel()
		s.stop()
		s.Stdin.Close()
		s.Wait()
		s.Stdout.Close()
		s.Stderr.Close()
	})
	return s.Wait()
}

func validateStream(r Request) error {
	if err := validate(r); err != nil {
		return err
	}
	if len(r.Stdin) != 0 || r.OutputStore != nil || r.CaptureLimit != 0 {
		return errors.New("protocol streams require pipe input and consumer-owned output bounds")
	}
	return nil
}
func (h Host) OpenStream(ctx context.Context, r Request) (*Stream, error) {
	if err := validateStream(r); err != nil {
		return nil, err
	}
	return openStream(ctx, h.Workspace, environment(r.Env), r.Argv[0], r.Argv[1:], func() error { return nil })
}
func (c Container) OpenStream(ctx context.Context, r Request) (*Stream, error) {
	if err := validateStream(r); err != nil {
		return nil, err
	}
	plan, err := c.prepare(ctx, r)
	if err != nil {
		return nil, err
	}
	return openStream(ctx, "", environment(nil), plan.docker, plan.args, plan.cleanup)
}
func openStream(ctx context.Context, dir string, env []string, name string, args []string, cleanup func() error) (stream *Stream, err error) {
	owned, cancel := context.WithCancel(ctx)
	cmd := process.Command(owned, name, args...)
	cmd.Dir = dir
	cmd.Env = env
	var pipes []*os.File
	retained := false
	defer func() {
		if !retained {
			cancel()
			for _, p := range pipes {
				p.Close()
			}
			err = errors.Join(err, cleanup())
		}
	}()
	input, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	defer func() {
		if !retained {
			input.Close()
		}
	}()
	output, outWriter, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	pipes = append(pipes, output, outWriter)
	diagnostic, errWriter, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	pipes = append(pipes, diagnostic, errWriter)
	cmd.Stdout = outWriter
	cmd.Stderr = errWriter
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	outWriter.Close()
	errWriter.Close()
	s := &Stream{Stdin: input, Stdout: output, Stderr: diagnostic, cancel: cancel, stop: func() { process.KillGroup(cmd) }, done: make(chan struct{})}
	retained = true
	go func() {
		defer close(s.done)
		defer cancel()
		runErr := cmd.Wait()
		if owned.Err() != nil {
			runErr = nil
		}
		s.err = errors.Join(runErr, process.KillGroup(cmd), cleanup())
	}()
	return s, nil
}
