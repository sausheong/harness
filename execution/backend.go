// Package execution provides explicit host and container command boundaries.
// Admission belongs to the caller; choosing a backend never grants permission.
package execution

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sausheong/harness/process"
)

type Request struct {
	CaptureLimit int                    // zero uses 64 KiB; bounded protocol callers may request up to 16 MiB
	OutputStore  *process.ArtifactStore // caller-owned trusted capture store
	Argv         []string
	Env          map[string]string
	Stdin        []byte
}
type Result struct {
	StdoutTruncated, StderrTruncated bool
	StdoutBytes, StderrBytes         int64
	StdoutArtifact, StderrArtifact   process.ArtifactInfo
	Stdout, Stderr                   string
	ExitCode                         int
	Truncated                        bool
}
type Backend interface {
	Run(context.Context, Request) (Result, error)
	Boundary() string
}
type Host struct{ Workspace string }

func (Host) Boundary() string { return "unrestricted host; commands can access host resources" }
func (h Host) Run(ctx context.Context, r Request) (Result, error) {
	if err := validate(r); err != nil {
		return Result{}, err
	}
	return runCaptured(ctx, r.OutputStore, r.CaptureLimit, h.Workspace, environment(r.Env), r.Stdin, r.Argv[0], r.Argv[1:]...)
}

var imageID = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
var envName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

type Container struct {
	Resources                        []ReadOnlyResource // explicit pinned files mounted read-only at /harness-resources
	WorkerSHA256                     string             // required when WorkerPath is configured
	WorkerPath                       string             // optional trusted executable mounted read-only at /hand-worker
	Docker, Socket, Image, Workspace string
	Writable, Network                bool
}

func (c Container) Boundary() string {
	return fmt.Sprintf("container: workspace writable=%t; network=%t; other host paths unmounted", c.Writable, c.Network)
}
func validate(r Request) error {
	if r.CaptureLimit < 0 || r.CaptureLimit > 16<<20 {
		return errors.New("invalid capture limit")
	}
	if len(r.Argv) == 0 || r.Argv[0] == "" || len(r.Argv) > 1024 || len(r.Stdin) > 1<<20 {
		return errors.New("invalid or oversized execution request")
	}
	size := 0
	for _, a := range r.Argv {
		size += len(a)
		if strings.ContainsRune(a, 0) {
			return errors.New("NUL in argument")
		}
	}
	for k, v := range r.Env {
		size += len(k) + len(v)
		if !envName.MatchString(k) || strings.ContainsRune(v, 0) {
			return errors.New("invalid environment entry")
		}
	}
	if size > 1<<20 {
		return errors.New("execution arguments exceed 1 MiB")
	}
	return nil
}
func environment(values map[string]string) []string {
	env := []string{"PATH=/usr/local/bin:/usr/bin:/bin"}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		env = append(env, k+"="+values[k])
	}
	return env
}
func run(ctx context.Context, dir string, env []string, input []byte, name string, args ...string) (Result, error) {
	return runCaptured(ctx, nil, 0, dir, env, input, name, args...)
}
func runCaptured(ctx context.Context, store *process.ArtifactStore, limit int, dir string, env []string, input []byte, name string, args ...string) (Result, error) {
	cmd := process.Command(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = env
	cmd.Stdin = bytes.NewReader(input)
	type capture interface {
		Write([]byte) (int, error)
		Snapshot() (string, int64, bool)
	}
	if limit == 0 {
		limit = 64 << 10
	}
	var out, stderr capture = process.NewCapture(limit), process.NewCapture(limit)
	var diskOut, diskErr *process.OutputCapture
	if store != nil {
		diskOut, diskErr = store.Capture(limit), store.Capture(limit)
		out, stderr = diskOut, diskErr
	}
	cmd.Stdout = out
	cmd.Stderr = stderr
	err := process.Run(cmd)
	result := Result{ExitCode: -1}
	if diskOut != nil {
		diskOut.Close()
		diskErr.Close()
		result.StdoutArtifact = diskOut.Info()
		result.StderrArtifact = diskErr.Info()
	}
	a, ab, at := out.Snapshot()
	b, bb, bt := stderr.Snapshot()
	result.Stdout = a
	result.Stderr = b
	result.StdoutBytes = ab
	result.StderrBytes = bb
	result.Truncated = at || bt
	result.StdoutTruncated = at
	result.StderrTruncated = bt
	if cmd.ProcessState != nil {
		result.ExitCode = cmd.ProcessState.ExitCode()
	}
	return result, err
}

type containerCommand struct {
	docker  string
	args    []string
	cleanup func() error
}

func (c Container) Run(ctx context.Context, r Request) (result Result, err error) {
	plan, err := c.prepare(ctx, r)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, plan.cleanup()) }()
	return runCaptured(ctx, r.OutputStore, r.CaptureLimit, "", environment(nil), r.Stdin, plan.docker, plan.args...)
}
func (c Container) prepare(ctx context.Context, r Request) (plan *containerCommand, err error) {
	if err = ctx.Err(); err != nil {
		return
	}
	if err = validate(r); err != nil {
		return
	}
	if !filepath.IsAbs(c.Docker) || !filepath.IsAbs(c.Socket) || !imageID.MatchString(c.Image) {
		return nil, errors.New("container requires absolute Docker executable/socket and immutable sha256 image ID")
	}
	info, e := os.Stat(c.Socket)
	if e != nil || info.Mode()&os.ModeSocket == 0 {
		return nil, errors.New("local container socket unavailable; no host fallback")
	}
	workspace, e := filepath.EvalSymlinks(c.Workspace)
	if e != nil {
		return nil, e
	}
	workspace, e = filepath.Abs(workspace)
	if e != nil {
		return nil, e
	}
	info, e = os.Stat(workspace)
	if e != nil || !info.IsDir() || strings.ContainsAny(workspace, ",\n\r") {
		return nil, errors.New("invalid workspace mount")
	}
	// A separate config directory prevents ambient Docker auth/context inheritance.
	config, e := os.MkdirTemp("", "harness-docker-config-")
	if e != nil {
		return nil, e
	}
	defer func() {
		if plan == nil {
			os.RemoveAll(config)
		}
	}()
	configCanonical, e := filepath.EvalSymlinks(config)
	if e != nil {
		return nil, e
	}
	relative, e := filepath.Rel(workspace, configCanonical)
	if e != nil || (relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)) {
		return nil, errors.New("container state must be outside the mounted workspace")
	}
	if strings.ContainsAny(configCanonical, ",\n\r") {
		return nil, errors.New("container state path cannot be mounted safely")
	}
	base := []string{"--config", config, "--host", "unix://" + c.Socket}
	inspectCtx, inspectCancel := context.WithTimeout(ctx, 5*time.Second)
	inspected, inspectErr := run(inspectCtx, "", environment(nil), nil, c.Docker, append(append([]string{}, base...), "image", "inspect", "--format", `{{json (index .Config "Volumes")}}`, c.Image)...)
	inspectCancel()
	if inspectErr != nil {
		return nil, fmt.Errorf("inspect container image: %w: %s", inspectErr, inspected.Stderr)
	}
	if inspected.Truncated {
		return nil, errors.New("container image volume metadata exceeds limit")
	}
	if err = validateImageVolumes(inspected.Stdout); err != nil {
		return
	}
	name := "harness-exec-" + strings.ToLower(rand.Text())
	cleanup := func() error {
		defer os.RemoveAll(config)
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		res, e := run(cleanupCtx, "", environment(nil), nil, c.Docker, append(append([]string{}, base...), "rm", "--force", name)...)
		if e != nil {
			return fmt.Errorf("container cleanup failed for %s: %w: %s", name, e, res.Stderr)
		}
		return nil
	}
	network := "none"
	if c.Network {
		network = "bridge"
	}
	mount := "type=bind,src=" + workspace + ",dst=/workspace,bind-propagation=rprivate,bind-recursive=disabled"
	if !c.Writable {
		mount += ",readonly"
	}
	uid, gid := os.Getuid(), os.Getgid()
	if uid < 1 {
		uid, gid = 65534, 65534
	}
	args := append(append([]string{}, base...), "create", "--name", name, "--pull=never", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network", network, "--pids-limit=128", "--memory=512m", "--memory-swap=512m", "--cpus=2", "--init", "--no-healthcheck", "--log-driver=none", "--user", strconv.Itoa(uid)+":"+strconv.Itoa(gid), "--mount", mount, "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777", "--workdir", "/workspace", "--interactive", "--entrypoint", r.Argv[0])
	if c.WorkerPath != "" {
		worker, e := snapshotWorker(ctx, c.WorkerPath, c.WorkerSHA256, config)
		if e != nil {
			return nil, e
		}
		args = append(args, "--mount", "type=bind,src="+worker+",dst=/hand-worker,readonly")
	}

	if len(c.Resources) > 0 {
		resources, e := snapshotResources(ctx, c.Resources, config)
		if e != nil {
			return nil, e
		}
		args = append(args, "--mount", "type=bind,src="+resources+",dst=/harness-resources,readonly,bind-recursive=disabled")
	}

	for _, entry := range environment(r.Env) {
		args = append(args, "--env", entry)
	}
	args = append(args, c.Image)
	args = append(args, r.Argv[1:]...)
	// Join creation independently of caller cancellation before removing its name.
	// A daemon/transport failure is surfaced and cannot be treated as clean exit.
	createCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	created, createErr := run(createCtx, "", environment(nil), nil, c.Docker, args...)
	cancel()
	if createErr != nil {
		return nil, errors.Join(fmt.Errorf("container create failed: %w: %s", createErr, created.Stderr), cleanup())
	}
	if err = ctx.Err(); err != nil {
		return nil, errors.Join(err, cleanup())
	}
	return &containerCommand{docker: c.Docker, args: append(append([]string{}, base...), "start", "--attach", "--interactive", name), cleanup: cleanup}, nil
}

// Image-declared volumes otherwise create implicit writable disk mounts even
// with a read-only root filesystem. Require explicit workspace/scratch only.
func validateImageVolumes(raw string) error {
	var volumes map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &volumes); err != nil {
		return errors.New("invalid container image volume metadata")
	}
	if len(volumes) != 0 {
		return errors.New("container image declares implicit volumes; use an image without VOLUME declarations")
	}
	return nil
}
