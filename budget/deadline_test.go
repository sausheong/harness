package budget

import (
	"context"
	"errors"
	"github.com/sausheong/harness/session"
	"testing"
	"time"
)

func TestDeadlineExpiresAcrossRestartAndExplicitResume(t *testing.T) {
	ctx := context.Background()
	store := session.NewStore(t.TempDir())
	if err := store.Create("a", "time"); err != nil {
		t.Fatal(err)
	}
	s, err := store.LoadExclusive("a", "time")
	if err != nil {
		t.Fatal(err)
	}
	if err = DecideDeadline(ctx, s, time.Second); err != nil {
		t.Fatal(err)
	}
	bound, cancel, err := WithDeadline(ctx, s)
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	select {
	case <-bound.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("deadline did not cancel")
	}
	if !errors.Is(context.Cause(bound), ErrTimeExhausted) {
		t.Fatal(context.Cause(bound))
	}
	before, err := ReadDeadline(s)
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = store.LoadExclusive("a", "time")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	after, err := ReadDeadline(s)
	if err != nil || !after.Deadline.Equal(before.Deadline) {
		t.Fatal("restart reset deadline")
	}
	_, cleanup, err := WithDeadline(ctx, s)
	cleanup()
	if !errors.Is(err, ErrTimeExhausted) {
		t.Fatal("expired restart accepted")
	}
	if err = DecideDeadline(ctx, s, time.Hour); err != nil {
		t.Fatal(err)
	}
	resumed, cleanup, err := WithDeadline(ctx, s)
	defer cleanup()
	if err != nil || resumed.Err() != nil {
		t.Fatal("explicit resume failed")
	}
}
func TestDeadlineValidationAndParentCancellation(t *testing.T) {
	s := session.NewSession("a", "time")
	ctx := context.Background()
	for _, d := range []time.Duration{0, -1, 31 * 24 * time.Hour} {
		if err := DecideDeadline(ctx, s, d); err == nil {
			t.Fatal("invalid allowance accepted")
		}
	}
	if err := DecideDeadline(ctx, s, time.Hour); err != nil {
		t.Fatal(err)
	}
	parent, cancel := context.WithCancel(ctx)
	bound, cleanup, err := WithDeadline(parent, s)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	cancel()
	<-bound.Done()
	if !errors.Is(context.Cause(bound), context.Canceled) || errors.Is(context.Cause(bound), ErrTimeExhausted) {
		t.Fatal("user cancellation relabelled as budget exhaustion")
	}
}
