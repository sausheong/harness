package runtime

import (
	"context"
	"errors"
	"testing"
)

func TestSteeringPreservesProtocolCancellationCause(t *testing.T) {
	cause := errors.New("duplicate tool call ID")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	ctx = WithSteering(ctx, func(context.Context) (*SteeringMessage, error) { t.Fatal("called after cancellation"); return nil, nil })
	_, err := (&Runtime{}).applySteering(ctx, nil)
	if !errors.Is(err, cause) {
		t.Fatalf("lost cause: %v", err)
	}
}
