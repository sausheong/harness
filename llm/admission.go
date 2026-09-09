package llm

import (
	"context"
	"errors"
)

// CallAdmission reserves capacity before an individual provider attempt. A
// successful reservation returns a non-nil settlement callback, called exactly
// once before terminal delivery. Missing usage is unknown, never zero cost.
// Implementations must be concurrency safe and must not mutate the request.
// Settlement must retain uncertain in-flight charges after cancellation.
// A denial must leave no reservation behind. A settlement error must leave
// conservative durable state; it turns the attempt into a visible error.
type CallAdmission func(context.Context, ChatRequest, CallCategory, string) (func(RequestUsage) error, error)

type callAdmissionKey struct{}

// WithCallAdmission adds a guard; nested callers cannot remove earlier guards.
// Guards run in installation order. If a later guard denies admission, earlier
// reservations settle as not_dispatched, with no provider call made.
func WithCallAdmission(ctx context.Context, guard CallAdmission) context.Context {
	previous, _ := ctx.Value(callAdmissionKey{}).([]CallAdmission)
	guards := append([]CallAdmission(nil), previous...)
	if guard != nil {
		guards = append(guards, guard)
	}
	return context.WithValue(ctx, callAdmissionKey{}, guards)
}

func admitCall(ctx context.Context, req ChatRequest, record RequestUsage, guards []CallAdmission) (func(RequestUsage) error, error) {
	var settlements []func(RequestUsage) error
	settle := func(result RequestUsage) error {
		var failures []error
		for i := len(settlements) - 1; i >= 0; i-- {
			detached := result
			if result.Usage != nil {
				u := *result.Usage
				detached.Usage = &u
			}
			failures = append(failures, settlements[i](detached))
		}
		return errors.Join(failures...)
	}
	for _, guard := range guards {
		finish, err := guard(ctx, req, record.Category, record.ID)
		if err == nil && finish == nil {
			err = errors.New("admission guard returned no settlement callback")
		}
		if err != nil {
			record.Status = "not_dispatched"
			return nil, errors.Join(err, settle(record))
		}
		settlements = append(settlements, finish)
	}
	return settle, nil
}
