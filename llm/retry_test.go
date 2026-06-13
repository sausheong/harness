package llm

import (
	"errors"
	"fmt"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsRetryableModelErrorAnthropic429(t *testing.T) {
	err := &anthropic.Error{StatusCode: 429}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorAnthropic529(t *testing.T) {
	err := &anthropic.Error{StatusCode: 529}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorAnthropic400NotRetryable(t *testing.T) {
	// Bad-request, validation errors etc. — fallback model can't fix
	// these, so we must NOT swap models. Critical: a flag flip here
	// would silently mask config bugs by pretending the second call
	// succeeded against a different model.
	err := &anthropic.Error{StatusCode: 400}
	assert.False(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorAnthropic401NotRetryable(t *testing.T) {
	err := &anthropic.Error{StatusCode: 401}
	assert.False(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOpenAI429(t *testing.T) {
	err := &openai.APIError{HTTPStatusCode: 429}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOpenAI500(t *testing.T) {
	err := &openai.APIError{HTTPStatusCode: 500}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOpenAI503(t *testing.T) {
	err := &openai.APIError{HTTPStatusCode: 503}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOpenAI400NotRetryable(t *testing.T) {
	err := &openai.APIError{HTTPStatusCode: 400}
	assert.False(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOpenAIRequestError(t *testing.T) {
	// RequestError is the SDK's wrapper for non-API HTTP failures.
	err := &openai.RequestError{HTTPStatusCode: 429}
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorWrappedError(t *testing.T) {
	// errors.As must traverse fmt.Errorf %w wrappers — the runtime
	// wraps SDK errors in fmt.Errorf("llm error: %w", err), which
	// landed past the classifier in an earlier draft.
	wrapped := fmt.Errorf("transient: %w", &anthropic.Error{StatusCode: 529})
	assert.True(t, IsRetryableModelError(wrapped))
}

func TestIsRetryableModelErrorNil(t *testing.T) {
	assert.False(t, IsRetryableModelError(nil))
}

func TestIsRetryableModelErrorOtherErrorString429(t *testing.T) {
	// Substring fallback: an SDK that returns a plain wrapped error
	// without typed-error interfaces should still be classified
	// correctly. Belt-and-braces.
	err := errors.New("api request failed: 429 Too Many Requests")
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorOtherErrorStringRateLimit(t *testing.T) {
	err := errors.New("rate limit exceeded")
	assert.True(t, IsRetryableModelError(err))
}

func TestIsRetryableModelErrorUnrelatedError(t *testing.T) {
	err := errors.New("some other failure")
	assert.False(t, IsRetryableModelError(err))
}

func TestRetry_DigitsInRequestIDNotRetryable(t *testing.T) {
	err := errors.New("request failed: req_abc429def something went wrong")
	require.False(t, IsRetryableModelError(err))
}

func TestRetry_RealStatus429Retryable(t *testing.T) {
	err := errors.New("openai: error, status 429 too many requests")
	require.True(t, IsRetryableModelError(err))
}
