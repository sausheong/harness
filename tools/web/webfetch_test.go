package web

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWebFetch_BlocksPrivateHost(t *testing.T) {
	tool := &WebFetchTool{}
	res, err := tool.Execute(context.Background(),
		[]byte(`{"url":"http://169.254.169.254/latest/meta-data/"}`))
	require.NoError(t, err)
	require.NotEmpty(t, res.Error, "private metadata IP must be rejected")
}
