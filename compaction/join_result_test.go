package compaction

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestJoinRetainsCompletedResultUntilConsumed(t *testing.T) {
	mgr := &Manager{Summarizer: &Summarizer{Provider: usageSummaryProvider{}, Model: "summary"}, PreserveTurns: 1}
	sess := longSession()
	done := mgr.MaybeCompactAsyncContext(context.Background(), sess, ReasonManual)
	<-done
	require.False(t, mgr.HasInFlight(sess), "completed producer is not running")
	// Neither a legacy waiter nor another launch may erase an unconsumed result.
	res, found := mgr.WaitForInFlight(sess, time.Second)
	require.True(t, found)
	require.True(t, res.Compacted)
	require.Equal(t, done, mgr.MaybeCompactAsyncContext(context.Background(), sess, ReasonPreventive))
	res, found, err := mgr.JoinInFlight(context.Background(), sess)
	require.NoError(t, err)
	require.True(t, found)
	require.True(t, res.Compacted)
	require.Len(t, res.Requests, 1)
	_, found, err = mgr.JoinInFlight(context.Background(), sess)
	require.NoError(t, err)
	require.False(t, found)
	next := mgr.MaybeCompactAsyncContext(context.Background(), sess, ReasonPreventive)
	require.NotEqual(t, done, next)
	<-next
	_, found, err = mgr.JoinInFlight(context.Background(), sess)
	require.NoError(t, err)
	require.True(t, found)
}
