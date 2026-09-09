package runtime

func (r *Runtime) outputTokenLimit() int {
	if r.MaxOutputTokens > 0 {
		return r.MaxOutputTokens
	}
	return 8192
}
