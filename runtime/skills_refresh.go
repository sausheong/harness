package runtime

import "fmt"

// RefreshSkills refreshes discovery while idle. Run also refreshes between
// model requests, after preceding tool calls have joined. Providers must bound
// filesystem work; callers cannot refresh concurrently with an active run.
func (r *Runtime) RefreshSkills() error {
	if !r.runMu.TryLock() {
		return fmt.Errorf("runtime is already running")
	}
	defer r.runMu.Unlock()
	r.refreshSkills()
	return nil
}

func (r *Runtime) refreshSkills() {
	if r.refreshToolPrompt == nil {
		return
	}
	var names []string
	if r.Tools != nil {
		names = r.Tools.Names()
	}
	r.StaticSystemPrompt = r.refreshToolPrompt(names)
}
