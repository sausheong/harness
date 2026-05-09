package runtime

// ReviewPromptDefault is the standard reviewer instruction. Concise:
// asks the reviewer to identify durable preferences/facts/workflows
// from the just-finished conversation and save them via memory or
// skill tools.
//
// Use as ReviewSpec.Prompt when you don't have a domain-specific
// reviewer prompt.
const ReviewPromptDefault = `You are reviewing a conversation that just completed. Your job is to extract anything worth remembering for future sessions.

Look for:
- User preferences (style, format, tools, conventions) — save as memory
- Project facts (architecture decisions, constraints, naming) — save as memory
- Reusable workflows or techniques the agent figured out — save as a skill

Use the memory tool (action=save) for short, durable facts. Use the skill_manage tool (action=create) for procedural knowledge that has its own name and would benefit from a body.

Skip:
- Things obvious from reading the code
- One-off task details that won't recur
- Information that's already been saved (check the existing memory and skills first via action=list)

If nothing is worth saving, do nothing and finish. A clean review is a successful review.`

// ReviewPromptVerbose is the elaborated reviewer instruction. Adds
// explicit examples and a worked-out decision tree. Use for stronger
// reviewers (capable models that benefit from longer context) or in
// the early days of a deployment when you want the reviewer to
// over-explain its choices.
const ReviewPromptVerbose = `You are reviewing a conversation that just completed between a user and an agent. Your job is to extract anything worth remembering for future sessions.

## What to look for

1. **User preferences** — style, format, tools, conventions the user has expressed. Examples:
   - "User prefers terse explanations without summaries at the end"
   - "User wants commits without Co-Authored-By trailers"
   - "User runs Go tests with -race by default"
   Save these via the memory tool (action=save).

2. **Project facts** — architectural decisions, constraints, naming choices, dependency choices that explain the codebase's shape. Examples:
   - "This project uses RFC3339Nano for timestamps to preserve precision"
   - "The skills package mirrors the memory package's structural template"
   Save these via the memory tool (action=save).

3. **Reusable workflows** — procedural knowledge the agent figured out that has a name and would benefit from a body. Examples:
   - A debugging routine the agent invented for this codebase
   - A multi-step deployment process the agent walked through
   - A test strategy specific to a tricky concurrency scenario
   Save these via the skill_manage tool (action=create) with kebab-case names.

## Decision tree

- One-line fact, no body needed → memory
- Multi-paragraph technique with a name → skill
- Already in memory or skills (check via action=list first) → don't duplicate
- Trivial / one-off / obvious from the code → skip

## Worked example

Conversation: User and agent debugged a flaky test that turned out to be a 1-second mtime resolution issue.
Reviewer extracts:
- memory: "Filesystem mtime resolution is 1s on this project's macOS dev machines; tests checking mtime advance need time.Sleep(1100ms)"
- skill (name=debugging-mtime-flakes): full body explaining the symptom, the diagnostic process, and the fix pattern

## Closing

If nothing is worth saving, finish without calling any write tools. A clean review is a successful review — quality over quantity.`
