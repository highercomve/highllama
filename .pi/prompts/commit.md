---
description: Analyze git changes and write a commit message with title and body
argument-hint: "[scope]"
---
Write a commit message for the current changes in this repository.

1. Use the worker agent to inspect the changes: run `git status --short`, `git diff --cached`, and `git diff` as needed.
2. Summarize what changed and why in a concise conventional-commit format:

```
<scope> <imperative title under 50 chars>

- <what changed and why>
- <notable edge cases or follow-ups>
```

If a scope is provided (`$1`), use it; otherwise infer the scope from the changed files.
