# Git Discipline Reference

Use local commits as the safety rail for this refactor.

## Before Edits

- Run `git status --short`.
- Identify unrelated dirty files.
- Do not revert or stage unrelated user changes.

## Before Commit

- Inspect `git diff`.
- Inspect `git diff --cached --stat`.
- Stage only related files.
- Compile changed Python files.
- Update `IMPLEMENTATION_STEPS.md` if the plan/status changed.

## Commit Style

Use focused messages:

```text
Add demo 10.8.1 workflow notes
Extract geometry distance helpers
Add Sal config runtime adapter
```

Avoid mixing:

- behavior changes with large moves
- unrelated docs with code extraction
- edits to older demos

