---
name: mep-demo-10-8-1-refactor
description: Workflow for refactoring Demo 10.8.1 into a modular interactive routing app. Use when modifying demos/10.8.1, its vent_router package, config taxonomy, routing/geometry/graph/placement/UI boundaries, validation plan, or git workflow for this refactor.
---

# Demo 10.8.1 Refactor Workflow

Follow this workflow before changing code in `demos/10.8.1`.

## Start

1. Read `REFACTOR_SCOPE.md`.
2. Read `IMPLEMENTATION_STEPS.md`.
3. Run `git status --short` from the `MEP` repo root.
4. Identify unrelated dirty files and leave them untouched.
5. State the intended cohesive subsystem step before editing.

## Core Rules

- Keep the work scoped to Demo 10.8.1.
- Preserve `python main.py` as the app entrypoint until a deliberate migration changes it.
- Prefer medium-to-large coherent subsystem extractions over isolated helper moves when the dependency direction and adapter seam are already clear.
- Include adjacent pure helpers, adapters, and exports in the same commit when they form one stable boundary; do not artificially split them into helper-only commits.
- Keep each extraction reviewable: do not combine unrelated subsystems or alter behavior while moving code unless the commit explicitly declares that change.
- Keep pure computation independent from Pygame and filesystem adapters.
- Use integer millimetres internally wherever practical.
- Use English semantic config keys.
- Treat current distances as feasibility, heuristic, solver, or performance values unless a regulation source is explicit.
- Do not guess future Cli/Coc/San behavior.
- Commit coherent steps locally.

## References

- Read `references/config-taxonomy.md` when changing config keys, defaults, units, or metadata.
- Read `references/refactor-boundaries.md` when moving functions between modules.
- Read `references/validation.md` before choosing test/manual validation for a step.
- Read `references/git-discipline.md` before staging or committing.

## Finish

1. Run risk-proportionate validation from `references/validation.md`.
2. Update `IMPLEMENTATION_STEPS.md` with completed work, validation, and next step changes.
3. Stage only related files.
4. Commit with a focused message.
5. Report commit hash and any remaining risks.
