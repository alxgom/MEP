# Demo 10.8.1 Agent Instructions

These instructions apply when working inside `demos/10.8.1`.

## Required Context

Before changing code, read:

- `REFACTOR_SCOPE.md`
- `IMPLEMENTATION_STEPS.md`
- `.agents/skills/mep-demo-10-8-1-refactor/SKILL.md`

Read the workflow references only when relevant:

- `.agents/skills/mep-demo-10-8-1-refactor/references/config-taxonomy.md`
- `.agents/skills/mep-demo-10-8-1-refactor/references/refactor-boundaries.md`
- `.agents/skills/mep-demo-10-8-1-refactor/references/validation.md`
- `.agents/skills/mep-demo-10-8-1-refactor/references/git-discipline.md`

## Development Rules

- Keep the work specific to Demo 10.8.1.
- Preserve runnable behavior through `main.py` after each extraction.
- Prefer small behavior-preserving commits over broad rewrites.
- Keep internal geometry and routing units in integer millimetres where practical.
- Use English semantic config keys.
- Do not classify a value as `REGULATION` without an explicit legal/code/project-code source.
- Do not invent Cli/Coc/San behavior. Leave extension points and wait for concrete source/demo behavior.
- Do not touch unrelated dirty files from other demos.
- Update `IMPLEMENTATION_STEPS.md` when a meaningful step is completed or the plan changes.

## Validation

- Run `python -m py_compile` on changed Python files before committing.
- Add focused tests when extracted modules become importable and pure enough to test.
- Use manual Pygame validation at the milestones listed in `IMPLEMENTATION_STEPS.md`.

