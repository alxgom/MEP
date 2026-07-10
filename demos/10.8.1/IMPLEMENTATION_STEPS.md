# Demo 10.8.1 Implementation Steps

This file is the execution ledger for the Demo 10.8.1 refactor. Keep it current before and after each meaningful code move.

## Working Decisions

- Scope: Demo 10.8.1 only. Do not generalize to older demos.
- Placement: keep the local workflow in this folder for now. A future extraction to `Documents/interactive-routing-app` is plausible once the app stops behaving like a demo.
- Internal units: use integer millimetres wherever practical. UI can show metres when that improves readability.
- Config language: use English semantic keys.
- Regulation: do not classify existing values as regulation unless a legal/code/project-code source is explicit.
- UI ownership: most UI behavior and style is installation-domain scoped; shared tools like ruler, zoom, overlays, and base dwelling geometry styling are global UI.
- Cli/Coc/San: treat as future unknowns. Do not design by guessing their details; leave extension points and merge concrete behavior later.

## Validation Strategy

Use layered validation:

- Always: run `python -m py_compile` on changed Python files.
- For pure modules: add or run focused tests once extraction makes functions importable.
- For integration milestones: run the Pygame app manually and validate the interaction checklist.
- For visually heavy behavior: prefer manual validation at milestones over brittle screenshot tests until the app is modular enough to expose deterministic state.

Manual validation milestones:

- after first `main.py` entrypoint split
- after graph builders move
- after routing strategies move
- after UI event handling moves
- after config defaults are consumed by runtime
- before any project move outside `demos/10.8.1`

## Current Status

- Baseline copied from Demo 10.8 and committed.
- Refactor scope document created and committed.
- Initial config catalog created and committed.
- Local workflow and implementation tracking added before further code movement.
- First import-safe value object extracted: `EnvView` now lives in `vent_router.graphs.env`.

## Next Steps

1. Continue import-safe value objects before moving behavior:
   - machine specification
   - route/terminal specification
   - app/runtime state shell
2. Extract pure geometry helpers.
3. Extract graph builders behind a stable interface.
4. Extract routing backends and scoring.
5. Extract placement.
6. Extract UI drawing and event handling.

## Step Log

### 2026-07-10

- Created `demos/10.8.1` from current `demos/10.8`.
- Committed baseline: `0f8e0b2 Add demo 10.8.1 baseline`.
- Added refactor scope: `5e9f1d4 Scope demo 10.8.1 refactor`.
- Added initial config catalog: `f7cd803 Add demo 10.8.1 config catalog`.
- Added local workflow plan and project-local agent instructions.
- Updated taxonomy decisions from the design discussion: millimetre integers internally, no confirmed regulation values in current demo config, English config keys, and installation-scoped UI.
- Validation note: `quick_validate.py` could not run because the active Python environment does not have `yaml`/PyYAML. Manual skill-structure checks and Python compile checks were used instead.
- Extracted `EnvView` to `vent_router.graphs.env` and updated `main.py` to import it.
- Added focused pytest coverage for `EnvView` reference preservation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\graphs\__init__.py demos\10.8.1\vent_router\graphs\env.py demos\10.8.1\tests\conftest.py demos\10.8.1\tests\test_graph_env.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py`.

## Commit Checklist

Before each commit:

- `git status --short`
- inspect staged files
- compile changed Python files
- update this file if the step changes the refactor plan or status
- commit only related files

## Open Items

- Decide later whether collision policy remains top-level `COLLISIONS` or moves under a shared domain area.
- Decide later whether Demo 10.8.1 should move to a standalone folder such as `Documents/interactive-routing-app`.
- Obtain concrete Cli/Coc/San behavior only when ready to merge those demos.
