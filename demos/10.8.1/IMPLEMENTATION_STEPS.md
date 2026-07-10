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

## External References

- `C:\Users\ALEXIS GOMEL\Documents\Dwelling-export`: dwelling geometry export source used by real-dwelling loading.
- `C:\Users\ALEXIS GOMEL\Documents\letsmep-routing-core`: routing-core code that Demo 10.8.1 mimics or selectively ports for Sal behavior.
- `C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Familias`: currently available machine-family metadata by installation type. Sal has one machine now; Cli is expected to use more than one, so the refactor should keep machine catalogs plural.
- `C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Config\global_config.json`: mutable local-testing config used as a reference for parameter values when explicitly requested. Do not silently treat this as immutable truth; use it when the task asks to mimic or source a value from local testing config.

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
- Sal Ozeo Flat machine dimensions and simple policies now live in `vent_router.domain.machines`.
- Canonical large-duct route names now live in `vent_router.domain.routes`.
- First pure geometry helper extracted: `snap_to_integer_grid` now lives in `vent_router.geometry.shapely_utils`.
- Pure NumPy segment distance helpers now live in `vent_router.geometry.distances`.
- Boundary and line segment extraction helpers now live in `vent_router.geometry.segments`.
- Ray casting and ray intersection helpers now live in `vent_router.geometry.rays`.
- Axis-aligned segment normalization/relation/distance helpers now live in `vent_router.geometry.axis`.
- Pure route segment merging and metric helpers now live in `vent_router.routing.segments`.
- Pure route quality counters now live in `vent_router.routing.metrics`, with Sal-specific diameter and minimum-piece policies injected by `main.py`.

## Next Steps

1. Continue import-safe value objects before moving behavior:
   - app/runtime state shell
2. Continue extracting pure geometry helpers.
3. Continue extracting pure routing helpers.
4. Extract graph builders behind a stable interface.
5. Extract routing scoring and backend-agnostic quality evaluation.
6. Extract routing backends.
7. Extract placement.
8. Extract UI drawing and event handling.

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
- Extracted `MachineSpec` and `SAL_OZEO_FLAT_MACHINE` to `vent_router.domain.machines`.
- Kept old `main.py` machine constant names as compatibility aliases assigned from the machine spec.
- Moved route diameter and pin stub policies to the machine spec while preserving baseline behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\domain\__init__.py demos\10.8.1\vent_router\domain\machines.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py demos\10.8.1\tests\test_machines.py`.
- Added canonical `SHAFT_ROUTE_NAME`, `KITCHEN_ROUTE_NAME`, and `LARGE_DUCT_ROUTE_NAMES` in `vent_router.domain.routes`.
- Used `LARGE_DUCT_ROUTE_NAMES` for Sal machine large-duct policy and fixed-width route drawing.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\domain\__init__.py demos\10.8.1\vent_router\domain\machines.py demos\10.8.1\vent_router\domain\routes.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py demos\10.8.1\tests\test_machines.py`.
- Extracted `snap_to_integer_grid` to `vent_router.geometry.shapely_utils`.
- Added focused tests for polygon, line, and geometry collection snapping.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\shapely_utils.py demos\10.8.1\tests\test_shapely_utils.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `point_segment_min_distances`, `edge_segment_min_distances`, and `edge_parallel_segment_min_distances` to `vent_router.geometry.distances`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for projection, endpoint, empty-segment, sampled-edge, and parallel-overlap distance behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\distances.py demos\10.8.1\tests\test_distances.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `extract_boundary_segments` and `extract_line_segments` to `vent_router.geometry.segments`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for polygon boundaries, geometry collections, line strings, multi-line strings, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\segments.py demos\10.8.1\tests\test_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `cast_rays_numpy` and `ray_ray_intersections_numpy` to `vent_router.geometry.rays`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for rectangular boundary ray casting, ray intersections, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\rays.py demos\10.8.1\tests\test_rays.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `normalize_axis_segment`, `axis_segment_relation`, and `axis_segment_distance` to `vent_router.geometry.axis`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for normalization, zero/diagonal rejection, overlap, crossing, endpoint touch, and separated distance.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\axis.py demos\10.8.1\tests\test_axis.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `merged_axis_segments`, `merged_route_axis_segments`, `metric_route_segments`, and `point_is_segment_endpoint` to `vent_router.routing.segments`.
- Left `_route_axis_records` in `main.py` for now because it still depends on the active route-diameter policy.
- Added focused tests for route segment merging, route-name preservation, non-axis metric segments, and endpoint detection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\segments.py demos\10.8.1\tests\test_routing_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route quality counters to `vent_router.routing.metrics`: crossings, overlaps, clearance conflicts, turns, merged piece lengths, and short-piece counts.
- Kept `main.py` compatibility wrappers where current behavior depends on active Sal policies such as route diameter, clearance, and minimum piece factor.
- Added focused tests for metric crossing semantics, overlap counting, injected clearance policy, turn counting, and injected short-piece policy.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\metrics.py demos\10.8.1\tests\test_routing_metrics.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.

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
- Add a machine-catalog loader only when the refactor needs to consume the `Familias` metadata rather than the current hardcoded Sal machine spec.
- Add a local-testing config adapter only when a task explicitly asks to source values from `global_config.json`.
