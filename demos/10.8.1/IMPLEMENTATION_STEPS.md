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
- Validation pacing: use contract tests for public behavior and solver/geometry invariants. Do not add tests for simple forwarding wrappers or unchanged drawing moves; use compilation and the relevant manual milestone instead.
- Shared package boundary: use `mep_routing` for installation-neutral graph, geometry, routing, placement, and UI code. Do not organize those capabilities beneath a ventilation-specific package; future Sal/Cli/Coc/San adapters belong below explicit installation-domain boundaries.
- Graph subsystem: regular and baseline regular-grid construction are now moving into `mep_routing.graphs`; Hannan and epsilon builders remain the next part of this same subsystem move.
- Extracted regular-grid node/edge construction and wall filtering to `mep_routing.graphs.regular`; `main.py` now only supplies active geometry and commits graph state.
- Added one graph-builder contract test for integer-grid nodes and wall-crossing edge removal.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\regular.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\tests\test_graph_env.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Moved Sal machine catalog and named route classifications from shared `mep_routing.domain` to `mep_routing.installations.sal`.
- Made `MachineSpec` installation-neutral: its default large-route classification is empty and each installation supplies its own route policy.
- Validation: Python 3.11 `-m py_compile` for affected package, application, and test modules.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests\test_machines.py`.
- Moved Sal's sequential Shaft/Kitchen/small-duct orchestration into `mep_routing.installations.sal.strategies`.
- Kept `main.py` as an adapter for live graph state, solver callbacks, and UI-driven selection.
- Extracted dynamic machine-obstacle graph filtering to `mep_routing.graphs.dynamic`; `main.py` now only resolves protected terminal/access nodes and invalidates app caches.
- Added a contract test for blocked graph edges and protected machine access-node preservation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\dynamic.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\tests\test_graph_env.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted epsilon axis-generation policy to `mep_routing.graphs.epsilon`; `main.py` supplies active machine access points and commits the shared axis-grid result.
- All regular, Hannan, and epsilon graph construction now runs through `mep_routing.graphs`; dynamic machine blocking remains the next graph-state move.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\epsilon.py demos\10.8.1\mep_routing\graphs\hannan.py demos\10.8.1\mep_routing\graphs\__init__.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted Hannan static-axis generation to `mep_routing.graphs.hannan` with explicit geometry, terminal, shaft, and graph-spacing inputs.
- Kept `main.py` as the cache owner and active-state adapter; epsilon axis-generation policy remains in `main.py`.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\hannan.py demos\10.8.1\mep_routing\graphs\__init__.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted shared axis-grid node construction, visibility filtering, wall checks, required-node reconnection, and timing breakdowns to `mep_routing.graphs.axis_grid`.
- Hannan and epsilon builders now retain only their axis-generation policy and active app adapters.
- Added a contract test covering axis-grid visibility filtering and timing output.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\axis_grid.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\tests\test_graph_env.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.

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
- First import-safe value object extracted: `EnvView` now lives in `mep_routing.graphs.env`.
- Sal Ozeo Flat machine dimensions and route policies now live in `mep_routing.installations.sal`.
- First pure geometry helper extracted: `snap_to_integer_grid` now lives in `mep_routing.geometry.shapely_utils`.
- Pure NumPy segment distance helpers now live in `mep_routing.geometry.distances`.
- Boundary and line segment extraction helpers now live in `mep_routing.geometry.segments`.
- Polygon iteration and largest-polygon helpers now live in `mep_routing.geometry.polygons`.
- Ray casting and ray intersection helpers now live in `mep_routing.geometry.rays`.
- Axis-aligned segment normalization/relation/distance helpers now live in `mep_routing.geometry.axis`.
- Pure route segment merging and metric helpers now live in `mep_routing.routing.segments`.
- Pure route quality counters now live in `mep_routing.routing.metrics`, with Sal-specific diameter and minimum-piece policies injected by `main.py`.
- Route scoring and quality-summary formatting now live in `mep_routing.routing.scoring`, with runtime weights and Sal policies passed in explicitly.
- Route length in metres now lives in `mep_routing.routing.scoring`.
- Route hit testing now lives in `mep_routing.routing.hit_testing`, with the UI zoom-derived hit radius supplied by `main.py`.
- Selected route pin detection now lives in `mep_routing.routing.hit_testing`.
- Terminal validity classification now lives in `mep_routing.routing.terminal_validity`, with room geometry accessors and clearances supplied by `main.py`.
- Min-cost-flow graph primitives now live in `mep_routing.routing.flow`; higher-level Sal route construction remains in `main.py`.
- Min-cost-flow source normalization now lives in `mep_routing.routing.flow`, with `grid_kd` supplied by `main.py`.
- Small-pin min-cost-flow target-spec assembly now lives in `mep_routing.routing.flow`.
- Sal machine pin geometry, port access specs, and outward direction helpers now live in `mep_routing.domain.machines`.
- Topological placement distance-field and score aggregation helpers now live in `mep_routing.placement.fields`.
- Machine placement feasibility and candidate-room filtering now live in `mep_routing.placement.feasibility`.
- Core-like machine placement scoring primitives now live in `mep_routing.placement.scoring`.
- Machine rotation field scoring and field-alignment angle selection helpers now live in `mep_routing.placement.rotation`.
- Topological auto-placement node/rotation selection now lives in `mep_routing.placement.selection`.
- Core-like placement candidate selection now lives in `mep_routing.placement.selection`.
- Port stub segment construction now lives in `mep_routing.routing.segments`, with active graph nodes supplied by `main.py`.
- Route reconstruction from graph paths now lives in `mep_routing.routing.segments`, with active graph nodes and shaft-entry behavior supplied by `main.py`.
- Route clearance math, route axis records, and weighted edge-cost lookup now live in `mep_routing.routing.clearance`.
- Line-graph direction, path length, and target heuristic helpers now live in `mep_routing.routing.search`.
- Terminal edge blocking policy now lives in `mep_routing.routing.clearance`.
- Small-room routing order policy now lives in `mep_routing.routing.search`.
- Terminal node index collection now lives in `mep_routing.routing.search`, with terminals and KD-tree supplied by `main.py`.
- Heatmap and UI color-map helpers now live in `mep_routing.ui.colors`.
- Solution log KPI and auto-best comparison helpers now live in `mep_routing.ui.solution_logs`.
- Graph axis collection, epsilon expansion, boundary extension, and value merging helpers now live in `mep_routing.graphs.axes`.

## Next Steps

1. Continue import-safe value objects before moving behavior:
   - app/runtime state shell
2. Continue extracting pure geometry helpers.
3. Continue extracting pure routing helpers.
4. Extract graph builders behind a stable interface.
5. Extract routing backends.
6. Extract placement.
7. Extract UI drawing and event handling.

## Step Log

### 2026-07-10

- Created `demos/10.8.1` from current `demos/10.8`.
- Committed baseline: `0f8e0b2 Add demo 10.8.1 baseline`.
- Added refactor scope: `5e9f1d4 Scope demo 10.8.1 refactor`.
- Added initial config catalog: `f7cd803 Add demo 10.8.1 config catalog`.
- Added local workflow plan and project-local agent instructions.
- Updated taxonomy decisions from the design discussion: millimetre integers internally, no confirmed regulation values in current demo config, English config keys, and installation-scoped UI.
- Validation note: `quick_validate.py` could not run because the active Python environment does not have `yaml`/PyYAML. Manual skill-structure checks and Python compile checks were used instead.
- Extracted `EnvView` to `mep_routing.graphs.env` and updated `main.py` to import it.
- Added focused pytest coverage for `EnvView` reference preservation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\mep_routing\graphs\env.py demos\10.8.1\tests\conftest.py demos\10.8.1\tests\test_graph_env.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py`.
- Extracted `MachineSpec` to `mep_routing.domain.machines` and initially colocated the Sal Ozeo Flat catalog entry there.
- Kept old `main.py` machine constant names as compatibility aliases assigned from the machine spec.
- Moved route diameter and pin stub policies to the machine spec while preserving baseline behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\domain\__init__.py demos\10.8.1\mep_routing\domain\machines.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py demos\10.8.1\tests\test_machines.py`.
- Initially added canonical `SHAFT_ROUTE_NAME`, `KITCHEN_ROUTE_NAME`, and `LARGE_DUCT_ROUTE_NAMES` in `mep_routing.domain.routes`.
- Used `LARGE_DUCT_ROUTE_NAMES` for Sal machine large-duct policy and fixed-width route drawing.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\domain\__init__.py demos\10.8.1\mep_routing\domain\machines.py demos\10.8.1\mep_routing\domain\routes.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests\test_graph_env.py demos\10.8.1\tests\test_machines.py`.
- Extracted `snap_to_integer_grid` to `mep_routing.geometry.shapely_utils`.
- Added focused tests for polygon, line, and geometry collection snapping.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\shapely_utils.py demos\10.8.1\tests\test_shapely_utils.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted shared canvas toolbar geometry, slider conversion, slider drawing, and weight-view switch drawing to `mep_routing.ui.controls`.
- Kept live zoom, route-weight, and active-control state mutations in `main.py`.
- Added focused tests for toolbar bounds, switch bounds, slider clamping, and slider fractions.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\ui\controls.py demos\10.8.1\tests\test_ui_controls.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted generic allowed-region route validation to `mep_routing.routing.validation`.
- Kept the real-dwelling core-shaft metadata policy in `main.py` because it remains source-specific.
- Added focused tests for shaft-entry exemptions, outside-segment counts, and missing allowed-region handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\validation.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\tests\test_routing_validation.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted the pin min-cost-flow residual-network builder to `mep_routing.routing.flow`.
- Kept `main.py` responsible for active graph/KD-tree access, edge-weight overlays, runtime cost callbacks, and consuming solved flow paths.
- Added focused tests for a direct start-to-pin network and no-target handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\flow.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: Python 3.11 `-m pytest demos\10.8.1\tests`.
- Extracted `point_segment_min_distances`, `edge_segment_min_distances`, and `edge_parallel_segment_min_distances` to `mep_routing.geometry.distances`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for projection, endpoint, empty-segment, sampled-edge, and parallel-overlap distance behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\distances.py demos\10.8.1\tests\test_distances.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `extract_boundary_segments` and `extract_line_segments` to `mep_routing.geometry.segments`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for polygon boundaries, geometry collections, line strings, multi-line strings, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\segments.py demos\10.8.1\tests\test_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `cast_rays_numpy` and `ray_ray_intersections_numpy` to `mep_routing.geometry.rays`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for rectangular boundary ray casting, ray intersections, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\rays.py demos\10.8.1\tests\test_rays.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `normalize_axis_segment`, `axis_segment_relation`, and `axis_segment_distance` to `mep_routing.geometry.axis`.
- Kept `main.py` compatibility imports under the previous private helper names.
- Added focused tests for normalization, zero/diagonal rejection, overlap, crossing, endpoint touch, and separated distance.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\axis.py demos\10.8.1\tests\test_axis.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted `merged_axis_segments`, `merged_route_axis_segments`, `metric_route_segments`, and `point_is_segment_endpoint` to `mep_routing.routing.segments`.
- Left `_route_axis_records` in `main.py` for now because it still depends on the active route-diameter policy.
- Added focused tests for route segment merging, route-name preservation, non-axis metric segments, and endpoint detection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\segments.py demos\10.8.1\tests\test_routing_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route quality counters to `mep_routing.routing.metrics`: crossings, overlaps, clearance conflicts, turns, merged piece lengths, and short-piece counts.
- Kept `main.py` compatibility wrappers where current behavior depends on active Sal policies such as route diameter, clearance, and minimum piece factor.
- Added focused tests for metric crossing semantics, overlap counting, injected clearance policy, turn counting, and injected short-piece policy.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\metrics.py demos\10.8.1\tests\test_routing_metrics.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route scoring to `mep_routing.routing.scoring`: score weights, total length, quality counts, quality warnings, and conflict summary formatting.
- Kept `main.py` as a compatibility adapter for active slider weights and Sal-specific policies.
- Added focused tests for length totals, score composition, warning formatting, and baseline conflict-summary text.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\scoring.py demos\10.8.1\tests\test_routing_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route hit testing to `mep_routing.routing.hit_testing`.
- Kept `main.py` as the adapter for zoom-dependent hit radius.
- Added focused tests for nearest-route detection, misses outside radius, and route-name hit lookup.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\hit_testing.py demos\10.8.1\tests\test_routing_hit_testing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route length in metres to `mep_routing.routing.scoring.total_route_length_m`.
- Kept `main.py` KPI wrapper as an adapter.
- Added focused test for millimetre-to-metre conversion and empty routes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\scoring.py demos\10.8.1\tests\test_routing_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Created `mep_routing.ui` for pure UI utilities.
- Extracted Turbo/Viridis palettes, heatmap palette selection, score-to-heatmap normalization, and cool colormap to `mep_routing.ui.colors`.
- Kept `main.py` wrappers for active heatmap palette and scale mode.
- Added focused tests for color clamping, interpolation, palette selection, linear/log score scaling, and cool colormap behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\ui\__init__.py demos\10.8.1\mep_routing\ui\colors.py demos\10.8.1\tests\test_ui_colors.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted Hannan/Epsilon grid axis value merging to `mep_routing.graphs.axes.merge_close_values`.
- Kept `main.py` wrapper for existing grid builder call sites.
- Added focused tests for threshold merging, preserved axes, priority axes, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\mep_routing\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted graph axis collection helpers to `mep_routing.graphs.axes`: point axes, polygon vertex axes, and bounds axes.
- Kept `main.py` wrappers for current Hannan/Epsilon grid builder call sites.
- Added focused tests for rounded point axes, polygon exterior/interior vertices, and buffered bounds.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\mep_routing\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted polygon iteration to `mep_routing.geometry.polygons.iter_polygons`.
- Kept `main.py` wrapper for existing grid and drawing call sites.
- Added focused tests for polygons, multi-polygons, mixed geometry collections, `None`, and empty geometries.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\polygons.py demos\10.8.1\tests\test_polygons.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted largest-polygon selection to `mep_routing.geometry.polygons.largest_polygon`.
- Kept `main.py` wrapper for allowed-boundary axis extension.
- Added focused tests for largest-area selection and no-polygon inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\geometry\__init__.py demos\10.8.1\mep_routing\geometry\polygons.py demos\10.8.1\tests\test_polygons.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted min-cost-flow primitives to `mep_routing.routing.flow`: residual edge creation, successive-shortest-path min-cost flow, positive-flow edge lookup, and traced path reconstruction.
- Kept existing private helper names in `main.py` as imported aliases to avoid touching route-construction call sites.
- Added focused tests for lower-cost path choice, partial-flow behavior, traced state/target reconstruction, and incomplete trace handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\flow.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted min-cost-flow source normalization to `mep_routing.routing.flow.source_start_nodes`.
- Kept `main.py` as the adapter for the active `grid_kd`.
- Added focused tests for explicit node indices, empty sources, and KD-tree coordinate lookup.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\flow.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted machine pin geometry, port access specs, axis direction helpers, and outward-vector logic to `mep_routing.domain.machines`.
- Kept `main.py` wrappers for `get_machine_pins`, `get_port_access_specs`, and `get_outward_vector`.
- Added focused tests for unrotated Sal pin geometry, port access stubs, and outward direction by side/rotation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\domain\__init__.py demos\10.8.1\mep_routing\domain\machines.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted port stub segment construction to `mep_routing.routing.segments.add_port_stub_segment`.
- Kept `main.py` as the adapter for active `current_env.nodes`.
- Added focused tests for direct pin stubs, access-point bridge stubs, and no-op handling for missing pins or nodes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\segments.py demos\10.8.1\tests\test_routing_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted pure route clearance helpers to `mep_routing.routing.clearance`: buffered radius, required clearance, route axis records, and weighted edge-cost lookup.
- Kept `main.py` wrappers for current Sal buffer ratio and route-diameter policy.
- Added focused tests for radius rounding, clearance sums, axis-record extraction, and weighted edge-cost fallback behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\clearance.py demos\10.8.1\tests\test_routing_clearance.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted line-graph direction, path physical length, and target heuristic to `mep_routing.routing.search`.
- Kept `main.py` wrappers for active heuristic mode, machine center, and `estimate_turns`.
- Added focused tests for dominant-axis directions, Euclidean path length, Manhattan heuristic, bend-aware heuristic, machine-ring heuristic, disabled heuristic mode, and invalid nodes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\search.py demos\10.8.1\tests\test_routing_search.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted terminal-node index collection to `mep_routing.routing.search.terminal_node_indices`.
- Kept `main.py` wrapper signature unchanged even though the old helper does not use `pin_node_map`.
- Added focused test for KD-tree terminal lookup and shaft-node preservation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\search.py demos\10.8.1\tests\test_routing_search.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Manual launch found a startup crash after machine geometry extraction: placement code still referenced `_local_axis_to_world`.
- Restored `main.py` compatibility import for `local_axis_to_world` and exported it from `mep_routing.domain`.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\domain\__init__.py demos\10.8.1\mep_routing\domain\machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Startup smoke: running Demo 10.8.1 with the 10.7 venv stayed alive until timeout instead of exiting with a traceback.
- Extracted selected route pin detection to `mep_routing.routing.hit_testing.selected_pin_names`.
- Kept `main.py` wrapper unchanged for UI selection code.
- Added focused tests for selected route endpoint pins, last-three-segment matching, and empty context handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\hit_testing.py demos\10.8.1\tests\test_routing_hit_testing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted graph epsilon axis expansion and allowed-boundary interior axis extension to `mep_routing.graphs.axes`.
- Kept `main.py` wrappers for current Hannan/Epsilon grid builder call sites.
- Added focused tests for epsilon point expansion, geometry-driven epsilon axes, empty boundary handling, and clustered interior axes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\graphs\__init__.py demos\10.8.1\mep_routing\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted reusable Pygame drawing primitives to `mep_routing.ui.drawing`: geometry overlays, polygon hatch fill, dashed polylines, and outlined text.
- Kept `main.py` wrappers to inject current screen size, world-to-screen transform, and label halo color.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\ui\drawing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted terminal validity entry/reason classification to `mep_routing.routing.terminal_validity`.
- Kept `main.py` responsible for cache keys, room geometry accessors, active graph state, and UI drawing.
- Added focused tests for allowed nodes, clearance-blocked nodes, outside-room nodes, isolated nodes, no-boundary room behavior, and missing routing-region behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\terminal_validity.py demos\10.8.1\tests\test_terminal_validity.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Created `mep_routing.placement` and extracted topological auto-placement distance fields, score aggregation, and current weight presets.
- Kept `main.py` responsible for active KD-tree lookup, mode selection, and machine-position side effects.
- Added focused tests for Dijkstra fields, multi-source fields, weight presets, and weighted placement score aggregation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\fields.py demos\10.8.1\tests\test_placement_fields.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted core-like placement scoring primitives to `mep_routing.placement.scoring`: candidate room points, machine polygon construction, room area-out percentage, target angle, and candidate score tuple.
- Kept `main.py` responsible for Sal machine pins, active room/shaft/kitchen state, allowed-boundary distance, and final machine-position mutation.
- Added focused tests for candidate points, polygon/area scoring, signed target angle, and the core-like score tuple.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\scoring.py demos\10.8.1\tests\test_placement_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted rotation field room selection, weight presets, pin direction mapping, and rotation field score aggregation to `mep_routing.placement.rotation`.
- Kept `main.py` responsible for active machine pins, shaft target lookup, room target lookup, and final angle mutation.
- Added focused tests for large/small pin room ownership, weight modes, direction transform injection, and deterministic shaft alignment score.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\rotation.py demos\10.8.1\tests\test_placement_rotation.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted machine placement feasibility and candidate-room filtering to `mep_routing.placement.feasibility`.
- Kept `main.py` responsible for active Sal pins, routing region, obstacle collections, and machine dimensions.
- Added focused tests for covered-room preference, fallback room selection, routing-region containment, and wall/column/shaft obstacle rejection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\feasibility.py demos\10.8.1\tests\test_placement_feasibility.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted topological auto-placement node/rotation selection to `mep_routing.placement.selection`.
- Kept `main.py` responsible for active KD-tree lookup, machine pin generation, machine-position mutation, and grid rebuild.
- Added focused tests for pin-node lookup, rotation score policy, and first valid topological placement selection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\selection.py demos\10.8.1\tests\test_placement_selection.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted field-alignment rotation angle selection to `mep_routing.placement.rotation.select_field_alignment_rotation`.
- Kept `main.py` responsible for applying the selected angle and updating UI-visible rotation scores.
- Added focused tests for retaining the current orientation and switching orientation when the score exceeds epsilon.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\rotation.py demos\10.8.1\tests\test_placement_rotation.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted core-like placement candidate selection to `mep_routing.placement.selection.choose_core_like_machine_placement`.
- Removed dead placement compatibility wrappers from `main.py`.
- Added focused tests for no feasible core-like candidates and lowest-score feasible candidate selection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\placement\__init__.py demos\10.8.1\mep_routing\placement\selection.py demos\10.8.1\tests\test_placement_selection.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route reconstruction from graph paths to `mep_routing.routing.segments`: path-to-segments and route-order assembly.
- Kept `main.py` responsible for active environment nodes and shaft-entry segment policy.
- Added focused tests for shaft-entry segments, port stubs, total-node counts, and missing path/target handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\segments.py demos\10.8.1\tests\test_routing_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted small-pin min-cost-flow target-spec assembly to `mep_routing.routing.flow.small_pin_target_specs`.
- Reused shared route reconstruction when appending small MCF routes in `run_small_pin_min_cost_flow_routing`.
- Added focused test for small-pin target collection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\flow.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted terminal block edge normalization and terminal-node edge blocking to `mep_routing.routing.clearance`.
- Extracted small-room routing order policy to `mep_routing.routing.search.ordered_small_room_names`.
- Added focused tests for edge normalization/blocking and room filtering/sorting by machine distance.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\routing\__init__.py demos\10.8.1\mep_routing\routing\clearance.py demos\10.8.1\mep_routing\routing\search.py demos\10.8.1\tests\test_routing_clearance.py demos\10.8.1\tests\test_routing_search.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted solution log KPI calculation, metric lookup, and auto-best update selection to `mep_routing.ui.solution_logs`.
- Kept `main.py` responsible for snapshotting active app state, mutating log collections, and drawing the log panel.
- Added focused tests for KPI calculation, empty-route handling, metric lookup, improved best-log updates, and missing-history handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\ui\solution_logs.py demos\10.8.1\tests\test_solution_logs.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Reused the shared route reconstruction adapter in sequential routing for shaft, kitchen, and small-duct routes.
- This removes three local path-to-segment implementations while preserving sequential-routing order, weights, and terminal selection in `main.py`.
- Extracted solution-log widget drawing and hit testing to `mep_routing.ui.solution_logs`.
- Kept `main.py` responsible for active log state and restoring a selected snapshot.
- Kept Pygame imported only inside the drawing adapter so the widget's log-selection policy remains testable without Pygame.
- Added focused tests for manual-best values, visible-entry ordering, and button/row action dispatch.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\mep_routing\ui\solution_logs.py demos\10.8.1\tests\test_solution_logs.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.

- Extracted static wall/shaft distance fields, machine-envelope clearance weighting, and route-to-route interaction weighting to `mep_routing.routing.clearance`.
- Kept `main.py` as the adapter for active graph state, Sal dimensions/cost settings, and static-distance cache ownership.
- Added focused tests for static distance fields, wall/shaft blocking, machine hard/soft clearances, and overlap/crossing/clearance interaction costs.
- Validation: `python -m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\clearance.py tests\\test_routing_clearance.py`.
- Validation: `python -m pytest tests\\test_routing_clearance.py`.

- Extracted state-expanded A* and line-graph A*/GBFS super-sink searches to `mep_routing.routing.search`.
- Kept `main.py` responsible for backend selection, active heuristic mode, machine center, and the legacy routing-call signatures.
- Added focused route-selection tests proving both backends choose the lower-turn-cost directed pin path.
- Validation: `python -m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\search.py tests\\test_routing_search.py`.
- Validation: `python -m pytest tests\\test_routing_search.py`.

- Created `mep_routing.data_sources` and extracted real-dwelling wall derivation, obstacle subtraction, wall polygon construction, and initial machine-position selection.
- Kept `main.py` responsible for loading the external scenario, mutating active app state, and rebuilding the graph.
- Added focused tests for shared-boundary wall derivation, structural-obstacle subtraction, and wet-room-preferred initial machine selection.
- Validation: `python -m py_compile main.py mep_routing\\data_sources\\__init__.py mep_routing\\data_sources\\dwelling.py tests\\test_dwelling_data_sources.py`.
- Validation: `python -m pytest tests\\test_dwelling_data_sources.py`.

- Extracted synthetic dwelling generation and millimetre normalization to `mep_routing.data_sources.synthetic` behind an injected layout-provider interface.
- Kept `main.py` responsible for applying the generated scenario, resetting app caches, and rebuilding the interactive graph.
- Added a focused fake-provider contract test covering normalized rooms, doors, terminals, machine position, and routing region.
- Validation: `python -m py_compile main.py mep_routing\\data_sources\\__init__.py mep_routing\\data_sources\\synthetic.py tests\\test_synthetic_data_source.py`.
- Validation: `python -m pytest tests\\test_synthetic_data_source.py`.
- Synthetic smoke: Demo 10.8.1 generated an 8-room / 4-terminal scenario and rebuilt the regular and Hannan grids using the 10.7 virtual environment.

- Wired the existing Sal semantic config catalog into Demo 10.8.1 runtime defaults for graph resolution, clearances, solver penalties, UI slider bounds, paths, and input selection.
- Kept machine geometry owned by the Sal machine specification instead of duplicating it through config defaults.
- Added focused config tests for semantic/legacy key resolution and solver-versus-feasibility defaults.
- Validation: `python -m py_compile main.py mep_routing\\config\\__init__.py mep_routing\\config\\schema.py mep_routing\\config\\defaults_sal.py tests\\test_config.py`.
- Validation: `python -m pytest tests\\test_config.py`.
- Startup smoke: the 10.7 virtual environment imported `main.py` with the expected 200 mm grid and 4000 mm bend defaults.

- Extracted pure regular-grid heatmap interpolation and edge-weight logarithmic scale calculations to `mep_routing.ui.colors`.
- Kept `main.py` responsible for Pygame surfaces, active visualization state, and drawing.
- Added focused UI contracts for bilinear/fallback score interpolation and blocked-edge-excluding log scale calculation.
- Validation: `python -m py_compile main.py mep_routing\\ui\\__init__.py mep_routing\\ui\\colors.py tests\\test_ui_colors.py`.
- Validation: `python -m pytest tests\\test_ui_colors.py`.

- Extracted preferred-terminal node mapping and point/area selection mutations to `mep_routing.ui.terminal_selection`.
- Kept `main.py` responsible for room geometry, active graph state, and Pygame drawing.
- Added focused tests for deduplicated preference mapping and point/area selection transitions.
- Validation: `python -m py_compile main.py mep_routing\\ui\\terminal_selection.py tests\\test_ui_terminal_selection.py`.
- Validation: `python -m pytest tests\\test_ui_terminal_selection.py`.

- Extracted preferred-terminal area, preference-marker, and routed-endpoint drawing to `mep_routing.ui.terminal_selection`.
- Kept `main.py` responsible for active state, theme values, and world-to-screen transformation injection.
- Validation: `python -m py_compile main.py mep_routing\\ui\\terminal_selection.py`.

- Created `mep_routing.ui.terminal_validity` for terminal validity markers, overlays, and tooltips.
- Kept `main.py` responsible for cache lookup, KD-tree access, Pygame input, and viewport state.
- Validation: `python -m py_compile main.py mep_routing\\ui\\terminal_validity.py`.

- Extracted terminal candidate-node filtering, clearance, and nearest-first ordering to `mep_routing.routing.terminal_validity`.
- Kept `main.py` responsible for the active room/cover boundary collection and candidate cache.
- Added a focused candidate-node contract test.
- Validation: `python -m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\terminal_validity.py tests\\test_terminal_validity.py`.
- Validation: `python -m pytest tests\\test_terminal_validity.py`.

- Created `mep_routing.routing.terminal_regions` for room-cover intersections, terminal-valid-region construction, and boundary-segment assembly.
- Kept `main.py` responsible for app-state lookups and geometry cache ownership.
- Added focused geometry contracts for cover/routing constraints and combined boundary segments.
- Validation: `python -m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\terminal_regions.py tests\\test_terminal_regions.py`.
- Validation: `python -m pytest tests\\test_terminal_regions.py tests\\test_terminal_validity.py`.

- Extracted static wall/shaft clearance segment assembly and its geometry-sensitive cache key to `mep_routing.routing.clearance`.
- Kept `main.py` responsible for cache ownership and active scenario state.
- Added a focused contract for static constraints and cache-key geometry changes.
- Validation: `python -m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\clearance.py tests\\test_routing_clearance.py`.
- Validation: `python -m pytest tests\\test_routing_clearance.py`.

- Created `mep_routing.ui.heatmaps` for distance-field, edge-weight overlay, and legend drawing.
- Kept `main.py` responsible for visualization toggles, cache ownership, and coordinate/color adapters.
- Validation: `python -m py_compile main.py mep_routing\\ui\\heatmaps.py`.

- Created `mep_routing.ui.help` for contextual-help buttons/popups, transient messages, and the viewer legend.
- Kept `main.py` responsible for active help/status state and current theme/layout values.
- Validation: `python -m py_compile main.py mep_routing\\ui\\help.py`.

- Created `mep_routing.observability.history` for reusable history samples and buffer resets.
- Wired both buffer reset and route-history sampling through the new module.
- Extracted solution snapshot serialization and terminal-selection normalization to `mep_routing.observability.snapshots`.
- Added focused observability coverage.
- Validation: `python -m py_compile main.py mep_routing\\observability\\__init__.py mep_routing\\observability\\history.py tests\\test_history.py`.
- Validation: `python -m pytest tests\\test_history.py`.

- Created `mep_routing.ui.overlays` for wet-room accents and terminal-area drag rendering.
- Kept `main.py` responsible for active geometry, transforms, and colors.
- Validation: `python -m py_compile main.py mep_routing\\ui\\overlays.py`.

- Moved the contextual-help catalog into `mep_routing.ui.help` with the associated help renderers.
- Validation: `python -m py_compile main.py mep_routing\\ui\\help.py`.

- Created `mep_routing.ui.plots` and switched the live routing-history panel to its rendering adapter.
- Removed the previous plot body after compiling the live adapter; `ui.plots` is now the sole renderer.
- Validation: `python -m py_compile main.py mep_routing\\ui\\plots.py`.

- Integrated `RoutingHistory` and `SolutionLogSession` as the live owners of routing plot buffers, event markers, and manual/automatic solution logs.
- Removed the parallel history/log globals from `main.py`; the app now passes session-owned state to the existing plot and solution-log UI adapters.
- Validation: Demo 10.7 virtual environment `-m py_compile main.py mep_routing\\observability\\__init__.py mep_routing\\observability\\session.py`; ran both `test_observability_session.py` contracts directly because its environment does not include pytest.

- Created `mep_routing.graphs.runtime` for shaft runtime-node attachment, machine-pin direction filtering, graph adjacency/edge-coordinate assembly, and spatial-index creation.
- Simplified `main.py` graph commits and base regular-grid setup to assign the extracted runtime representation while retaining app-state lookup and cache invalidation there.
- Added focused runtime-graph contracts.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\graphs\\runtime.py mep_routing\\graphs\\__init__.py tests\\test_graph_runtime.py`; Python 3.11 `-m pytest tests\\test_graph_runtime.py`.

- Integrated Sal routing strategy dispatch into `main.py` through `mep_routing.installations.sal.orchestration`.
- Kept Pygame/app callbacks and negotiated-congestion execution in `main.py`; the extracted module now owns typed strategy selection, min-cost-flow dispatch, sequential room-order policy, and First Fit early exit.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile main.py mep_routing\\installations\\sal\\orchestration.py tests\\test_sal_orchestration.py`. Focused pytest could not run because that virtual environment does not include pytest; no Python 3.11 executable is installed in this workspace.

- Integrated Sal's negotiated-congestion routing behind `mep_routing.installations.sal.negotiated` and removed the legacy branch from `main.py`.
- Kept live graph state, elapsed-time measurement, and status formatting in `main.py`; the Sal module owns route-ordering, congestion history, pin selection, and candidate scoring through explicit callbacks.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile main.py mep_routing\\installations\\sal\\negotiated.py tests\\test_sal_negotiated.py`; direct focused negotiated-routing contract invocation passed. Focused pytest and Python 3.11 remain unavailable in this workspace.

- Integrated `SalFlowRuntime` as the live adapter for Sal min-cost-flow routing, leaving `main.py` with only the active-state factory and compatibility entrypoints.
- Validation: compile and focused flow-runtime contract test.

- Integrated shaft-entry geometry, entry-node ranking, and route-segment assembly through `mep_routing.routing.shaft_entries`.
- Kept `main.py` responsible for source-metadata normalization and the live geometry cache; the routing module now owns the reusable geometric policy.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\routing\\shaft_entries.py mep_routing\\routing\\__init__.py tests\\test_shaft_entries.py`; Python 3.11 `-m pytest tests\\test_shaft_entries.py tests\\test_routing_segments.py`.

- Integrated `mep_routing.ui.events` into the live keyboard routing-control cluster (`R`, `C`, `T`, `L`, `Y`, `Tab`, `A`, `P`, `U`, and `W`).
- Kept Pygame-side routing, graph rebuild, rotation application, and placement-field refresh in `main.py`; the UI module now owns transition and marker policy.
- Preserved the prior no-history behavior for auto-placement mode controls.
- Validation: `python -m py_compile main.py mep_routing\\ui\\events.py tests\\test_ui_events.py`; ran both UI event contracts directly because the available default Python environment does not include pytest.

- Integrated Hannan and epsilon variant assembly through `mep_routing.graphs.variants`, including live machine access points, required-node preservation, and variant timing diagnostics.
- Removed unused graph-axis forwarding wrappers from `main.py`; it now owns cache/state selection and graph-runtime commit only.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\graphs\\variants.py mep_routing\\graphs\\__init__.py tests\\test_graph_variants.py`; Python 3.11 `-m pytest tests\\test_graph_variants.py tests\\test_graph_axes.py tests\\test_graph_env.py`.

- Integrated the `ui.events` canvas-gesture and panel-interaction transitions into the live Pygame loop.
- Kept Pygame hit testing, cursor changes, slider values, terminal callbacks, solution-log restoration, routing solves, and history writes in `main.py`; `ui.events` now owns pointer/control transition ordering and transient drag state.
- Preserved the original slider priority, control-panel priority, and no-history auto-placement behavior.
- Validation: `python -m py_compile main.py mep_routing\\ui\\events.py tests\\test_ui_events.py`; six focused UI event contracts invoked directly because pytest is unavailable in the active Python environment.

- Integrated prepared synthetic and real dwelling state through `mep_routing.data_sources.scenario`.
- `main.py` now owns source selection, fallback messaging, cache resets, graph rebuild, and auto-placement while the data-source module owns scenario normalization, accent generation, and real-scenario preparation.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\data_sources\\scenario.py mep_routing\\data_sources\\__init__.py tests\\test_scenario_preparation.py`; Python 3.11 `-m pytest tests\\test_scenario_preparation.py tests\\test_synthetic_data_source.py tests\\test_dwelling_data_sources.py`.

- Integrated the Sal routing controller as the live solver entrypoint. `main.py` now provides preflight, active-graph, and live-callback adapters while the controller owns the strategy dispatch and result/status policy.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\installations\\sal\\controller.py mep_routing\\installations\\sal\\__init__.py tests\\test_sal_controller.py`; Python 3.11 `-m pytest tests\\test_sal_controller.py tests\\test_sal_orchestration.py tests\\test_sal_negotiated.py` (7 passed).

- Integrated core-like and topological machine-placement coordination through `mep_routing.placement.runtime`.
- Kept `main.py` responsible for live state commits, graph rebuilding, and status output; the placement module now owns selection and score-field coordination through explicit callbacks.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile main.py mep_routing\\placement\\runtime.py tests\\test_placement_runtime.py`; direct focused placement-runtime contracts invoked because pytest is unavailable in that environment.

- Integrated terminal candidate/cache and preferred-start orchestration through `mep_routing.routing.terminal_runtime`.
- Kept `main.py` responsible for scenario construction, graph commits, terminal rendering, and snapshot calls; the routing runtime now owns terminal geometry caches, candidate selection, preference state, and explicit invalidation.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile main.py mep_routing\\routing\\__init__.py mep_routing\\routing\\terminal_runtime.py` plus 13 direct terminal contracts from `test_terminal_runtime.py`, `test_terminal_regions.py`, `test_terminal_validity.py`, and `test_ui_terminal_selection.py`.

- Integrated routing edge-weight assembly through `mep_routing.routing.weight_runtime`.
- Kept `main.py` responsible for live graph/geometry context, the existing display globals, and terminal-block exclusions; the routing runtime now owns static distance caching, clearance and interaction composition, and overlay calculation.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\routing\\weight_runtime.py mep_routing\\routing\\__init__.py tests\\test_weight_runtime.py`; Python 3.11 `-m pytest tests\\test_weight_runtime.py tests\\test_routing_clearance.py` (14 passed).

- Integrated the five-card dashboard sidebar through `mep_routing.ui.sidebar`.
- Kept `main.py` responsible for current display-value derivation, help-button state, and slider callbacks; the UI module now owns the fixed Pygame card layout and text rendering.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\ui\\sidebar.py tests\\test_ui_sidebar.py`; Python 3.11 `-m pytest tests\\test_ui_sidebar.py` (2 passed).

- Integrated the central-plan render pass through `mep_routing.ui.canvas`.
- Kept `main.py` responsible for selection, scene preparation, heatmap/guide decisions, and toolbar/ruler rendering; the UI module now owns the stable drawing order for canvas primitives and overlays.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\ui\\canvas.py`.

- Integrated regular, Hannan, epsilon, and dynamic-obstacle graph lifecycle coordination through `mep_routing.graphs.lifecycle`.
- Kept `main.py` responsible for live graph assignment, weight-runtime cache resets, terminal-runtime updates, and status output; the graph module now owns graph-mode construction, pin/shaft runtime assembly, and Hannan-template caching.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile main.py mep_routing\\graphs\\__init__.py mep_routing\\graphs\\lifecycle.py` plus 24 direct focused graph contracts from `test_graph_runtime.py`, `test_graph_variants.py`, `test_graph_env.py`, `test_graph_axes.py`, and `test_graph_lifecycle.py`.

- Restored missing startup defaults for the selected graph and grid visibility, restored the shared polygon iterator adapter, and routed canvas toolbar drawing through `mep_routing.ui.canvas_tools`.
- Validation: Demo 10.7 Python 3.12 virtual environment kept the interactive application running through a 19-second startup smoke without an exception.

- Added a compact `SalInstallationDefinition` and integrated it as the live owner of Sal's current machine catalog, route diameter classification, supported graph modes, supported solver strategies, and default selections.
- Removed the duplicated machine, graph-label, strategy-label, default-index, and large-route policy declarations from `main.py` while preserving the existing UI labels and selector indices.
- Validation: Python 3.11 `-m py_compile main.py mep_routing\\installations\\sal\\__init__.py mep_routing\\installations\\sal\\definition.py tests\\test_sal_definition.py`; Python 3.11 `-m pytest tests\\test_sal_definition.py tests\\test_machines.py tests\\test_sal_orchestration.py tests\\test_sal_controller.py` (15 passed).
- Startup smoke: Demo 10.8.1 remained running after 8 seconds with definition-backed startup defaults.

- Moved Sal's search-backend and heuristic selections into `SAL_INSTALLATION` and routed the live solver through the shared `SearchBackend` dispatcher.
- Removed the parallel State A*, Line-Graph A*, and Line-Graph GBFS wrappers from `main.py`; the compatibility entrypoint still owns weight-overlay recording and supplies live heuristic inputs.
- Validation: Demo 10.7 Python 3.12 virtual environment `-m py_compile` for the application, Sal definition, routing exports, and focused tests; eight direct backend/Sal/controller contracts passed because pytest is unavailable; the interactive app remained alive through a 10-second startup smoke.

- Integrated one immutable `SalSolverPolicy` snapshot through the live controller, sequential, flow, negotiated-congestion, search, edge-weight, and scoring paths.
- Removed parallel derived solver-penalty globals; the policy now owns their existing formulas while the UI retains only its mutable bend, crossing-multiplier, minimum-piece, and heuristic controls.
- Moved negotiated iteration, present/history congestion, and large-route favouring constants into explicit policy values.
- Validation: Python 3.11 compilation and 42 focused Sal, weight, clearance, search, and scoring contracts passed. Demo 10.7 Python 3.12 virtual environment remained running after an 8-second startup/initial-solve smoke.

- Moved negotiated-congestion iteration, congestion history, route rebuilding, scoring, and early-exit behavior into installation-neutral `mep_routing.routing.negotiated`.
- Reduced `installations.sal.negotiated` to Sal route-start, port-eligibility, terminal-node, and large-route preference adaptation using `SalRoutePlan` and `SalSolverPolicy`.
- Preserved Sal's public negotiated context/result names, deterministic port order, missing-Kitchen failure, and live controller/main signatures.
- Validation: Python 3.11 compilation; focused neutral negotiated, Sal negotiated/controller, route-plan, and policy contracts; headless startup smoke.

- Integrated `SalPreparedRoutingProblem` as the single post-shaft handoff shared by sequential, flow, and negotiated solver dispatch.
- Removed the duplicate `SalFlowRoutingRequest`; controller-facing callbacks now consume the prepared problem plus only a room order or large-route preference where required, while lower-level algorithms retain their established signatures.
- Added thin prepared-problem entrypoints to `SalFlowRuntime` and kept graph/search callbacks in the existing runtime adapters.
- Validation: Demo 10.7 Python 3.12 virtual environment compiled the application, prepared/controller/orchestration/flow modules, and focused tests; direct controller/flow contracts passed. Pytest remains unavailable in that environment. Startup and initial solve remained running after 8 seconds.

## Commit Checklist

- Integrated `SalStrategyRuntime` and `solve_prepared_strategy` as the single post-preparation strategy boundary used by the live Sal controller.
- Replaced seven solver callback fields on `SalRoutingControllerContext` with one runtime object and removed duplicate flow, negotiated, and sequential branching from the controller.
- Removed orchestration dispatch/category helpers made obsolete by the strategy dispatcher while retaining deterministic sequential-order policy helpers.
- Validation: focused Sal strategy/controller/orchestration compilation and contracts plus a headless interactive startup smoke.

- Integrated `SalRoutePlan` as the single live owner of Sal shaft, kitchen, wet-room, and machine-port topology.
- The Sal controller now builds one plan per solve and threads it through sequential, negotiated-congestion, and min-cost-flow execution; shared routing helpers receive the plan's shaft and port names explicitly.
- Preserved existing small-room keyword filtering, distance ordering, large/small port tie-break order, and each strategy's missing-Kitchen behavior.
- Validation: Python 3.11 `-m py_compile` on `main.py` and the changed Sal modules; Python 3.11 focused Sal/shared routing suite (50 passed); startup smoke completed without an exception.

- Integrated Sal min-cost-flow execution through the shared `RoutingProblem` / `SolverResult` pin-flow solver while preserving the existing tuple-return adapter for current strategies.
- Exported the shared routing request/result and pin-flow payload contracts; Sal still owns terminal-to-node preparation, weight-overlay recording, and legacy incomplete-input outcomes.
- Validation: Demo 10.7 Python 3.12 virtual environment compilation plus 11 shared routing-flow and 3 Sal flow-runtime contracts invoked directly because pytest is unavailable; the interactive app remained alive through a 10-second startup smoke.

- Removed unused Sal flow compatibility wrappers left behind after prepared-strategy integration and added persistent traceback capture for uncaught interactive crashes.
- Validation: all four real dwellings loaded and routed; six dwelling-cycle events including wraparound and eight source-mode toggles completed in the headless Pygame loop without an exception.

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
