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

- Ported routing-core's sparse simple orthogonal grid into the installation-neutral graph subsystem with explicit required points, obstacles, axis policy, and blocked-edge cost inputs.
- Added a fourth selectable Sal graph mode backed by a Sal-owned input adapter for terminal/access nodes and column, shaft, and machine obstacles; retained Hannan as the default and did not introduce Steiner behavior into Sal.
- Replaced catch-all graph selection with an explicit `GraphMode` contract so unsupported integer modes fail instead of silently selecting epsilon.
- Validation: Python compilation and the full 300-test Demo 10.8.1 suite.

- Added an installation-neutral `RoutingWorkspace` that owns the selected graph, dynamic environment, base placement graph, spatial index, static-clearance cache, and edge-weight overlay for one dwelling.
- Removed the parallel graph/cache/overlay globals from `main.py`; dwelling replacement now atomically replaces the workspace before rebuilding placement and selected graphs.
- Preserved the unfiltered committed graph separately from the dynamically filtered machine-obstacle environment and synchronized `TerminalRuntime` after both transitions.
- Validation: compilation, eight focused workspace/lifecycle/Sal-application contracts, and a 12-second headless startup/initial-solve smoke.

- Added a concrete `SalApplicationAdapter` as the live boundary between interactive graph/session hooks and Sal policy, route-plan, flow-runtime, strategy-runtime, and controller assembly.
- Removed direct Sal controller and flow wiring from `main.py`; the UI-facing solve wrapper now only resets overlays, invokes the adapter, and applies its result.
- Kept graph lifecycle, routing weight runtime, scoring, and UI state in `main.py` for the next workspace/session boundary rather than introducing a generic installation framework.
- Validation: compilation, two application-boundary contracts, six existing controller/flow contracts, and a 12-second headless startup/initial-solve smoke.

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

- Integrated prepared sequential and negotiated execution through the existing `SalFlowRuntime`, including its terminal-node and blocking adapters.
- Removed the duplicate sequential wrapper, negotiated-context factory, and controller-local strategy closures from `main.py`; no additional runtime abstraction was introduced.
- Validation: focused Sal runtime/strategy/controller contracts plus dwelling-cycle and startup smoke.

- Integrated the shared `RoutingRuntime` as the live owner of edge-clearance weights, overlays, and backend search dispatch.
- Removed the duplicate weight/search adapters from `main.py`; Sal flow and shaft callbacks now construct their runtime only after graph refresh, preventing a solve from retaining the previous dwelling's graph.
- Reset routing overlays and static-clearance cache when applying a dwelling and at solve start so cycled dwellings cannot retain stale edge identifiers.
- Validation: Python compilation; 24 focused routing-runtime, Sal flow/controller/strategy tests; six solve events and eight dwelling/source-cycle events in one headless Pygame process; five-second startup smoke.

- Extracted machine-placement orchestration into `PlacementApplicationAdapter`: topological/core dispatch, candidate scoring, rotation-field scoring, and timing now run without mutating the interactive application.
- Kept graph rebuilds and UI state application in `main.py`; installation-specific machine dimensions continue to enter through the selected `MachineSpec` without adding a speculative placement framework.
- Validation: Python compilation, 24 focused placement contracts, and a 12-second headless startup smoke through the default core-like placement path.

- Added `MachineRoutingSession` as the live boundary between a selected machine and the routing workspace, owning preflight geometry, graph refresh behavior, and pin-access snapping.
- Bound route diameters to the selected `MACHINE_SPEC` rather than Sal's default machine so alternate machine choices cannot mix geometry and duct sizes.
- Validation: Python compilation, 11 focused machine-session and Sal application/runtime contracts, and a 12-second headless startup solve.

- Extracted Sal's live weight/search runtime composition into `SalLiveRoutingSession`, keeping one policy snapshot aligned with the selected machine, graph backend, cache, and overlay.
- Validation: Python compilation, 12 focused machine/Sal contracts, and a 12-second headless startup solve.

- Moved Sal application-hook and adapter assembly behind the existing live session; `main.py` now supplies only a compact bundle of demo-specific route callbacks.
- Validation: Python compilation, 13 focused machine/Sal contracts, and a 12-second headless startup solve.

- Added vertical machine-clearance feasibility from exported cover voids, using the Ozeo family's 197.504 mm `DimensionZ` plus Sal's 20 mm slab allowance.
- Placement scoring remains unchanged in shallow covered zones, while machine candidates and routing preflight reject positive-area overlap; the placement heatmap marks those zones with a gray hatch.
- Heatmap rendering now fills excluded zones opaque gray with dashed hatching and normalizes its palette/colorbar only across placeable graph nodes.
- Aligned the active Ozeo plan and connector diameters with the AzureFile family JSON (459 x 512 x 197.504 mm, 80/125 mm ducts).
- Validation: Python compilation, focused machine/placement/Sal contracts, one real shallow-cover extraction check, and headless startup solves.

- Aligned both automatic-placement modes with machine-only vertical-clearance exclusions and made no-feasible placement outcomes explicit.
- Tightened shared placement feasibility so the full machine footprint must remain inside the routing envelope, fit within one room, and avoid wall centerlines and buffered wall geometry.
- Validation: Python compilation and 24 focused placement contracts.

- Extracted `SalRouteAnalysis` as the installation-owned route-quality boundary for selected-machine diameters, clearance, minimum-piece policy, scoring, warnings, and conflict summaries.
- The live Sal session now produces the analysis snapshot used by solver callbacks and UI compatibility functions; shared routing metrics remain installation-neutral.
- Validation: Python compilation and 20 focused Sal policy/application/live-session and routing metric/scoring contracts.

- Extracted `SalRouteMaterializer` as the live path-to-geometry boundary for source-node lookup, shaft-entry segments, machine-pin stubs, and multi-route assembly.
- `main.py` now supplies the current graph nodes, spatial index, and shaft-entry callback without owning Sal route-materialization policy.
- Validation: Python compilation and 28 focused Sal materializer/application/session/strategy and shared routing-segment contracts.

- Added `SalInteractiveSolver` as the installation-owned interactive solve facade, including live machine-session construction, callback adaptation, overlay lifecycle, and result application.
- Removed the obsolete source-node and path-materialization wrappers from `main.py`; the entrypoint now supplies one explicit live callback snapshot and consumes the solver result.
- Validation: Python compilation and 14 focused Sal interactive/live-session/application/materialization/analysis and machine-session contracts.

- Added `CanvasUiRuntime` as the live owner of viewport transforms, responsive canvas layout, zoom/pan state, canvas control hit rectangles, and distance/edge heatmap caches and rendering.
- Kept routing choices and domain toggles in `main.py`; the runtime delegates to the existing focused Pygame renderers rather than duplicating drawing behavior.
- Validation: Python compilation, six focused viewport/frame contracts, and a 16-second headless startup/initial-solve smoke without an exception.

- Added `WorkbenchEventAdapter` as the Pygame boundary for raw event classification, hit testing, installation-header dragging, transient gesture state, and mouse/keyboard command execution.
- Reduced `main.py` to the outer loop, domain composition, and rendering; each frame now synchronizes one compact event-state snapshot from the adapter.
- Validation: Python compilation, focused UI transition/event-adapter contracts, and a headless startup/initial-solve smoke.

- Made shared allowed-region validation accept blocked solver results with `routes=None`, so dwellings with no feasible route remain visible with their blocked execution status instead of crashing during frame rendering.
- Validation: Python compilation and focused routing-validation/Sal route-analysis contracts.

- Completed the Sal solver-composition boundary: `main.py` retains only live settings/session snapshots, low-level dwelling/graph callbacks, UI metrics, and the single `solve_ventilation_routing` entrypoint.
- Made Sal route materialization lazy so paths always use the graph produced by the solver's machine-aware refresh rather than a pre-solve graph snapshot.
- Removed residual flow/search imports and dead solver compatibility wrappers from `main.py`; strategy dispatch, flow algorithms, route planning, scoring policy, materialization, machine routing, and overlay lifecycle now remain under shared routing or `installations.sal`.
- Deliberately rejected a large application-state DTO: the two small live factories are clearer until a concrete second installation reveals the shared runtime contract.
- Validation: Python compilation, 114 focused Sal/shared-routing contracts, and a headless startup/initial-solve smoke that remained alive without an exception.

- Added an installation routing-order header above the canvas with separate state for routing participation, active UI installation, and execution order.
- Sal is the only available runtime today; Cli, San, and Coc are visible reorderable placeholders but cannot be enabled or selected until their runtime boundaries exist.
- Switch clicks toggle participation, pill clicks select the active installation, and pill drags reorder future multi-installation execution. Disabling Sal now suppresses routing without discarding its active UI context.
- Reserved a responsive header band above the plan so installation controls do not cover dwelling geometry or existing canvas tools.
- Validation: Python compilation, 11 focused installation-header/Sal/UI-event contracts, and an eight-second headless startup/initial-solve smoke.

- Added `ActiveDwellingSession` as the shared application boundary pairing one immutable `PreparedDwelling` with fresh `RoutingWorkspace` and `TerminalRuntime` instances.
- Sal live solving, route analysis, materialization, and callback construction now consume the active dwelling session rather than assembling their own geometry/runtime view from parallel globals.
- Existing UI readers temporarily receive aliases from the session; this keeps the migration reviewable while allowing Cli to start from one coherent dwelling contract.
- Validation: Python compilation, 12 focused dwelling-session/scenario/workspace/Sal/machine-session contracts, and a headless startup/initial-solve smoke.

Before each commit:

- `git status --short`
- inspect staged files
- compile changed Python files
- update this file if the step changes the refactor plan or status
- commit only related files

## Open Items

- Added installation-owned UI capability and projection snapshots derived from the runtime registry and keyed `WorkbenchState` results.
- Canvas projection keeps every enabled solved installation visible in routing order while only the active projection exposes effective editing capabilities; matching event and terminal-tool guards no longer identify Sal/Cli directly.
- Validation: Python compilation and focused installation-projection contracts.

- Added `DemoUiFacade` as the compatibility owner of installation-header layout, machine selection, terminal/canvas tools, slider drawing, ruler behavior, UI cache invalidation, heatmap rendering, and route-width projection.
- `main.py` retains the existing workbench callback names as bound facade methods and reached 542 physical lines.
- Validation: Python compilation, focused UI-facade contracts, and a generated-dwelling solve.

- Added `DwellingRuntime` as the owner of synthetic/real source preparation and fallback, active-session publication, terminal/placement binding order, Cli initial placement, clearance derivation, and initial graph construction. `main.py` retains four event-adapter aliases.
- Validation: Python compilation, `git diff --check`, and a generated synthetic dwelling Sal solve (8 rooms, 4 terminals, 5 routes). `main.py` is 686 physical lines after this boundary.

- Added `InteractiveObservationController` as the compatibility boundary for KPI capture, history/log updates, reproducible UI-state snapshots, snapshot restoration, and observation chrome callbacks.
- Plot drawing now belongs to `WorkbenchChromeRuntime`; `main.py` preserves the workbench callback surface through bound controller methods and reached 791 physical lines.
- Validation: Python compilation and a focused observation snapshot round-trip contract.

- Added `DemoRoutingController` as the cohesive owner of Sal graph construction/dynamic obstacles, solver settings and live sessions, route analysis/materialization, placement transitions, interactive solving, and ordered Sal/Cli execution. `main.py` retains bound callback names for the existing event/UI adapters.
- Validation: Python compilation, `git diff --check`, controller import, and a generated-dwelling initial Sal solve. `main.py` is 899 physical lines after this boundary.

- Extracted Pygame initialization, font construction, event-loop ownership, canvas/context projection, sidebar rendering, placement explorer rendering, and frame timing into `ui.workbench_loop`; `main.py` now delegates application execution through one injected module facade.
- Validation: Python compilation, `git diff --check`, and an eight-second headless startup/initial-routing smoke. `main.py` is 1,092 physical lines after this boundary.

- Replaced the temporary Cli grille heuristic with the concrete Demo `10.8_cli` placement policy: 1000/1050 mm supply/return sizes, eligible wall extraction, wall margins and overlap checks, same-wall and different-wall alternatives, machine-aligned supply candidates, partial-false-ceiling detection, and horizontal returns.
- Grille placement now receives the selected Cli machine pose explicitly. Cli placement scores supply and return distances from their respective machine connectors, and Cli's core-simple graph adds a 600 mm scaffold so valid legacy grille connection points remain connected on irregular routing regions.
- Validation: Python compilation, seven focused direct Cli grille/placement/runtime contracts, partial-false-ceiling horizontal-return coverage, and a generated-dwelling supply/return solve using the auto-placed machine.

- Extracted ordered multi-installation execution from `main.py`: capability-specific callbacks remain concrete, while a small coordinator owns enabled/order traversal, prior-result obstacle projection, active-result selection, and result storage.
- Extracted non-active canvas layer projection into a pure UI module. `main.py` now supplies installation visuals instead of assembling `CanvasContextLayer` instances inline and is 1,179 physical lines.
- Validation: Python compilation and two direct focused execution-order/context and active/context projection contracts.
- Final smoke correction: Cli grille routing points now land 300 mm inside the routable region, avoiding isolated boundary nodes on synthetic dwellings. If prior 2D route obstacles disconnect the entire Cli supply graph, Cli explicitly reports a context-relaxation warning and retains both installations' visible results rather than erasing the later run.
- Final validation: 12-second headless startup/initial-solve smoke, direct supply/return solve on a generated dwelling, and dual-installation execution in both routing orders.

- Cli now solves and materializes separate supply and return-air trees with connector/grille dimensions, machine stubs, and explicit duct segment roles.
- Enabled installations execute in header order and receive earlier machine footprints and buffered routes as contextual obstacles; Cli placement also follows the selected PEAD footprint on dwelling and machine changes.
- Validation: Python compilation and five direct focused Cli materialization, placement, grille, runtime-context, and installation-context contracts (the available project environments do not contain pytest).

- Added Cli-owned grille preparation with paired supply/return wall grilles, routing-region connection points, dimensions, roles, and room identity.
- Added Cli-owned PEAD placement using the selected machine footprint, routing-region feasibility, fixed obstacles, contextual obstacles, and grille-distance scoring; it does not consume Sal placement fields or weights.
- Validation: Python compilation and five focused Cli grille, placement, geometry, terminal, and runtime contracts.

- Split canvas projection into one active installation layer plus subdued context layers for every other enabled installation. Each layer contributes its own routes, terminals, and machine outline; the active layer alone contributes selection, connector emphasis, placement guides, and editing interactions.
- Added independent Sal and Cli machine poses. Mouse drag, wheel rotation, and Cli `R` rotation now mutate only the active machine; Sal-only sliders and routing commands are rejected while Cli is active instead of changing hidden context state.
- Validation: Python compilation, full Demo suite (330 passed), and a timed Pygame smoke that enabled Cli beside Sal, rendered both runs, activated Cli, and exited cleanly.

- Fixed the first live Cli activation crash by normalizing Cli routes to the canvas sequence contract and making machine-pin rendering accept installation-specific connector sets instead of Sal's six fixed pins.
- Cli canvas projection now disables Sal placement guides while retaining the Cli body and air/freon connector markers.
- Validation: Python compilation, full Demo suite (330 passed), and a timed headless Pygame activation/render smoke that selected Cli and exited cleanly.

- Added the first live Cli vertical slice without adapting it to Sal's interfaces: Cli owns its selected PEAD geometry, supply connector access, occupied-room terminal preparation, core-simple graph inputs, and Kou Steiner solve runtime.
- Cli can now be enabled, activated, reordered, rerun after machine selection, and projected onto the canvas/sidebar independently of Sal; the shared application loop only sequences enabled installation runtimes and selects the active result.
- Cli supply terminals currently represent accessible routing-region connection points near eligible rooms. Detailed grille placement, return-air routing, Cli-specific machine placement, duct sizing/materialization, scoring, and cross-installation collision feedback remain separate future capabilities.
- Validation: Python compilation, 14 focused Cli/catalog/header/Pygame-event contracts, the full Demo 10.8.1 suite (329 passed), and a real exported dwelling Cli solve (two rooms, six Steiner edges).

- Added `WorkbenchCommandController` as the UI application boundary for routing keyboard transitions, canvas toolbar actions, and solver-control slider/reset effects.
- The Pygame event adapter retains the same module callback names through bound controller methods while `main.py` no longer owns command decoding or transition side effects.
- Validation: Python compilation and focused workbench-command contracts; `main.py` reached 1,195 physical lines.

- Added `PlacementSession` as the bound owner of active-dwelling placement feasibility, vertical-clearance exclusions, weight policy, cached topological fields, automatic placement, and field-alignment rotation.
- `main.py` now applies placement outcomes to live machine coordinates while the reusable session owns calculation state; selected-machine and dwelling inputs remain explicit at the binding seam.
- Validation: Python compilation and focused placement-session contracts.

- Added `RoutingObservationRuntime` as the application-facing owner of routing history, snapshots, manual logs, and automatic-best markers; `main.py` now only supplies the live UI state mapping and restore application.
- Added `WorkbenchChromeRuntime` for solution-log hit state, help controls, transient messages, and viewer legend rendering, and routed the Pygame event adapter through that state owner.
- Validation: Python compilation, focused observation/chrome contracts, and the Demo 10.8.1 test suite.

- Added `TerminalUiRuntime` as the bound adapter for terminal interaction, preferred-terminal rendering, room/shaft hit testing, and validity-cache ownership.
- Removed terminal geometry and drawing pass-through wrappers from `main.py`; the entrypoint now binds the active dwelling once and supplies the runtime directly to canvas hooks.
- Validation: Python compilation and focused terminal-runtime contracts.

- Extracted dwelling source-frame normalization, routing-core shaft-entry specification parsing, and per-dwelling shaft-entry caches from `main.py`.
- Added installation-scoped graph profiles plus shared graph/terminal runtime configuration, keeping Sal's port and simple-graph callbacks explicit for future Cli composition.
- Added `GraphBuildSession` as the application facade for base, selected, and dynamic graph construction; `main.py` now retains only live machine inputs and compatibility aliases.
- Centralized fresh `ActiveDwellingSession` construction and real-dwelling loader adaptation without moving UI fallback policy or compatibility-global mutation into reusable modules.
- Validation: Python compilation, four focused source-frame/graph-session contracts, and the full Demo 10.8.1 suite (315 passed).

- Added `InstallationUiSession` as the application-owned machine-selection boundary, with selected keys and dropdown state scoped per installation plus an explicit runtime-invalidation result for future interactive runtimes.
- Cli is now activatable for configuration and exposes all seven PEAD machines in the sidebar while its routing switch remains unavailable; selecting a Cli machine persists across Sal/Cli switches, does not reroute, and reports that the interactive runtime is pending.
- Sidebar projection no longer presents Sal placement, frame, or rotation as Cli state, and Pygame dropdown hits use the same bounds contract as rendering.
- Validation: Python compilation and focused installation-session/header/dropdown/sidebar/frame/Pygame-event contracts.

- Added an immutable shared catalog contract and explicit Sal/Cli machine catalogs. Sal exposes its existing Ozeo machine as a one-entry, non-selectable catalog; Cli exposes the seven connector-rich Mitsubishi PEAD definitions from Demo 10.8_cli without adapting them to Sal's symmetric `MachineSpec` geometry.
- Added pure shared dropdown state, transition, and layout contracts plus a thin Pygame renderer. The sidebar accepts machine selector data but suppresses the control for Sal's single option; Cli routing and installation activation remain disabled.
- Added a minimal Cli installation definition that composes its PEAD catalog, core-simple supply graph mode, and shared Steiner solver while explicitly declaring that no interactive Cli runtime is available yet.
- Validation: Python compilation and focused machine-catalog/dropdown/sidebar contracts.

- Added an installation-neutral Kou-style multi-terminal Steiner solver under shared routing, with immutable results, input validation, deterministic traversal, and no module-global graph state. It is exported for future Cli use but deliberately has no Sal integration.
- Validation: Python compilation and focused shared Steiner routing contracts.

- The redundant routing KPI sidebar card was removed; live plot values now share each plot header, while crossing counts remain execution warnings.
- The active dwelling moved from machine placement into a canvas dropdown below the zoom controls, allowing direct selection of any configured real dwelling or a new synthetic dwelling.
- Plot headers now format current values with metric-specific units and precision; the duplicated edge-weight row was removed from the solver card.
- Plot text uses one Arial-compatible family: 16 px bold titles, 14 px bold current values, 13 px minimum labels, and 11 px annotations, all with lining figures.
- Real dwelling choices are discovered from every row in the configured dwellings-export SQLite catalog; selection retains project identity and no longer depends on a hardcoded case list in `main.py`.

- Added a keyed `WorkbenchState` with independent Sal/Cli machine poses and results plus one dwelling-session owner. Removed the corresponding flat globals and migrated routing, activation, observation, commands, events, and canvas rendering to the state boundary while keeping machine selection in `InstallationUiSession`.
- Validation: Python compilation and seven focused state, command, observation, and UI-facade contracts.

- Added a narrow installation runtime lifecycle with concrete Sal and Cli adapters. The shared contract covers only solve context, machine pins, and declared interaction capabilities; each installation retains its own graph, placement, and solver implementation.
- Added a registry/coordinator for ordered context-aware execution and keyed `WorkbenchState` result publication. `DemoRoutingController` no longer assembles per-run solver/pin lambdas or owns result replacement.
- Validation: Python compilation and focused installation coordinator order, contextual obstacle, stale-result, and active-result contracts.

- Added immutable `DemoConfig` and `WorkbenchTheme` composition objects. Stable Sal defaults, graph/terminal construction settings, mode vocabularies, filesystem inputs, panel dimensions, and the complete canvas/sidebar palette no longer live as constant blocks in `main.py`.
- Routing, dwelling activation, commands, observation, event handling, and rendering now consume the typed objects directly; mutable solver choices remain live workbench state rather than being mislabeled as configuration.
- Validation: Python compilation, nine focused config/theme and affected-controller contracts, and an eight-second headless startup/initial-routing smoke.

- Added `DemoWorkbenchApplication` as the concrete surface consumed by the event adapter and render loop. Routing, dwelling, command, UI, observation, and chrome controllers are named boundaries instead of dozens of callbacks flattened onto `main.py`.
- Removed the compatibility aliases for graph building, machine pins, solving, dwelling activation, placement transitions, drawing delegates, and observation callbacks. Remaining composition data is temporarily proxied through one explicit migration seam.
- Validation: Python compilation, the full 347-contract Demo suite, and an eight-second headless startup/initial-routing smoke.

- Exported the application, routing, dwelling, command, facade, and chrome controllers through their package boundaries; `main.py` now imports composition services from those public surfaces.
- Removed obsolete imports and corrupted separator comments, reducing `main.py` to 230 lines, and added `main.smoke_test()` backed by the typed `run_workbench_smoke()` application entrypoint.
- Validation: Python compilation, focused application/smoke contracts, the full 350-contract Demo suite, and the non-interactive smoke entrypoint against a real exported dwelling.

- Removed installation suffixes from context terminal labels so room identity remains installation-neutral on the canvas.
- Combined machine selection, pose/frame information, and automatic-placement controls into one sidebar card. Single-machine catalogs such as Sal now retain a visible fixed selector while multi-machine catalogs remain expandable.
- Validation: Python compilation, focused canvas/dropdown/facade/event contracts, the full 350-contract Demo suite, and an eight-second headless startup smoke.

- Added Demo-owned Cli state and a Cli-only sidebar toggle for return ventilation. The option is passed into the Cli runtime, where disabling it omits return grilles, terminals, routing, and materialized return segments; routing core remains unchanged.
- Disabled installations no longer project machine pins, and disabled active installations reject machine drag and wheel rotation while preserving their stored pose for later re-enabling.
- Validation: Python compilation, 20 focused Cli/state/projection/facade/event/frame contracts, the full 353-contract Demo suite, and a real-dwelling dual-installation smoke confirming supply-only Cli routing plus empty disabled-Cli machine pins.

- Replaced read-only solver rows with installation-scoped dropdown controls for strategy, router, heuristic, graph, and terminal starts. Sal retains its existing catalogs and hotkeys; Cli owns independent indices and the same `C/T/L/Y/Tab` interaction surface.
- Enabled all four graph builders already shared by the application for Cli. Cli's implemented multi-terminal Steiner strategy/backend/heuristic remain honest fixed one-entry controls until additional legacy backends are ported.
- Validation: Python compilation, focused dropdown/hotkey/Cli-runtime contracts, the full 359-contract Demo suite, a real exported dwelling routed successfully with every Cli graph mode, and an eight-second headless UI startup smoke.

- Added three Cli-owned multi-terminal backends: the shared Kou port, its nonnegative-heuristic variant, and a bend/crossing-aware line-state metric-closure MST. Sal continues to consume only its existing solvers.
- Gave Cli independent bend and crossing weights. These are consumed by the line-state backend; the two Kou variants intentionally preserve their unweighted legacy behavior.
- Added Cli machine auto-placement interaction through the existing installation placement runtime, with installation-specific mode text and hotkeys rather than exposing Sal-only heatmap and scoring controls.
- Scoped route validation/KPIs, debug overlays, and grille controls in `CLI_FOLLOWUP_SCOPE.md`; no speculative controls or validation policy were implemented.
- Validation: Python compilation, 42 focused Cli/backend/UI/placement contracts, and the full 366-contract Demo suite.

- Hardened canvas projection for disabled or stale active installations whose machine-pin map is intentionally empty; machine outlines and placement guides now render only when their required installation-specific pins exist.
- Solver dropdown controls now render closed controls first and the open option sheet last. Opening a solver or machine selector also dismisses the other selector family, preserving one readable overlay.
- Validation: Python compilation, 15 focused frame/facade/event/projection contracts, the full 368-contract Demo suite, and an eight-second headless startup smoke.

- Refined the compact workbench chrome without changing plots, weight-view controls, dwelling presentation, grid overlays, or colorbar behavior. Enabled installation pills are light, disabled pills remain subdued, the active pill remains blue, and a three-dot grip hints at the existing drag-to-reorder interaction without changing the pill geometry.
- Increased sidebar/card surface separation, made execution status a flatter non-interactive readout without a help affordance or decorative status rail, and limited semantic colour to success/failure text and warnings.
- Moved terminal controls nearer the canvas edge while preserving most of the vertical colorbar reserve. Shaft routing now follows the shaft's red semantic colour, and the active machine has a stronger blue outline.
- Validation: Python compilation, 18 focused UI/config contracts, the full 368-contract Demo suite, and an eight-second headless startup smoke.

- Split routing-order pill activation from its dedicated three-dot drag grip. The grip uses a hand cursor on hover; dragging lifts the pill under the pointer with a shadow and amber outline while the ordered row reflows, then snaps the pill into place on release.
- Reserved a separate semantic status line so bold success/failure emphasis no longer overlaps wrapped solver diagnostics such as permutation/iteration counts.
- Cli return ventilation now defaults off while retaining the existing per-session toggle and runtime behavior.
- Validation: Python compilation, focused header/event/sidebar/state contracts, the full 370-contract Demo suite, and an eight-second headless startup smoke.

- Added one theme-owned installation colour map. Non-active routes, terminals, and machines retain their installation colour blended toward a subdued context tone instead of collapsing to neutral grey.
- Split the lower canvas legend into stable room semantics, enabled-installation colours with an active label, and a terminal-validity legend that appears only with its toggleable view. Plot and colorbar rendering remain unchanged.
- Validation: Python compilation, focused projection/frame/theme/chrome contracts, the full 370-contract Demo suite, and an eight-second headless startup smoke.

- Regrouped the compact canvas toolbar so zoom, reset, ruler, and the explicit `Actual Ø` display toggle read as one viewport/measurement cluster. Weights remains separated with its existing small/big pipe POV switch unchanged.
- Reworked solution-log rows into readable metric names and two-line summaries with units and full words; selected solutions now use the same blue interaction emphasis as the active workbench controls.
- Validation: Python compilation, 16 focused control/log/chrome/event contracts, the full 370-contract Demo suite, and an eight-second headless startup smoke.

- Replaced the permanent 56 px vertical colorbar gutter with one compact horizontal bottom-right canvas overlay contract. Distance palettes retain low/high and palette labels; edge weights retain ratio endpoints and the blocked-edge key.
- The reclaimed width is now part of the canvas viewport, so terminal tools align near the actual right canvas edge. Heatmap calculation, toggles, palettes, dwelling/grid rendering, and KPI plots are unchanged.
- Validation: Python compilation, 12 focused viewport/colorbar/config/facade contracts, the full 373-contract Demo suite, direct off-screen rendering of both colorbar types, and an eight-second headless startup smoke.

- Cleaned the five retained KPI plots without changing their data or layout: best/log event diamonds remain, but persistent marker captions no longer collide with traces and each other. Minimum footers now use metric-aware units and integer formatting for score/turns.
- Validation: Python compilation, six focused plot/history contracts, the full 374-contract Demo suite, a marker-dense off-screen plot render, and an eight-second headless startup smoke.

- Added a compact KPI scope selector in the right rail. `All installations` is the default aggregate history; enabled installations receive their own histories from the same executions, so Sal can be compared before and after Cli joins the routing context without changing solver behavior or routing order.
- Aggregate samples combine enabled route sets and sequential solver elapsed time. Installation samples retain their own result and elapsed time. Solution logs remain aggregate, and interaction markers are mirrored into every visible scope.
- Validation: Python compilation and focused observation/plot contracts. The direct Pygame application smoke is unavailable in this shell because its interpreter does not include Pygame.

- Added Coc as a real installation runtime using dwelling-export's selected kitchen exhaust connector, dedicated Coc shaft entry, appliance footprint, and 125 mm duct metadata. Coc is a fixed two-terminal plan route: it has no machine placement, machine editing, height solving, or downstream 3D behavior.
- Coc owns its graph/strategy control catalog. The routing-core-style simple graph remains the default, with an explicit regular-grid option and a reported regular-grid fallback for sparse-axis disconnections.
- Ordered-installation context is applied during Coc search by a NumPy graph-occupancy filter that eliminates covered vertices and edges from numeric obstacle bounds. Shapely remains in normalized dwelling/graph construction adapters, not Coc obstacle avoidance or path search.
- Validation: Python compilation, 14 focused Coc/graph/state/execution/frame contracts, the full 379-contract Demo suite, and direct routing of all seven exported dwelling cases (7/7 successful, four using the reported regular-grid fallback). The Pygame launch smoke remains unavailable in this shell because its interpreter does not include Pygame.

- Ported routing-core's Cli supply sizing as an installation-owned, replaceable policy. The supply tree is rooted at the machine, accumulates conditioned-room polygon area in post-order, allocates verified machine airflow proportionally, and applies the core constant-friction formula with 150 mm height and 50 mm upward width increments.
- Each sized supply segment retains downstream area, airflow, equivalent diameter, section, and nominal/actual velocity for comparison with future policies. Return sizing remains intentionally unchanged because routing-core provides no equivalent return-tree baseline.
- Added verified routing-core project-input airflow/loss data for PEAD M71JA and M100JA. Other PEAD models remain explicitly unsourced and retain connector dimensions rather than receiving inferred engineering values.
- Validation: Python compilation and focused Cli sizing/runtime contracts.

- Added a separate Cli `Core ducts` canvas control without changing the lightweight `Actual Ø` stroke view. The overlay lazily projects completed supply segments into routing-core-compatible conducto rectangles and max-connected-width union squares.
- Polygon projection is installation-owned, measured independently, and cached by `CliRoutingResult`; repeated render frames do not rebuild Shapely geometry, and a new solve invalidates the cache. Conducto and union polygons remain separate artifacts for comparison with future geometry procedures.
- Validation: Python compilation, focused conducto/union/cache and toolbar contracts, and the full Demo suite (386 passed).

- Ported routing-core's supply endpoint materialization around the sized tree. Supply routing now starts 500 mm from the machine air connector; the final 200 mm uses the physical connector width, while a remaining run longer than 150 mm uses full-flow calculated sizing.
- Grille connections now retain calculated room-flow sizing until the final 100 mm physical-width stub. Machine and grille stubs inherit the core duct height of 150 mm; endpoint sizing artifacts remain attached to calculated branch segments for comparison and alternative-policy development.
- Return endpoint behavior remains intentionally unchanged because the current port scope is supply-only.
- Validation: Python compilation, focused endpoint sizing/length contracts, and the full Demo suite (387 passed).

- Replaced the hardcoded 1000 mm Cli supply-grille width with a replaceable routing-core grille-sizing policy. It allocates machine airflow by conditioned room area, evaluates 3/3.5/4 m/s face velocities with 50% effective area, rounds upward in 100 mm increments, and caps width at 1000 mm.
- Per-grille airflow, selected width, 150 mm height, and actual face velocity are retained on each supply connection. Calculated width now participates in placement feasibility and physical endpoint materialization; unsourced machines retain the previous fallback width.
- The `Core ducts` baseline intentionally retains routing-core's max-width square union at the grille transition. Its oversized footprint is now documented as a core geometry artifact for the planned alternative geometry procedure, rather than being mistaken for a sizing error.
- Validation: Python compilation, focused grille policy/runtime contracts, and the full Demo suite (388 passed).

- Scoped the edge-weight diagnostic to the active installation workspace. Weight refresh and rendering no longer read the shared Sal workspace while Cli is active, so a disabled Sal machine cannot leave a ghost clearance field on Cli's graph view.
- Cli no longer advertises a routing-weight projection it does not implement. Its solver-owned bend and crossing costs are unchanged; the global weight overlay is empty for Cli until a Cli-specific projection is added.
- Validation: Python compilation, focused capability/workspace-overlay contracts, and the full Demo suite (390 passed).

- Added a Cli-owned 75 mm minimum centerline-clearance field derived from the ported 150 mm minimum duct width. Wall-parallel edges and edges within 75 mm of the active machine receive the routing-core-compatible near-hard cost of 1,000,000 without rebuilding the selected graph.
- All three Cli tree backends consume a copied weighted adjacency. The same field is retained in the Cli workspace for the active-installation `Weights` diagnostic; its refresh path no longer invokes Sal policy.
- Validation: Python compilation, focused Cli clearance/runtime/projection contracts, and the full Demo suite (392 passed).

- Hardened the shared machine-clearance field for Cli's hard-only 75 mm policy. A zero soft margin now skips soft-band arithmetic, preventing boundary-tolerance cases during machine dragging from dividing by zero and emitting non-finite solver costs.
- Validation: Python compilation, focused clearance and Cli drag-equivalent routing contracts, and the full Demo suite (393 passed).

- Replaced Cli grille dots with thick plan-line symbols derived from each connection's physical width and wall normal. Calculated supply widths and configured return widths now scale with the viewport; ceiling returns without a wall normal use a deterministic horizontal symbol.
- The installation projection carries the same grille geometry for active and subdued context layers while routing continues to consume the existing point terminals.
- Validation: Python compilation, focused grille projection/frame contracts, and the full Demo suite (394 passed).

- Unified Cli grille symbols with the installation's blue supply-output colour instead of falling back to white room-terminal colours. Wall grilles now include routing-core-style 500 mm airflow arrows: supply points into the room and return points toward the ceiling routing side.
- Horizontal ceiling returns intentionally omit a plan arrow because their airflow direction is vertical.
- Validation: Python compilation, focused grille colour/airflow projection/frame contracts, and the full Demo suite.

- Replaced the Sal-derived score in Cli KPI histories with a Cli-owned post-route score: physical route length plus the active Cli bend penalty per turn and `bend penalty × crossing multiplier` per crossing. It deliberately excludes Sal diameters, clearances, overlaps, and short-piece policy.
- Aggregate KPI history now sums each enabled installation's independently evaluated score instead of applying one installation's policy to mixed routes.
- Validation: Python compilation, focused Cli score and aggregate-history contracts, and the full Demo suite (395 passed).

- Extended solution-log snapshots with installation-header state and a Cli-owned configuration block: machine pose/family, return toggle, graph/backend modes, placement mode, bend penalty, and crossing multiplier.
- Clicking a Cli log now restores those values before rerunning, so the active Cli controls and machine configuration visibly match the saved solution. Legacy snapshots retain their previous Sal-only fallback behavior.
- Validation: Python compilation, focused Cli snapshot/restore contracts, and the full Demo suite (395 passed).

- Replaced the binary Cli core-polygon toggle with an installation-owned `Off / Core / Aligned` selector. Core remains the routing-core comparison baseline; aligned geometry is a separately cached, post-route projection over the same sized supply skeleton.
- The first aligned policy compresses unchanged collinear runs into trivial propagation. It creates decisions only at section changes, bends, and junctions; asymmetric tees favor directional continuity weighted by downstream airflow, offset the dominant run away from a secondary branch, and carry that face through downstream reductions. Ordinary bends remain centered in this first experiment.
- Polygon work remains lazy and outside the routing loop. Alignment state selection is numeric; Shapely is used only when the selected overlay is rendered. The selector is retained in solution-log snapshots.
- Validation: Python compilation, focused aligned-geometry/cache/selector/snapshot contracts, the full Demo suite (400 passed), a non-interactive startup solve, and a real dual-installation solve whose 17 Cli supply segments produced the aligned projection in about 1.2 ms before caching.

- Corrected aligned offsets from the provisional `side * child_width / 2` displacement to face continuity based on `parent_offset + side * (parent_width - child_width) / 2`. A dominant straight leaf now retains the selected parent face through tees and later reductions without the artificial lateral jump; the machine-side first reduction remains centered as the lower-loss, simpler installation preference.
- Hardened the projection for Regular Grid endpoint snapping. Machine, tree, and grille adapters may materialize as separate geometric components, so alignment now orients every component deterministically instead of assuming one connected tree. It deliberately does not invent bridging ducts across snapped gaps.
- Validation: Python compilation, focused face-offset/centered-first-reduction/disconnected-adapter contracts, the full Demo suite (402 passed), and a live Regular Grid Cli solve with the Aligned overlay selected (11 supply segments, 14 polygons, no crash).

- Changed the centered machine-side reduction from a forced state to a preference. The aligned projection evaluates center first, then both face-aligned offsets when the centered candidate's following straight run overlaps the routing boundary, walls, columns, shafts, machine footprint, or retained earlier-installation obstacles.
- Constraint evaluation remains lazy with the polygon overlay and does not change the routing skeleton. Intentional connector contact is excluded by evaluating the reduction's downstream run; if every candidate is obstructed, the centered invalid outcome is retained instead of silently inventing geometry.
- Validation: Python compilation, focused centered/obstacle-forced/runtime-constraint contracts, the full Demo suite (404 passed), and a live Regular Grid Cli solve with Aligned geometry enabled.

- Reversed the shared sidebar dropdown palette while a selector is open: the active control and downward option sheet use light surfaces with dark text, while the selected row retains a distinct blue-grey fill. Closed controls keep the previous dark workbench styling.
- Replaced the permanent KPI-scope pills with the same dropdown interaction in the right rail. It opens downward over the plots, renders after the chart cards so it remains legible, dismisses competing machine/solver selectors, and retains the same aggregate/installation history semantics.
- Validation: Python compilation, focused dropdown-palette/KPI-layout/KPI-controller/event contracts, the full Demo suite (406 passed), an off-screen visual check of both open option sheets, and a timed headless workbench render that completed without UI errors.

- Added experimental Sal `State A* (hard min pieces)` and `Line graph A* (hard min pieces)` backends alongside the unchanged baseline choices. Both enforce existing diameter/factor endpoint and internal centerline minima, including materialized port stubs, snap adapters and shaft-entry geometry. No fitting deductions, soft length penalties or later sanitization are included.
- The shared pure search tracks only capped straight-run length and first-piece status; line-graph labels also retain the incoming directed edge. Cost/length dominance preserves more expensive arrivals when they permit a later bend that a cheaper short arrival cannot take. Immediate reversals are excluded. Saturation bounds length-history distinctions, not absolute total memory; global self-intersection/elementary-path constraints remain outside this experiment.
- Added per-scene numeric edge-geometry reuse, cached relaxed heuristics and a bounded 64-entry exact weighted-request cache across routing-order retries. Prior-route weights, route identity, starts, eligible pins and bend cost are part of request identity. Caches do not survive a scene recompute.
- Sequential and negotiated Sal routing pass explicit route identities into search. Min-cost-flow combinations report unsupported instead of silently routing unconstrained flow stages. Defaults and graph discretization are unchanged. The current catalog's 80/125 mm ducts determine minima; older README machine tables are explicitly identified as historical.
- Validation: compilation of changed Python files, 435 passing Demo tests (including exhaustive small-grid cost comparisons, longer-arrival preservation, collinear merging, endpoint/snap/shaft adapters, fractional thresholds and unsupported flow combinations), and Pygame dummy-video keyboard cycling plus rendered dropdown/route checks for both new backends.
- Added `benchmarks/benchmark_sal_min_pieces.py` and checked-in summary. Seven exported dwellings, six poses per dwelling, three measured repetitions after warmup: 126 samples per backend, including blocked configurations. Final Sal-only scene recompute median/p95/max: state hard minima 82.9/158.2/192.1 ms; line-graph hard minima 81.1/155.1/204.1 ms. State had zero samples above 200 ms; line graph had one. Both produced zero short pieces in every successful sample (69/126), compared with 90/126 successful baseline samples. These differing feasibility sets must not be interpreted as a like-for-like speed ranking.
- Timings exclude initial dwelling preparation, rendering and plot updates. System load caused substantial variation between repeated benchmark passes, including slower passes exceeding 200 ms for both baselines and experiments. No fixed latency guarantee or timeout is imposed. Next experiments can compare decreasing length penalties or a lazy straight-run graph while retaining this hard-minimum baseline.

- Corrected Sal machine proximity weighting for legal pin approaches. The shared weight builder now exempts diameter-matched, collinear outward rays beyond physical access points from the soft machine penalty. Hard machine clearance, static constraints, existing edge costs, route interactions and proximity costs elsewhere remain active. Every Sal backend receives this same correction; no search-loop state or graph subdivision changed.
- Added a regression covering identical graph/edge fields across all five backends, valid straight access selection, preserved hard blocks and other soft costs, and connector-diameter matching. At the reported `(4561, 5250, 90 degrees)` real-dwelling pose, all backends now route Washroom and Bathroom straight (650 and 1749 mm). The shaft can still differ because its physical entry has separate minimum-piece feasibility requirements. Shared weighting does not imply equal intermediate routes or equal subsequent interaction fields when prior routes differ.
- Validation: changed Python compilation, 437 passing Demo tests, full-scene graph/static weight equality at the reported pose, and a rendered Pygame check. Ten warmed runs per backend at that pose: median/max scene times were state A* 66.7/70.3 ms, line A* 70.8/78.6 ms, GBFS 70.1/72.9 ms, state hard minimums 66.4/74.8 ms, and line hard minimums 66.4/74.0 ms. These are pose-specific timings; the earlier seven-dwelling benchmark summary predates this weighting correction.

- Corrected shared Sal interaction geometry after the crossing comparison at `(4217, 5539, 180 degrees)`. Runtime records now retain physical diagonal adapters and merge axis fragments before building weights; the legacy axis-only record helper remains available. Diagonal intersection, overlap and exact segment distance use NumPy during preparation, including crossings between sampling locations, with no additional search-state memory.
- Crossing weights at candidate-edge endpoints now use half the penalty, so a route traversing two adjoining edges through one crossing node pays one configured crossing penalty. This remains a local edge approximation: a route starting or ending exactly at a crossing node receives only its incident share, and route-proximity costs remain edge-based. No hard maximum-crossing constraint or multiplier change was introduced. Normalized prior axis geometry removes duplicate crossing/clearance contributions caused by collinear route fragments.
- The ordinary A* shaft route had a diagonal connector that was absent from interaction weights, while the constrained shaft route had a detected axis crossing. After correction, ordinary state/line A* and both hard-minimum variants all choose the same Bath detour at that pose. Its alternative straight crossing now adds 20,000 instead of 160,000. GBFS still follows its heuristic and can select the crossing; this is independent of the length experiment.
- Validation: changed Python compilation, 467 passing Demo tests, a 203-edge independent Shapely oracle for diagonal crossing/overlap/clearance, subdivision/reversal-sensitive crossing examples, and identical crossing-versus-detour switch thresholds across all four A* backends. Both reported real poses were reproduced, including retained straight small-pin access at 90 degrees. Ten warmed full-scene runs per backend at the straight-access pose remained below 79 ms maximum; median times were 69-73 ms. Prior benchmark summaries predate these weighting corrections.

- Added two experimental Cli Directed Steiner supply backends: `Steiner: min bends` and `Steiner: min bends + tees`. The per-solve minimum is the calculated full-flow trunk width (600 mm for the sourced M71JA inputs); machine/grille face widths and unsourced machine guesses are excluded. Return routing retains its existing line-graph MST behavior. Dropdown and keyboard selection enable wall candidates and display the active minimum.
- The pure numeric solver preserves baseline weighted arcs, turn costs and terminal costs while carrying capped physical run length independently. Physical machine/grille adapters count, collinear graph/size subdivisions do not reset runs, and immediate reversals cannot manufacture length. Tee merges reset every outgoing branch, retain distinct physical departure directions through zero-length wall aliases, and support more than two children. Compact direction masks and shared merge calculations reduce the initial tee-mode overhead.
- Reconstructed and materialized physical trees are checked separately. An invalid physical union is blocked without trying alternate trees, relaxing lengths or routing unconstrained supply. This expanded-state DP does not certify physical infeasibility or exact physical-tree optimality; five bends-only poses and nine tee-mode poses in the 42-pose benchmark rejected their chosen physical union. A snapped machine access, unsourced sizing or center-grille start configuration reports unsupported rather than silently changing the constraint.
- Validation: changed Python compilation, 496 passing Demo tests, exhaustive small-graph physical-tree cost comparison, identical prepared graph/cost/candidate fields across the baseline and both experiments, short endpoint/bend/tee and adapter regressions, ternary branching, dropdown/keyboard behavior, and Pygame off-screen renders of all three backends and the open selector. Opened a visible `Cli hard lengths` workbench with Cli and the bends mode selected.
- Added `benchmarks/benchmark_cli_hard_lengths.py` and checked-in summary. Seven exported dwellings, six poses each, three measured repetitions after warmup: 126 samples per backend on Core Simple with M71JA and returns disabled. Final median/p95/max scene ms: baseline 55.1/113.4/156.7; bends 61.6/138.2/146.9; bends + tees 75.6/204.8/213.5. Ten tee samples exceeded 200 ms; none did for the other two. Successful samples were 123/102/90, so the differing feasible/accepted sets prevent a simple speed ranking. Maximum hard states: 2,271/3,144; subset labels: 29,151/42,246; tee branch labels: 35,123.
- A slower repeat of the same benchmark also slowed the unchanged baseline: median/p95/max ms 158.0/327.5/368.1, bends 176.3/372.5/397.3, and tees 205.1/498.9/552.8. The final repeat returned to the lower timing range. These are observations, not a fixed 200 ms guarantee. Next useful algorithmic work is retaining alternative physical-tree candidates after an expanded-state union rejection and reducing branch-label work on larger scenes.

- Added Cli `Steiner: min + fittings` as a separate backend over the existing hard bends-and-tees solver. Its scalar objective adds 1e-6 mm equivalent per physical elbow, tee or cross; minimum lengths, weighted graph inputs, terminal candidates, primary bend costs, return routing and KPI semantics retain their existing behavior. The perturbation can prefer a slightly higher primary cost within its tiny fitting correction; it is not a strict lexicographic objective.
- Reused the existing per-candidate departure masks and scalar DP labels. Physical elbow charges are prepared with graph transitions and terminal adapters; branch summaries exclude the shared origin, and the full junction mask receives one charge when lifted, including multiway branches. Zero-length wall aliases, collinear grid splits, section changes and straight connections add no fitting charges. A small shared mask-charge cache avoids repeated classification in the subset loop. No fitting-history state or per-label count was added.
- Added pure physical fitting diagnostics and sidebar E/T/X counts for the hard-length modes. The final count is reconstructed independently from centerline incidence; search diagnostics recover the primary score by subtracting the perturbation with normal floating point roundoff. Backend capabilities now declare directed/hard-length indices for consistent dropdown and keyboard selection. Existing physical-union rejection limitations remain explicit.
- Validation: changed Python compilation, 518 passing Demo tests, exhaustive small-graph perturbed-cost comparison, physical port classification, epsilon versus meaningful/near primary differences, prefix/diagonal/grille adapters, zero aliases, cross accounting through binary subset merges, matching materialized fitting counts, and identical graph/weight/candidate fields. The controlled tie reproduces the requested preference: two elbows and one tee become one elbow and one tee at primary cost 13,400. Pygame renders verify the new option, active minimum, fitting breakdown and status.
- Performance: seven dwellings, six poses and three measured repetitions after warmup, 126 samples per mode. Existing/perturbed median scene times 72.1/71.7 ms, p95 199.1/200.4 ms, maxima 221.4/214.8 ms, and 5/8 samples above 200 ms. Both accepted 90 samples, rejected 27 physical unions, and had identical state counts at every pose (maximum 3,144), with matching fitting/primary costs on the accepted scene pairs. These scenes demonstrate comparable latency, not a measured speedup or universal latency guarantee.
- A frozen four-room request (317 graph nodes, alternating mode order, five warmups and 20 measured samples per mode) gave pure-solver medians 108.76/108.88 ms. Separately measured peak Python allocations were approximately 20.98/20.83 MB; both modes retained 42,246 subset labels and 35,123 branch labels. Timing/allocation variation is not evidence of a memory reduction. Recorded summary: `benchmarks/cli_fittings_summary.json`.

- Added the requested `0002_real_c90 / A_A1_P1_V02` case using its canonical source ID `A_A1_P01_V02`. Exported a new fixture through the current routing-core interpreter and appended it to the shared `Dwelling-export/data/dwellings.sqlite` database; the catalog now has eight cases. A SQLite backup was retained in this task's output folder, and all seven existing fixture records were verified unchanged.
- The existing database discovery automatically exposes the case in the selector; no app loader or solver changes were needed. The fixture and sample rebuild entry are saved in Dwelling-export commit `f3a712d`, alongside a narrowly guarded settings-import fallback for routing-core's move to `afd_routing.legacy.info.settings`. The pre-existing `.pytest_tmp/` directory was left untouched. Historical seven-case performance summaries predate this catalog addition.
- Validation: database integrity/foreign-key checks; valid geometry in all three routing frames; 8 rooms, 6 covers, 2 shafts and 6 columns; current exporter/rebuild Python compilation and 9 passing exporter tests. The Pygame app discovers all eight cases and renders the new dwelling. Both Cli hard bends-and-tees backends route its 3 supply grilles at the automatically chosen pose. Opened a fresh app window with the new dwelling and fitting-perturbation backend selected.

- Added `SOLVERS.md` as the central guide to the current Sal, Cli and Coc solvers: algorithms, objectives, state/memory choices, reasons for each experiment, strategy/backend distinctions, hard-length rules, failure semantics and benchmark sources. Linked it from the README and recorded the unresolved graph-preparation/timing findings from the recent audit.
- Added a Cli hard-minimum slider shared by `Steiner: min bends`, `Steiner: min bends + tees` and `Steiner: min + fittings`. Following the user's clarification, the control multiplies the calculated full-flow trunk width by 0.25–3.00x in 0.05x steps; default/reset is 1.00x. Its label shows both the factor and resulting minimum in mm, and every setting follows subsequent sizing changes. The factor is Cli-owned, retained across backend switches, passed through context retries and saved/restored in solution logs. Older Cli logs restore 1.00x. Supply sizing, primary costs, return routing and Sal's independent diameter-factor control retain their existing semantics.
- Moved the Cli fitting breakdown to the solver summary row to accommodate the slider without overlapping controls. No new solver state or search-loop bookkeeping was introduced.
- Validation: changed Python compilation and 530 passing Demo tests, including recalculated sizing, context retries, factor clamping and log restoration. Pygame rendering and injected click/drag/release/reset events verified the factor/mm labels, hidden controls for baseline routers, reset to 1.00x, and independent Sal settings. A 24-request sweep over the new dwelling and the larger four-room B_B case exercised all three hard variants at 0.25x, 0.50x, 1.00x and 3.00x (150/300/600/1800 mm for these inputs). Observed scene times ranged from 39 to 232 ms, including blocked candidates; this is a smoke sweep, not a performance ranking or latency guarantee. On the new dwelling, the fitting-aware mode selected 2 fittings at 0.50x versus 3 at 1.00x.

- Added the user-scoped medial-axis grid MVP as `GraphMode.MEDIAL_AXIS = 4`, independently selectable from routing strategy and state-expanded/line-graph representation. Sal and Cli expose the additional grid; existing enum values/default selections remain stable. The builder approximates polygon medial axes with sampled-boundary SciPy Voronoi ridges, discards same-boundary-feature ridges, preserves loops and simplifies degree-two chains to 5% of sample spacing after geometry/wall checks. Default boundary spacing is 200 mm (semantic shared graph config); no new dependency was installed. Simplified chords are explicit physical edges, not hidden polyline bends.
- Required terminal, machine access and shaft-face points attach exactly through bounded local visible projections. The machine body is included as a polygon hole to generate pose-specific detours. Geometry/pose/spacing templates are cached with a 16-entry bound, and identical solves reuse their skeleton. Sampling and the bounded attachment policy can lose useful routing choices; blocked is not a continuous-space infeasibility claim.
- Added hashable arbitrary headings plus numeric direction comparison/reversal. Both standard Sal A* backends and min-cost flow retain a fixed bend cost, now recognizing direction changes between diagonals. Minimum-piece searches and physical turn/piece metrics understand diagonal headings in this mode. Euclidean target distance replaces orthogonal heuristic estimates for Medial (Zero remains diagnostic), with the effective label visible in the dropdown. Cli Kou uses Euclidean distance; its line-graph metric MST accepts the grid. Directed wall-candidate/hard Steiner modes explicitly report unsupported instead of being silently substituted. Coc's options remain unchanged.
- Kept the angle-weighted follow-up deferred, per the user's fixed-penalty MVP. Inspected routing-core's `afd_routing/mep/routing/scoring.py:angle_aware_bend_turn_factor` (quadratic `x*(1.48-0.48*x)`, capped above 90 degrees) and its post-materialization Clima candidate-scoring call. The demo's geometric heading comparison follows the core's normalized-vector absolute tolerance (1e-4), without importing its materialization/scoring architecture.
- Reused numeric general-segment interaction fields for diagonal candidate edges against cardinal prior routes. Added an AABB broad phase and prebuilt segment geometry to crossing diagnostics after profiling showed repeated GEOS intersections dominated the unsimplified skeleton run. Existing grids retain their axis interaction fast path. Post-route overlap/clearance KPI counts still have orthogonal assumptions; Sal exposes this MVP limitation in warnings and README documents the scope.
- Validation: 33 changed/new Python modules compiled; all 545 Demo tests passed using the Pygame Python 3.12 runtime, with the already-installed pure pytest package loaded from the core environment (plugin autoload disabled). New contracts cover centreline geometry/determinism, holes/cycles, disconnected regions, cached-template immutability, exact endpoints, machine detours, protected-access collision filtering, fixed diagonal turn costs in both search representations, Euclidean heuristics, diagonal physical pieces, minimum-piece diagonal searches and cardinal/diagonal route interactions. The CLI graph-mode contract and cycle-hotkey test cover the new option. Scripted SDL dummy rendering verified Sal/Cli startup and the grid/heuristic labels; normal and blocked scenes render successfully. A Sal smoke sweep exercised all seven strategies on the first real dwelling.
- Recorded `benchmarks/medial_grid_results.json` with the reproducible `benchmark_medial_grid.py`: two real dwellings plus seeded synthetic, three poses, both standard A* representations and two measured repeats after warmup (36 samples per grid). All sampled requests returned routes, with warnings retained. Regular/Hannan/Medial median full-scene times were 57.035/84.650/28.954 ms, p95 890.673/256.695/470.906 ms, and median graph node counts 634/656/98. Preparation/automatic placement and rendering are excluded; graph refresh, weights and routing are included. These are exploratory machine-dependent observations, not a claim of uniformly interactive latency or improved physical route quality.

- Following the user's clarification that Figure 5(d), rather than a Voronoi medial axis, is the intended linear construction, added `GraphMode.CELL_SKELETON = 5` and the separate `Cell Skeleton (orthogonal)` Grid option for Sal/Cli. Kept the Voronoi mode and all existing selector values/defaults. Grid generation remains independent of representation and routing strategy; no hybrid was added in this step.
- Added the pure `graphs/cells.py` builder: compressed boundary-coordinate rectangles, wall-band partition features, greedy rectangular merges, cell centres, positive-length shared-side portals and explicit orthogonal elbow nodes. All crossings/T junctions and overlapping spans are split into a planar candidate graph. Required points attach via validated local L shapes, machine geometry is a free-space hole, and short transverse wall crossings follow the existing policy. This is an adaptation of the survey's principle, not an implementation algorithm attributed to the illustrative Figure 5.
- Exact rectangular decomposition applies to rectilinear input. Sloped/curved boundaries receive local conservative subdivision controlled by `DOMAIN.SOLVER.GRAPH.CELL_BOUNDARY_RESOLUTION_MM` (200 mm default); unresolved fragments are excluded and the console reports excluded area/attachments. Sixteen geometry/pose/resolution templates are cached. Sparse centre/portal choices and excluded fragments can disconnect physically routable situations; failures are retained.
- Added cell metadata to the graph view and the `G` overlay, drawing pale rectangular cell outlines beneath the candidate edges. Separated exact endpoint attachment semantics from the Euclidean heuristic flag; cell grids use the normal orthogonal heuristics and fixed bend costs. Machine filtering preserves cell metadata and cannot bypass body collisions via protected access endpoints. Cli's existing directed/hard modes remain selectable with the orthogonal grid.
- Validation: all 556 Demo tests pass, including new exact-coverage/non-overlap, hole connectivity, conservative slope refinement, unequal-side portals, corner-only contact, traversable wall bands, exact disconnected endpoints, explicit crossing/T-junction nodes, cached-cell immutability, machine detours, and heuristic/metadata lifecycle contracts. Scripted SDL dummy rendering verified the new selector and cell/skeleton overlay. First-real-dwelling smoke solves succeeded with all five Sal backends; Cli Kou, metric MST and directed-wall baseline solved, while a hard-length request remained blocked.
- Extended the existing grid benchmark with opt-in `--include-cells`, preserving its original default comparison. `benchmarks/cell_grid_results.json` records 36 warm requests per grid over two real dwellings, seeded synthetic, three poses and two A* representations. Cell skeleton routed 24/36 requests (all real cases), with 12 synthetic shaft failures; the other grids returned routes in 36/36. The synthetic cell graph's shaft-face component was disconnected from the machine component despite near-zero excluded boundary area, illustrating the sparse centre/portal/wall-policy limitation. Cell median/p95 scene times were 72.842/85.010 ms, with median 157 nodes; failures are included, machine load differs from the earlier run, and no speed/quality superiority is claimed.

- Added `GraphMode.HANNAN_MEDIAL = 6` as the separate Sal/Cli **Hanan + Voronoi (local links)** grid. Retains Hanan axes and sampled-Voronoi edges; each Voronoi node receives up to four nearest legal cardinal projections onto existing Hanan edges within one scaffold spacing (600 mm default). Following the user's clarified scope, the final implementation uses local segments rather than extending Voronoi coordinates into global Hanan axes. Existing standalone grids, values and defaults remain available.
- Added pure `graphs/hybrid.py` assembly: bounded numeric projection candidates, existing geometry/wall validation, target-edge splitting, STRtree intersection queries for added edges against Hanan and each other, shared junctions and collinear overlap deduplication. Inputs/cached Voronoi geometry are not mutated. Machine geometry is a hole in both inputs. No distant or diagonal endpoint fallback is invented when local connections fail. This remains a grid-construction experiment independent of routing representation and strategy.
- Reused Euclidean heuristics, fixed actual-heading bend costs and diagonal route metrics in both standard A* representations; the sidebar displays the effective Euclidean mode and preserves diagnostic Zero. Cli's Kou variants and line-graph metric MST support the hybrid; its directed wall/hard-length backends explicitly report unsupported, matching their existing diagonal-grid limitation. Orthogonal overlap/clearance KPI limitations remain visible. No angle-weighted cost was added.
- Validation: changed/new Python compilation and 565 passing tests. Added contracts for short local attachments without axis growth, edge splitting and switching between Hanan and diagonal edges in both A* representations, natural crossings without added links, collinear overlaps, radius bounds, holes/disconnected regions, rejected long wall crossings, deterministic immutable inputs, machine holes and lifecycle heuristics. SDL dummy rendering verified the merged grid and selector; the pictured first real case used 986 runtime nodes and routed in about 172 ms. Standalone modes remain selectable.
- Added opt-in `--include-hybrid` to the existing benchmark and recorded `benchmarks/hannan_medial_grid_results.json`: 18 warm requests per grid, two real dwellings plus seeded synthetic, three poses and both normal A* representations. All four grids returned routes in all sampled requests, with warnings retained. Hybrid median/p95 scene times were 177.423/955.962 ms with median 1,064 nodes; Hannan was 85.656/257.218 ms with 656 nodes. At 18 samples the p95 is the slowest request; preparation/rendering are excluded and these observations do not establish uniformly interactive latency or physical route-quality superiority.

- Enriched the existing Cell Skeleton mode with wall-side shifted samples and a minimum retained cell area, following the user's request for more local candidate points and less diagonal-boundary fragmentation. Used the stated starting assumptions: both normal wall offsets, existing wall clearance (100 mm default), existing regular-grid spacing (200 mm) and a configurable 40,000 mm² / 0.04 m² cell floor. The mode and its selector value remain orthogonal and independent of strategy/backend.
- Added pure `graphs/wall_grid.py`: clip wall offsets to allowed geometry, sample locally and try validated orthogonal L steps along diagonal walls. Extracted the existing orthogonal path assembler into `graphs/orthogonal.py` for shared reuse. Generalized the existing local union as `join_grids_locally` (preserving its original Hanan/Voronoi import name), adding cardinal attachments to isolated base nodes. Wall nodes attach to cell edges within one scaffold spacing (600 mm), shared crossings/attachments become nodes, and still-isolated required points may attach locally to the combined grid. No global wall axes or unchecked diagonal connections are added; machine and wall policies remain active.
- Added shared semantic config `DOMAIN.SOLVER.GRAPH.CELL_MIN_AREA_MM2` (40,000 default; zero disables). Mixed boundary rectangles stop before splitting below the floor; fully covered input rectangles can first merge, then remaining undersized cells are removed. The final floor therefore covers both recursion and initial wall-coordinate slivers. The area value participates in the bounded cell-template cache key, and diagnostics expose discarded-small-cell counts, excluded partition area, wall nodes and wall links. Excluded cell area can still contain legal wall-grid candidates; the floor can remove useful narrow cell passages.
- Validation: 13 changed/new Python files compiled; 574 tests pass. New contracts cover area-bounded diagonal refinement, initial thin wall bands, area-sensitive caching, both wall offsets and sample spacing, orthogonal diagonal-wall steps, holes/components, wall-to-isolated-centre links, explicit shared intersections, machine interiors and required points whose small cell was excluded. SDL dummy rendering verified cell outlines plus wall candidates. The first real dwelling changed from 31 to 23 cells and uses 200 runtime nodes; its rendered solve took about 67 ms with route-quality warnings retained. Cli Kou, line-graph metric MST and directed Steiner smoke requests solved; a hard-length request remained blocked.
- Recorded `benchmarks/cell_wall_grid_results.json` without overwriting earlier cell-only results: 18 warm requests per grid over two real dwellings, seeded synthetic, three poses and both standard A* representations. All four grids routed all requests, including the synthetic shaft poses that the prior cell-only experiment left disconnected. Enriched-cell median/p95 scene times were 75.326/903.382 ms, with median 199 nodes. At 18 samples p95 is the slowest request; preparation/rendering are excluded, warnings are retained and no general route-quality or latency advantage is claimed.

- Applied the user's clarification to reuse the existing Hanan wall-shifted X/Y axes in Cell Skeleton and retain the confirmed 0.04 m² area floor. Extracted the unchanged wall midpoint/normal-offset calculation into `hannan.build_wall_shift_axes`, now shared by Hanan static templates and the cell mode. The cell wall layer merges these coordinates under the existing 120 mm rule and uses the existing axis-grid geometry checks. It uses wall-origin axes only; other Hanan axis sources remain in Hanan mode.
- Replaced and removed the sampled-wall-run builder `graphs/wall_grid.py` and its spacing argument/metadata. Retained local connections and shared junctions between wall-axis and cell grids. Still-isolated cell centres, as well as required points, can project onto nearby combined-grid edges within the same scaffold-distance bound. The mode remains orthogonal; the accepted area-floor setting and other grid/solver defaults are unchanged.
- Validation: six changed live Python files compiled and 575 tests passed. Updated contracts verify explicit midpoint/normal coordinates, Hanan priority/preservation behavior, wall-axis merging, cell-centre connections without vertex alignment, holes/components, machine interiors, shared crossings and required points in excluded cells. A real-scene before/after replay confirmed identical Hanan static-axis dictionaries with shifting both disabled and enabled. SDL dummy rendering verified the cell overlay and wall-axis layer (5x5 axes, 13 wall nodes, 156 runtime nodes); the pictured scene solved in about 44 ms with physical warnings retained. Cli Kou, line-graph metric MST and directed Steiner smoke requests solved; a hard-length request remained blocked.
- Recorded `benchmarks/cell_hanan_axes_results.json`, preserving the earlier sampled-wall-run results: 18 warm requests per grid over two real dwellings, seeded synthetic, three poses and both standard A* representations. Current cell mode routed 12/18 (all real cases); six synthetic shaft requests remained blocked. Other grids routed 18/18. Cell median/p95 times were 36.363/45.461 ms with median 155 nodes, including blocked requests. At 18 samples p95 is the slowest request; the differing coverage prevents a speed-superiority claim. The wall-axis variant deliberately retains its disconnected synthetic outcomes without falling back to another grid.

- Added the separate Sal/Cli **Hanan + Cell Skeleton** option as `GraphMode.HANNAN_CELL = 7`, retaining existing modes, selector values and defaults. It merges the full Hanan input with the current cell mode (including wall-shifted axes and the confirmed 0.04 m² area floor), preserving legal physical edges from both inputs. Crossings/attachments become shared nodes and collinear spans are deduplicated; local cardinal links use the existing scaffold-distance bound. Cell coordinates do not generate additional global Hanan axes, and isolated cell centres are retained. Machine holes are applied to both inputs before assembly; pin/shaft runtime constraints are committed once.
- The combined mode remains orthogonal with the normal heuristics, fixed bend costs and exact port checks. Cli's directed and hard-length/fitting backends remain selectable. The cell overlay metadata survives graph commitment and dynamic filtering. Diagnostics show each source's node/edge counts, local links and merged counts. Added a fully keyed 16-entry immutable union cache after measuring repeated preparation at about 230 ms; the displayed pose then prepared in about 42 ms after an initial 238 ms union. The cache keys include both input topologies, geometry, wall shapes/thickness policy and connection radius; returned edge/diagnostic containers cannot mutate cached templates.
- Validation: eight changed/new Python files compiled and 584 tests passed. New contracts cover preservation of both physical edge sets and input nodes without axis expansion, explicit junctions/overlap removal, routing between Hanan/cell points with both search representations, machine holes/disconnected regions, deterministic source immutability, exact ports and overlay lifecycle, single-layer cases and cache invalidation/immutability. SDL dummy rendering verified the new selector and G overlay: first-real scene 935 runtime nodes, 1,699 grid edges before shaft commitment and about 111 ms warm solve. Cli Kou, metric MST, directed Steiner and all three hard bends/tees/fittings smoke requests returned supply trees; hard tees/fittings also solved with immutable cached unions.
- Added opt-in `--include-hanan-cells` to the existing benchmark and recorded `benchmarks/hannan_cell_grid_results.json`: 18 warm requests per grid across two real dwellings, seeded synthetic, three poses and both normal A* representations. The merged mode routed 18/18, including the six synthetic requests blocked on standalone Cell; Hanan also routed 18/18. Merged median/p95 scene times were 114.236/531.834 ms with median 935 nodes; Hanan 85.470/264.028 ms with 656 nodes; Cell 36.857/47.106 ms with 155 nodes including failures. At 18 samples p95 is the slowest request. Physical warnings are retained; cold/changed-pose preparation, density and route-quality limits remain explicit, with no universal latency or quality claim.

- Profiled the merged grid after the user's interaction-latency report. A first-real cold build made about 6,800 GEOS intersection calls; repeated-pose union caching hid that work. Added a numeric path for exact nonzero horizontal/vertical segments in the shared local union. Vectorized bounding-box candidates replace repeated GEOS calls; overlap orientation, pair order, shared junctions, node indices and sorted edge records are preserved. Diagonals, near-cardinal slants and degenerate segments retain the GEOS routine. No grid thinning, connection-policy, area-floor or solver changes were made. Temporary arrays are linear in segment count, but pair scanning remains quadratic in the worst case.
- Added `benchmarks/benchmark_grid_build.py` and `benchmarks/grid_build_results.json` to distinguish changed-pose construction from warm scene solving. Two real dwellings, seeded synthetic, three poses, three repetitions and alternating implementation order give 27 samples per path per grid. Each cold build clears cell-template and union caches; its immediate repeat is warm. Static Hanan axes and dwelling preparation are excluded, runtime assembly included, routing/rendering excluded. Every cold/warm result matches exact node arrays, edge records and cell overlays against the retained GEOS path. Merged cold median/p95/max changed from 229.295/370.474/370.926 ms to 108.825/179.438/185.462 ms; warm median from 41.882 to 37.653 ms. Cell cold median changed from 19.182 to 14.094 ms; unchanged Hanan controls measured 29.564/28.828 ms. This is a measured construction improvement, not a universal interaction bound; routing and rebuilding source layers still contribute latency.
- Validation: three changed/new Python files compiled and 593 tests passed. Added independent GEOS comparisons covering randomized fractional cardinal segments, reversed horizontal/vertical overlaps, duplicate spans, endpoints, crossings, local projections, exact ordering and fallback selection for diagonal/near-cardinal/zero-length inputs. Existing physical-grid, cache, both search representations and installation contracts remain passing. First-real Sal scene smoke requests routed all four terminals using both A* representations; startup construction took about 116 ms and cached builds 37-39 ms. Full warm scene solves remained about 110-112 ms, as routing itself was not optimized.

- Split cell decomposition into a cached fixed dwelling partition and local machine-cell replacements, following the user's selected approach. Both standalone Cell and Hanan + Cell now request the base template before subtracting the machine body. Only base cells with positive-area body overlap are repartitioned; boundary-only contact leaves them intact. Untouched rectangles are retained, replacement cells follow the existing 0.04 m² floor/resolution, and shared portals are rematched. A separate bounded 16-entry pose-patch cache prevents dragging from evicting the dwelling's 16-entry base cache. Region/wall/resolution/floor changes invalidate the base; ports/terminals do not. Wall-axis construction, attachments, grid unions and runtime constraints remain live. This deliberately changes candidate geometry from global per-pose decomposition; it is not an exact topology-preserving optimization.
- Added base/rebuilt cell counts to diagnostics and console reports; the first real scene patches 1 of 20 base cells. Added `benchmarks/benchmark_cell_cache.py` and `benchmarks/cell_cache_results.json`, replaying the pre-change builders from local commit `914d637`. Two real dwellings, seeded synthetic, three new poses after one priming pose and three repetitions give 27 samples per path/grid, with alternating path order. The new path has 27 dwelling-template hits and zero misses versus 27 misses previously. Standalone partition median changed from 3.461 to 0.979 ms; complete Cell construction from 14.140 to 11.685 ms; merged construction from 107.901 to 104.922 ms. Unchanged Hanan controls were 30.226/29.852 ms. Runtime assembly is included, dwelling preparation/routing/rendering excluded. The modest merged benefit reflects continued union cost and includes changed local candidate geometry.
- Validation: seven changed/new Python files compiled and 600 tests passed. Contracts cover cache reuse across 30 distinct poses, independent bounded pose caching, both graph modes requesting the same fixed region, invalidation on dwelling/wall/settings changes, unchanged untouched cells, rematched cross-boundary portals, exact rectangular free-space coverage, fully covered cells, outside/boundary-touching machines, rotated bodies with area-limited refinement, excluded-area accounting and live attachment changes. Existing machine-detour and both search/installation contracts pass. SDL dummy rendering verified the merged selector, cells and routed first-real scene; Cli directed and all hard bends/tees/fittings smoke requests returned supply trees on the merged grid.
- Recorded `benchmarks/cell_cached_scene_results.json`: 18 warm requests per grid across two real dwellings, seeded synthetic, three poses and both standard A* representations. Standalone Cell retained 12/18 successful requests (all real cases; six synthetic shaft failures); merged retained 18/18. Cell median/p95 scene times were 30.779/38.765 ms with median 155 nodes; merged 107.727/524.015 ms with median 937 nodes. At 18 samples p95 is the slowest request. Physical warnings remain visible. Updated the earlier intersection benchmark to clear the new machine-cell cache when measuring cold builds; prior result files remain historical. Further merged-grid latency work should target union/source-grid assembly rather than claiming that partition caching removes its cost.

- Added CLI's independent `[V]` body-fit overlay. The first placement experiment evaluates only the machine body at the current cardinal rotation on a 50 mm raster. Rasterize rooms separately within the routing region; require each free tile to be fully covered and disjoint from static walls, columns and shafts. A summed-area table checks each rectangular footprint window conservatively over the entire colored center tile. Green means the body fits in plan; red may include borderline feasible positions. Cache the room tiles for the active dwelling and the eroded masks by oriented machine dimensions. Keep SAL's score overlay independent and leave CLI ranking, routes, inlet/outlet, vertical and other-installation checks outside this experiment.
- Validation: seven changed Python files compiled and 603 tests passed. Tests cover no fit across room boundaries, thin obstacles, sloped walls, exact geometry of whole green tiles, rotation, machine dimensions and dwelling invalidation. SDL rendering exercised CLI `V`, rotation and zoom. On the first real dwelling, exact checks confirmed all 1,494 green tiles at 90° and 1,064 at 180°. Median over 30 mask-only measurements: 23.8 ms initial raster plus erosion, 0.16 ms for the second orientation from the same raster, and under 0.001 ms for a cached lookup; rendering and routing excluded.

- Updated CLI's body-fit overlay to screen against the full allowed routing region, including its traversable interior walls. The global footprint erosion still excludes columns, shafts and any area outside the region. A separate room-and-wall-clear erosion distinguishes green (clear in one room) from yellow (otherwise fits, including a wall overlap or a room boundary). Red still means the conservative raster did not clear the footprint. The change is display-only; placement ranking, routing, and access rules are unchanged.
- Validation: three changed Python files compiled and 604 tests passed. Tests cover interior walls and wall-free room boundaries, valid space outside room polygons, and wall areas excluded from the allowed region. Exact geometry checks verified all green and yellow tiles at 90° and 180° on the first real dwelling, and the Pygame render showed the three-color legend. At 90°, the map retained 1,494 clear tiles and added 632 yellow tiles.

- Fixed CLI routing from an out-of-bounds machine body when the supply access point remained inside, reproduced with `Steiner: min + fittings` at one millimetre of body overhang. All CLI backends now require the exact rotated footprint to be covered by the allowed routing region before preparing or searching the graph. The blocked result clears prior routes and shows a specific status, while preserving the pose so the user can drag it back inside. Boundary contact is allowed, holes in allowed geometry are respected, and traversable interior walls do not create new exclusions. Automatic placement reuses the same footprint construction without changing its ranking.
- Validation: regressions cover the inside-outlet/outside-body case for Kou and hard-length fittings, boundary contact, recovery, and a hole under the footprint with all four corners inside. The first real dwelling routed at (6000, 5200) mm and 90°, blocked at (6000, 5330) with the outlet still inside, and routed again after returning to the initial pose. SDL rendering verified the blocked scene.

- Extended CLI's `[V]` placement mask to screen the machine body and the supply extension together. The extension starts at the rotated catalog outlet, uses its width, and reaches the same 500 mm access point as routing. Two signed rectangular windows over one summed-area table preserve the narrower outlet shape, without treating the whole area beside it as occupied. Cache the static dwelling raster and each machine's four cardinal orientations separately; the return extension and solver behavior are outside this overlay change. Green and yellow retain their existing meaning for the combined footprint.
- Validation: seven changed Python files compiled and 609 tests passed. Regression cases cover opposite outlet directions, boundary and shaft collisions confined to the extension, wall contact producing yellow, and the narrower extension clearing an obstacle that a single bounding box would hit. Exact geometry checks verified every green/yellow tile on the first real dwelling at all four orientations; SDL rendering verified the updated legend and rotation.

- When CLI's placement heatmap is visible, draw a dashed perimeter around the body and supply extension and make the active machine fill translucent. The perimeter and raster use the same footprint geometry before tile rounding. Verified the outline, heatmap visibility through the fill, rotation, and restoration of the opaque fill with the overlay off in Pygame. Six changed Python files compiled; 609 tests passed.

- Separated CLI placement feasibility from room coloring. The combined body and supply extension still determine validity, but only the body determines whether a valid tile is green (clear in one room) or yellow. Regression tests cover a green body with an extension crossing a traversable wall into another room, and a blocked extension preventing green. Three changed Python files compiled; 609 tests passed. On the first real dwelling at 90°, the valid count stayed at 1,203 and the green count rose from 421 to 592; verified the new legend in Pygame.

- Added a fixed Cell Skeleton partition over the CLI placement heatmap, independently of the selected routing grid. The overlay reuses the dwelling cell template before cutting the live machine. It marks cell centers green or yellow when they pass the current rotation's footprint mask and gray otherwise. The machine and its footprint outline still draw above the grid. `[Shift+G]` toggles the placement cells while the heatmap is on; `[G]` retains the routing-grid toggle. Candidate markers do not move or snap the machine.
- Validation: six changed Python files compiled and 609 tests passed. Pygame smoke checks verified the layered view, independent toggles without a routing solve, and stable partition across rotation. On the first real dwelling the 20 cells offered 6, 6, 5, and 4 footprint candidates at 0°, 90°, 180°, and 270°. At 90°, four of the six candidates also routed with the hard-length fittings backend on Core Simple.

- Corrected the CLI placement overlay to show the actual static Cell Skeleton nodes and edges in addition to cell outlines. It uses the routing mode's Cell Skeleton builder and options, including wall-shifted Hanan connections, with no live machine or endpoint attachments. Cell center candidate rings and the independent toggle are unchanged. On the first real dwelling, all 98 nodes and 114 edges matched the Cell Skeleton routing builder with those same exclusions. The cached overlay stayed fixed on rotation, the SDL render showed the connections, four changed Python files compiled, and 609 tests passed.

- Added the room-aware CLI placement preview from the room-wall audit. Partition each room clipped to the allowed region, then partition the remaining allowed area separately. Earlier room polygons own any overlap; both area and resolution limits still apply. Rejoin shared portals and keep the same wall checks and wall-shifted edges. Cache both preview variants per dwelling. Room-aware is the default when the CLI heatmap is on; `[Ctrl+G]` compares the original, `[Shift+G]` hides the preview, and `[G]` continues to toggle the live routing grid. The live grid and solver are unchanged.
- Validation: nine changed Python files compiled and 611 tests passed. Added geometry checks for sloped boundaries, an unmodeled gap between rooms, non-overlapping cells, retained area limits, and connections across the partition. Pygame verified the overlay switch without rerouting. On the first real dwelling at 90°, the original/room-aware previews had 20/29 cells and 6/9 valid centers; 4/8 of those centers routed with the same Core Simple grid and `Steiner: min + fittings` backend. This is one dwelling and rotation, not a general routing advantage.

- Added an X at the CLI machine body center while the placement heatmap is on. It uses the same pose coordinates as the mask, independent of the supply extension. Pygame smoke checks confirmed the marker stays centered when rotated, follows movement, and is hidden with the heatmap off. Three Python files compiled; 12 focused frame/projection tests passed.

- Expanded CLI placement candidates to the Cartesian product of the cell-center X and Y coordinates. Both preview variants use unique axes and retain only intersections that pass the current rotation's footprint mask. Results are cached by mask and cells. Skeleton edges and routing stay unchanged; the overlay shows compact candidate rings and a count in the legend.
- Validation: four changed Python files compiled and 612 tests passed. The product test covers recombined coordinates, duplicate axes, and rejection by the mask. Pygame verified the denser preview, rotation and the comparison toggle. At 90°, the first real dwelling had 92 valid product candidates (66 green, 26 yellow) instead of 9 original cell centers; the second had 40 instead of 3. These counts measure footprint feasibility, not routing success.

- Moved the CLI supply root far enough along the centered outlet to meet the straight-run minimum and clear the machine body at a turn. The placement raster, automatic placement screen, and solver require the extension and at least one duct-width mitered turn envelope to fit. The outline shows the usable turn envelopes. Green still classifies the body alone after the exit passes. Cell Skeleton receives the usable turn endpoints as local attachment points.
- Validation: 614 tests passed, including rejection of a straight extension whose two turns are blocked, acceptance with one open side, and raster invalidation when the extension length changes. Headless Pygame rendered the real-dwelling CLI overlay and produced 635 valid raster tiles at 0° and 929 at 90°. The turn envelope is a demo geometry rule.

- Aligned plan-view CLI grille options with routing core's wall support, partial cover edge and horizontal return rules. A supply frame must fit its clipped span and its 400 mm routing anchor needs 240 mm of clearance inside the allowed region. Both fixed placement and directed routing use the same paired options. The directed graph adds the approved anchors and chooses one supply per room with its paired return. A room without a pair is reported as blocked. Synthetic full-coverage cases retain wall options where no cover records describe a soffit.
- Validation: 617 tests passed. Headless Pygame routed and rendered two supply/return pairs on the first real dwelling with Core Simple and Steiner min + fittings. Hanan + Cell also routed this pose. Cell Skeleton's hard-length solve was blocked at the same pose before and after the change. This step covers plan geometry. Core elevation and grille frame checks require separate validation.

- Added CLI virtual-source placement as a separate `P` mode. The cached Hanan + Cell Skeleton supplies candidate centers. The current rotation's raster and exact body/exit checks screen the centers. Private, outgoing-only outlet states enter one shared Core Simple hard-length Steiner search over pose-independent cover-aware supply choices. Ranking uses the complete grille set at a single outlet. Up to five ranked poses are verified with the actual body and ordered-installation obstacles. A verified placement selects Core Simple, Wall Candidates and Steiner min + fittings for the live route. Press `R` to run at the next rotation, and `V` to see the hybrid grid and raster candidates.
- Validation: 619 tests passed. The solver test rejects a virtual source whose separate exits can each serve only part of the demand. Headless Pygame exercised `P`, `R`, and the hybrid overlay. On four real dwellings, the experiment ranked 51–106 outlets in roughly 270–655 ms per run, including verification and a cold hybrid-grid build. The first-ranked outlet verified on all four. The shared score covers supply routing at the current rotation. The pose-specific verification checks planar centerlines against the selected body and obstacles.

- Separated CLI placement controls. `P` selects the mode, `A` runs it at the displayed angle, and `R` reruns Virtual source at the next angle. Pressing `A` from Manual selects Virtual source. The legacy workflow also honors the current angle on an explicit run. Headless Pygame confirmed different verified Virtual source positions at 90° and 180°, and a legacy placement at 270°. All 620 tests passed.

- Added a 0–500 mm yellow-placement penalty slider to CLI Virtual source, defaulting to 0. The candidate's virtual-source edge carries the penalty once when its raster tile is yellow. The slider reruns placement on release at the current rotation, and the reset button restores 0. Solution logs retain the value. Headless Pygame verified the 500 mm endpoint, single rerun on release, and reset. The ranking regression checks that only yellow candidates pay the selected amount.

- Raised the CLI yellow-placement slider limit to 1500 mm. The placement message reports the winning zone, its applied penalty and the resulting ranking score. At 90° on the first dwelling, the best green score was 4000 mm above the best yellow score. On dwelling `01_01_P04_V06` at 180°, dragging from 0 to 1500 mm and releasing switched the verified placement from yellow to green. All 621 tests passed.

- Fixed the yellow-placement slider hit area to include its knob at both endpoints. On `01_01_P04_V06` at 180°, a headless Pygame test grabbed the knob at 1500 mm, dragged to 0 and back, and verified the machine moved between the green and yellow positions after each release. All 622 tests passed.

- The CLI Virtual source sidebar shows the score gap between the top green and yellow candidates at the selected penalty. The gap updates during the drag, and placement reruns on release. On `A_A1_P00_V01` at 90°, the live event test reduced the yellow lead from 5058 mm to 3558 mm and kept the winning pose at (7737, 4418) mm. All 622 tests passed.

- Raised the yellow-placement penalty limit to 6000 mm. On `A_A1_P00_V01` at 90°, dragging from 1500 to 6000 mm and releasing moved the machine from the yellow pose at (7737, 4418) mm to the verified green pose at (8400, 4418) mm and routed all three supply grilles. All 622 tests passed.

- CLI Virtual source now uses the shared graph to shortlist placements and compares the first verified supply route in each zone using the Workbench Score. It tries up to five poses per zone and adds the slider value once to the yellow pose. On `A_A1_P00_V01` at 90°, the verified scores were 22,206 for yellow and 22,681 for green. The solver switched at 475 mm, and a Pygame slider test moved the machine at 509 mm. All 622 tests passed.

- Decide later whether collision policy remains top-level `COLLISIONS` or moves under a shared domain area.
- Decide later whether Demo 10.8.1 should move to a standalone folder such as `Documents/interactive-routing-app`.
- Obtain concrete Cli/Coc/San behavior only when ready to merge those demos.
- Add a machine-catalog loader only when the refactor needs to consume the `Familias` metadata rather than the current hardcoded Sal machine spec.
- Add a local-testing config adapter only when a task explicitly asks to source values from `global_config.json`.
