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
- Polygon iteration and largest-polygon helpers now live in `vent_router.geometry.polygons`.
- Ray casting and ray intersection helpers now live in `vent_router.geometry.rays`.
- Axis-aligned segment normalization/relation/distance helpers now live in `vent_router.geometry.axis`.
- Pure route segment merging and metric helpers now live in `vent_router.routing.segments`.
- Pure route quality counters now live in `vent_router.routing.metrics`, with Sal-specific diameter and minimum-piece policies injected by `main.py`.
- Route scoring and quality-summary formatting now live in `vent_router.routing.scoring`, with runtime weights and Sal policies passed in explicitly.
- Route length in metres now lives in `vent_router.routing.scoring`.
- Route hit testing now lives in `vent_router.routing.hit_testing`, with the UI zoom-derived hit radius supplied by `main.py`.
- Selected route pin detection now lives in `vent_router.routing.hit_testing`.
- Terminal validity classification now lives in `vent_router.routing.terminal_validity`, with room geometry accessors and clearances supplied by `main.py`.
- Min-cost-flow graph primitives now live in `vent_router.routing.flow`; higher-level Sal route construction remains in `main.py`.
- Min-cost-flow source normalization now lives in `vent_router.routing.flow`, with `grid_kd` supplied by `main.py`.
- Sal machine pin geometry, port access specs, and outward direction helpers now live in `vent_router.domain.machines`.
- Topological placement distance-field and score aggregation helpers now live in `vent_router.placement.fields`.
- Machine placement feasibility and candidate-room filtering now live in `vent_router.placement.feasibility`.
- Core-like machine placement scoring primitives now live in `vent_router.placement.scoring`.
- Machine rotation field scoring and field-alignment angle selection helpers now live in `vent_router.placement.rotation`.
- Topological auto-placement node/rotation selection now lives in `vent_router.placement.selection`.
- Port stub segment construction now lives in `vent_router.routing.segments`, with active graph nodes supplied by `main.py`.
- Route clearance math, route axis records, and weighted edge-cost lookup now live in `vent_router.routing.clearance`.
- Line-graph direction, path length, and target heuristic helpers now live in `vent_router.routing.search`.
- Terminal node index collection now lives in `vent_router.routing.search`, with terminals and KD-tree supplied by `main.py`.
- Heatmap and UI color-map helpers now live in `vent_router.ui.colors`.
- Graph axis collection, epsilon expansion, boundary extension, and value merging helpers now live in `vent_router.graphs.axes`.

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
- Extracted route scoring to `vent_router.routing.scoring`: score weights, total length, quality counts, quality warnings, and conflict summary formatting.
- Kept `main.py` as a compatibility adapter for active slider weights and Sal-specific policies.
- Added focused tests for length totals, score composition, warning formatting, and baseline conflict-summary text.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\scoring.py demos\10.8.1\tests\test_routing_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route hit testing to `vent_router.routing.hit_testing`.
- Kept `main.py` as the adapter for zoom-dependent hit radius.
- Added focused tests for nearest-route detection, misses outside radius, and route-name hit lookup.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\hit_testing.py demos\10.8.1\tests\test_routing_hit_testing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted route length in metres to `vent_router.routing.scoring.total_route_length_m`.
- Kept `main.py` KPI wrapper as an adapter.
- Added focused test for millimetre-to-metre conversion and empty routes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\scoring.py demos\10.8.1\tests\test_routing_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Created `vent_router.ui` for pure UI utilities.
- Extracted Turbo/Viridis palettes, heatmap palette selection, score-to-heatmap normalization, and cool colormap to `vent_router.ui.colors`.
- Kept `main.py` wrappers for active heatmap palette and scale mode.
- Added focused tests for color clamping, interpolation, palette selection, linear/log score scaling, and cool colormap behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\ui\__init__.py demos\10.8.1\vent_router\ui\colors.py demos\10.8.1\tests\test_ui_colors.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted Hannan/Epsilon grid axis value merging to `vent_router.graphs.axes.merge_close_values`.
- Kept `main.py` wrapper for existing grid builder call sites.
- Added focused tests for threshold merging, preserved axes, priority axes, and empty inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\graphs\__init__.py demos\10.8.1\vent_router\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted graph axis collection helpers to `vent_router.graphs.axes`: point axes, polygon vertex axes, and bounds axes.
- Kept `main.py` wrappers for current Hannan/Epsilon grid builder call sites.
- Added focused tests for rounded point axes, polygon exterior/interior vertices, and buffered bounds.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\graphs\__init__.py demos\10.8.1\vent_router\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted polygon iteration to `vent_router.geometry.polygons.iter_polygons`.
- Kept `main.py` wrapper for existing grid and drawing call sites.
- Added focused tests for polygons, multi-polygons, mixed geometry collections, `None`, and empty geometries.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\polygons.py demos\10.8.1\tests\test_polygons.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted largest-polygon selection to `vent_router.geometry.polygons.largest_polygon`.
- Kept `main.py` wrapper for allowed-boundary axis extension.
- Added focused tests for largest-area selection and no-polygon inputs.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\geometry\__init__.py demos\10.8.1\vent_router\geometry\polygons.py demos\10.8.1\tests\test_polygons.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted min-cost-flow primitives to `vent_router.routing.flow`: residual edge creation, successive-shortest-path min-cost flow, positive-flow edge lookup, and traced path reconstruction.
- Kept existing private helper names in `main.py` as imported aliases to avoid touching route-construction call sites.
- Added focused tests for lower-cost path choice, partial-flow behavior, traced state/target reconstruction, and incomplete trace handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\flow.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted min-cost-flow source normalization to `vent_router.routing.flow.source_start_nodes`.
- Kept `main.py` as the adapter for the active `grid_kd`.
- Added focused tests for explicit node indices, empty sources, and KD-tree coordinate lookup.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\flow.py demos\10.8.1\tests\test_routing_flow.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted machine pin geometry, port access specs, axis direction helpers, and outward-vector logic to `vent_router.domain.machines`.
- Kept `main.py` wrappers for `get_machine_pins`, `get_port_access_specs`, and `get_outward_vector`.
- Added focused tests for unrotated Sal pin geometry, port access stubs, and outward direction by side/rotation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\domain\__init__.py demos\10.8.1\vent_router\domain\machines.py demos\10.8.1\tests\test_machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted port stub segment construction to `vent_router.routing.segments.add_port_stub_segment`.
- Kept `main.py` as the adapter for active `current_env.nodes`.
- Added focused tests for direct pin stubs, access-point bridge stubs, and no-op handling for missing pins or nodes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\segments.py demos\10.8.1\tests\test_routing_segments.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted pure route clearance helpers to `vent_router.routing.clearance`: buffered radius, required clearance, route axis records, and weighted edge-cost lookup.
- Kept `main.py` wrappers for current Sal buffer ratio and route-diameter policy.
- Added focused tests for radius rounding, clearance sums, axis-record extraction, and weighted edge-cost fallback behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\clearance.py demos\10.8.1\tests\test_routing_clearance.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted line-graph direction, path physical length, and target heuristic to `vent_router.routing.search`.
- Kept `main.py` wrappers for active heuristic mode, machine center, and `estimate_turns`.
- Added focused tests for dominant-axis directions, Euclidean path length, Manhattan heuristic, bend-aware heuristic, machine-ring heuristic, disabled heuristic mode, and invalid nodes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\search.py demos\10.8.1\tests\test_routing_search.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted terminal-node index collection to `vent_router.routing.search.terminal_node_indices`.
- Kept `main.py` wrapper signature unchanged even though the old helper does not use `pin_node_map`.
- Added focused test for KD-tree terminal lookup and shaft-node preservation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\search.py demos\10.8.1\tests\test_routing_search.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Manual launch found a startup crash after machine geometry extraction: placement code still referenced `_local_axis_to_world`.
- Restored `main.py` compatibility import for `local_axis_to_world` and exported it from `vent_router.domain`.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\domain\__init__.py demos\10.8.1\vent_router\domain\machines.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Startup smoke: running Demo 10.8.1 with the 10.7 venv stayed alive until timeout instead of exiting with a traceback.
- Extracted selected route pin detection to `vent_router.routing.hit_testing.selected_pin_names`.
- Kept `main.py` wrapper unchanged for UI selection code.
- Added focused tests for selected route endpoint pins, last-three-segment matching, and empty context handling.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\hit_testing.py demos\10.8.1\tests\test_routing_hit_testing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted graph epsilon axis expansion and allowed-boundary interior axis extension to `vent_router.graphs.axes`.
- Kept `main.py` wrappers for current Hannan/Epsilon grid builder call sites.
- Added focused tests for epsilon point expansion, geometry-driven epsilon axes, empty boundary handling, and clustered interior axes.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\graphs\__init__.py demos\10.8.1\vent_router\graphs\axes.py demos\10.8.1\tests\test_graph_axes.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted reusable Pygame drawing primitives to `vent_router.ui.drawing`: geometry overlays, polygon hatch fill, dashed polylines, and outlined text.
- Kept `main.py` wrappers to inject current screen size, world-to-screen transform, and label halo color.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\ui\drawing.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted terminal validity entry/reason classification to `vent_router.routing.terminal_validity`.
- Kept `main.py` responsible for cache keys, room geometry accessors, active graph state, and UI drawing.
- Added focused tests for allowed nodes, clearance-blocked nodes, outside-room nodes, isolated nodes, no-boundary room behavior, and missing routing-region behavior.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\routing\__init__.py demos\10.8.1\vent_router\routing\terminal_validity.py demos\10.8.1\tests\test_terminal_validity.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Created `vent_router.placement` and extracted topological auto-placement distance fields, score aggregation, and current weight presets.
- Kept `main.py` responsible for active KD-tree lookup, mode selection, and machine-position side effects.
- Added focused tests for Dijkstra fields, multi-source fields, weight presets, and weighted placement score aggregation.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\fields.py demos\10.8.1\tests\test_placement_fields.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted core-like placement scoring primitives to `vent_router.placement.scoring`: candidate room points, machine polygon construction, room area-out percentage, target angle, and candidate score tuple.
- Kept `main.py` responsible for Sal machine pins, active room/shaft/kitchen state, allowed-boundary distance, and final machine-position mutation.
- Added focused tests for candidate points, polygon/area scoring, signed target angle, and the core-like score tuple.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\scoring.py demos\10.8.1\tests\test_placement_scoring.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted rotation field room selection, weight presets, pin direction mapping, and rotation field score aggregation to `vent_router.placement.rotation`.
- Kept `main.py` responsible for active machine pins, shaft target lookup, room target lookup, and final angle mutation.
- Added focused tests for large/small pin room ownership, weight modes, direction transform injection, and deterministic shaft alignment score.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\rotation.py demos\10.8.1\tests\test_placement_rotation.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted machine placement feasibility and candidate-room filtering to `vent_router.placement.feasibility`.
- Kept `main.py` responsible for active Sal pins, routing region, obstacle collections, and machine dimensions.
- Added focused tests for covered-room preference, fallback room selection, routing-region containment, and wall/column/shaft obstacle rejection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\feasibility.py demos\10.8.1\tests\test_placement_feasibility.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted topological auto-placement node/rotation selection to `vent_router.placement.selection`.
- Kept `main.py` responsible for active KD-tree lookup, machine pin generation, machine-position mutation, and grid rebuild.
- Added focused tests for pin-node lookup, rotation score policy, and first valid topological placement selection.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\selection.py demos\10.8.1\tests\test_placement_selection.py`.
- Validation: `python -m pytest demos\10.8.1\tests`.
- Extracted field-alignment rotation angle selection to `vent_router.placement.rotation.select_field_alignment_rotation`.
- Kept `main.py` responsible for applying the selected angle and updating UI-visible rotation scores.
- Added focused tests for retaining the current orientation and switching orientation when the score exceeds epsilon.
- Validation: `python -m py_compile demos\10.8.1\main.py demos\10.8.1\vent_router\placement\__init__.py demos\10.8.1\vent_router\placement\rotation.py demos\10.8.1\tests\test_placement_rotation.py`.
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
