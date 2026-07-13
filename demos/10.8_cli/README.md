# MEP Clima Placement Dashboard (Demo 10.8_cli)

Interactive Pygame prototype for exploring Clima machine, grille placement, and indoor supply-routing ideas. Demo 10.8_cli starts as an exact copy of Demo 10.8, then switches the first milestone to Clima: it derives supply/return grille targets from covered non-service rooms, places the machine, builds the routing grid, and routes a first indoor supply-air tree approximation from the machine `air_out` connector to impulsion grilles.

## Run

Use the existing Demo 10.7 virtual environment while 10.8_cli is still a refactor copy:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.8_cli"
..\10.7\.venv\Scripts\python.exe main.py
```

Or install the dependencies and run with another Python 3.10+ interpreter:

```powershell
pip install pygame numpy shapely scipy
python main.py
```

## Controls

Machine placement:

- Left mouse drag: move the ventilation machine.
- Mouse wheel while dragging or hovering the machine: rotate.
- `R`: rotate 90 degrees clockwise.
- `A`: cycle automatic machine placement modes.
- `P`: cycle placement mode.
- `W`: toggle placement scoring weights.

Dwelling and routing setup:

- `D`: toggle dwelling source between Real DB and Random Synthetic.
- `Space`: advance the active source. In Real DB mode this cycles the configured real scenarios; in Random Synthetic mode this generates a new random layout.
- `O`: cycle the real-dwelling routing frame.

Routing and display:

- `Tab`: cycle routing grid.
- `K`: cycle Clima supply backend.
- `G`: toggle grid nodes and edges.
- `J`: toggle routing-core parity debug overlay. Blue shows the raw Steiner tree with source-rooted direction arrows, yellow shows temporary terminal access links and bridge points, green shows connector stubs.
- `V`: toggle the placement heatmap.
- `M`: toggle the modified-edge-weight overlay.
- `H`: toggle heatmap scale between linear and logarithmic.
- `B`: toggle heatmap palette between Turbo and Viridis.
- `F11`: toggle fullscreen.
- On-canvas `+`, `-`, and `1:1` buttons: zoom in up to 6x, zoom out, and reset the drawing zoom.
- Hold `Shift` and use the mouse wheel to zoom at the cursor.
- Hold `Shift` and left-drag, or hold the middle mouse button, to pan the zoomed view.
- On-canvas `Weights` button: toggle the modified-edge-weight overlay.
- On-canvas `Ruler` button: turn the pointer into a crosshair; click and hold on the canvas to measure a distance in mm. `Esc` exits ruler mode.
- Right-side `Grille` tool: click inside a room to add the nearest valid room start node as a preferred grille-routing node; `Ctrl+click` removes an existing preferred point.
- Right-side `Grille area` tool: drag a rectangle to add every valid room start node inside the area; `Ctrl+drag` removes preferred points inside the area.
- Preferred areas are drawn as translucent filled regions with dotted borders. Nodes inside preferred areas are hollow; the node actually used by the routed solution is filled.
- Right-side `Log` button: store the current session-local routing state, add a marker to the KPI plots, and list the logged KPIs. Click a log row to restore that configuration and recompute the route.
- Click a duct or room to highlight its route. Other routes keep their width but turn black; unrelated rooms and pins are desaturated. Click empty canvas, or press `Esc` when not in ruler mode, to clear selection.
- The solver card includes sliders for `Min pieces factor`, bend weight, and crossing multiplier. Bend weight moves in 100-unit integer steps. The bend and crossing sliders include icon-only reset buttons.

## Current Solvers

The default Clima supply solver is `Routing-Core Port: Kou Steiner`. `solve_clima_routing()` connects the PEAD `air_out` connector to every `* Supply` grille target, then sends those graph terminal nodes through a lightweight port of the routing-core Steiner tree step: metric closure, Kou-style MST expansion, redundant-leaf pruning, collinear degree-2 simplification, and best-direction scoring. Return grilles remain visible metadata and refrigeration/common-area routing remains pending.

This is a routing-core behavior port of the tree algorithm, not a full route-core runtime clone. The app still uses its interactive NumPy graph builders and Shapely-free graph solve surface instead of routing-core's full dynamic grid/`Tramo`/`Conducto` pipeline. That split is intentional for this branch: the default lane should track core behavior where it has been explicitly ported, while keeping the visualizer responsive enough to test graph construction and placement changes.

The core-port adapter now uses routing-core's direction-aware terminal adjustment before graph construction: terminals within `EPSILON = 0.3 m` share a coordinate only when their connector vector permits motion on that axis. It then uses those adjusted Steiner terminals for the machine `air_out` access and grille offset points. The physical connector stubs remain attached to the original connector offsets. If an adjusted terminal is not already a graph node, the adapter adds temporary orthogonal graph links, including a validated one-bend bridge when needed. This mirrors core's behavior of rebuilding the grid around Steiner points and avoids misleading diagonal snap segments. The adapter keeps `min_value=0.0` because the current routing-core Clima call does not pass a nonzero `min_value` into `route_steiner`.

`Routing-Core Port: Simple Grid` is the corresponding NumPy port of routing-core's `_create_grid_dynamic_simple` input geometry. It uses exact supply Steiner terminals, the core 200 mm inset allowed region, inward allowed-boundary corner axes, and Cli shaft/column obstacles buffered from the active machine air-connector width with core `BUFFER_RATIO = 0.6`. It also ports the buffered machine body and connector obstacle polygons, including core's 20 mm terminal-side cut so a legal Steiner edge remains available. It retains the core behavior of making an invalid edge very expensive rather than removing it. The refrigeration-side obstacle uses `freon1`, matching routing-core's first `SupplyHydronic` connector selection; this should be revisited when refrigeration routing is implemented. The existing Hannan and epsilon grids remain separate interactive alternatives because they add scaffolding, wall shifts, and recovery links not present in the simple core grid.

The visualizer additionally hard-removes core-simple graph edges that cross the physical machine body. Routing-core represents all invalid simple-grid edges with `1e6` cost; retaining that fallback in an interactive preview can display an illegal route when no legal connection exists. The physical connector stub remains outside the graph, so this guard blocks only equipment traversal and reports a disconnected graph instead of drawing a duct through the machine.

Use `[J]` to inspect the core-port layers separately from the final route: exact terminals, temporary access links, bridge points, source-rooted directed tree, and connector stubs. This is a visual parity/debug aid only; it does not change the solver.

## Backend Comparison Harness

`compare_clima_backends.py` measures the existing routing path without opening the Pygame UI. By default it runs every configured real dwelling against every graph builder and all three supply backends, then writes CSV to standard output. It fixes placement to `Routing-Core Workflow` and uses routing frame `dominant_walls`, so each backend receives the same generated scenario and machine placement for a given graph.

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.8_cli"
..\10.7\.venv\Scripts\python.exe compare_clima_backends.py
```

Use indices to narrow a run while investigating a case. This example compares the two Kou variants on the second real dwelling with the Hannan graph (indices are zero-based):

```powershell
..\10.7\.venv\Scripts\python.exe compare_clima_backends.py --scenarios 1 --graphs 1 --backends 0,1
```

`--output path\to\report.csv` writes the same data to a CSV file. The report includes status, solve time, route length, turns, geometric validation counts, diagonal and outside-allowed counts, experimental short-piece count, and the core-port debug layer counts. The latter fields remain zero for the L(G) approximation because it does not generate the core-port overlay.

`Routing-Core Variant: Kou nonnegative heuristics` keeps the same core-port tree pipeline but removes the negative A* heuristic pairs from the 9-direction sweep, using only `(0,0)`, `(0,1)`, `(1,0)`, and `(1,1)`. It is intentionally labeled as a variant, not the core-parity default.

The previous line-graph metric-closure MST remains in code as `Core Approximation: L(G) MST` for comparison. It solves pairwise group paths on directed edge states `(u, v)` so turns are transition costs in `L(G)`, then unions the selected metric-closure paths into one route named `Supply Air Tree`. It is not the default core-port lane.

The visible app surface is now Cli-only. Inherited Sal strategy/backend/heuristic controls are hidden because they describe many-to-one Sal experiments, not the current supply-air tree problem. The first refactor keeps the old code internally while the Cli app stabilizes; later cleanup should move or delete those inactive Sal branches instead of maintaining them in this file.

## Heuristics and Constants

The main routing weights are:

| Name | Value | Use |
| --- | ---: | --- |
| `C_BEND` | `0-10000` mm slider, default `4000` | Added when a route changes direction. |
| `CROSSING_PENALTY` | multiplier slider, default `5 * C_BEND` | Added for allowed perpendicular duct crossings. |
| `CLEARANCE_PENALTY` | `CROSSING_PENALTY` | Added for route-route or machine soft-clearance violations. |
| `OVERLAP_BLOCK_WEIGHT` | `1e9` | Effective hard block for exact duct overlaps and impossible edges. |
| `OVERLAP_SCORE_PENALTY` | `50 * C_BEND` | Post-route score penalty for same-axis duct overlaps that still appear in fallback or simultaneous-routing results. |
| `SHORT_PIECE_SCORE_PENALTY` | `2 * C_BEND` | Post-route score penalty per short physical duct piece. |

The geometry and clearance parameters are:

| Name | Value | Use |
| --- | ---: | --- |
| `GRID_SPACING` | `200` mm | Regular grid spacing. |
| `HANNAN_SCAFFOLD_SPACING` | `600` mm | Static Hannan connectivity scaffold. |
| `CORE_EPSILON_GRID_MM` | `200` mm | Core-like epsilon axis offset for the experimental epsilon grid. |
| `WALL_THICKNESS` | `150` mm | Wall display/filter thickness. |
| `ROUTING_WALL_CLEARANCE_MM` | `100` mm | Minimum node/edge clearance from allowed-area boundaries and walls. |
| `PATINEJO_CLEARANCE_MM` | `200` mm | Hard clearance from shafts for non-shaft ducts. |
| `MACHINE_CLEARANCE_SOFT_MARGIN_MM` | `150` mm | Soft-cost band outside machine-body hard clearance. |
| `DUCT_BUFFER_RATIO` | `1.05` | Core-like radius inflation before clearance tests. |
| `MIN_DISTANCE_MACHINE_CONNECTOR_AIRE_MM` | `700` mm | Core `MIN_DISTANCE_MACHINE_CONNECTOR_AIRE`, used for the `air_out` and visible `air_in` connector stubs. |
| `MIN_DISTANCE_MACHINE_CONNECTOR_FRIGO_MM` | `25` mm | Core `MIN_DISTANCE_MACHINE_CONNECTOR_FRIGO`, used for the visible `Freon1`/`Freon2` connector stubs. |
| `MIN_SIZE_CONDUCTO_AIRE_MM` | `150` mm | Core `MIN_SIZE_CONDUCTO_AIRE`; a machine connector is split after its 200 mm first tract only when the remaining tract exceeds this value. |
| `TRANSLATION_MACHINE_MM` | `250` mm | Core `TRANSLATION_MACHINE`; placement candidates include the room centroid plus one offset in each routing-frame direction. |

The machine and connector parameters are derived from the indoor Clima family metadata:

| Name | Value | Use |
| --- | ---: | --- |
| `MACHINE_BODY_W` | selected PEAD type width | Display/collision body width. |
| `MACHINE_BODY_H` | `700` mm for PEAD | Display/collision body height. |
| `MACHINE_OVERALL_W` | selected PEAD type width | Kept equal to body width for the current indoor model. |
| `MACHINE_SMALL_DUCT_D` | selected PEAD air connector height, currently `178` mm | Active air-routing clearance diameter. |
| `MACHINE_LARGE_DUCT_D` | selected PEAD air connector height, currently `178` mm | Compatibility value for copied routing experiments. |

The interactive ranges are:

| Name | Value | Use |
| --- | ---: | --- |
| `MIN_PIECE_FACTOR_DEFAULT` | `1.05` | Default short-piece factor matching routing-core local config. |
| `MIN_PIECE_FACTOR_MIN` | `0.50` | Minimum slider value. |
| `MIN_PIECE_FACTOR_MAX` | `2.00` | Maximum slider value. |
| Zoom range | `0.5x` to `6.0x` | Canvas zoom clamp. |

The final cost score is currently:

```text
score =
    total_length_mm
  + C_BEND * turns
  + CROSSING_PENALTY * crossings
  + OVERLAP_SCORE_PENALTY * same_axis_overlaps
  + CLEARANCE_PENALTY * clearance_conflicts
  + SHORT_PIECE_SCORE_PENALTY * short_pieces
```

## Minimum Duct Pieces

Demo 10.8 reports and scores short duct pieces using the same core-style thresholds:

```text
grille segment minimum = diameter * factor
internal segment minimum = diameter * 2.0 * factor
```

The default factor is `1.05`, matching `FACTOR_CONDUCTO_MINIMA_LONGITUD` in the local routing-core config. The sidebar slider sweeps this factor from `0.50` to `2.00`.

The diameter-based short-piece count is an inherited demo experiment, not a routing-core Clima validity rule. Routing-core adds the grille and machine connector conductos, then `simplificar_tramo` removes only near-zero conductos below `TOLERANCIA_SIMPLIFICACION_TRAMO = 0.005 m`. The demo retains its merged physical-piece count for the L(G) approximation, but it is not scored or reported as a validation warning for either Routing-Core Port backend.

## Grille Preferences

The Cli app routes supply-air grilles from the graph node nearest to the placed grille point, rather than exposing the old Sal room-start selector. The node must still be a valid graph node inside the room and inside the false-ceiling region.

Preferred grille points override the automatic nearest-grille node. When a room has one or more preferred square markers, the route group uses only those mapped graph nodes; rooms without preferred markers use the nearest valid graph node to the generated grille point. Candidate room nodes are not drawn by default; use `G` to inspect graph nodes and edges when needed.

`ROUTING_WALL_CLEARANCE_MM` currently defaults to 100 mm. It insets candidate routing nodes from the false-ceiling boundary for both regular and Hannan grids; edge validation still uses the actual false-ceiling geometry.

## Routing Grids

Demo 10.8 supports three grid modes:

- Regular 200 mm Grid
- Hannan Grid (numpy)
- Epsilon Grid (core-like numpy)

The Hannan grid follows a simple axis construction inspired by `letsmep-routing-core`: it builds X/Y axes from relevant room, obstacle, machine, and terminal geometry, then constructs axis-aligned graph nodes and edges.

This is comparable to the routing-core simple grid at the level of "axis lines from terminals, allowed boundaries, obstacles, and connector geometry", but it is not a byte-for-byte port. The demo keeps a precomputed static Hannan template and overlays dynamic machine axes for interactivity; routing-core rebuilds a simpler grid from supplied points, allowed-boundary offsets, and obstacle-buffer axes.

The epsilon grid is an experimental routing-core-inspired alternative. It mirrors the core idea of adding axes at geometry/connector coordinates plus `+/- EPSILON`, but it deliberately keeps the demo's NumPy graph build and edge filtering instead of porting the core Shapely-heavy grid internals line-for-line. This keeps it interactive and makes the difference explicit.

## Validation Warnings

The status card shows warning-only route validation:

- crossings
- same-axis duct overlaps
- clearance conflicts
- short duct pieces
- segments outside allowed geometry
- missing routing-core shaft entry metadata

Overlap, clearance, crossing, and short-piece counts contribute to score where listed above, but warnings do not reject a route or trigger retries. Routing-core rejects some of these cases because it can keep trying alternate configurations; the demo keeps them visible because it is interactive.

## TODO: Core Connector Heuristics

The following routing-core connector-placement checks should remain TODOs until we decide how much should be hard feasibility versus placement/routing penalties:

- `MIN_DISTANCE_MACHINE_PATINEJO`
- `MIN_DISTANCE_MACHINE_BIG_PIPE`
- `MIN_DISTANCE_MACHINE_NORMAL_PIPE`
- `MIN_DISTANCE_CONECTOR_ALLOWED`
- `MIN_DISTANCE_CONECTOR_ESPACIO`

Some of these already exist in demo form as softer heuristics: pin projection lengths, machine clearance fields, allowed-boundary distance in Routing-Core Workflow placement, and diameter-specific route clearance. The open question is whether to expose them as warnings, placement penalties, or hard invalidation.

## Machine Model

The active machine list is the indoor Mitsubishi PEAD family exported under `Cli_UnidadInterior_Mitsubishi_PEAD_*`. ERST20D metadata exists, but it is not selectable in this demo because it has no rectangular HVAC `SupplyAir` connector; current routing-core air routing would reject it for this phase.

The canonical demo connector names are:

- `air_out`: exported `Supply Air1`, `DomainHvac`, `DuctSystemType=SupplyAir`, `Direction=Out`. This is the only air connector used as the source for the current core-like route-to-rooms phase.
- `air_in`: exported `Supply Air2`, `DomainHvac`, `DuctSystemType=SupplyAir`, `Direction=In`. It is drawn and kept as metadata, but return routing is not active because current core creates retorno grilles without routing a return duct tree in `RouteClimaLivingRouting`.
- `freon1` and `freon2`: exported refrigeration connectors, `DomainPiping`, `Direction=Bidirectional`, `PipeSystemType=SupplyHydronic`. They are drawn for the later machine-to-Cli-shaft refrigeration phase.

The family JSON connector offsets are insertion-relative. The demo still treats `machine_cx/machine_cy` as the body center because that is how the inherited interactive dragging and collision code works. To keep behavior stable, `get_machine_pins()` converts exported insertion-relative offsets into body-centered local coordinates before applying the current machine rotation. This is an intentional demo adaptation, not a core behavior change.

Routing-Core Workflow uses the first `SupplyHydronic` connector (`freon1` in the current family metadata), matching routing-core. Its final candidate ranking is the routing-core order: minimum sum of `air_out` connector distances to the centroids of all conditioned rooms, then maximum forward `air_out` clearance inside the selected room, then maximum forward refrigeration-side clearance inside that room. The two clearance rays use a NumPy segment-intersection port of routing-core's polygon-boundary calculation. Full routing is not rerun for every placement candidate; a candidate that later cannot route remains invalid in the interactive result rather than changing the placement sort.

Pipe rendering can be toggled from fixed-width strokes to real-diameter strokes with the on-canvas `Diam` button or `[X]`. In real-diameter mode, each route is drawn as `diameter_mm * current_px_per_mm`, so zooming changes the visible stroke width consistently with the plan scale.

The edge-weight overlay has a viewpoint toggle next to `Weights`. The `Small`/`Big` pill, or `[N]`, currently switches between the same selected PEAD air connector height because this first Clima pass has only one active indoor air-duct size. The field is recomputed against the current walls, shaft, machine, and routed ducts without rerunning the route solver.

Sidebar cards show only the main shortcuts. Use each card's `?` button to open a compact binding popup for the less common controls.

## Machine Placement Modes

`[P]` cycles through placement modes:

- Manual
- Routing-Core Workflow (default)

The Routing-Core Workflow mode mirrors the core placement phase at demo scale: it generates candidates from machine-room centroids, optional +/-250 mm translations along routing-frame axes, and four connector-aligned rotations. Candidates are feasibility-filtered against room/obstacle geometry and the routed `air_out` access point must be inside the core allowed region. Candidates are then sorted with routing-core's final Clima machine sort: connector-to-conditioned-room-centroid distance, air-side room clearance, and refrigeration-side room clearance. It does not rerun the full route for each candidate; this keeps the demo interactive while preserving the placement candidate and ranking behavior.

Routing-core selects pure Clima fancoil machine rooms from bathroom wet spaces (`InfoClima.get_optional_rooms_machine`). Combined Cli+Sal uses a wider bathroom/laundry set in `InfoClimaSalubridad`, but this Cli demo keeps the pure-Clima bathroom behavior unless exported metadata overrides the candidate flag.

The inherited Topological Fields placement mode was removed from the Cli demo because it optimizes the machine like a many-to-one hub. That is useful for Sal-style problems, but it is the wrong abstraction for the indoor Cli supply phase: the machine has one active ventilation output and should behave closer to a source/terminal for the supply tree. Placing it between grilles can block or bias the route in ways core routing would not intentionally select.

Cli supply grilles use their generated grille connector point to find nearby connected graph nodes. This intentionally bypasses the inherited Sal-style interior room-start filter, because Cli grilles are wall-adjacent by construction and can otherwise lose all valid start nodes before routing begins. The rendered supply tree includes direct connector tramos from the grille connector point to the selected graph node so the route remains visually attached to the terminal.

The demo no longer creates fallback supply/return terminals from room representative points when no valid core-like grille option exists. Those fallback points had no connector vector and could appear as disconnected route stubs. A room with no valid grille candidate is skipped for the current supply route instead of being routed from a fake terminal.

The routed supply tree now follows the core connector-point split at demo scale: the L(G) terminal is the grille Steiner point at `MIN_DISTANCE_REJA` from the grille connector along the transformed connector vector, the last `SIZE_FIRST_TRAMO_REJA` is kept as a separate grille-width connector tramo, and the machine air connector uses `MIN_DISTANCE_MACHINE_CONNECTOR_AIRE` with its final `SIZE_FIRST_TRAMO_MAQUINA_AIRE` split. This keeps the interactive geometry closer to `RouteClimaLivingRouting` while still drawing a single blue supply tree.

Route validation excludes only the physical segment between a supply grille and its inward Steiner point from the `outside allowed` warning. A grille connector may intentionally start on the allowed-area boundary, while routing-core models the connector tramo into the room. Raw tree edges, terminal-access bridges, and machine stubs are still checked against the allowed geometry.

Grille and machine connector directions are also fed into the L(G) metric closure. For wall-mounted grilles, the placement code stores the room-inward normal only as placement metadata; routing uses the opposite connector vector, matching core's `reja.R @ connector.orientation` convention. Selected grille markers draw supply and return arrows with the same direction semantics used in core export images. The demo applies a finite bend-like penalty when the first/last graph edge does not match the connector vector. This differs from a hard validity filter on purpose: with the interactive Hannan/epsilon graph, a strict direction requirement can remove all connected metric-closure edges for otherwise routable cases.

## TODO: Outdoor and Common Areas

Outdoor units and common-area routing are intentionally out of scope for this indoor placement/routing pass. Future interactive work should add outdoor unit selection, refrigerant pair matching by connector size/type rather than name, common-area shaft-to-outdoor routing, and multi-dwelling aggregation.

The placement heatmap is independent from the active placement mode. `[V]` displays the placement score field for Manual and Routing-Core Workflow modes.


## Route Interaction Model

Sequential routing keeps the super-sink connection model, but applies one canonical route interaction layer while each subsequent route is solved:

- Same-axis duct overlap is effectively blocked.
- Perpendicular crossings are allowed but penalized.
- Near parallel or adjacent segments inside the core-like buffered-radius band are penalized.
- Actual perpendicular crossings are not counted again as clearance conflicts.
- The old strict/relaxed sequential reservation pass has been removed; there is no separate `block_nodes` weight mode.
- Negotiated congestion applies the same overlap, crossing, and buffered-clearance interaction layer to its currently negotiated paths, in addition to present/history congestion weights.

The interaction weights are applied during traversal and are reflected in the final score. If a simultaneous or fallback strategy still produces a same-axis overlap, it is reported and receives `OVERLAP_SCORE_PENALTY`. Short-piece penalties are post-route score terms only. Static geometry clearance is also applied as an edge field before routing:

- Terminal candidates are disconnected from their virtual room source when they are inside either the 100 mm regulation terminal clearance or `BUFFER_ROOM_TERMINALES_AIRE_MM`.
- Walls/allowed-area exteriors are hard-blocked below the buffered duct radius.
- Patinejo/shaft clearance is hard-blocked below `max(PATINEJO_CLEARANCE_MM, buffered duct radius)` for non-shaft ducts. The current value is 200 mm, matching `BUFFER_ALLOWED_PATINEJO = 0.2 m` from the local routing-core config.
- The shaft route itself is exempt from the patinejo clearance field so it can enter the shaft.

Machine-body clearance is added as a diameter-specific edge field: edges inside the buffered duct radius are blocked and edges in the nearby soft band receive an added cost. Min-cost-flow variants use the same bend and interaction constants where they inject previously selected routes into a later stage.

The duct clearance model follows the routing-core style rather than the previous fixed maintenance margin:

```text
buffered_radius_mm = ceil(diameter_mm / 2 * 1.05)
required_clearance_mm = buffered_radius_a + buffered_radius_b
```

## Plots

The right dashboard tracks duct length, cost score, turns, turns per metre, and solver time in milliseconds. Solver time measures the routing solve path, not full frame rendering or UI drawing.

Each plot highlights the best observed minimum and labels the current value as the percentage above that minimum.

Manual solution logs are drawn as diamond markers with `L1`, `L2`, etc. Automatic best-metric logs replace their previous entry when a new minimum is found for score, length, turns, crossings, short pieces, or solver time.

## Low-Priority TODO

- Refactor UI option names into provenance groups: Routing-Core Port, Core Approximation, Baseline, and Experimental. Keep the core-port lane available as the default while preserving graph-builder and heuristic knobs for comparison.
- Add soft preference semantics where non-preferred room starts remain available with a penalty.
- Add optional export for session logs if visual exploration later needs CSV/JSON output.
- Revisit routing-core alignment after the demo placement workflow stabilizes.
- Scope machine placement and rotation changes separately from the current terminal-preference work.
- Add a NumPy-only detector for edges parallel to walls that run inside wall gaps or wall cavities.
- Add click-to-enlarge behavior for dashboard plots, opening the selected plot in a popup.
