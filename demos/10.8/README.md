# MEP Ventilation Router Dashboard (Demo 10.8)

Interactive Pygame prototype for routing ventilation ducts from a compact ventilation machine to room starts, machine pins, and shafts. Demo 10.8 uses Demo 10.79 as its base, preserving the line-graph backends, min-cost-flow strategies, real/synthetic dwelling source selection, and room-node-set starts.

## Run

Use the existing Demo 10.7 virtual environment while 10.8 is still a refactor copy:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.8"
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
- `T`: toggle room starts between Room node set and Centroid terminal.

Routing and display:

- `C`: cycle solver strategy.
- `L`: cycle routing backend between state-expanded A*, line graph `L(G)` A*, and line graph `L(G)` GBFS.
- `Y`: cycle A* heuristic mode.
- `Tab`: cycle routing grid.
- `G`: toggle grid nodes and edges.
- `V`: toggle the placement heatmap.
- `M`: toggle the modified-edge-weight overlay.
- `N`: toggle the modified-edge-weight view between small and big duct weights. The same control is also available as the on-canvas switch next to `Weights`.
- `H`: toggle heatmap scale between linear and logarithmic.
- `B`: toggle heatmap palette between Turbo and Viridis.
- `F11`: toggle fullscreen.
- On-canvas `+`, `-`, and `1:1` buttons: zoom in up to 6x, zoom out, and reset the drawing zoom.
- Hold `Shift` and use the mouse wheel to zoom at the cursor.
- Hold `Shift` and left-drag, or hold the middle mouse button, to pan the zoomed view.
- On-canvas `Weights` button: toggle the modified-edge-weight overlay.
- On-canvas `Ruler` button: turn the pointer into a crosshair; click and hold on the canvas to measure a distance in mm. `Esc` exits ruler mode.
- Right-side `Terminal` tool: click inside a room to add the nearest valid room start node as a preferred terminal; `Ctrl+click` removes an existing preferred terminal.
- Right-side `Term. area` tool: drag a rectangle to add every valid room start node inside the area; `Ctrl+drag` removes preferred terminals inside the area.
- Preferred areas are drawn as translucent filled regions with dotted borders. Nodes inside preferred areas are hollow; the node actually used by the routed solution is filled.
- Right-side `Log` button: store the current session-local routing state, add a marker to the KPI plots, and list the logged KPIs. Click a log row to restore that configuration and recompute the route.
- Click a duct or room to highlight its route. Other routes keep their width but turn black; unrelated rooms and pins are desaturated. Click empty canvas, or press `Esc` when not in ruler mode, to clear selection.
- The solver card includes sliders for `Min pieces factor`, bend weight, and crossing multiplier. Bend weight moves in 100-unit integer steps. The bend and crossing sliders include icon-only reset buttons.

## Current Solvers

The strategy selector includes:

- Greedy (Dual-Sort)
- First Fit
- Best Fit
- Negotiated Congestion
- Negotiated Congestion (Favour Large)
- Min-Cost Flow (Small Pins)
- Min-Cost Flow (Two-Stage)

The router backend selector is independent of strategy:

- State-expanded A*: routes with `(node, incoming_direction)` states.
- Line graph `L(G)` A*: routes with directed-edge states, so turns are modeled as transitions from one physical edge to the next.
- Line graph `L(G)` GBFS: uses the same super-sink target model and accumulated bend/crossing/clearance weights, but orders expansion by the line-graph heuristic only.

The heuristic selector is also independent of strategy where A* is used:

- Pin + bends: current baseline, using Manhattan distance to candidate pin access nodes plus estimated remaining bend penalty.
- Pin distance: Manhattan distance to candidate pin access nodes without bend estimation.
- Machine envelope: pin-agnostic lower bound using distance to the machine centre minus the maximum L1 radius of the active pin access candidates.
- Zero: Dijkstra-style diagnostic mode with no remaining-distance estimate.

The min-cost-flow strategies from Demo 10.78 remain available:

- Min-Cost Flow (Small Pins): keeps the normal shaft route and kitchen route, then solves only the small duct rooms jointly.
- Min-Cost Flow (Two-Stage): evaluates big-first and small-first stage orders and keeps the lower final score.

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

The machine and connector parameters are:

| Name | Value | Use |
| --- | ---: | --- |
| `MACHINE_BODY_W` | `410` mm | Display/collision body width. |
| `MACHINE_BODY_H` | `460` mm | Display/collision body height. |
| `MACHINE_OVERALL_W` | `511` mm | Connector envelope width. |
| `MACHINE_SMALL_DUCT_D` | `90` mm | Bathroom/toilet/washroom duct diameter. |
| `MACHINE_LARGE_DUCT_D` | `125` mm | Kitchen and shaft duct diameter. |
| `SMALL_PIN_STUB_LENGTH` | `100` mm | Small duct outside connector projection. |
| `LARGE_PIN_STUB_LENGTH` | `250` mm | Kitchen/shaft outside connector projection. |

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
terminal segment minimum = diameter * factor
internal segment minimum = diameter * 2.0 * factor
```

The default factor is `1.05`, matching `FACTOR_CONDUCTO_MINIMA_LONGITUD` in the local routing-core config. The sidebar slider sweeps this factor from `0.50` to `2.00`.

Short-piece handling is intentionally post-route, matching routing-core's current validation phase. The demo merges consecutive collinear graph edges into physical duct pieces before counting, so the metric is not inflated by grid discretization. The count contributes to `Total Cost Score` and is shown as `Short Pieces` in the KPI card, but it no longer expands A* search state. This keeps First Fit/Best Fit interactive while preserving comparability with routing-core validation.

## Room Starts

The default start mode is `Room node set`. Instead of snapping each wet room to a single centroid terminal, each route starts from a virtual source connected to every valid graph node inside that room and inside the false-ceiling region. This keeps the current room-to-machine routing direction but removes unnecessary dependence on a centroid point.

Preferred terminals keep the same virtual-source abstraction. When a room has one or more preferred square markers, the virtual source connects only to those mapped graph nodes with zero cost; the other room nodes are disconnected for that route. Rooms without preferred markers keep the full room-node set. Candidate room nodes are not drawn by default; use `G` to inspect graph nodes and edges when needed.

`Centroid terminal` is kept as a comparison mode. It uses the previous single nearest grid node for each terminal.

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

The default machine is based on the S&P Ozeo Flat family dimensions used for the current test scenario:

- Overall envelope: 511 x 460 mm
- Body envelope used for collision and display: 410 x 460 mm
- Large ducts: 125 mm
- Small ducts: 90 mm
- Small pin projection: 100 mm
- Large/kitchen/shaft pin projection: 250 mm
- Duct buffer ratio: 1.05, using integer `ceil(diameter / 2 * ratio)` buffered radii

Small duct pin behavior constrains exits to the allowed side directions with a short outside allowance. Large duct connections enter perpendicular to the machine side.

Pipe rendering can be toggled from fixed-width strokes to real-diameter strokes with the on-canvas `Diam` button or `[X]`. In real-diameter mode, each route is drawn as `diameter_mm * current_px_per_mm`, so zooming changes the visible stroke width consistently with the plan scale.

The edge-weight overlay has a viewpoint toggle next to `Weights`. The `Small`/`Big` pill, or `[N]`, switches the displayed field between a generic 90 mm duct and a generic 125 mm duct. The field is recomputed against the current walls, shaft, machine, and routed ducts without rerunning the route solver.

Sidebar cards show only the main shortcuts. Use each card's `?` button to open a compact binding popup for the less common controls.

## Machine Placement Modes

`[P]` cycles through placement modes:

- Manual
- Topological Fields
- Routing-Core Workflow (default)

The Routing-Core Workflow mode mirrors the core placement phase at demo scale: it generates candidates from machine-room centroids, optional +/-100 mm translations along routing-frame axes, and four connector-aligned rotations. Candidates are feasibility-filtered against room/obstacle geometry, then sorted by core-like machine metrics: percentage outside room, connector angular alignment to patinejo/kitchen, connector clearance to allowed boundaries, and distance to main targets. It does not run full routing for each candidate; this keeps the demo interactive while preserving the core placement workflow structure.

The placement heatmap is independent from the active placement mode. `[V]` displays the placement score field for Manual, Topological Fields, and Routing-Core Workflow modes.


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

- Add soft preference semantics where non-preferred room starts remain available with a penalty.
- Add optional export for session logs if visual exploration later needs CSV/JSON output.
- Revisit routing-core alignment after the demo placement workflow stabilizes.
- Scope machine placement and rotation changes separately from the current terminal-preference work.
- Add a NumPy-only detector for edges parallel to walls that run inside wall gaps or wall cavities.
