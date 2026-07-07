# MEP Ventilation Router Dashboard (Demo 10.79)

Interactive Pygame prototype for routing ventilation ducts from a compact ventilation machine to room starts, machine pins, and shafts. Demo 10.79 uses Demo 10.78 as its base, preserving the line-graph backends and min-cost-flow strategies, then adds real/synthetic dwelling source selection and room-node-set starts.

## Run

Use the existing Demo 10.7 virtual environment while 10.79 is still a refactor copy:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.79"
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
- `H`: toggle heatmap scale between linear and logarithmic.
- `B`: toggle heatmap palette between Turbo and Viridis.
- On-canvas `+`, `-`, and `1:1` buttons: zoom in, zoom out, and reset the drawing zoom.
- Hold `Shift` and left-drag, or hold the middle mouse button, to pan the zoomed view.
- On-canvas `Weights` button: toggle the modified-edge-weight overlay.
- On-canvas `Ruler` button: turn the pointer into a crosshair; click and hold on the canvas to measure a distance in mm. `Esc` exits ruler mode.
- Click a duct or room to highlight its route. Other routes keep their width but turn black; unrelated rooms and pins are desaturated. Click empty canvas, or press `Esc` when not in ruler mode, to clear selection.

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

## Room Starts

The default start mode is `Room node set`. Instead of snapping each wet room to a single centroid terminal, each route starts from a virtual source connected to every valid graph node inside that room and inside the false-ceiling region. This keeps the current room-to-machine routing direction but removes unnecessary dependence on a centroid point.

`Centroid terminal` is kept as a comparison mode. It uses the previous single nearest grid node for each terminal.

`ROUTING_WALL_CLEARANCE_MM` currently defaults to 100 mm. It insets candidate routing nodes from the false-ceiling boundary for both regular and Hannan grids; edge validation still uses the actual false-ceiling geometry.

## Routing Grids

Demo 10.79 supports two grid modes:

- Regular 200 mm Grid
- Hannan Grid (numpy)

The Hannan grid follows a simple axis construction inspired by `letsmep-routing-core`: it builds X/Y axes from relevant room, obstacle, machine, and terminal geometry, then constructs axis-aligned graph nodes and edges.

This is comparable to the routing-core simple grid at the level of "axis lines from terminals, allowed boundaries, obstacles, and connector geometry", but it is not a byte-for-byte port. The demo keeps a precomputed static Hannan template and overlays dynamic machine axes for interactivity; routing-core rebuilds a simpler grid from supplied points, allowed-boundary offsets, and obstacle-buffer axes.

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
- Proximity
- Topological Fields
- Routing-Core Workflow

The Routing-Core Workflow mode mirrors the core placement phase at demo scale: it generates candidates from machine-room centroids, optional +/-100 mm translations along routing-frame axes, and four connector-aligned rotations. Candidates are feasibility-filtered against room/obstacle geometry, then sorted by core-like machine metrics: percentage outside room, connector angular alignment to patinejo/kitchen, connector clearance to allowed boundaries, and distance to main targets. It does not run full routing for each candidate; this keeps the demo interactive while preserving the core placement workflow structure.

The placement heatmap is independent from the active placement mode. `[V]` displays the placement score field for Manual, Proximity, Topological Fields, and Routing-Core Workflow modes.


## Route Interaction Model

Sequential routing keeps the super-sink connection model, but applies route interaction weights while each subsequent route is solved:

- Same-axis duct overlap is effectively blocked.
- Perpendicular crossings are allowed but penalized.
- Near parallel or adjacent segments inside the core-like buffered-radius band are penalized.
- Actual perpendicular crossings are not counted again as clearance conflicts.
- Legacy sequential reservations only block exact used graph edges; they no longer block every edge adjacent to a routed path node.
- Negotiated congestion applies the same overlap, crossing, and buffered-clearance interaction layer to its currently negotiated paths, in addition to present/history congestion weights.

The interaction weights are applied during traversal and are reflected in the final score. Static geometry clearance is also applied as an edge field before routing:

- Walls/allowed-area exteriors are hard-blocked below `max(ROUTING_WALL_CLEARANCE_MM, buffered duct radius)`.
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
