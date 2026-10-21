# MEP Ventilation Router Dashboard (Demo 10.8)

Interactive Pygame prototype for routing ventilation ducts from a compact ventilation machine to room starts, machine pins, and shafts. Demo 10.8 uses Demo 10.79 as its base, preserving the line-graph backends, min-cost-flow strategies, real/synthetic dwelling source selection, and room-node-set starts.

See [SOLVERS.md](SOLVERS.md) for a comparative guide to every current solver:
what it does, how it works, why it exists, its objective and its limitations.

## Run

Use the existing Demo 10.7 virtual environment while 10.8 is still a refactor copy:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.8.1"
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
- The solver card includes sliders for `Min pieces factor` (Sal), `Hard min` as a trunk-width multiplier (Cli hard-length modes), bend weight, and crossing multiplier. Cli's multiplier ranges from 0.25x to 3.00x in 0.05x steps; the label also shows the resulting mm minimum, and reset restores 1.00x. Bend weight moves in 100-unit integer steps. The bend and crossing sliders include icon-only reset buttons.

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

The baseline backends handle short pieces post-route, matching routing-core's current validation phase. The demo merges consecutive collinear graph edges into physical duct pieces before counting, so the metric is not inflated by grid discretization. The count contributes to `Total Cost Score` and is shown as `Short Pieces` in the KPI card. The experimental hard-minimum backends below enforce the same centerline criteria during search.

### Hard Minimum-Piece Experiment

With Sal active, select `First Fit` and choose either `State A* (hard min pieces)` or `Line graph A* (hard min pieces)` in the Router dropdown. `[L]` also cycles to these two additional choices. Existing backends and the default router remain available.

Both experiments accumulate physical length until a direction change, reject a turn that closes a short piece, and use the smaller endpoint minimum at the start/end. Length saturates at the largest relevant minimum; only nondominated cost/length arrivals are retained per state. State A* groups arrivals by node/direction, while line-graph A* also retains the incoming directed edge. Physical machine stubs, snap adapters and shaft-entry segments participate in the checks. These are centerline experiments, not fitting or fabrication-length checks, and no later sanitization is performed.

Thresholds follow the current machine catalog and `Min pieces factor` slider. The checked-in Ozeo Flat now has 80 mm small ducts and 125 mm large ducts: at factor 1.05, endpoint/internal minima are 84/168 mm and 131.25/262.5 mm respectively. Some older dimension tables in this README describe an earlier machine baseline; the catalog is authoritative.

Sequential and negotiated strategies use these backends. Min-cost-flow combinations explicitly report unsupported because flow stages bypass the selected A* backend. A failed search/order reports blocked; it does not establish that every possible joint installation configuration is infeasible. There is no automatic relaxation, new grid discretization or fixed time cutoff.

The runtime reuses numeric edge geometry within each scene solve, caches relaxed heuristics per arrival direction, and memoizes up to 64 exact weighted route requests across routing-order retries. These caches are discarded on the next scene solve. Saturation limits length-history distinctions, not absolute total memory.

Repeat a no-window comparison, including graph refresh, clearance weights and all Sal routes:

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_sal_min_pieces.py --cases 7 --repeats 3 --output sal_min_pieces_results.json
```

The benchmark uses exported cases when available, otherwise one synthetic scene. It compares both baseline A* representations with both hard-minimum variants at the same machine poses, records materialized short-piece counts, and reports median/p95/max scene latency. Drawing and KPI plotting are outside the measured scene solve.

## Experimental Cli Directed Steiner Minimum Lengths

With Cli active, the Router dropdown offers `Steiner: min bends` and
`Steiner: min bends + tees`, alongside `Directed Steiner arborescence (exact)`.
Selecting a directed router sets Starts to `Wall candidates`; `[L]` also cycles
through the choices. Keep the same machine pose, graph and bend weight when
comparing them. These experiments constrain the supply tree; optional return
routing retains the existing line-graph MST backend.

The default minimum for each solve is the calculated full-flow trunk width,
using the existing duct-sizing policy. PEAD M71JA currently gives 600 mm.
The `Hard min` slider multiplies it by 0.25–3.00x in 0.05x steps and displays
the resulting minimum in mm. Reset restores 1.00x. The setting always follows
the recalculated trunk width. It is
shared by all three hard-length variants and saved in solution logs; it does
not change duct sizing or the graph's costs. Sal's separate factor slider keeps
its existing diameter-based rules.
Machine/grille face widths do not determine this minimum. Models without
sourced airflow and pressure-loss inputs report unsupported. A graph that snaps
the machine access away from its physical point also reports unsupported;
Core Simple includes the exact access node used in the comparison benchmark.

The bends mode measures uninterrupted centerline runs along every root-to-grille
path. The tees mode also ends the incoming piece at each branch and starts every
outgoing piece afresh, including straight continuation. A grille takeoff on a
continuing trunk is a branch. Collinear graph subdivisions and duct-size changes
do not end a run. Physical machine and grille adapters count toward the lengths;
fitting deductions and fabrication lengths are outside this lab experiment.

Both modes use the baseline's weighted adjacency, bend costs and terminal costs.
Physical distances are separate from costs, and attained length saturates at the
minimum. Tee merges retain occupied departure directions and require distinct
physical branches, including through zero-length wall split edges. An independent
check validates the reconstructed and materialized physical tree before accepting it.

These are experimental state-expanded subset solvers. The chosen expanded-state
tree can still collapse into overlapping physical paths or an invalid physical
arborescence. Such candidates are reported as blocked; the solver does not
enumerate alternative trees after that rejection. **Blocked does not certify
infeasibility**, and these variants do not claim exact physical-tree optimality.
There is no fallback to unconstrained supply routing or length relaxation.

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_cli_hard_lengths.py --cases 7 --repeats 3 --output cli_hard_lengths_results.json
```

The benchmark measures Cli scene recomputation, including graph construction,
wall candidates, search and sizing, at identical poses. It excludes initial
dwelling preparation, automatic placement, rendering and history. Checked-in
`benchmarks/cli_hard_lengths_summary.json` records both the final run and a slower
repeat. Timings vary substantially with machine load; saturation does not impose
an absolute memory cap, and terminal subsets still grow exponentially with room count.

### Fitting perturbation

`Steiner: min + fittings` uses the same hard minimum lengths at bends and tees,
with the scalar search objective `existing cost + 1e-6 * physical fittings`.
Compare it with `Steiner: min bends + tees` at the same pose. The sidebar shows
the total and its elbow/tee/cross breakdown; each component counts as one.
Reducers, straight couplings and machine/grille connections do not add fittings.
A grille takeoff on a continuing trunk counts as a junction.

The solver reuses its existing candidate-specific direction masks and retains
one scalar per label. Elbow charges are prepared with transitions, zero-length
wall aliases carry no physical charge, and a completed branch receives one
tee/cross charge irrespective of its number of binary subset merges. Branch
summaries exclude their shared origin, preventing elbow-plus-tee double counting.
Final physical fitting counts are computed independently from the reconstructed
centerlines, including machine and grille adapters. Diagnostics recover the
primary search score by subtracting the counted perturbation, subject to floating
point roundoff. The workbench KPI scoring formula retains its existing semantics.

This is a near-tie preference, not strict lexicographic ordering: a sufficiently
small primary-cost difference can be outweighed by fewer fittings. The weight is
fixed in `mep_routing/installations/cli/fittings.py`. It is a lab preference,
not a pressure-loss or fabrication-cost model. The existing physical-union
rejection limitations also apply to this variant.

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_cli_hard_lengths.py --cases 7 --repeats 3 --backends 6 7 --output cli_fittings_results.json
```

The recorded comparison is in `benchmarks/cli_fittings_summary.json`. The real
scenes had matching fitting counts for both modes; a controlled regression with
equal primary cost demonstrates the intended two-elbows-plus-tee to one-elbow-plus-tee
choice. No additional routing states are introduced.

## Room Starts

The default start mode is `Room node set`. Instead of snapping each wet room to a single centroid terminal, each route starts from a virtual source connected to every valid graph node inside that room and inside the false-ceiling region. This keeps the current room-to-machine routing direction but removes unnecessary dependence on a centroid point.

Preferred terminals keep the same virtual-source abstraction. When a room has one or more preferred square markers, the virtual source connects only to those mapped graph nodes with zero cost; the other room nodes are disconnected for that route. Rooms without preferred markers keep the full room-node set. Candidate room nodes are not drawn by default; use `G` to inspect graph nodes and edges when needed.

`Centroid terminal` is kept as a comparison mode. It uses the previous single nearest grid node for each terminal.

`ROUTING_WALL_CLEARANCE_MM` currently defaults to 100 mm. It insets candidate routing nodes from the false-ceiling boundary for both regular and Hannan grids; edge validation still uses the actual false-ceiling geometry.

## Routing Grids

Demo 10.8.1 exposes eight grid modes for Sal and Cli:

- Regular 200 mm Grid
- Hannan Grid (numpy)
- Epsilon Grid (core-like numpy)
- Routing-Core Port: Simple Grid
- Medial Axis (sampled Voronoi)
- Cell Skeleton (orthogonal)
- Hanan + Voronoi (local links)
- Hanan + Cell Skeleton

The Hannan grid follows a simple axis construction inspired by `letsmep-routing-core`: it builds X/Y axes from relevant room, obstacle, machine, and terminal geometry, then constructs axis-aligned graph nodes and edges.

This is comparable to the routing-core simple grid at the level of "axis lines from terminals, allowed boundaries, obstacles, and connector geometry", but it is not a byte-for-byte port. The demo keeps a precomputed static Hannan template and overlays dynamic machine axes for interactivity; routing-core rebuilds a simpler grid from supplied points, allowed-boundary offsets, and obstacle-buffer axes.

The epsilon grid is an experimental routing-core-inspired alternative. It mirrors the core idea of adding axes at geometry/connector coordinates plus `+/- EPSILON`, but it deliberately keeps the demo's NumPy graph build and edge filtering instead of porting the core Shapely-heavy grid internals line-for-line. This keeps it interactive and makes the difference explicit.

### Medial-axis grid MVP

Choose `Medial Axis (sampled Voronoi)` in **Grid**, or cycle with `Tab`.
It generates candidate nodes and edges independently of the routing strategy
and graph/search representation. It approximates the free-space medial axis
using SciPy Voronoi ridges of sampled outer/hole boundaries, discarding ridges
between samples on the same boundary segment. This is not an exact segment-site
Voronoi construction or a raster skeleton.

`DOMAIN.SOLVER.GRAPH.MEDIAL_BOUNDARY_SAMPLE_SPACING_MM` defaults to 200 mm.
Degree-two chains are simplified with a tolerance of 5% of that spacing
(10 mm by default); junctions, endpoints and closed cycles remain. Every
replacement chord is checked against the allowed geometry and current wall
policy, and is an explicit straight edge with its own heading. Terminals,
machine access points and shaft faces attach to up to three nearby visible
skeleton projections (at most 24 segment candidates per endpoint). Missing
machine access points are not silently snapped elsewhere.

The machine body is a hole in this mode's free space, so changing its pose
regenerates detours. Up to 16 geometry/pose/spacing templates are cached for
repeated solves. Internal walls retain the existing short transverse-crossing
policy. Boundaries/obstacles follow the current inset routing region.

Sal's state-expanded, line-graph and minimum-piece backends accept arbitrary
headings, with a **fixed penalty for each actual direction change**. The
normalized-vector comparison uses routing-core's `flow_turn` tolerance (1e-4).
Angle-weighted costs are deferred; routing-core's quadratic angle score is not
enabled. This grid uses Euclidean target-distance heuristics, retaining `Zero`
for diagnostic searches, and displays the effective override in the sidebar.
The source grid is still selectable independently of strategy/backend.

Cli's two Kou variants use Euclidean distance in this grid, and its line-graph
metric MST accepts diagonal headings. Directed wall-candidate and hard Steiner
backends explicitly report unsupported for this MVP. Coc's grid choices are
unchanged. The existing `G` overlay draws the candidate skeleton and attachments.

Crossing detection and numeric route-interaction fields include diagonal
segments. Post-route overlap/clearance conflict counts still use the existing
orthogonal logic; the Sal warnings flag this limitation. Minimum-piece modes
can block on the skeleton's short segments. Success means a solver returned a
route on this approximate graph, not that every construction constraint passed.

Run a fixed-pose grid comparison with the demo interpreter:

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_medial_grid.py --cases 2 --repeats 2 --output benchmarks\medial_grid_results.json
```

The recorded run covers two real dwellings plus a seeded synthetic dwelling,
three poses, and both standard A* representations (36 warm samples per grid).
All three grids routed all sampled requests. Median scene times were
57.0/84.7/29.0 ms for Regular/Hannan/Medial; p95 was 890.7/256.7/470.9 ms.
Median node counts were 634/656/98. These are exploratory, machine-dependent
results including graph refresh and routing, excluding rendering and initial
dwelling preparation/automatic placement; they do not establish universal
speed or route-quality superiority.

### Cell-based orthogonal skeleton

Choose **Grid > Cell Skeleton (orthogonal)** for the experiment based on the
cell-to-skeleton construction in Figure 5(d) of the
[Blokland et al. survey](https://doi.org/10.1007/s43069-023-00208-5).
This is a demo adaptation of that principle; the figure does not prescribe
the particular decomposition and connection rules implemented here.
The separate sampled-Voronoi option remains available.

The builder partitions the inset free space using boundary coordinates, merges
adjacent rectangles, and retains internal-wall bounding bands as partition
features. Cells contain centre nodes; neighboring cells connect through the
midpoint of their shared boundary interval. Required elbow nodes make every
candidate edge horizontal or vertical. Corner-only cell contact does not create
a connection. The final assembly splits all segment crossings and shared spans
into explicit graph nodes/edges. Terminal, machine-access and shaft-face points
receive validated local L-shaped attachments, and the machine body is a hole.
Wall crossings retain the current short transverse-crossing filter.

`DOMAIN.SOLVER.GRAPH.CELL_MIN_AREA_MM2` defaults to **40,000 mm² (0.04 m²)**.
Mixed boundary cells stop before subdivision would create children below this
area. Fully covered initial rectangles can still merge; any remaining cell
below the area floor is excluded after merging. This also limits tiny cells
created by closely spaced wall-bound coordinates before recursion. Zero disables
the area floor, allowing exact rectangular coverage for rectilinear input.
Sloping/curved boundaries additionally stop at
`DOMAIN.SOLVER.GRAPH.CELL_BOUNDARY_RESOLUTION_MM` (default 200 mm).
The console reports excluded partition area, small-cell removals and endpoint
attachment counts. These limits can remove narrow passages from the cell layer.

The cell skeleton is enriched with the existing **Hanan wall-shifted axes**.
For each wall segment, its midpoint is offset to both normal sides by the
wall-clearance distance (100 mm default). The two points contribute rounded
integer X/Y coordinates, using the same shared generator as Hanan. Coordinates
merge under the existing 120 mm axis rule; their Cartesian intersections and
horizontal/vertical edges pass the cell mode's geometry/wall checks. Diagonal
walls contribute orthogonal axes. This wall layer uses only wall-shifted axes;
the other Hanan terminal/cover/scaffold axis sources remain part of Hanan mode.

Wall-grid nodes receive up to four local cardinal connections to nearby
cell-skeleton edges within one scaffold spacing (600 mm default). Target edges
and all crossings are split into shared nodes. The wall-side sampled runs from
the earlier experiment have been replaced by this Hanan axis construction.

A required point in an excluded boundary fragment may attach to a nearby cell
within one boundary resolution if a legal L shape exists. Still-isolated
cell centres and required points can attach locally to the combined wall/cell
grid. Invalid or
disconnected points receive no distant fallback. The wall layer can provide
legal candidate edges in areas excluded from the rectangular partition.

Press `G` to see pale cell outlines plus the centre/portal skeleton. Routing
strategy, backend and heuristic controls remain independent; this orthogonal
grid keeps the normal Manhattan/bend heuristics and fixed bend penalties.
Cli's directed wall backends can use this grid, although sparse candidate
locations may block hard-length modes. Geometry/pose/area-floor cell templates
have a 16-entry cache; the wall-axis layer and local links use the live geometry.
Exact access-point checks apply without enabling the Voronoi grid's
Euclidean heuristic override. Coc's choices are unchanged.

Compare all four experimental/baseline grids with:

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_medial_grid.py --cases 2 --repeats 1 --include-cells --output benchmarks\cell_hanan_axes_results.json
```

The recorded comparison has 18 warm requests per grid over two real dwellings,
seeded synthetic, three poses and both standard A* representations. The cell
grid routed all 12 real-dwelling requests; six synthetic shaft requests remained
blocked. Other grids routed all 18 requests. Physical warnings and failures
are retained. Cell-grid median/p95 scene times were 36.4/45.5 ms, with median
155 nodes. These timings include blocked requests and do not show a speed
advantage. At 18 samples the p95 is the slowest request. These machine-dependent
timings include graph refresh and routing, exclude preparation/rendering and
do not establish route-quality or latency superiority. The earlier cell-only
comparison remains in `benchmarks/cell_grid_results.json`, and the earlier
sampled-wall-run comparison in `benchmarks/cell_wall_grid_results.json`.
Neither describes the current wall-axis construction. Hard-length searches
can still block.

### Hanan + Cell Skeleton

Choose **Grid > Hanan + Cell Skeleton** to merge the full Hanan grid with the
current cell skeleton, including its Hanan wall-shifted axes. Both inputs remain
available as separate grid options. The merged mode retains their legal edge
geometry: crossings and attachments become shared nodes, and overlapping spans
are split and deduplicated. Cell nodes can also connect to nearby Hanan edges
through cardinal links within one scaffold spacing (600 mm default). Cell
coordinates do not add full-dwelling Hanan axes; even isolated cell centres
remain visible if no legal connection exists.

The cell layer keeps the confirmed **0.04 m²** minimum area and boundary
resolution. Machine geometry is excluded from both inputs; runtime port and
shaft constraints are applied once to the merged result. The grid remains
orthogonal, using normal Manhattan/bend heuristics and fixed bend costs in
both A* representations. Cli's directed and hard bends/tees/fittings backends
can use it. Subdividing a straight physical edge does not add a physical bend.
Press `G` to see the merged candidates and the cell outlines.

A bounded 16-entry cache reuses immutable unions for identical input nodes,
edges, allowed geometry, walls, wall policy and connection radius. Changing a
machine pose or terminal geometry changes those keys. Consumers receive their
own edge list. Initial and changed-pose preparation still performs the union;
warm timings do not describe every drag update.

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_medial_grid.py --cases 2 --repeats 1 --include-cells --include-hanan-cells --output benchmarks\hannan_cell_grid_results.json
```

The recorded run has 18 warm requests per grid: two real dwellings, seeded
synthetic, three poses and both standard A* representations. The merged mode
routed 18/18, including the six synthetic requests blocked on standalone Cell;
Hanan also routed 18/18. Warnings remain visible. Merged median/p95 scene times
were 114.2/531.8 ms, with median 935 nodes; Hanan was 85.5/264.0 ms with 656
nodes, and Cell 36.9/47.1 ms with 155 nodes including its failures. At 18 samples
p95 is the slowest request. Preparation of the displayed pose was about 238 ms
initially and 42 ms on reuse; its warm scene solve was about 111 ms. These
machine-dependent observations do not establish route-quality superiority or
uniform interaction latency. The denser graph remains an explicit experiment.

### Hanan + Voronoi with local links

Choose **Grid > Hanan + Voronoi (local links)** to retain the Hanan grid and
Voronoi diagonals in one connected candidate grid. The standalone Hannan,
Voronoi and cell-skeleton modes remain available. Grid construction remains
independent of routing strategy and graph/search representation.

The Hanan axes remain unchanged. Each Voronoi node may connect in up to four
cardinal directions to the nearest legal projection onto a Hanan edge, within
one Hanan scaffold spacing (600 mm by default). These are short local segments;
they do not extend an axis across the dwelling. The builder splits the target
edge at the attachment point. Natural intersections between the two grids,
and between their new links, are also split into shared junctions. Collinear
overlaps are split and deduplicated. Nodes retain the common float32 millimetre
representation so diagonal intersections can be shared without integer snapping.

Local links obey the allowed-region and short wall-crossing checks. No distant
fallback is added when a local connection is blocked or out of range. The
machine body is cut from both inputs. The sampled Voronoi geometry uses its
existing bounded cache; Hanan construction and the local union are rebuilt
with the current required points. Console diagnostics show skeleton-node,
local-link and total topology counts. Press `G` to inspect the merged grid.

Both normal A* representations retain fixed bend costs and use Euclidean target
distance (or the diagnostic Zero heuristic), as in the standalone Voronoi mode.
Cli's Kou variants and line-graph metric MST support the hybrid. Its directed
wall-candidate/hard Steiner backends explicitly report unsupported for grids
with diagonals. Existing diagonal overlap/clearance metric limitations apply.
Sampling and bounded attachment can still disconnect useful routing choices.

Run a fixed-pose comparison with:

```powershell
..\10.7\.venv\Scripts\python.exe benchmarks\benchmark_medial_grid.py --cases 2 --repeats 1 --include-hybrid --output benchmarks\hannan_medial_grid_results.json
```

The recorded run has 18 warm requests per grid over two real dwellings, seeded
synthetic, three poses and both standard A* representations. All requests
returned routes, retaining physical warnings. The hybrid median/p95 times were
177.4/956.0 ms, with median 1,064 nodes; Hannan was 85.7/257.2 ms with 656 nodes.
These machine-dependent observations include graph refresh and routing, not
rendering or initial preparation. With only 18 samples the reported p95 is
the slowest sample; no uniform interaction latency or route-quality benefit
is established.

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
- Add click-to-enlarge behavior for dashboard plots, opening the selected plot in a popup.
- Replace or augment centroid-field rotation with a graph/allowed-space-aware rotation heuristic, since local egress and surrounding routable space dominate machine orientation quality.
