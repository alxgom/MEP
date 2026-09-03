# MEP Ventilation Router Dashboard (Demo 10.78)

Interactive Pygame prototype for routing ventilation ducts from a compact ventilation machine to room terminals and shafts. Demo 10.78 starts from the Demo 10.75 routing-backend refactor and adds an experimental two-stage min-cost flow strategy for assigning and routing duct pins without enumerating room permutations.

## Run

Use the existing Demo 10.7 virtual environment while 10.78 is still a refactor copy:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.78"
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
- `Space`: generate a new random apartment layout.

Routing and display:

- `C`: cycle solver strategy.
- `L`: cycle routing backend between state-expanded A*, line graph `L(G)` A*, and line graph `L(G)` GBFS.
- `Tab`: cycle routing grid.
- `G`: toggle grid nodes and edges.
- `V`: toggle the placement heatmap.
- `H`: toggle heatmap scale between linear and logarithmic.
- `B`: toggle heatmap palette between Turbo and Viridis.
- `W`: toggle placement scoring weights.

## Current Solvers

The strategy selector currently includes:

- Greedy (Dual-Sort)
- First Fit
- Best Fit
- Negotiated Congestion
- Negotiated Congestion (Favour Large)
- Min-Cost Flow (Small Pins)
- Min-Cost Flow (Two-Stage)

The sequential strategies route the shaft first, then the kitchen, then the remaining room ducts according to the strategy ordering or permutation. Best Fit evaluates permutations using the same final score terms used for reporting, rather than choosing by length only.

The min-cost flow strategies are experimental:

- Min-Cost Flow (Small Pins): keeps the normal shaft route and normal kitchen route, then solves only the small duct rooms jointly with min-cost flow. This preserves the previous baseline for comparing whether joint small-pin assignment is useful.
- Min-Cost Flow (Two-Stage): evaluates two global stage orders and keeps the lower final score.

The two-stage strategy compares:

- Big-first: route the shaft and kitchen as large ducts, then route the small ducts jointly against the selected large routes.
- Small-first: route the small ducts jointly first, then route the shaft and kitchen against those selected small routes.

The large-duct stage tries both large-pin assignments, shaft-left/kitchen-right and shaft-right/kitchen-left, and both large-route orders. The second route in each large candidate receives diameter-aware interaction weights from the first route. For interactivity it reuses the current shaft entry candidate instead of testing every shaft boundary node.

The small-duct stage routes each room as one unit of flow and each small pin accepts one unit, so it can assign pins and paths without trying every room order.

Successful two-stage status text reports the winning global stage order, large-pin assignment, and large-route order. This is intentionally diagnostic so route quality can be compared while the strategy remains experimental.

Costs include physical edge weights and bend penalties. The large-duct order test and the small-duct stage inject already chosen routes as size-aware interaction weights, so large/large and large/small clearance and crossing penalties are present without building full Shapely duct buffers in the solve loop.

This is not a full multi-commodity optimizer. It is intended as a timing and route-quality experiment for the pin assignment/order problem, not yet as a replacement for the A* best-fit score.

Future min-cost-flow variants worth testing:

- Layered graph by pipe size: build separate large-duct and small-duct graph layers, with shared capacity or penalty links between layers. This would make diameter costs explicit during a single solve, but the graph would be larger and tuning shared-layer penalties may be delicate.
- Iterative min-cost flow: solve all ducts, measure overlaps/crossings/clearance conflicts, increase penalties on the offending edges, and re-solve a small number of times. This is close to negotiated congestion but avoids explicit room permutations.

The router backend selector is independent of strategy:

- State-expanded A*: routes with `(node, incoming_direction)` states.
- Line graph `L(G)` A*: routes with directed-edge states, so turns are modeled as transitions from one physical edge to the next.
- Line graph `L(G)` GBFS: uses the same super-sink target model and accumulated bend/crossing/clearance weights, but orders expansion by the line-graph heuristic only. This follows the high-speed Demo 08 variant and is faster but not optimal.

## Routing Grids

Demo 10.75 supports two grid modes:

- Regular 200 mm Grid
- Hannan Grid (numpy)

The Hannan grid follows a simple axis construction inspired by `letsmep-routing-core`: it builds X/Y axes from relevant room, obstacle, machine, and terminal geometry, then constructs axis-aligned graph nodes and edges. The current implementation keeps this numeric and vectorized where practical so grid rebuilds remain interactive.

## Machine Model

The default machine is based on the S&P Ozeo Flat family dimensions used for the current test scenario:

- Overall envelope: 511 x 460 mm
- Body envelope used for collision and display: 410 x 460 mm
- Large ducts: 125 mm
- Small ducts: 80 mm
- Clearance margin used by the lightweight routing penalty: 30 mm

Small duct pin behavior constrains exits to the allowed side directions with a short outside allowance. Large duct connections enter perpendicular to the machine side.

## Route Interaction Model

Sequential routing keeps the super-sink connection model, but applies route interaction weights while each subsequent route is solved:

- Same-axis duct overlap is effectively blocked.
- Perpendicular crossings are allowed but penalized.
- Near parallel or adjacent segments inside the diameter-clearance band are penalized.
- Actual perpendicular crossings are not counted again as clearance conflicts.

The interaction weights are applied during A* traversal and are also reflected in the final score. This keeps route selection and score reporting aligned for Dual-Sort, First Fit, and Best Fit.

The clearance check is intentionally lightweight. It uses axis-aligned segment records and NumPy masks against candidate grid edges instead of building Shapely duct buffers inside the interactive routing loop.

## Plots

The right dashboard tracks:

- Duct length
- Cost score
- Turns
- Turns per metre
- Solver time in milliseconds

Solver time measures the routing solve path, not full frame rendering or UI drawing.

Each plot highlights the best observed minimum and labels the current value as the percentage above that minimum. This makes the dashboard show the current penalty relative to the best state found during the interaction.

Turn counts use graph and pin-access directions. Regular-grid diagonal snap artifacts at pins are ignored, while the large shaft connector is counted because it represents a real visible connection.

## Notes

Negotiated congestion modes are still separate experimental strategies. They do not yet share every sequential scoring detail from the Dual-Sort, First Fit, and Best Fit path.
