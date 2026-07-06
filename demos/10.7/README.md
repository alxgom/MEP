# MEP Ventilation Router Dashboard (Demo 10.7)

Interactive Pygame prototype for routing ventilation ducts from a compact ventilation machine to room terminals and shafts. Demo 10.7 focuses on fast Hannan-style grid routing, visible route-cost heatmaps, sequential crossing penalties, and a lightweight diameter-clearance penalty.

## Run

Use the demo-local virtual environment when available:

```powershell
cd "C:\Users\ALEXIS GOMEL\Documents\mep_alexis_prehire\MEP\demos\10.7"
.\.venv\Scripts\python.exe main.py
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

The sequential strategies route the shaft first, then the kitchen, then the remaining room ducts according to the strategy ordering or permutation. Best Fit evaluates permutations using the same final score terms used for reporting, rather than choosing by length only.

## Routing Grids

Demo 10.7 supports two grid modes:

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

## Notes

Negotiated congestion modes are still separate experimental strategies. They do not yet share every sequential scoring detail from the Dual-Sort, First Fit, and Best Fit path.
