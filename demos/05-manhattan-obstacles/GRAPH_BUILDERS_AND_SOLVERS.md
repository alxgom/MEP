# Graph Builders & Solvers — Reference Manual

> **Demo 05 — Manhattan Steiner with Obstacles**
> All code lives in `demos/05-manhattan-obstacles/`.

---

## Architecture Overview

The system is split into two independent layers. Any solver works with any environment.

```
┌──────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│         ENVIRONMENT (Layer 1)        │    │            SOLVER (Layer 2)              │
│                                      │    │                                          │
│  GridEnvironment         (Dense)     │──▶ │  solve_mst()              Baseline       │
│  EscapeGraphEnvironment  (Sparse)    │──▶ │  solve_kou()              Classical      │
│                                      │    │  solve_greedy()           Heuristic      │
│  Common interface:                   │    │  solve_fast_corner()      Heuristic      │
│    .nodes          np.ndarray        │    │  solve_prune()            Heuristic      │
│    .n_nodes        int               │    │  solve_stochastic_kou()   Stochastic     │
│    .node_map       dict              │    │  solve_anisotropic_kou()  Stochastic+A*  │
│    .dist_matrix    np.ndarray V×V    │    │  solve_monte_carlo()      Evolutionary   │
│    .predecessors   np.ndarray V×V    │    │                                          │
│    .adj_matrix     CSR sparse        │    │  All share _compute_geodesic_mst()       │
│    .get_path(u,v)  List[int]         │    │  as the common MST evaluation engine.    │
└──────────────────────────────────────┘    └──────────────────────────────────────────┘
```

The APSP (`dist_matrix` + `predecessors`) is computed **once at environment build time**.
All solvers do O(1) distance lookups from this table. No Dijkstra runs at solve time —
*except* `solve_anisotropic_kou()`, which uses NetworkX A* on-demand per trial.

---

## Layer 1: Graph Builders (environment.py)

### 1A. `GridEnvironment` — Dense Hanan Grid

**Theory:** Hanan's Theorem (1966). The optimal Rectilinear Steiner Minimal Tree (RSMT)
has all Steiner points at intersections of horizontal/vertical lines through terminal
and obstacle-corner coordinates.

**Construction:**
1. Collect X-coords: terminal X ∪ obstacle left/right edges
2. Collect Y-coords: terminal Y ∪ obstacle top/bottom edges
3. Cross product → all Hanan intersections
4. Remove nodes strictly inside obstacles
5. Connect horizontally and vertically adjacent nodes; skip edges that pierce obstacles
6. Run SciPy APSP (Dijkstra from every node)

**Node count:** O(|T| × |O|) — grows quadratically with obstacles.

**When to use:** Small to medium problems; when you want the densest possible
candidate graph (maximum Steiner search coverage).

```python
env = GridEnvironment(terminals, obstacles)
# env.n_nodes ≈ |T|² to |T|×|O| depending on layout
```

---

### 1B. `EscapeGraphEnvironment` — Sparse Escape Graph

**Theory:** From Blokland (2023) / VLSI routing literature. Only nodes where routing
*decisions* happen need to be represented: obstacle corners, terminals, and the points
where orthogonal rays from those locations terminate or intersect.

**Construction (5 steps):**
1. **Interest Points** = terminals ∪ all obstacle corners
2. **Ray Projection**: from every interest point, cast 4 orthogonal rays (N/S/E/W).
   Each ray stops at the first obstacle face or the padded bounding box (±80 units).
3. **Intersection Discovery**: find all T-intersections between horizontal and vertical
   ray segments (brute-force O(|rays|²), fast in practice because rays are few).
4. **Node Set** = ray endpoints ∪ intersection points ∪ interest points,
   filtered to exclude nodes strictly inside obstacles.
5. **Edge Set**: connect consecutive nodes along each ray segment;
   midpoint-check blocks obstacle-piercing edges.

**Node count:** O(|T| + |O|) — sparse, grows linearly in typical layouts.

**Expected advantage:** ≥40% node reduction → ≥2× APSP speedup vs. Hanan Grid
on dense obstacle maps. *(Target from topology_faceoff.py benchmark.)*

**When to use:** Large problems; when APSP build time is the bottleneck;
foundation for Demo 07 multi-net routing.

```python
env = EscapeGraphEnvironment(terminals, obstacles)
# env.n_nodes << GridEnvironment.n_nodes for same inputs
```

---

### Shared Interface

Both environments expose an identical API so solvers are environment-agnostic:

| Attribute | Type | Description |
|---|---|---|
| `nodes` | `np.ndarray [V, 2]` | (x, y) coordinates of all graph nodes |
| `n_nodes` | `int` | Total node count V |
| `node_map` | `dict (x,y)→int` | Fast coordinate-to-index lookup |
| `terminal_node_indices` | `List[int]` | Indices of terminal nodes in `nodes` |
| `dist_matrix` | `np.ndarray [V,V]` | Precomputed geodesic distances (APSP) |
| `predecessors` | `np.ndarray [V,V]` | Predecessor matrix for path backtracing |
| `adj_matrix` | `csr_matrix [V,V]` | Sparse weighted adjacency matrix |
| `get_path(u, v)` | `List[int]` | Backtrace shortest path via predecessors |

---

## Layer 2: Solvers (solver.py)

All solvers are methods of `ObstacleSteinerSolver`. The core engine is
`_compute_geodesic_mst(node_indices)` which slices the APSP dist_matrix to the
active node set and runs SciPy's `minimum_spanning_tree`.

```python
solver = ObstacleSteinerSolver(env)
result = solver.solve_kou()
# result = {"weight": float, "steiner_indices": List[int], "segments": List[Tuple[int,int]]}
```

---

### Solver 1: `solve_mst()` — Geodesic MST Baseline

**Also known as:** Shortest Path Heuristic (SPH)

**What it does:**
1. Computes MST on the terminal-only submatrix of `dist_matrix`
2. Expands each MST edge back to its actual grid path for drawing

**What it does NOT do:** Add any Steiner junction nodes.

**Relation to KMB:** This is KMB steps 1–3 without the re-MST (step 4) or
pruning (step 5). Path expansion is for display only — intermediate nodes are
not fed back into the optimizer.

**Quality:** Always obstacle-aware (uses geodesic distances). Best possible
tree if junctions are only allowed at terminal locations.

**Complexity:** O(|T|²) lookups + O(|T| log |T|) Kruskal.

**Visualizer key:** `1`

---

### Solver 2: `solve_kou()` — Kou-Markowsky-Berman (KMB '81)

**Approximation ratio:** ≤ 2(1 − 1/|S|) of optimal Steiner tree (proven).

**The 5 steps:**

```
Step 1+2:  MST on terminal metric closure (geodesic distances)
             → |T|-1 edges, each weighted by obstacle-aware distance

Step 3:    Expand each MST edge to actual grid path
             → harvest all intermediate grid nodes as Steiner candidates

Step 4:    Re-MST on terminals ∪ candidates (TRUE distances)
             → eliminates cycles that arise when expanded paths share nodes

Step 5:    Prune: iteratively remove non-terminal degree-1 leaves
             → clean up dangling segments
```

**Why step 4 matters:** If two terminal-pair paths share an intermediate node J,
the expanded graph has a cycle through J. The re-MST collapses this to the
optimal sub-tree through J, potentially saving that shared segment.

**Complexity:** O(|T|²) + O(|T| log |T|) + O(|T| × path_len) + re-MST.

**Visualizer key:** `6`

---

### Solver 3: `solve_greedy()` — Iterated 1-Steiner

**Strategy:** Bottom-up. Evaluate every non-terminal grid node as a candidate
Steiner point. Add the one with maximum weight reduction. Repeat.

**Candidate search:** O(V) nodes per round × `max_steiner` rounds = O(V × S) MST evaluations.

**Quality:** Typically best deterministic quality, but slowest on large grids.

**Complexity:** O(V × max_steiner × |T|²).

**Visualizer key:** `2`

---

### Solver 4: `solve_fast_corner()` — L-Bend Corner Heuristic

**Strategy:** Only tests L-bend corners of current MST edges — the two axis-aligned
corner nodes connecting any pair of non-collinear terminals in the MST.

```
T1 ──────── T2           Corner candidates:
|                           C1 = (T1.x, T2.y)
T3                          C2 = (T2.x, T1.y)
```

**Why it works:** On Hanan grids, the optimal Steiner junction for a pair of terminals
is almost always at an L-bend corner. This exploits the geometric structure instead
of searching blindly.

**Stochastic variant:** `stochastic=True` → randomly picks from Top-3 gain candidates
instead of always taking the best. Provides cheap stochastic escape.

**Complexity:** O(|MST_edges| × max_steiner × |T|²) — much faster than solve_greedy.

**Visualizer key:** `4`

---

### Solver 5: `solve_prune()` — Dense Grid Pruning (Top-Down)

**Strategy:** Start with ALL grid nodes in the MST (the densest possible Steiner tree),
then iteratively remove non-terminal nodes with degree ≤ 2 (passing-through nodes
that contribute no branching).

**Why degree ≤ 2:** A node on a "trunk" (degree 2) is worth keeping only if its
geodesic path is shorter than the direct path between its two MST neighbors.
The iterative pruning naturally discovers this.

**When it wins:** Finds Steiner junctions that bottom-up methods miss because it
considers ALL possible junction combinations simultaneously.

**Complexity:** O(V²) initial MST + O(V × pruning_rounds).

**Visualizer key:** `3`

---

### Solver 6: `solve_stochastic_kou()` — Stochastic KMB

**Parameters:** `n_trials=20`, `perturbation=0.10`

**Core idea:** The KMB local minimum trap: on Hanan grids, many paths between
terminal pairs have identical cost. Dijkstra always picks the same one (by index order).
These equal-cost paths have different shapes → different intermediate nodes → different
Steiner candidates. By perturbing the terminal metric closure, we change which MST
topology is chosen in step 2, which paths are expanded in step 3, and ultimately which
intermediate nodes the re-MST in step 4 can use.

**Implementation:**
```python
noise = Uniform(1-p, 1+p, shape=(|T|, |T|))
noise = (noise + noise.T) / 2          # keep symmetric
perturbed_dist = base_terminal_dist * noise
# MST on perturbed → different topology → different path expansion
# Re-MST + prune with TRUE distances → evaluate fairly
# Keep best across n_trials
```

**Key invariant:** Evaluation always uses TRUE geodesic distances. The perturbation
only affects which topological path the MST chooses; the scoring is unbiased.

**Complexity:** n_trials × O(KMB).

**Visualizer key:** `7`

---

### Solver 7: `solve_anisotropic_kou()` — Anisotropic KMB with NetworkX A*

**Parameters:** `w_x=1.0`, `w_y=2.0`, `n_trials=15`, `sigma=0.3`

**Core idea:** Replace KMB Step 3's Dijkstra path expansion with NetworkX A*
using a directionally-weighted heuristic:

```
h(u, v) = w_x × |Δx| + w_y × |Δy|
```

This biases A* toward different axis preferences. Even for terminal pairs with identical
geodesic distance, A* with different (w_x, w_y) traces physically different routes —
horizontal-biased vs vertical-biased — surfacing different intermediate Steiner nodes.

**When w_y > w_x:** A* penalises vertical movement → prefers horizontal runs.
This mirrors real MEP cost asymmetry (vertical risers cost more: floor penetrations,
support clamps, gravity constraints on drainage slopes).

**Admissibility:** The heuristic is admissible when w_x, w_y ≤ min edge cost.
Since edge costs in our grid are ≥ 1 pixel, setting w_x=1, w_y=2 is inadmissible
for the heuristic alone, but Step 1+2 use TRUE distances so the overall KMB
approximation bound still applies to the final solution quality.

**Stochastic mode (n_trials > 1, sigma > 0):**
Each trial samples (w_x ± σ, w_y ± σ), giving a different directional bias and
a different set of harvested intermediate nodes. The best result by TRUE weight is kept.
This combines topology diversity (like Stochastic KMB) with geometric diversity
(horizontal vs. vertical path shape variation).

**Why NetworkX A* instead of APSP predecessors:**
The precomputed predecessor matrix is tied to a single Dijkstra run. A* with a
custom heuristic can return a different (still valid) path of equal cost but different
shape. This is impossible with the APSP table without recomputing it.

**Complexity:** n_trials × (O(|T|²) APSP slice + O(|T| × A*_cost) + O(re-MST)).

**Visualizer key:** `8`

---

### Solver 8: `solve_monte_carlo()` — Evolutionary Population

**Strategy:** Genetic-style search over random Steiner node sets.

```
Population: N individuals, each = terminals + random grid nodes
Each generation:
  1. Evaluate all by MST weight
  2. Keep top-2 as elite parents
  3. Mutate: replace one random Steiner node with a random grid node
  4. Repeat for max_generations rounds
Winner: best individual + leaf pruning
```

**When it wins:** Highly degenerate topologies where geometric heuristics fail.
Acts as a sanity check — if solve_monte_carlo beats all geometric methods,
it suggests the search space has non-obvious structure.

**Complexity:** O(population × generations × |T+S|²).

**Visualizer key:** `5`

---

## Solver Comparison Table

| Solver | Key | Geometric Bias | Stochastic | Formal Guarantee | Speed |
|---|---|---|---|---|---|
| `solve_mst()` | 1 | None (APSP only) | ❌ | ≤ 2× MST | ⚡⚡⚡ |
| `solve_kou()` | 6 | Path expansion | ❌ | ≤ 2(1-1/\|S\|) | ⚡⚡ |
| `solve_greedy()` | 2 | None | ❌ | None | 🐢 |
| `solve_fast_corner()` | 4 | L-bend corners | ❌ | None | ⚡⚡ |
| `solve_fast_corner(stochastic)` | — | L-bend corners | ✅ Top-3 | None | ⚡⚡ |
| `solve_prune()` | 3 | None (top-down) | ❌ | None | 🐢 |
| `solve_stochastic_kou()` | 7 | Path expansion | ✅ Topology | Per-trial KMB | ⚡ |
| `solve_anisotropic_kou()` | 8 | Directional A* | ✅ Directional | Per-trial KMB | ⚡ |
| `solve_monte_carlo()` | 5 | None (random) | ✅ Full random | None | 🐢 |

---

## KPI Definitions

Every solver returns the same dict structure:

```python
{
    "weight":          float,        # Total path weight (sum of all segment lengths)
    "steiner_indices": List[int],    # Node indices of Steiner junction points
    "segments":        List[Tuple],  # (u_idx, v_idx) pairs of adjacent grid segments
}
```

**Saving %** displayed in the visualizer is always relative to `solve_mst()` (the baseline):
```
Saving = (MST_weight - Solver_weight) / MST_weight × 100%
```

---

## VLSI Connection

This entire stack is ported from VLSI (chip design) routing literature:

| VLSI Term | Our MEP Term |
|---|---|
| Net | Pipe network (group of terminals) |
| Gate pin | Terminal (sink/source fixture) |
| Blockage | Obstacle (beam, wall) |
| Metal layer | Floor / routing layer |
| Hanan Grid | Same — adopted directly |
| Escape Graph | Same — from VLSI papers |
| KMB Algorithm | Same — designed for VLSI in 1981 |
| Rip-up & Re-route | Demo 07 multi-net recovery |
| Global routing | Coarse path assignment |
| Detailed routing | Exact pipe placement |

The physics intuition: minimising total wire length on a chip ≡ minimising total
pipe length in an MEP system. Both are weighted Steiner tree problems on
rectilinear graphs with obstacle constraints.
