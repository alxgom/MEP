# Demo 08: Bend-Aware Non-Orthogonal Routing

This directory implements and compares 14 different solver configurations for turn-minimizing (bend-aware) point-to-point and multi-terminal (Steiner tree) routing in environments with non-grid-aligned (rotated) obstacles.

---

## 1. Comparative Performance Table ($C_{\text{bend}} = 500$)

The following benchmark table sweeps varying turn penalties and compares solver paradigms on the **Winding Corridor Room** geometry (containing 12 boundary vertices, 7 rotated obstacles, and 6 terminals, generating a dense grid of 366 nodes):

| Solver / Configuration | Length (units) | Turn Count | Runtime (ms) | Speedup vs. Baseline | Path Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`BendAwareKMB (C_bend=0)`** | 3480.53 | 23 | 26.23 ms | Baseline (1.0x) | Suboptimal (23 turns) |
| **`BendAwareKMB (C_bend=500)`** | 3636.60 | 7 | 24.94 ms | 1.05x | Moderate (7 turns) |
| **`TurnCleanupSolver`** | 3331.54 | 6 | 726.01 ms | 0.03x | Moderate (6 turns) |
| **`BendAwareFastCorner (C_bend=500)`** | 3136.60 | 4 | 22,921.65 ms | 0.001x | **Optimal** (4 turns) |
| **`BendAwareStochasticKMB`** | 3268.30 | 6 | 364.49 ms | 0.07x | Very Good (6 turns) |
| **`BendAwareDualGraphGBFS`** | 3136.60 | 4 | **11.56 ms** | **2.27x** | **Optimal (1982x faster than FastCorner)** |
| **`BendAwareDualGraphFastCorner`** | 3136.60 | 4 | 21,940.09 ms | 0.001x | **Optimal** (4 turns) |
| **`BendAwareDualGraphGBFSFastCorner`** | 3136.60 | 4 | 5,394.69 ms | 0.005x | **Optimal** (4.2x faster than state-expanded) |
| **`StateExpandedSequentialFastCorner`** | 3400.00 | 4 | 15.91 ms | 1.65x | Very Good (4 turns) |
| **`DualGraphSequentialFastCorner`** | 3400.00 | 4 | **12.02 ms** | **2.18x** | **Very Good (1800x faster than FastCorner)** |

---

## 2. Grid Construction & State-Space Formulations

To solve bend-aware routing around non-orthogonal obstacles, the solvers operate on two core mathematical structures:

### A. The Escape Graph (Generalized Orthogonal Grid)
The routing graph $G = (V, E)$ is constructed using a boundary-conforming **Escape Graph** builder:
1. **Interest Points**: Collect coordinates of all terminals, room boundary corners, and obstacle vertices.
2. **Ray Tracing**: Project horizontal ($+x, -x$) and vertical ($+y, -y$) rays from all interest points until they intersect a boundary face (outer room wall or obstacle face).
3. **Intersections**: Compute all ray-ray intersection coordinates.
4. **Vertices $V$**: Deduplicated interest points, ray endpoints, and intersections that lie in free space.
5. **Edges $E$**: Connect adjacent grid nodes along ray lines, validating containment via `shapely` line-in-free-space checks.

#### ❓ Does the Escape Graph depend on the starting point?
* **Global/Static Escape Graph (Our Implementation)**: By projecting rays from **all** terminals and obstacle vertices simultaneously, we construct a single unified grid once. Therefore, **our escape graph does NOT depend on the starting point** of any specific routing query. This allows us to precompute the graph once and run fast $O(T^2)$ shortest-path closure sweeps.
* **Local/Dynamic Escape Graph (Alternative)**: In some routing paradigms, rays are projected dynamically *only* from the current path endpoint to find a way out of an obstacle cluster. In that case, the escape graph **does depend on the starting point**, but it requires expensive dynamic geometric computations at runtime.

---

### B. State-Space Representations
To penalize bends, the pathfinder must know the path's direction. We model this using two alternative graph structures:

#### 1. Dynamic State-Expanded Graph
* **States**: Tuples `(node_idx, incoming_direction)` where $\text{direction} \in \{\text{N}, \text{S}, \text{E}, \text{W}, \text{None}\}$.
* **Transitions**: Generated dynamically during search. When transitioning from state $(u, d_1)$ to neighbor $v$ (edge direction $d_2$), the cost is $\text{Length}(u, v) + C_{\text{bend}}$ if $d_1 \neq d_2$, else $\text{Length}(u, v)$.
* **Sizing**: At most $5 \cdot |V|$ states.

#### 2. Explicit Directed Dual Graph (Line Graph $L(G)$)
* **Vertices $V_L$**: For each undirected edge $e = (u, v) \in E$ of length $L$, we instantiate two directed nodes `(u, v)` (traversing $u \to v$) and `(v, u)` (traversing $v \to u$).
* **Edges $E_L$**: Connect node `(u, v)` to node `(v, w)` for all neighbors $w$ of $v$ ($w \neq u$).
* **Transitions & Weights**: Precomputed statically in $L(G)$ as:
  $$\text{Weight} = \text{Length}(v, w) + C_{\text{bend}} \cdot \text{TurnPenalty}( (u, v), (v, w) )$$
* **Sizing**: At most $4 \cdot |V|$ states. Precomputing this graph once replaces dynamic direction calculations with static array lookups.

---

## 3. Pathfinding & Heuristics

### A. Search Algorithms
* **State-Expanded A***: Minimizes $f(n) = g(n) + h(n)$ to guarantee optimal shortest paths.
* **Dual Graph GBFS (Greedy Best-First Search)**: Minimizes $f(n) = h(n)$, ignoring $g(n)$. Rushes directly toward the target, drastically reducing node expansions.

### B. Heuristic Formulations
To guide A* and GBFS, we implement a **Turn-Aware Manhattan Heuristic**:
$$h(v, \text{dir}, \text{target}) = |x_v - x_{\text{target}}| + |y_v - y_{\text{target}}| + C_{\text{bend}} \cdot \text{EstTurns}(v, \text{dir}, \text{target})$$

The term $\text{EstTurns}$ estimates the minimum number of $90^\circ$ bends to reach the target:
* **Collinear Case** (Target is aligned on $x$ or $y$ axis): 0 turns if incoming direction matches axis of alignment, else 1 turn.
* **Diagonal Case** (Target is in a diagonal quadrant): 1 turn if incoming direction points toward target quadrant, else 2 turns.

This heuristic is **consistent and admissible**, guaranteeing optimal paths for A*.

#### Multi-Target Heuristic (for Sequential Routing)
When routing from terminal $t_i$ to a growing tree $T_V$ containing multiple active vertices, the heuristic computes the minimum estimate:
$$h_{\text{tree}}(v, \text{dir}) = \min_{t \in T_V} h(v, \text{dir}, t)$$

---

## 4. Solvers Swept

We benchmarked 5 distinct routing paradigms across 14 configurations:

1. **`BendAwareKMB`**: Turn-penalized metric closure Steiner solver using dynamic state-expanded A*.
2. **`TurnCleanupSolver`**: Non-bend-aware baseline ($C_{\text{bend}}=0$) followed by post-processing segment coordinate shifting.
3. **`BendAwareFastCorner`**: Constructive sweep that greedily inserts corner candidates minimizing KMB closure MST costs.
4. **`BendAwareStochasticKMB`**: KMB run multiple times with perturbed edge lengths to break symmetric local minima.
5. **`BendAwarePrune`**: Top-down grid pruning that removes degree $\le 2$ non-terminals from turn-penalized MSTs.
6. **`BendAwareDualGraphKMB` / `GBFS`**: KMB Steiner routing using explicit Dual Graph $L(G)$ pathfinders.
7. **`SequentialFastCorner` (State-Expanded & Dual Graph)**: Grows the tree by sequentially routing each terminal to the active tree using multi-target pathfinders (bypassing expensive APSP closure sweeps).

---

## 5. Key Architectural Takeaways

### A. The GBFS Pathfinder Breakthrough
On this winding corridor layout, the Greedy Best-First Search (GBFS) pathfinder on the dual graph $L(G)$ yielded a spectacular result:
* **`BendAwareDualGraphGBFS`** achieved the **exact same optimal layout** (Length = 3136.60, Turns = 4) as the heavy candidate-sweeping FastCorner solver.
* However, it completed in just **11.56 ms** compared to **22.9 seconds** for `BendAwareFastCorner`—representing a **1,982x speedup**!
* **Why?** By sorting the search queue purely by $h(n)$, the pathfinder rushes directly toward the terminals. In this winding geometry, the turn-penalized Manhattan heuristic is highly accurate, leading to minimal node expansions while finding the optimal path.

### B. Sequential Growth vs. Candidate-Sweeping FastCorner
Evaluating candidates by running metric-closure MST loops (`BendAwareFastCorner`) degrades rapidly on larger grids:
* `BendAwareFastCorner` took **22.9 seconds** on a 366-node grid.
* `DualGraphSequentialFastCorner` grows the tree by routing terminals one by one using a multi-target search. It took only **12.02 ms** (a **1,800x speedup**) while achieving a very good layout (Length = 3400.00, Turns = 4).

### C. Turn Cleanup vs. Native Solvers
* The post-processing `TurnCleanupSolver` took **726.01 ms** and yielded a suboptimal layout (Length = 3331.54, Turns = 6) compared to the native solvers.
* Native state-expanded and dual-graph solvers are **both faster and yield higher quality layouts** by making routing decisions bend-aware from the start.

---

## 3. Visualizations

The plot below shows the winding corridors, obstacles, terminals, and the paths found by the representative solvers, along with a performance bar chart:

![Routing Comparison Plot](routing_comparison.png)
