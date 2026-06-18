# Demo 08 Instructions: Bend-Aware Non-Orthogonal Routing

You are the Implementation Agent (coder) for **Demo 08: Bend-Aware Non-Orthogonal Routing**. Your task is to implement the environment, graph builder, pathfinder, and solvers in `demos/08-bend-aware-non-orthogonal/`.

## 🛠️ Required Files

You must implement the following three files in `demos/08-bend-aware-non-orthogonal/`:
1.  `environment.py`: Geometry representations, generalized orthogonal ray tracing, graph construction.
2.  `solver.py`: Bend-aware state-expanded $A^*$, native `BendAwareKMB` solver, and `FastCornerWithCleanup` post-processing.
3.  `main.py`: A script to generate a non-square room with rotated obstacles, run both solvers, and print length/turn metrics.

---

## 1. Environment Specifications (`environment.py`)

### 1.1 Geometry Representation
*   Use `shapely.Polygon` for the room (e.g., L-shaped: `Polygon([(0,0), (1000,0), (1000,500), (500,500), (500,1000), (0,1000)])`).
*   Use `shapely.Polygon` for obstacles (e.g., rotated rectangle: rotate a polygon by $30^\circ$ using `shapely.affinity.rotate`).
*   Verify that any routing path segment `shapely.LineString([p1, p2])` lies entirely inside the room and does not intersect any obstacles.

### 1.2 Generalized Orthogonal Grid Builder
To generate a grid that conforms to arbitrary obstacles:
1.  **Collect Interest Points:** 
    *   All terminals.
    *   All vertices of the room polygon and all obstacle polygons.
2.  **Ray Tracing:**
    *   For each interest point $(x, y)$, project rays horizontally ($+x, -x$) and vertically ($+y, -y$).
    *   Use Shapely's `intersection` or `boundary` checks to find where each ray first hits the room outer wall or an obstacle face.
    *   Collect these ray termination coordinates.
3.  **Intersections:**
    *   Find all intersection points between the horizontal and vertical rays.
4.  **Nodes:**
    *   The node set of the graph consists of all unique interest points, ray termination points, and ray-ray intersections that lie within the free space (room minus obstacles).
5.  **Edges:**
    *   Connect adjacent nodes along the horizontal and vertical ray segments, checking that the segment `shapely.LineString([node_a, node_b])` is fully contained in the free space.

---

## 2. Solver Specifications (`solver.py`)

### 2.1 State-Expanded $A^*$ Pathfinder
Implement an $A^*$ search where states are tuples: `(node_index, incoming_direction)`.
*   **Directions:** `N`, `S`, `E`, `W`, or `None` (for starting node).
*   **Transitions:**
    *   From state `(u, dir_1)` to neighbor `v` (direction `dir_2` from $u$ to $v$):
    *   If `dir_1 != dir_2` and `dir_1 is not None`, the edge cost is `distance + C_bend`.
    *   Otherwise, it is just `distance`.
*   **Heuristic:** Manhattan distance to target + estimated turn penalty to align with target.

### 2.2 Native `BendAwareKMB` Solver
1.  Compute the turn-penalized shortest path between all pairs of terminals.
2.  Build the metric closure.
3.  Compute the MST on the metric closure.
4.  Expand MST edges back to grid paths, deduplicate overlap segments, and prune leaves.

### 2.3 `FastCornerWithCleanup` Solver
1.  Solve routing using standard metric closure KMB or a fast corner heuristic (pure distance-based).
2.  Apply **Post-Processing Turn Cleanup**:
    *   Loop through the tree segments and identify elbows (degree-2 turns).
    *   Attempt to shift the coordinate line of one segment to align it with an adjacent segment (merging two turns into zero or one).
    *   Use Shapely to verify that the shifted segments remain in free space.
    *   Commit shifts that reduce turn count without violating obstacle bounds.

### 2.4 Explicit Directed Dual Graph Solver (`BendAwareDualGraphKMBSolver`)
1.  **Dual Graph Construction**: Represent each directed edge in $G$ as a state `(u, v)`. Build static transitions `(u, v) -> (v, w)` with cost `dist(v, w) + C_bend` if `direction(u, v) != direction(v, w)`, else `dist(v, w)`.
2.  **Dual Graph A* Pathfinder**: Run A* starting with all states `(start, neighbor)` and terminating at any state `(neighbor, target)`. Use the turn-minimizing Manhattan heuristic.
3.  **KMB Routing**: Build metric closure using the dual graph pathfinder, run Kruskal's MST, expand paths, and prune non-terminal leaves.

### 2.5 Explicit Directed Dual Graph GBFS Solver (`BendAwareDualGraphGBFSSolver`)
1.  **Dual Graph GBFS Pathfinder**: Run Greedy Best-First Search (GBFS) starting with all states `(start, neighbor)` and terminating at any state `(neighbor, target)`. Sort the priority queue purely by the heuristic estimate $h(n)$ (Manhattan distance + turn penalty estimate) to target, ignoring the cost $g(n)$ paid so far.
2.  **KMB Routing**: Build metric closure using this GBFS pathfinder, run Kruskal's MST, expand paths, and prune non-terminal leaves.

### 2.6 Dual Graph FastCorner Solver (`BendAwareDualGraphFastCornerSolver`)
1.  **FastCorner candidate evaluation**: Sweep all grid nodes as candidates.
2.  **Evaluation cost**: Run the KMB metric closure and MST cost estimation, using `dual_graph_astar` as the pathfinder.
3.  **Iteration**: Greedily insert candidates that minimize cost, up to `max_steiner` nodes.

### 2.7 Dual Graph GBFS FastCorner Solver (`BendAwareDualGraphGBFSFastCornerSolver`)
1.  Same as `BendAwareDualGraphFastCornerSolver`, but using `dual_graph_gbfs` as the pathfinder to construct the metric closure and estimate costs.

### 2.8 State-Expanded Sequential FastCorner Solver (`StateExpandedSequentialFastCornerSolver`)
1.  **Sequential tree growth**: Start with terminal 1. For each subsequent terminal:
    *   Find the shortest path from the terminal to any vertex currently in the active tree, using a multi-target version of `state_expanded_astar` (which stops when a popped state belongs to the active tree's vertices).
    *   Add the path's nodes/segments to the active tree.
2.  **Refinement**: Identify candidates that can bridge tree branches at lower length + turn cost, inserting them as Steiner points.

### 2.9 Dual Graph Sequential FastCorner Solver (`DualGraphSequentialFastCornerSolver`)
1.  Same as `StateExpandedSequentialFastCornerSolver`, but routing terminals sequentially using a multi-target version of `dual_graph_astar` to search from the terminal to the active tree states.

---

## 3. Main Runner (`main.py`)
Write a demonstration script that:
1.  Initializes a **Winding Corridor Room**:
    `room = Polygon([(0, 0), (1200, 0), (1200, 400), (800, 400), (800, 800), (1200, 800), (1200, 1200), (0, 1200), (0, 800), (400, 800), (400, 400), (0, 400)])`
2.  Places **7 rotated obstacles** along the corridors:
    - Obstacle 1: Center `(200, 200)`, size `(100, 60)`, rotated `15` degrees.
    - Obstacle 2: Center `(600, 200)`, size `(80, 120)`, rotated `-30` degrees.
    - Obstacle 3: Center `(1000, 200)`, size `(60, 60)`, rotated `45` degrees.
    - Obstacle 4: Center `(1000, 1000)`, size `(120, 80)`, rotated `60` degrees.
    - Obstacle 5: Center `(600, 1000)`, size `(80, 80)`, rotated `-15` degrees.
    - Obstacle 6: Center `(200, 1000)`, size `(60, 100)`, rotated `75` degrees.
    - Obstacle 7: Center `(600, 600)`, size `(100, 100)`, rotated `30` degrees.
3.  Generates **6 terminal points**:
    - `(100, 100)`
    - `(1100, 100)`
    - `(100, 1100)`
    - `(1100, 1100)`
    - `(600, 500)`
    - `(600, 700)`
4.  Runs all KMB, FastCorner, and Sequential solvers (excluding `BendAwarePruneSolver` from the automatic sweep to avoid timeouts).
5.  Prints a comparative table of **Total Length**, **Turn Count**, and **Runtime** for each configuration to demonstrate the Pareto frontier.

