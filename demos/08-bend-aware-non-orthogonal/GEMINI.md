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

---

## 3. Main Runner (`main.py`)
Write a demonstration script that:
1.  Initializes an L-shaped room.
2.  Places three rotated obstacles inside.
3.  Generates terminal points.
4.  Runs `BendAwareKMB` with varying $C_{\text{bend}}$ values (e.g., 0, 50, 200, 1000).
5.  Runs `FastCornerWithCleanup`.
6.  Prints a comparative table of **Total Length**, **Turn Count**, and **Runtime** for each configuration to demonstrate the Pareto frontier.
