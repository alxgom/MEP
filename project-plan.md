2# Comprehensive Software Project Plan

## Hybrid Geometric Steiner Tree Solver & Architectural Piping Optimizer

---

## 1. Overview

This document consolidates all ideas from our multi-session discussion into a unified software development plan covering two closely related projects:

### Project A: Hybrid Geometric Steiner Tree Solver (2D)
A heuristic solver for the Euclidean Steiner Tree Problem using a variational approach — combining discrete topology search (MST) with continuous coordinate optimization (gradient descent + Simulated Annealing). The algorithm treats Steiner points as particles in a zero-length spring network, exploiting the physics analogy of energy minimization.

### Project B: Architectural Piping Optimizer (3D)
An extension of Project A into three-dimensional architectural environments, incorporating fluid dynamics (Darcy-Weisbach head loss), gravity constraints, collision detection with structural obstacles, and orthogonal (Manhattan) routing. Uses A* and Sequential A* on Hanan Grids for pathfinding within constrained building geometry.

Both projects share a common mathematical core: **graph-based optimization on spatial networks with physical constraints**.

---

## 2. Graph Theory Foundations (Common to Both Projects)

### 2.1 Graph Fundamentals
- **Graph** G = (V, E): vertices (points) and edges (connections).
- **Weighted graph**: each edge e ∈ E has weight w(e) ≥ 0 (Euclidean distance in 2D, or 3D Manhattan/euclidean distance in 3D).
- **Path**: sequence of edges connecting two vertices without repeating vertices.
- **Cycle**: a path that starts and ends at the same vertex.
- **Tree**: connected, acyclic graph with exactly |V| − 1 edges.
- **Connected graph**: path exists between every pair of vertices.

### 2.2 Key Problems
- **Minimum Spanning Tree (MST)**: Connect all vertices with minimum total weight. Polynomial time — O(E log E) via Kruskal's, O(E + V log V) via Prim's.
- **Shortest Path (A*/Dijkstra)**: Find minimum-cost path between two vertices in a weighted graph.
- **Steiner Tree Problem (STP)**: Connect a subset S ⊆ V (terminals) with minimum total weight, optionally introducing Steiner vertices V_s. **NP-hard**.
- **Rectilinear Steiner Tree (RST)**: Variant where all edges must be axis-aligned (horizontal/vertical). Relevant for architectural routing.

### 2.3 Formal Steiner Tree Definition
- **Input**: Graph G = (V, E), weights w: E → ℝ⁺, terminal set S ⊆ V.
- **Output**: Tree T = (V_T, E_T) such that S ⊆ V_T, minimizing W(T) = Σ_{e ∈ E_T} w(e).
- **Steiner vertices**: V_T \ S (auxiliary junctions that reduce total cost).
- **Bound**: At most |S| − 2 Steiner points are ever needed.
- **Approximation**: MST of terminals alone costs ≤ 2× the optimal Steiner tree.

---

## 3. Project A: Hybrid Geometric Steiner Tree Solver (2D)

### 3.1 Algorithm: Hybrid Variational Approach

**Core Insight**: Treat Steiner points as particles connected by zero-length springs (constant tension). The system's "energy" is the total tree length. Minimize this energy via gradient descent, with Simulated Annealing to escape local minima caused by MST topology changes.

#### Phase A: Initialization
1. **Input**: Terminal coordinates V_t = {(x₁, y₁), (x₂, y₂), ...}
2. **Initialize Steiner vertices** V_s: Place N_s = |V_t| − 2 random points within the convex hull of V_t.
3. **Hyperparameters**:
   - Initial temperature T₀ (e.g., 1.0)
   - Cooling rate α (e.g., 0.99 per iteration)
   - Minimum temperature T_min (e.g., 1e-6)
   - Learning rate η (e.g., 0.1)
   - Merge threshold ε (e.g., 1% of bounding box diagonal)
   - MST recalculation interval k (e.g., 10 steps)
   - Max iterations

#### Phase B: Optimization Loop

Repeat until convergence (ΔL < threshold) or T ≤ T_min:

**Step 1 — Topology Quench (Discrete)**
- Build complete weighted graph over V = V_t ∪ V_s.
- Edge weights = Euclidean distances between all pairs.
- Compute MST (Kruskal's with DSU) → edge set E_MST.
- **Optimization**: Recalculate MST only every k steps to save computation.

**Step 2 — Gradient Update (Continuous)**
- For each Steiner vertex vᵢ ∈ V_s:
  - Compute force (negative gradient) from MST neighbors:
    ```
    Fᵢ = Σ_{j ∈ neighbors(i)} (xⱼ − xᵢ) / ‖xⱼ − xᵢ‖
    ```
  - At equilibrium, Fᵢ = 0 → edges meet at 120° (Fermat point condition).
  - Update: `xᵢ ← xᵢ + η · Fᵢ`
- All gradient calculations are embarrassingly parallel.

**Step 3 — Simulated Annealing (Stochastic Exploration)**
- Propose random thermal displacement: `xᵢ ← xᵢ + 𝒩(0, σ² · T)`
- Compute ΔL = new_total_length − old_total_length.
- Metropolis criterion: ΔL < 0 → always accept; ΔL > 0 → accept with P = exp(−ΔL / T).
- Cool: `T ← α · T`
- Helps escape local minima at MST topology boundary "kinks."

**Step 4 — Annihilation / Merging (Pruning)**
- If dist(vᵢ, vⱼ) < ε for any two vertices: merge them, remove redundant vertex.
- Naturally optimizes Steiner point count as temperature drops.

**Step 5 — Convergence Check**
- If |L − L_prev| < convergence_threshold: stop.

### 3.2 Outputs
- Final Steiner point coordinates.
- Edge list of the Steiner Tree.
- Total tree length vs. MST comparison.
- Convergence trace (L vs. iteration).
- Visualization.

---

## 4. Project B: Architectural Piping Optimizer (3D)

### 4.1 Problem Definition
Given a 3D architectural model with structural obstacles (beams, columns, walls, ducts), connect fixture terminals (sinks, faucets, outlets) with piping that:
1. Minimizes material cost (pipe length + diameter sizing).
2. Maintains adequate pressure at all endpoints.
3. Respects gravity constraints (drainage slope ≥ 1-2%).
4. Avoids collisions with structural obstacles.
5. Prefers orthogonal routing (Manhattan/rectilinear geometry).

### 4.2 Mathematical Model

**Decision Variables**: Topology (E), junction coordinates (x,y,z), pipe diameters (Dᵢ), flow rates (Qᵢ).

**Objective — Total Annualized Cost**:
```
J = α·(Material Cost) + β·(Pumping Energy) + γ·(Space Occupancy)

Material Cost     = Σ Cₘ · Lᵢ · Dᵢᵃ
Pumping Energy    = Σ ΔPᵢ · Qᵢ / η
ΔP from Darcy-Weisbach: h_f = f · (L/D) · (v²/2g),  v = Q/A
Minor Losses:      h_L = K · (v²/2g)
```

**Physical Constraints**:
1. Mass Balance (Kirchhoff): Σ Q_in − Σ Q_out = 0 at every junction.
2. Energy Balance (Loop Law): Σ h_L = 0 in every closed loop.
3. Darcy-Weisbach head loss (non-convex, v² dependence).
4. Minor loss coefficients at fittings (90° bend K≈0.3–1.1, Y-junction K≈0.2).
5. Gravity: drainage pipes must maintain downward slope (DAG constraint).
6. Obstacle avoidance: pipes cannot intersect structural elements.

### 4.3 Algorithm Architecture

**Layer 1 — Topology Search**: Hanan Grid + Sequential A*
- Construct Hanan Grid from terminal coordinates + obstacle boundaries.
- Sequential A*: Connect terminals one-by-one to growing network using A* on the grid.
- A* heuristic: 3D Manhattan distance + turn penalty + obstacle proximity penalty.

**Layer 2 — Coordinate Optimization**: Variational approach from Project A
- Gradient descent on junction coordinates within fixed topology.
- Simulated Annealing for escaping local minima.
- Annihilation for pruning unnecessary junctions.
- Constraint: all points must avoid obstacles (repulsive potential).

**Layer 3 — Hydraulic Sizing**: Pipe diameter optimization
- Hardy Cross Method or Gradient Method for flow distribution.
- Economic diameter: balance material cost vs. pumping cost per segment.

### 4.4 A* vs. MST — The Key Distinction

| Feature | MST | Sequential A* |
|---------|-----|---------------|
| **Goal** | Minimize total network length (global) | Find valid path for each net (local) |
| **Obstacles** | Difficult to integrate | Naturally avoids obstacles |
| **Engineering Rules** | Ignores slopes, bend radii | Penalties for turns, slopes, wall proximity |
| **Complexity** | O(E log V) — fast | O(k · E log V) — moderate |
| **Node Ordering** | None (global) | High dependency (order matters) |

**A* is the industry choice for architectural routing** because buildings have obstacles, codes, and physical constraints. MST provides the theoretical minimum but cannot respect "no-go" zones.

**Solving the Net Ordering Problem**:
- **Critical Path First**: Route most constrained pipes first.
- **Shortest-to-Longest**: Start with closest terminal pairs.
- **Iterative Rip-up and Re-route**: If algorithm gets stuck, rip up an existing pipe, route the new one, then retry the ripped pipe. This is the industry gold standard.

### 4.5 Industry Context
| Aspect | Residential/Commercial | Industrial |
|--------|----------------------|------------|
| Primary Goal | Code compliance + space | Flow efficiency + cost |
| Math Level | Geometric (A*, Dijkstra) | Hydraulic (NLP/Gradient) |
| Tools | Revit, AutoCAD | AVEVA E3D, AFT Fathom |
| Steiner Points | Forced into 90° | Optimized 45°/90° mix |

---

## 5. A* Graph Search — Deep Dive

### 5.1 Core Algorithm
- **f(n) = g(n) + h(n)**
  - g(n): actual cost from start to n.
  - h(n): heuristic estimate from n to goal.
- If h(n) is **admissible** (never overestimates) and **consistent**, A* finds optimal path.
- For 3D plumbing: h(n) = 3D Manhattan distance (admissible for rectilinear movement).

### 5.2 Engineering the Cost Function for Piping
```
g(n) = Σ [Segment_Length + Turn_Penalty + Obstacle_Penalty + Slope_Penalty]
```
- 90° elbow: +high penalty (expensive fitting, pressure drop).
- Passing through wall: +very high penalty (requires sleeve).
- Collision with beam: +∞ (forbidden).

### 5.3 Spatial Partitioning for Collision Detection

The "needle in a haystack" problem: with N pipe segments and M structural obstacles, naive collision checking is O(N×M). For 10,000 objects that's 100 million checks — far too slow for millisecond response times. Spatial partitioning reduces this to O(log N).

#### 5.3.1 Octrees (Recursive Cubing)
- Divide the 3D building bounding box into 8 octants.
- Recursively subdivide any octant containing too many objects.
- High resolution where objects are dense (equipment rooms), low resolution in empty hallways.
- **Use case**: Tracing a pipe segment through empty space — quickly skip large empty volumes.
- **Physics analogy**: Mesh refinement in CFD — adaptive resolution.

#### 5.3.2 kd-trees (Intelligent Splitting)
- Binary tree where each node splits space along one axis (X, Y, or Z alternately).
- Split plane chosen to balance objects on both sides (unlike Octree's fixed midpoints).
- **Use case**: Static environments (finished building models), nearest-neighbor searches (finding closest terminal/connection point to a Steiner point).
- **Advantage over Octree**: Tighter fit to actual object distribution.

#### 5.3.3 Bounding Volume Hierarchies (BVH) — The "Russian Doll"
- Wraps each object in a simple bounding box (AABB or OBB).
- Groups of boxes wrapped in larger parent boxes, forming a tree.
- **Two-phase collision detection**:
  - **Broad Phase**: Check against parent boxes. If no overlap, discard entire subtree.
  - **Narrow Phase**: Only perform expensive precise geometry checks on surviving candidates.
- **Use case**: Dynamic environments — if a pipe moves, just update its box and parents (fast).
- **Direct analogy to video game engines** (same tech used in Elden Ring, Call of Duty for real-time physics).

#### 5.3.4 AABB vs. OBB
- **AABB (Axis-Aligned Bounding Box)**: Aligned to X/Y/Z axes. Fast to compute but wasteful for diagonal objects.
- **OBB (Oriented Bounding Box)**: Rotated to fit object tightly. Tighter fit (better for pipes at angles) but more expensive collision math.

#### 5.3.5 Selection Strategy for the Project

| Method | Best For | Why |
|--------|----------|-----|
| **Octree** | Sparse architectural spaces | Fast ray-tracing through empty space |
| **kd-tree** | Finding closest connection points | Optimized for point-based spatial queries |
| **BVH** | Complex geometry collision checks | Scales with object count, not empty space |

**Implementation note**: For the architectural routing algorithm, a **hybrid approach** is most practical:
- Use a **kd-tree** over terminal/Steiner point coordinates for nearest-neighbor queries in the optimization loop.
- Use a **BVH** over structural obstacles for collision detection during A* pathfinding.
- Use an **Octree** only if the building volume is very large and mostly empty.

#### 5.3.6 Integration with A* and Annihilation
- A*: Each time A* considers a step, it queries the BVH for obstacle collision → O(log M) instead of O(M).
- Annihilation: Finding "nearby nodes to merge" via kd-tree nearest-neighbor → O(log N) instead of O(N).
- The "ghost gradient" from MST lag optimization is acceptable because step sizes are small → spatial queries remain valid.

### 5.4 Optimizations for Millisecond Response
- **Priority Queue**: Binary heap or Fibonacci heap for A* — O(log n) extraction.
- **Hierarchical Pathfinding**: Coarse grid → fine grid (reduces A* search space by ~90%).
- **Jump Point Search**: Skips uniform-cost regions in straight aisles/hallways.
- **Hanan Grid**: Only search through terminal-aligned and obstacle-aligned planes → reduces nodes from millions to thousands.
- **Parallelization**: NumPy vectorized operations for gradient/force calculations; potential GPU offloading for large instances.

---

## 6. Benchmarking & Comparison Plan

### 6.1 Competitors

| Tier | Method | Purpose |
|------|--------|---------|
| **T1: Baseline** | **Pure MST** on terminals | Zero-effort baseline. |
| **T2: Classical** | **Smith-Hwang-Richards** (120° Fermat) | Geometric heuristic comparison. |
| **T3: Exact Solver** | **ILP / GeoSteiner** for N < 20 | Measures optimality gap. |
| **T4: Industry** | **Sequential A* on Hanan Grid** | Compares against industry standard. |

### 6.2 KPIs

| Metric | Definition |
|--------|-----------|
| **Optimality Gap** | (L_Hybrid − L_Exact) / L_Exact × 100% |
| **Convergence Speed** | Seconds/iterations to ΔL < ε |
| **Topology Flips** | Number of MST "snaps" |
| **Pruning Rate** | Initial N_s vs. Final N_s |
| **Wall-clock Time** | Total execution time |
| **Route Validity** | % of paths without collisions (3D), checked via BVH spatial partitioning |
| **Pressure Compliance** | % of endpoints meeting minimum pressure (3D) |

### 6.3 Test Scenarios

| Scenario | Tests |
|----------|-------|
| Equilateral Triangle | 120° convergence; L ≈ 1.732L_side |
| Square Grid | Symmetric diagonal Steiner connections |
| Perturbed Lattice | Robustness to irregularity |
| High-Density Cluster (50+) | Scalability + annihilation |
| Collinear Points | Degenerate → simple path |
| Random Uniform | General-case baseline |
| 3D Building + Obstacles | Full architectural scenario |
| High-Density A* | Rip-up/re-route under congestion |

### 6.4 Tournament Script Output
- Side-by-side visualizations with lengths and Steiner point counts.
- Summary table with all KPIs per method.
- Convergence curves.

---

## 7. Job Interview Preparation

### 7.1 Requirement Mapping

| Requirement | Covered In |
|------------|-----------|
| Diseñar algoritmos de enrutado automático | §4–5 (A*, Sequential A*, Hanan Grids, Steiner heuristics) |
| Geometría computacional 3D | §4.4–4.5 (Octrees, BVH, GJK, transformation matrices) |
| Modelar problemas vía grafos | §2–3 (Graph theory, MST, Steiner, A*) |
| Milisegundos / soluciones óptimas | §5.3 (Priority queues, JPS, hierarchical search, KPI tracking) |
| Arquitectura escalable | §8 Phase 8 (Design patterns, Git, CI/CD) |

### 7.2 Key Interview Concepts

1. **MST vs. A***: MST gives global minimum connectivity (no obstacles). A* gives valid obstacle-aware paths (local). Production uses Sequential A* or hybrid.
2. **Net Ordering Problem**: Early pipes block later ones → rip-up/re-route, priority queues.
3. **Hanan Grids**: Reduce continuous 3D search to discrete grid aligned with terminals.
4. **Steiner Tree**: NP-hard, n−2 bound, 120° rule, MST ≤ 2× approximation.
5. **Gradient + SA**: Physics-inspired optimization — particles, springs, temperature.
6. **Hard vs. Soft MST**: Speed vs. differentiability trade-off.
7. **Fluid Dynamics**: Darcy-Weisbach, minor losses, equivalent length K factors.
8. **Spatial Partitioning**: Octrees, kd-trees, BVH — O(log N) collision detection vs. O(N) brute force.
9. **Design Patterns**: Strategy (swap algorithms), Factory (pipe types).

### 7.3 Sample Answers

**Q**: A* vs MST — which is better?
**A**: *"They solve different problems. MST gives global minimum connectivity but ignores obstacles. A* finds valid obstacle-aware paths. In architectural routing we use Sequential A* on Hanan Grids with rip-up/re-route to handle the ordering dependency. The Steiner approach comes in when optimizing junction placement."*

**Q**: How do you handle routing failures in dense environments?
**A**: *"Iterative rip-up and re-route: if A* fails, rip an existing lower-priority pipe, retry, then re-route the ripped pipe. Combined with critical-path-first ordering and Hanan Grid pruning, this gives millisecond-level solutions."*

**Q**: How do you optimize Steiner point placement?
**A**: *"We treat Steiner points as particles in a spring network. Gradient descent using analytical forces (sum of unit vectors toward MST neighbors) moves them to equilibrium. Simulated Annealing with Metropolis criterion escapes local minima. Merging nearby points (annihilation) naturally prunes unnecessary junctions."*

**Q**: In a 3D building with thousands of obstacles, how do you achieve millisecond-level collision detection?
**A**: *"Spatial partitioning. I'd use a BVH over the obstacle geometry for broad-phase collision detection during A* pathfinding — checking against parent boxes before doing expensive per-triangle tests. For finding nearby Steiner points to merge, a kd-tree gives O(log N) nearest-neighbor queries instead of O(N²) pairwise checks. The key insight is reducing O(N×M) brute-force to O(log N + log M) per query."*

### 7.4 Production Readiness Checklist

- [ ] Git branching + PR workflow
- [ ] Unit tests (pytest, one per function)
- [ ] Type hints throughout
- [ ] CI/CD pipeline
- [ ] Docstrings + README with examples
- [ ] Profiling (cProfile, line_profiler)
- [ ] Error handling for degenerate inputs
- [ ] Clean API with sensible defaults

---

## 8. Development Roadmap (10 Weeks)

| Phase | Week | Deliverable | Status |
|-------|------|-------------|--------|
| **1: Foundations** | 1–2 | Graph class, Kruskal's MST, DSU, distance functions | ✅ Done |
| **2: Steiner Core** | 2–3 | Steiner vertex init, gradient descent, main loop | ✅ Done |
| **3: Simulated Annealing** | 3–4 | Metropolis criterion, cooling schedule | ✅ Done |
| **4: Annihilation** | 4 | Distance merging, dynamic graph updates | ✅ Done |
| **5: Optimization + Viz** | 5 | Periodic MST recalc, convergence logging, visualization | ✅ Done |
| **6: 3D Piping** | 6–8 | Hanan Grid, Sequential A*, Darcy-Weisbach | ⏳ Pending |
| **7: Benchmarking** | 8–9 | Tournament script, unit tests, brute-force verification | ⏳ Pending |
| **8: Production** | 9–10 | Clean API, CI/CD, documentation, Docker | ⏳ Pending |

---

## 9. Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Core** | Python 3.10+ | Language |
| **Math** | NumPy | Vectorized position/force calculations |
| **Graph** | Custom DSU + Heapq | Kruskal's, A* priority queue |
| **Geometry** | SciPy (ConvexHull, spatial) | Convex hull, spatial queries |
| **Viz** | Matplotlib / PyVista | 2D plots and 3D rendering |
| **Optimization** | Numba / JAX (optional) | JIT compilation of hot loops |
| **Spatial Partitioning** | scipy.spatial.KDTree (built-in) / Custom BVH | Collision detection, nearest-neighbor queries |
| **Testing** | pytest | Unit tests |
| **CI/CD** | GitHub Actions | Automated testing |
| **Viz (3D)** | PyVista / Open3D | 3D obstacle/rendering |
| **Optional** | NetworkX | Graph analysis and validation |

---

## 10. Summary of All Discussion Snippets

| # | Key Content |
|---|------------|
| 1 | Graph fundamentals, MST via Kruskal's, MST vs Steiner distinction |
| 2 | Formal Steiner definition (NP-hard), soap films, Fermat point, 120° rule, approximation ratios |
| 3 | Variational approach: Steiner points as particles, gradient descent, MST topology snapping |
| 4 | Simulated Annealing (Metropolis), MST lag optimization, annihilation/merging, Boltzmann soft MST |
| 5 | Hard vs Soft MST trade-off, analytical gradient (sum of unit vectors), MD analogy |
| 6 | Piecewise differentiability, topology boundaries, neuro-symbolic RL / straight-through estimator |
| 7 | Benchmarking plan: Pure MST vs Smith-Hwang-Richards vs ILP exact, tournament spec |
| 8 | Architectural piping: orthogonal routing, Darcy-Weisbach, gravity, minor losses, BIM tools |
| 9 | 90° vs 45° junctions (K factors), multivariable cost function, cavitation |
| 10 | Industry reality: residential (code-driven, Revit/A*) vs industrial (hydraulic NLP, AFT Fathom) |
| 11 | A* vs MST for routing, Hanan Grids, Sequential A*, rip-up/re-route, net ordering problem |
| 12 | Spatial partitioning: Octrees, kd-trees, BVH — broad phase vs. narrow phase, O(log N) collision detection |

---

## 11. Final Notes

- All conversation snippets have been incorporated into this plan. The plan is ready for implementation.
- The physics-to-code translation is the core intellectual contribution: energy ↔ cost, force ↔ gradient, temperature ↔ annealing, equilibrium ↔ convergence.
- Project A (2D Steiner) should be completed first as a foundation for Project B (3D piping).
- The benchmarking framework (§6) is critical for validating that the heuristic genuinely outperforms baselines.
- For the job interview: the math/physics intuition is already strong; the key gap is production engineering (Git, testing, clean code) and the ability to articulate the physics-to-engineering translation clearly.