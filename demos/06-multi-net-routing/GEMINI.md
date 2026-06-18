# Demo 06: Multi-Net Routing Evolution

## Active Mandate: Staged Optimization Pipeline & Baselines

To improve performance and maintain high optimality, Demo 06 must be updated to support a **Staged Optimization Pipeline** and specific **Speed Baselines**. 

### 1. New Speed Baselines
Implement these variants to establish the "lower bound" of intelligence vs. speed:
*   **Baseline_MST_Perm:** Route using Pure MST (0 kicks) + Global Permutation Search.
*   **Baseline_MST_Ripup:** Route using Pure MST (0 kicks) + Surgical Rip-up.

### 2. Staged Optimization (The Pipeline)
Refactor `solver.py` to implement the following 3-stage lifecycle for all high-quality solvers:

1.  **Stage 1: Drafting:** In the main routing loop, replace multi-kick Steiner search with **Pure MST** or **1-Steiner**. (Goal: Extreme Speed).
2.  **Stage 2: Coordination:** Use Negotiated Congestion or Surgical Rip-Up to resolve all collisions using the fast draft topologies.
3.  **Stage 3: Steiner Polishing:** Once the global topology is feasibility-locked, invoke the full **Steiner Kick** (`_solve_fast_corner`) on the valid paths.
    *   **CRITICAL FEASIBILITY CHECK:** During polishing, every potential Steiner junction must be validated against the environment's `locked_edges`. If a junction would create a collision, it must be discarded.

### 3. New Algorithm: Inference-Degree Ordering
Implement a solver that calculates the **Interference Degree** (blocking score) for every net and routes them in a single sorted sequence, bypassing the $N!$ permutation search.

### 4. Mandatory KPI: Execution Time
The visual dashboard must now display **Time (ms)** alongside **Length** and **Issues** for every panel.

### 5. New Solver: Dynamic Escape Graph Sequential Solver
Implement a routing option `solve_dynamic_escape_sequential()` that routes nets sequentially, but constructs a custom, augmented escape graph for each net:
*   Route nets sequentially sorted by bounding box area.
*   For each net:
    *   Construct a local `MultiNetEnvironment` where coordinates of the current net's terminals and all segments/nodes of previously routed nets' paths are added as interest points (forcing grid lines).
    *   Lock the edges and nodes of the previously routed paths in the local environment to prevent collisions.
    *   Solve the current net on this dynamic local grid.
    *   Store the path in physical coordinates and project it back to the global environment's node indices.
*   Wire this solver into `main.py` as a 5th comparison panel to evaluate length, turns, collisions, and runtime against the static grid solvers.

*Goal:* Establish whether the Staged Pipeline achieves near-permutation quality in a fraction of the time.

---

## Future Research & Development (Literature-Backed)

Beyond Steiner Healing, the following enhancements (derived from Blokland et al., 2023) should be prioritized once instructed by the user:

### 1. Macro-Pipe Clustering (Strategy: Yuan et al. [37])
*   **Concept:** Pre-cluster terminals into group "flows" using geometric clustering.
*   **Implementation:** Route the cluster as a single "macro-trunk" before branching to individual terminals. This simulates industrial pipe racks and reduces spatial fragmentation.

### 2. Parallelism Score (Strategy: Dong & Lin [39])
*   **Metric:** Implement a KPI that measures the % of total pipe length that runs parallel and adjacent to existing segments.
*   **Objective:** Reward solvers that "bundle" pipes together, which is critical for real-world installation feasibility.

### 3. Bend-Continuity Index
*   **Metric:** Calculate the ratio of straight segments to 90° elbows.
*   **Heuristic:** Add a "Turn Penalty" to the A* cost function to minimize elbows, reducing hydraulic head loss (Darcy-Weisbach).

### 4. Boltzmann Stochastic Selection
*   **Algorithm:** Upgrade the Steiner Kick to use a temperature-based probability distribution ($P(c) \propto e^{gain(c)/T}$) instead of random top-K selection.
*   **Benefit:** Improves global exploration and helps the Negotiated solver escape topological local minima in dense maps.
