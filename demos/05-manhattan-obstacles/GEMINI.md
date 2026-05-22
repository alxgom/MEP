# Demo 05: Manhattan Obstacles Evolution

## Active Mandate: Escape Graph Benchmark (Non-Destructive)

To justify the topological foundation for Demo 07, Demo 05 must conduct a formal, side-by-side benchmark of **Escape Graphs** vs. **Hanan Grids**.

### 1. Research Goal
Empirically prove that Escape Graphs (EGs) reduce node count and pathfinding latency without sacrificing the optimality of the Steiner Tree.

### 2. Implementation Rules
*   **Safety:** Do NOT modify `GridEnvironment`. Implement `EscapeGraphEnvironment` as a new parallel class in `environment.py`.
*   **Logic:** EGs must trace rays from terminals/corners and terminate them at the first obstacle boundary encountered (Ray-Tracing).
*   **Isolation:** All results MUST be saved to a fresh `topology_benchmark.db`. The original `benchmark_results.db` is strictly READ-ONLY.

### 3. Task for Data Exploration Agent:
Create `data_exploration/topology_faceoff.py` to:
1. Initialize `topology_benchmark.db`.
2. Run a sweep of seeds (N=20 to N=100 terminals).
3. For each seed, execute both topologies and log:
    * `node_count`, `edge_count`
    * `apsp_time`, `solver_time`
    * `total_path_weight`
4. Generate a `topology_report.md` (following `explore_data.md`) comparing the results.

*Takeaway:* If weights are identical and node count is significantly lower, we will officially switch to Escape Graphs for all multi-net development.
