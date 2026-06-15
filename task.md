# Task List: Phase 0 — Escape Graph Benchmarking

- `[x]` **Task 0.1: Initialize Environment & Scaffolding**
    - `[x]` Read existing `GridEnvironment` to ensure exact matching of APIs in `demos/05-manhattan-obstacles/environment.py`.
- `[x]` **Task 0.2: Implement EscapeGraphEnvironment**
    - `[x]` Write the `EscapeGraphEnvironment` parallel class in `demos/05-manhattan-obstacles/environment.py`.
    - `[x]` Implement orthogonal ray-tracing from terminals and obstacle corners.
    - `[x]` Terminate rays at first obstacle collision or padded boundary box.
    - `[x]` Formulate nodes (terminals, corners, ray termination, ray intersections).
    - `[x]` Establish adjacent edges along ray segments, filtering out obstacle intersections.
    - `[x]` Compute sparse matrix and APSP (All-Pairs Shortest Path) with predecessors.
- `[x]` **Task 0.3: Build Benchmarking & System**
    - `[x]` Create `demos/05-manhattan-obstacles/data_exploration/topology_faceoff.py` for headless statistical sweeps.
    - `[x]` Set up isolated `topology_benchmark.db` database logging.
    - `[x]` Code sweeps for terminal counts $N \in [20, 100]$.
    - `[x]` Run comparison tests and calculate KPI averages.
    - `[x]` Format results and auto-generate `demos/05-manhattan-obstacles/data_exploration/topology_report.md`.
- `[x]` **Task 0.4: Verification & Testing**
    - `[x]` Write test assertions in a test script (or `rigor_test.py` extension) to verify topology properties.
    - `[x]` Verify that Escape Graph node count is strictly less than Hanan Grid node count.
    - `[x]` Verify that Steiner path length is identical (confirming routing optimality).

