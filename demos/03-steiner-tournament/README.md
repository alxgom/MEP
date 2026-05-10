# Demo #3: Steiner Tournament (Heavy Benchmark)

A high-performance arena for statistically comparing Steiner Tree metaheuristics on complex terminal sets.

## Key Solvers

| Solver | Strategy | Verdict |
| :--- | :--- | :--- |
| **Iterated 1-Steiner** | Greedy greedy addition of Fermat points. | **The Quality Leader.** Perfect solutions, but slow. |
| **Delaunay Centroid Kick** | Injects points at triangulation centroids. | **Efficiency King.** Fastest high-quality solver for $N > 50$. |
| **Reactive Quenching** | Uses MST stability to trigger cooling. | **Precision King.** Reaches $10^{-10}$ Fermat deviation. |
| **Hybrid Reactive-Delaunay** | Multi-stage: Grid -> Kick -> Quench. | Robust global search; beats greedy on mid-sized maps. |
| **Monte Carlo Population** | Evolutionary survival of 5 universes. | Highly stable; clones top topologies to escape minima. |
| **Pure MST** | Standard Kruskal’s (no junctions). | Baseline for quantifying "Steiner Gain." |

## Optimization Engine

The core of this demo is built on a fully vectorized **NumPy and SciPy** engine:
- **Topology:** `scipy.sparse.csgraph.minimum_spanning_tree` provides C-optimized MST calculations.
- **Physics:** Gradients and displacements are calculated as matrix operations, eliminating Python `for` loops.
- **Scaling:** Features a `Heavy Benchmark` that runs 20 unique $N=70$ maps in parallel using `ProcessPoolExecutor`.

## Documentation & Logging

We followed an evidence-based development process:
- [**data_exploration.md**](./data_exploration.md): Contains the "Master Improvement Log" and records of all failed/successful experiments.
- [**REPORT_HEAVY.md**](./REPORT_HEAVY.md): Comprehensive statistical results from the 180-job parallel run.

## Running

```bash
# Standard Tournament (Classic cases + small random)
python run_tournament.py

# Heavy Benchmark (N=70 parallel execution)
python run_heavy_benchmark.py
```
