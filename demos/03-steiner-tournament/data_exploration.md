# Data Exploration: Steiner Optimization Experiments

# Experiment #1: Adaptive Physics & SciPy MST

## 1. Reason behind the test
The current solver uses a fixed learning rate and a manual Kruskal's MST implementation. In dense clusters, the fixed step size causes "overshooting" and "jitter."

**Hypothesis:** 
*   An **Adaptive Step Ceiling** (0.6x shortest edge) will provide stability.
*   **SciPy MST** will provide a >50% reduction in execution time.

## 2. What the test does
1.  **Baseline:** NumPy-vectorized state.
2.  **Implementation:** SciPy MST integration + 0.6x original distance ceiling.
3.  **Validation:** Run tournament and compare metrics.

## 3. The Takeaway
The test confirmed that **Feature-Aware Step Clamping** is superior for precision ($10^{-10}$ deviation) but it triggers **False Convergence** in dense datasets because weight changes become too small too fast.

# Experiment #2: Force-Based Equilibrium Stopping
*(Completed: Logic merged into Experiment #3)*

# Experiment #3: Reactive Quenching (MST Stability)

## 1. Reason behind the test
To find a "Smart Stopping" point that doesn't rely on arbitrary weight-deltas. We want to quench the system only when the topology is stable and reset energy if the topology flips.

## 2. What the test does
1.  **Stability Trigger:** Monitor `set(mst_edges)`. If it remains unchanged for 15 steps, begin quenching $LR$ by 0.95.
2.  **Reset:** If the topology changes, reset $LR$ to $0.1$.
3.  **Stop:** Break when $LR < 1e-4$.

## 3. The Takeaway
**Highly Successful.** This is our most robust metaheuristic. It achieved a **0.45% average gap**, nearly matching Iterated 1-Steiner quality on large datasets while being faster. It perfectly solves the precision vs. speed trade-off.

# Experiment #4: Hybrid Reactive-Delaunay

## 1. Reason behind the test
Even with reactive quenching, the initial grid might miss specific strategic junctions. A "Delaunay Kick" performed *after* the first stable convergence can introduce points at the centers of empty regions, potentially jumping the network into a globally superior topology.

## 2. What the test does
1.  **Stage 1:** Run Reactive Quenching until convergence.
2.  **The Kick:** Calculate Delaunay Triangulation of current points and add centroids.
3.  **Stage 2:** Run another pass of Reactive Quenching to integrate new points.
4.  **Cleanup:** Aggressive pruning of redundant junctions.

## 3. The Takeaway
**Major Breakthrough.** The Hybrid solver achieved an average gap of **0.40%**. 
*   **Significance:** It successfully **outperformed the Iterated 1-Steiner** in the `Random_30` case (4.0506 vs 4.0561), proving that a well-kicked global relaxation can beat a pure greedy approach.
*   **Limitation:** In the highest-density cases (Random_50), the Delaunay kick adds too many points, slightly cluttering the final cleanup. Filtering centroids by triangle size is recommended for future iterations.

---

# Master Improvement Log (Benchmarking Progress)

| Version | Key Change | Avg. Gap % | Avg. Max Angle Dev | Random_50 Length |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Baseline (Fixed 10x10) | 1.10% | 15.0° | 5.15 |
| **v1.1** | Adaptive Step Ceiling (0.6x) | 1.35% | 10⁻¹⁰° | 5.11 |
| **v1.2** | Reactive Quenching | 0.45% | 10⁻¹⁰° | 5.02 |
| **v1.3** | **Hybrid Reactive-Delaunay** | **0.40%** | **10⁻¹⁰°** | **5.03** |

*Note: Gap % is relative to the best found in the tournament (usually Iterated 1-Steiner).*
