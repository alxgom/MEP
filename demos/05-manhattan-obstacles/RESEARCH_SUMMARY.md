# Manhattan Steiner Optimization: 2D Research Summary

## 1. Project Overview
This phase focused on the **Rectilinear Steiner Tree with Obstacles (RSTPO)** problem, transitioning from Euclidean geometry to the orthogonal constraints required for architectural piping and MEP optimization.

## 2. Core Technical Evolution
- **Hanan Grid Integration:** Implemented the theorem that optimal rectilinear Steiner points lie on the intersections of terminal coordinates.
- **Geodesic Distance Engine:** Developed an obstacle-aware distance metric using SciPy's All-Pairs Shortest Path (APSP) on a constrained Hanan visibility grid.
- **NumPy/SciPy Engine:** Refactored the core logic for 10x performance gains via vectorization and sparse matrix operations.
- **Statistical Framework:** Built a parallelized benchmark runner ($N=40$ to $100$) backed by a WAL-SQLite database for large-scale analysis.

## 3. Key Mathematical Takeaways (Benchmark N=70-100)
- **Density Convergence:** As terminal density increases in a fixed domain, the gain over the MST baseline decays. This is because terminal proximity naturally mimics the Hanan grid, reducing the "topological surface area" available for Steiner junctions to provide savings.
- **Geometric Masking:** Static obstacles create "routing skeletons." Large gains are only possible in open regions; in narrow corridors or shadow zones, the routing is functionally solved by the environment, masking the solver's effectiveness.
- **The N=50 Complexity Peak:** We identified a "sweet spot" for metaheuristics around 50 terminals, where topological complexity is high enough to benefit from exploration but not yet suppressed by high-density convergence.

## 4. Advanced Heuristics (Next-Gen Ideas)

### **A. Boltzmann Selection (The "Heat" Kick)**
Currently, the "Stochastic Kick" picks randomly from the Top-K candidates.
- **Proposed Change:** Implement **Boltzmann/Softmax Selection**. 
- **The Logic:** Every candidate $i$ is chosen with probability $P_i = \frac{e^{Gain_i / T}}{\sum e^{Gain_j / T}}$, where $T$ is the temperature.
- **Benefit:** High temperature allows for chaotic exploration to find non-obvious "jumps," while quenching (lowering $T$) locks the system into the final high-precision Fermat topology.

### **B. Constant-Density Domain Scaling**
To find a clear "scaling law" for Steiner trees, the domain must grow with $N$.
- **The Plan:** Scale the bounding box area linearly with the number of terminals.
- **Obstacle Randomization:** To avoid "skeleton bias," obstacles must be randomized and scaled in size and quantity alongside the domain. This ensures that the solver is tested against fresh geometric challenges in every trial.

## 5. How to Run the Dashboard
The research phase includes an interactive **Gradio Dashboard** for live statistical and topological analysis.

```bash
cd demos/05-manhattan-obstacles
python gradio_dashboard.py
```
- **Access:** Visit `http://localhost:7865` in your browser.
- **Features:** 
    - **Live Charts:** View efficiency gains and win probabilities as data streams from the SQLite database.
    - **Map Drill-Down:** Select any configuration ID to see the **side-by-side comparison** of the Greedy path (Red) vs. the Stochastic Winner (Green), including highlighted junctions.
    - **Concurrency:** Uses SQLite WAL mode to allow reading stats while a benchmark run is writing.

## 6. Transition to Phase 6: 3D Piping
The lessons from this 2D phase—**Hanan Grids, Geodesic APSP, and Stochastic Kick-Relax cycles**—provide the foundational components for 3D architectural piping. The third dimension adds vertical complexity (floor-to-floor) but follows the same fundamental Manhattan logic established here.
