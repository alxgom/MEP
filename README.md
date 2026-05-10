# Comprehensive Steiner Optimization Project

This repository contains a multi-session evolution of solvers for the Euclidean Steiner Tree Problem, ranging from basic visualizations to high-performance parallelized metaheuristics.

## Project Demos

### 1. [01-MST Visualizer](./demos/01-mst-visualizer/)
- **Core:** Basic Kruskal’s algorithm visualization.
- **Goal:** Understand terminal-only connectivity.

### 2. [02-Steiner Playground](./demos/02-steiner-playground/)
- **Core:** Physics-based relaxation using gradient descent and Simulated Annealing.
- **Goal:** Interactive exploration of 120° Fermat junctions.

### 3. [03-Steiner Tournament](./demos/03-steiner-tournament/) (The Crown Jewel)
- **Core:** A high-performance benchmarking arena featuring 8 different solvers.
- **Key Tech:** NumPy Vectorization, SciPy Graph MST, Multiprocessing.
- **Highlights:** 
    - **Reactive Quenching:** Uses MST stability as a trigger for precision relaxation ($10^{-10}$ deviation).
    - **Delaunay Kicks:** Uses triangulation centroids to escape local minima, achieving <0.2% gap from optimal.
    - **Heavy Benchmark:** Statistical evaluation over 20 unique 70-terminal maps.

## Technical Achievements
- **Vectorized Physics:** Moved all pairwise distance and force calculations into optimized C-arrays via NumPy.
- **SciPy Integration:** Utilized Sparse CSR MST solvers for a 40% performance boost over pure Python loops.
- **Evidence-Based Dev:** Followed a strict "Reason/Hypothesis/Takeaway" workflow documented in `data_exploration.md`.
