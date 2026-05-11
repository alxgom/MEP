# Multi-Net Obstacle Routing Research Summary (v1.0)

## 1. Problem: Spatial Complexity & Resource Competition
Following Demo 06, we identified that purely dynamic obstacle avoidance (pipes as obstacles) is insufficient for real-world architectural scenarios which contain static beams, walls, and restricted zones. Furthermore, "selfish" shortest paths often lead to high fragmentation and inefficient use of space.

## 2. Objective
Demo 07 integrates static obstacles into the multi-net environment and evaluates literature-backed heuristics for:
1. **Parallelism (Bundling):** Using potential energy to cluster pipes.
2. **Interference Degree:** Ordering net routing based on how much they block others.
3. **Space Pruning:** Using Bounded Area Search to manage grid dimensionality.

## 3. Data Exploration Status
All hypothesis testing is documented in `data_exploration/` following the `explore_data.md` workflow.
