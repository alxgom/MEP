shoe"""
Manhattan Steiner Tournament
============================
Benchmarks different rectilinear routing strategies.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from solvers.manhattan_mst_solver import ManhattanMSTSolver
from solvers.hanan_greedy_solver import HananGreedySolver
from solvers.hanan_prune_solver import HananPruneSolver
from solvers.fast_corner_solver import FastCornerHananSolver
from solvers.soft_physics_solver import SoftPhysicsManhattanSolver
from manhattan_visualizer import plot_manhattan_tree

# Port some datasets from previous demos
DATASETS = {
    "Triangle": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]],
    "Square": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
    "Pentagon": [[0.5, 0.8], [0.8, 0.5], [0.7, 0.2], [0.3, 0.2], [0.2, 0.5]],
    "Random_10": np.random.RandomState(42).rand(10, 2).tolist(),
    "Random_30": np.random.RandomState(99).rand(30, 2).tolist(),
    "Random_50": np.random.RandomState(50).rand(50, 2).tolist()
}

SOLVERS = {
    "Manhattan MST": ManhattanMSTSolver,
    "Greedy Hanan": HananGreedySolver,
    "Hanan-to-Prune": HananPruneSolver,
    "Fast Corner Kick": FastCornerHananSolver,
    "Soft-Physics (L1.1)": SoftPhysicsManhattanSolver
}

def main():
    print(f"{'Algorithm':<20} | {'Dataset':<10} | {'Length':<10} | {'Time (s)':<10}")
    print("-" * 58)
    
    results = []
    
    for ds_name, terminals in DATASETS.items():
        best_weight = float('inf')
        case_results = []
        
        for algo_name, solver_cls in SOLVERS.items():
            solver = solver_cls(terminals, seed=42)
            
            start = time.time()
            res = solver.solve()
            duration = time.time() - start
            
            weight = res["mst_weight"]
            if weight < best_weight:
                best_weight = weight
                
            entry = {
                "algorithm": algo_name,
                "dataset": ds_name,
                "length": weight,
                "time": duration,
                "steiner_count": len(res.get("steiner_points", [])),
                "points": solver.points.tolist(),
                "edges": solver.mst_edges
            }
            case_results.append(entry)
            print(f"{algo_name:<20} | {ds_name:<10} | {weight:<10.4f} | {duration:<10.4f}")

        for r in case_results:
            r["gap_pct"] = (r["length"] - best_weight) / best_weight * 100 if best_weight > 0 else 0
            results.append(r)

    # 1. Summary Report
    df = pd.DataFrame(results)
    summary = df.groupby("algorithm").agg({
        "length": "mean",
        "gap_pct": "mean",
        "time": "mean",
        "steiner_count": "mean"
    }).sort_values("gap_pct")
    
    print("\n--- Summary ---")
    print(summary)
    
    # 2. Visualizations
    # Plot the results for "Random_10" as an example
    r10_results = [r for r in results if r["dataset"] == "Random_10"]
    for res in r10_results:
        output_name = f"plot_{res['algorithm'].lower().replace(' ', '_')}_r10.png"
        plot_manhattan_tree(
            np.array(DATASETS["Random_10"]),
            np.array(res["points"][len(DATASETS["Random_10"]):]),
            res["edges"],
            title=f"{res['algorithm']} on Random_10\\nLength: {res['length']:.4f}",
            save_path=output_name
        )

if __name__ == "__main__":
    main()
