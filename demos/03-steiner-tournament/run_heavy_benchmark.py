"""
Heavy Benchmark Parallel Runner
===============================
Runs 20 unique N=70 maps in parallel using ProcessPoolExecutor.
Exports full data for the dashboard.
"""

import os
import time
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from solvers.variational_solver import VariationalSolver
from solvers.iterated_steiner_solver import Iterated1SteinerSolver
from solvers.grid_prune_solver import GridPruneSolver
from solvers.adaptive_grid_solver import AdaptiveGridPruneSolver
from solvers.reactive_solver import ReactiveQuenchingSolver
from solvers.hybrid_solver import HybridReactiveDelaunaySolver
from solvers.monte_carlo_solver import MonteCarloSolver
from solvers.delaunay_kick_solver import DelaunayKickSolver
from solvers.mst_solver import PureMSTSolver
from datasets import get_heavy_benchmark_suite

# Dictionary of available solvers
SOLVERS = {
    "Pure MST (No Steiner)": PureMSTSolver,
    "Variational (Baseline)": VariationalSolver,
    "Iterated 1-Steiner": Iterated1SteinerSolver,
    "Grid-to-Prune (Fixed 10x10)": GridPruneSolver,
    "Adaptive Grid-to-Prune": AdaptiveGridPruneSolver,
    "Reactive Quenching Grid": ReactiveQuenchingSolver,
    "Hybrid Reactive-Delaunay": HybridReactiveDelaunaySolver,
    "Monte Carlo Population": MonteCarloSolver,
    "Delaunay Centroid Kick": DelaunayKickSolver
}

def run_single_job(args):
    """Worker function for the parallel executor."""
    algo_name, map_data, seed = args
    solver_cls = SOLVERS[algo_name]
    terminals = map_data["terminals"]
    case_name = map_data["name"]
    
    solver = solver_cls(terminals, seed=seed)
    
    start_time = time.time()
    try:
        res = solver.solve()
    except Exception as e:
        return {"error": f"Error in {algo_name} on {case_name}: {e}"}
    duration = time.time() - start_time
    
    return {
        "algorithm": algo_name,
        "case": case_name,
        "length": res["mst_weight"],
        "time": duration,
        "steiner_count": len(res.get("steiner_points", [])),
        "max_120_dev": res.get("max_120_deviation", 0.0),
        "points": solver.points.tolist(),
        "edges": solver.mst_edges
    }

def main():
    print("Initializing Heavy Benchmark (20 maps, N=70)...")
    maps = get_heavy_benchmark_suite()
    
    # Prepare jobs
    jobs = []
    for m in maps:
        for algo in SOLVERS.keys():
            jobs.append((algo, m, 42)) # Fixed seed for repeatability
            
    print(f"Total jobs to execute: {len(jobs)}")
    print(f"Executing in parallel using {os.cpu_count()} workers...")
    
    results = []
    with ProcessPoolExecutor() as executor:
        for i, res in enumerate(executor.map(run_single_job, jobs)):
            if "error" in res:
                print(res["error"])
            else:
                results.append(res)
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(jobs)} jobs completed...")

    # Post-process for Optimality Gap
    df = pd.DataFrame(results)
    best_weights = df.groupby("case")["length"].transform("min")
    df["gap_pct"] = (df["length"] - best_weights) / best_weights * 100
    
    # Save raw results for dashboard
    with open("heavy_results.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print("Raw results saved to heavy_results.json")
    
    # Generate Markdown Report
    summary = df.groupby("algorithm").agg({
        "length": "mean",
        "gap_pct": "mean",
        "time": "mean",
        "steiner_count": "mean",
        "max_120_dev": "mean"
    }).sort_values("gap_pct")
    
    report = "# Heavy Steiner Tournament Benchmarking Report (N=70)\n\n"
    report += f"**Dataset:** 20 unique random maps, 70 terminals each.\n"
    report += f"**Parallelism:** ProcessPoolExecutor ({os.cpu_count()} cores).\n\n"
    report += "## Executive Summary\n"
    report += summary.to_markdown() + "\n\n"
    
    report += "## Length Results per Case\n"
    pivot = df.pivot(index="case", columns="algorithm", values="length")
    report += pivot.to_markdown() + "\n\n"

    with open("REPORT_HEAVY.md", "w") as f:
        f.write(report)
        
    print("\nBenchmark Complete! Report saved to REPORT_HEAVY.md")

if __name__ == "__main__":
    main()
