"""
Tournament Framework
====================
Orchestrates runs between different solvers and records metrics.
"""

import time
import json
from typing import List, Dict, Any, Type
from solvers.base_solver import BaseSteinerSolver
from solvers.variational_solver import VariationalSolver
from solvers.iterated_steiner_solver import Iterated1SteinerSolver
from solvers.grid_prune_solver import GridPruneSolver
from solvers.adaptive_grid_solver import AdaptiveGridPruneSolver
from solvers.reactive_solver import ReactiveQuenchingSolver
from solvers.hybrid_solver import HybridReactiveDelaunaySolver
from solvers.monte_carlo_solver import MonteCarloSolver
from solvers.delaunay_kick_solver import DelaunayKickSolver
from datasets import get_benchmark_suites

from solvers.mst_solver import PureMSTSolver

class Tournament:
    def __init__(self):
        self.solvers: Dict[str, Type[BaseSteinerSolver]] = {
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
        self.results = []

    def run(self):
        suites = get_benchmark_suites()
        
        print(f"{'Algorithm':<25} | {'Dataset':<15} | {'Length':<10} | {'Time (s)':<10}")
        print("-" * 65)
        
        for suite_name, cases in suites.items():
            for case in cases:
                case_name = case["name"]
                terminals = case["terminals"]
                
                # Find best weight among all solvers for this case to calculate gap
                case_results = []
                best_weight = float('inf')
                
                for algo_name, solver_cls in self.solvers.items():
                    # Create a fresh solver instance
                    solver = solver_cls(terminals, seed=42)
                    
                    start_time = time.time()
                    try:
                        res = solver.solve()
                    except Exception as e:
                        print(f"Error in {algo_name} on {case_name}: {e}")
                        continue
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    weight = res["mst_weight"]
                    
                    if weight < best_weight:
                        best_weight = weight
                    
                    result_entry = {
                        "algorithm": algo_name,
                        "suite": suite_name,
                        "case": case_name,
                        "length": weight,
                        "time": duration,
                        "iterations": res.get("iterations", 0),
                        "steiner_count": len(res.get("steiner_points", [])),
                        "max_120_dev": res.get("max_120_deviation", 0.0),
                        "points": [list(p) for p in solver.points],
                        "edges": list(solver.mst_edges)
                    }
                    case_results.append(result_entry)
                    
                    print(f"{algo_name[:25]:<25} | {case_name[:15]:<15} | {weight:<10.4f} | {duration:<10.4f}")

                # Calculate optimality gap relative to best found in this tournament
                for r in case_results:
                    r["gap_pct"] = (r["length"] - best_weight) / best_weight * 100 if best_weight > 0 else 0
                    self.results.append(r)
        
        self.save_results()

    def save_results(self, filename="tournament_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    t = Tournament()
    t.run()
