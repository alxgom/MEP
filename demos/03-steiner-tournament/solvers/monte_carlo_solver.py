"""
Monte Carlo Population Steiner Solver
=====================================
Global evolutionary search with random topological mutations.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class MonteCarloSolver(BaseSteinerSolver):
    """
    Maintains a population of solutions and uses mutations to escape local minima.
    """

    def solve(
        self,
        population_size: int = 5,
        generations: int = 10,
        steps_per_gen: int = 40,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        
        # Initialize population
        population = []
        for _ in range(population_size):
            solver = BaseSteinerSolver(self.terminals, seed=self.rng.randint(0, 10**6))
            # Initial random Steiner points
            n_steiner = max(0, self.n_terminals - 2)
            margin = 0.05
            new_pts = [
                [
                    solver.rng.uniform(solver.min_x + margin * solver.bb_diag, solver.max_x - margin * solver.bb_diag),
                    solver.rng.uniform(solver.min_y + margin * solver.bb_diag, solver.max_y - margin * solver.bb_diag)
                ]
                for _ in range(n_steiner)
            ]
            if new_pts:
                solver.points = np.vstack([solver.points, np.array(new_pts)])
                solver.velocities = np.zeros_like(solver.points)
                
            solver.compute_mst()
            population.append(solver)

        best_solver = None
        best_weight = float('inf')

        for gen in range(generations):
            # 1. Evolution: optimize each member
            for solver in population:
                for _ in range(steps_per_gen):
                    solver.compute_mst()
                    solver.apply_physics_step(learning_rate=learning_rate)
                    solver.merge_points()
                    solver.prune_redundant()
                
                w = solver.compute_mst()
                if w < best_weight:
                    best_weight = w
                    # Best way to 'copy' a solver with numpy arrays
                    best_solver = BaseSteinerSolver(self.terminals)
                    best_solver.points = solver.points.copy()
                    best_solver.velocities = solver.velocities.copy()
                    best_solver.compute_mst()

            # 2. Selection & Mutation: replace worst half with mutated best
            population.sort(key=lambda s: s.mst_weight)
            
            # Keep top half, mutate them to fill bottom half
            half = population_size // 2
            for i in range(half, population_size):
                # Pick a winner to clone
                parent = population[self.rng.randint(0, half - 1)]
                
                # Manual clone for numpy
                child = BaseSteinerSolver(self.terminals)
                child.points = parent.points.copy()
                child.velocities = parent.velocities.copy()
                child.rng = parent.rng # Keep the same RNG instance for this run
                
                # Apply mutation: random jump to one Steiner point
                if len(child.points) > child.n_terminals:
                    idx = child.rng.randint(child.n_terminals, len(child.points) - 1)
                    child.points[idx] += [
                        child.rng.gauss(0, 0.1 * child.bb_diag),
                        child.rng.gauss(0, 0.1 * child.bb_diag)
                    ]
                
                child.compute_mst()
                population[i] = child

        # Final cleanup for the best
        best_solver.final_cleanup(iterations=100)

        self.points = best_solver.points
        self.mst_edges = best_solver.mst_edges
        self.mst_weight = best_solver.mst_weight
        self.adj = best_solver.adj

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": generations * steps_per_gen,
            "max_120_deviation": self.get_max_angle_deviation()
        }
