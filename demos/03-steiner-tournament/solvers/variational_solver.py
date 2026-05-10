"""
Variational Steiner Solver (Baseline)
=====================================
Initializes N Steiner points and optimizes via gradient descent + Simulated Annealing.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class VariationalSolver(BaseSteinerSolver):
    """
    Standard physics-based solver using random initialization and Simulated Annealing.
    """

    def solve(
        self,
        n_steiner: Optional[int] = None,
        max_iterations: int = 200,
        learning_rate: float = 0.05,
        sa_initial_temp: float = 0.02,
        sa_cooling: float = 0.98,
        convergence_threshold: float = 1e-5
    ) -> Dict[str, Any]:
        # If n_steiner not provided, use heuristic: n_terminals - 2
        if n_steiner is None:
            n_steiner = max(0, self.n_terminals - 2)
            
        # Initialize random points
        margin = 0.05
        new_pts = [
            [
                self.rng.uniform(self.min_x + margin * self.bb_diag, self.max_x - margin * self.bb_diag),
                self.rng.uniform(self.min_y + margin * self.bb_diag, self.max_y - margin * self.bb_diag)
            ]
            for _ in range(n_steiner)
        ]
        if new_pts:
            self.points = np.vstack([self.points, np.array(new_pts)])
            self.velocities = np.zeros_like(self.points)

        self.compute_mst()
        temp = sa_initial_temp
        prev_weight = float('inf')
        
        for i in range(max_iterations):
            self.iteration = i
            self.compute_mst()
            
            # 1. Physics Step
            max_force = self.apply_physics_step(learning_rate=learning_rate)
            
            # 2. Simulated Annealing (Thermal perturbation)
            if temp > 1e-6:
                for j in range(self.n_terminals, len(self.points)):
                    self.points[j][0] += self.rng.gauss(0, temp * 0.5)
                    self.points[j][1] += self.rng.gauss(0, temp * 0.5)
                temp *= sa_cooling

            # 3. Annihilation / Merging
            self.merge_points()
            
            # 4. Pruning (only after some stabilization)
            if i > 20:
                self.prune_redundant()
            
            curr_weight = self.compute_mst()
            
            # Convergence check
            if abs(prev_weight - curr_weight) < convergence_threshold and temp < 1e-4:
                break
            prev_weight = curr_weight

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": self.iteration,
            "max_120_deviation": self.get_max_angle_deviation()
        }
