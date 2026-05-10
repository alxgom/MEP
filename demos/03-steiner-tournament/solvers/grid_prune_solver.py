"""
Grid-to-Prune Steiner Solver
============================
Initializes a dense grid of Steiner points and prunes them via annihilation.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class GridPruneSolver(BaseSteinerSolver):
    """
    Over-saturates the space with a grid and lets annihilation/pruning find the tree.
    """

    def solve(
        self,
        grid_resolution: int = 10,
        max_iterations: int = 300,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        # Initialize grid
        dx = (self.max_x - self.min_x) / (grid_resolution + 1)
        dy = (self.max_y - self.min_y) / (grid_resolution + 1)
        
        # Pythonic grid generation
        new_points = [
            [self.min_x + i * dx, self.min_y + j * dy]
            for i in range(1, grid_resolution + 1)
            for j in range(1, grid_resolution + 1)
        ]
        self.points = np.vstack([self.points, np.array(new_points)])
        self.velocities = np.zeros_like(self.points)

        self.compute_mst()
        prev_weight = float('inf')
        for i in range(max_iterations):
            self.iteration = i
            self.compute_mst()
            
            # Physics
            self.apply_physics_step(learning_rate=learning_rate)
            
            # Aggressive merging
            self.merge_points(threshold=0.01 * self.bb_diag)
            
            # Pruning
            if i > 30:
                self.prune_redundant()
            
            curr_weight = self.compute_mst()
            if abs(prev_weight - curr_weight) < 1e-6 and i > 100:
                break
            prev_weight = curr_weight

        self.final_cleanup(iterations=200)

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": self.iteration,
            "max_120_deviation": self.get_max_angle_deviation()
        }
