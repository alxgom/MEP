"""
Adaptive Grid-to-Prune Steiner Solver
=====================================
Initializes a grid whose density is derived from the terminal-only MST statistics.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class AdaptiveGridPruneSolver(BaseSteinerSolver):
    """
    Grid-to-Prune strategy where the resolution adapts to the terminal distribution.
    """

    def solve(
        self,
        max_total_points: int = 1600,
        max_iterations: int = 300,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        # 1. Compute Base MST to find the absolute smallest feature size
        self.compute_mst()
        if not self.mst_edges:
            return {"mst_weight": 0, "steiner_points": [], "iterations": 0}
            
        # NumPy distance calculation for MST edges
        edge_lengths = [np.linalg.norm(self.points[u] - self.points[v]) for u, v in self.mst_edges]
        min_dist = min(edge_lengths)
        
        # 2. Determine resolution (R x R grid)
        # We use the larger dimension of the bounding box to define the 'step'
        max_dim = max(self.max_x - self.min_x, self.max_y - self.min_y)
        
        if min_dist < 1e-6:
            res = 10
        else:
            res = int(max_dim / min_dist)
            
        # Floor of 10 (never less dense than baseline)
        # Cap of sqrt(max_total_points) to keep performance stable
        res_limit = int(math.sqrt(max_total_points))
        res = max(10, min(res, res_limit))
        
        # 3. Initialize R x R grid (mapping to bounding box shape)
        dx = (self.max_x - self.min_x) / (res + 1)
        dy = (self.max_y - self.min_y) / (res + 1)
        
        # Pythonic grid generation
        new_points = [
            [self.min_x + i * dx, self.min_y + j * dy]
            for i in range(1, res + 1)
            for j in range(1, res + 1)
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
            "grid_size": (res, res),
            "max_120_deviation": self.get_max_angle_deviation()
        }
