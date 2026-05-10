"""
Reactive Quenching Steiner Solver
=================================
Dynamically adjusts learning rate based on MST topology stability.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class ReactiveQuenchingSolver(BaseSteinerSolver):
    """
    Solver that quenches only when the topology is stable and resets on 'flips'.
    """

    def solve(
        self,
        max_iterations: int = 1500,
        initial_lr: float = 0.1,
        cooling_rate: float = 0.95,
        stability_threshold: int = 15,
        min_lr: float = 1e-4
    ) -> Dict[str, Any]:
        
        # 1. Initialize with Adaptive Grid logic (our best starting point)
        self.compute_mst()
        if not self.mst_edges:
            return {"mst_weight": 0, "steiner_points": [], "iterations": 0}
            
        edge_lengths = [np.linalg.norm(self.points[u] - self.points[v]) for u, v in self.mst_edges]
        min_dist = min(edge_lengths)
        max_dim = max(self.max_x - self.min_x, self.max_y - self.min_y)
        res = max(10, min(int(max_dim / min_dist), 40))
        
        dx = (self.max_x - self.min_x) / (res + 1)
        dy = (self.max_y - self.min_y) / (res + 1)
        new_pts = [[self.min_x + i * dx, self.min_y + j * dy] for i in range(1, res + 1) for j in range(1, res + 1)]
        self.points = np.vstack([self.points, np.array(new_pts)])
        self.velocities = np.zeros_like(self.points)

        # 2. Optimization Loop with Reactive Quenching
        current_lr = initial_lr
        stable_steps = 0
        self.compute_mst()
        
        for i in range(max_iterations):
            self.iteration = i
            
            # Record current topology
            old_edges = set(self.mst_edges)
            
            # Physics + Maintenance
            self.apply_physics_step(learning_rate=current_lr)
            self.merge_points()
            self.prune_redundant()
            
            # Re-compute MST to check for flips
            self.compute_mst()
            new_edges = set(self.mst_edges)
            
            # REACTIVE LOGIC:
            if old_edges == new_edges:
                stable_steps += 1
            else:
                # Topology flipped! Found a new "well". Reset energy to explore it.
                stable_steps = 0
                current_lr = initial_lr
                
            # Only quench if we are stable
            if stable_steps >= stability_threshold:
                current_lr *= cooling_rate
                
            if current_lr < min_lr:
                break

        self.final_cleanup(iterations=200)

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": self.iteration,
            "max_120_deviation": self.get_max_angle_deviation()
        }
