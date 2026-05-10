"""
Hybrid Reactive-Delaunay Steiner Solver
=======================================
Combines the stability of Reactive Quenching with the topological exploration 
of Delaunay Centroid Kicks.
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class HybridReactiveDelaunaySolver(BaseSteinerSolver):
    """
    Two-stage solver: Reactive Quenching -> Delaunay Kick -> Reactive Quenching.
    """

    def _run_reactive_pass(
        self, 
        max_iterations: int = 1000, 
        initial_lr: float = 0.1, 
        cooling_rate: float = 0.95,
        stability_threshold: int = 15,
        min_lr: float = 1e-4
    ):
        """Standard reactive quenching loop."""
        current_lr = initial_lr
        stable_steps = 0
        self.compute_mst()
        
        for i in range(max_iterations):
            old_edges = set(self.mst_edges)
            
            self.apply_physics_step(learning_rate=current_lr)
            self.merge_points()
            self.prune_redundant()
            
            self.compute_mst()
            new_edges = set(self.mst_edges)
            
            if old_edges == new_edges:
                stable_steps += 1
            else:
                stable_steps = 0
                current_lr = initial_lr # Reset energy on flip
                
            if stable_steps >= stability_threshold:
                current_lr *= cooling_rate
                
            if current_lr < min_lr:
                break

    def solve(self, **kwargs) -> Dict[str, Any]:
        # 1. Initialize with Adaptive Grid (Same as Reactive solver)
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

        # STAGE 1: Initial Reactive Convergence
        self._run_reactive_pass()
        
        # STAGE 2: THE KICK (Delaunay Centroids)
        # We only add centroids for the current stable topology
        pts = np.unique(self.points, axis=0)
        if len(pts) >= 3:
            try:
                tri = Delaunay(pts)
                centroids = pts[tri.simplices].mean(axis=1)
                self.points = np.vstack([self.points, centroids])
                self.velocities = np.zeros_like(self.points)
                
                # STAGE 3: Re-convergence
                self._run_reactive_pass(max_iterations=500)
            except Exception:
                pass # Skip kick if triangulation fails

        self.final_cleanup(iterations=200)

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": self.iteration, # Iteration count from the passes
            "max_120_deviation": self.get_max_angle_deviation()
        }
