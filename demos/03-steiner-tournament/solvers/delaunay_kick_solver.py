"""
Delaunay Centroid Kick Steiner Solver
=====================================
Uses Delaunay Triangulation to identify candidate regions for new Steiner points.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import Delaunay
from .base_solver import BaseSteinerSolver


class DelaunayKickSolver(BaseSteinerSolver):
    """
    Standard variational solver + geometric "kicks" using Delaunay centroids.
    """

    def _optimize(self, iterations: int = 100):
        for _ in range(iterations):
            self.compute_mst()
            self.apply_physics_step()
            self.merge_points()
            self.prune_redundant()

    def solve(
        self,
        kicks: int = 3,
        max_iterations_per_kick: int = 100,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        
        # Initial optimization (start with MST midpoints)
        self.compute_mst()
        new_pts = [
            (self.points[u] + self.points[v]) / 2
            for u, v in self.mst_edges
        ]
        if new_pts:
            self.points = np.vstack([self.points, np.array(new_pts)])
            self.velocities = np.zeros_like(self.points)
            
        self._optimize(max_iterations_per_kick)

        for k in range(kicks):
            # Compute Delaunay Triangulation
            # Remove duplicate points which can cause Delaunay errors
            pts = np.unique(self.points, axis=0)
            
            if len(pts) < 3:
                continue
                
            try:
                tri = Delaunay(pts)
                
                # Add centroids of all triangles
                centroids = pts[tri.simplices].mean(axis=1)
                
                self.points = np.vstack([self.points, centroids])
                self.velocities = np.zeros_like(self.points)
                
                self._optimize(max_iterations_per_kick)
            except Exception as e:
                # Likely collinear points or other geometric degeneracy
                self._optimize(max_iterations_per_kick)
                continue

        self.final_cleanup(iterations=100)

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": kicks * max_iterations_per_kick,
            "max_120_deviation": self.get_max_angle_deviation()
        }
