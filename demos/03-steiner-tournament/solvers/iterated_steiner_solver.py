"""
Iterated 1-Steiner Solver
=========================
Greedy heuristic: iteratively adds the single best Steiner point.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_solver import BaseSteinerSolver


class Iterated1SteinerSolver(BaseSteinerSolver):
    """
    Industry-standard greedy approach. Adds one Steiner point at a time.
    """

    def _refine_point(self, pt_idx: int, iterations: int = 50) -> float:
        """Helper to locally optimize a single Steiner point's position."""
        lr = 0.05
        for _ in range(iterations):
            self.compute_mst()
            fx, fy = self._compute_point_gradient(pt_idx)
            self.points[pt_idx][0] += fx * lr
            self.points[pt_idx][1] += fy * lr
        return self.compute_mst()

    def solve(
        self,
        max_steiner: Optional[int] = None,
        candidate_samples: int = 20
    ) -> Dict[str, Any]:
        if max_steiner is None:
            max_steiner = max(0, self.n_terminals - 2)
            
        self.compute_mst()
        best_overall_weight = self.mst_weight
        
        for s_idx in range(max_steiner):
            best_addition_weight = best_overall_weight
            best_pt = None
            
            # Pythonic candidate generation
            candidates = [
                (self.points[u] + self.points[v]) / 2
                for u, v in self.mst_edges
            ]
            candidates.extend([
                np.array([self.rng.uniform(self.min_x, self.max_x), self.rng.uniform(self.min_y, self.max_y)])
                for _ in range(candidate_samples)
            ])
                
            # Test each candidate
            for cand in candidates:
                # Save state
                original_points = self.points.copy()
                original_velocities = self.velocities.copy()
                
                # Add candidate and refine
                self.points = np.vstack([self.points, cand])
                self.velocities = np.vstack([self.velocities, [0.0, 0.0]])
                new_weight = self._refine_point(len(self.points) - 1)
                
                if new_weight < best_addition_weight - 1e-6:
                    best_addition_weight = new_weight
                    best_pt = self.points[-1].copy()
                
                # Restore state
                self.points = original_points
                self.velocities = original_velocities

            if best_pt is not None:
                self.points = np.vstack([self.points, best_pt])
                self.velocities = np.vstack([self.velocities, [0.0, 0.0]])
                best_overall_weight = best_addition_weight
                self.compute_mst()
            else:
                # No improvement found
                break

        # Final cleanup refinement (aggressive pruning)
        self.final_cleanup(iterations=150)

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": s_idx + 1,
            "max_120_deviation": self.get_max_angle_deviation()
        }
