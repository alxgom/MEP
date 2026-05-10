"""
Fast MST-Corner Manhattan Solver
================================
A targeted heuristic that only tests the 'L-corners' of current MST edges.
This is the Manhattan equivalent of the 'Delaunay Kick'.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .manhattan_base_solver import ManhattanBaseSolver


class FastCornerHananSolver(ManhattanBaseSolver):
    """
    Reduced complexity Greedy solver. Instead of searching the full N^2 Hanan grid,
    it only tests candidates generated from the 'corners' of current MST edges.
    """

    def _get_mst_corner_candidates(self) -> np.ndarray:
        """Extract the two L-corners for every edge in the current MST."""
        candidates = []
        for u, v in self.mst_edges:
            p1, p2 = self.points[u], self.points[v]
            
            # If not already orthogonal, generate corners
            if not np.isclose(p1[0], p2[0]) and not np.isclose(p1[1], p2[1]):
                c1 = [p1[0], p2[1]]
                c2 = [p2[0], p1[1]]
                candidates.append(c1)
                candidates.append(c2)
                
        if not candidates:
            return np.array([])
            
        # Deduplicate and remove points already in self.points
        candidates = np.unique(np.array(candidates), axis=0)
        
        # Filter out points already in current point set
        mask = [not any(np.allclose(c, p) for p in self.points) for c in candidates]
        return candidates[mask]

    def solve(
        self,
        max_steiner: Optional[int] = None
    ) -> Dict[str, Any]:
        
        if max_steiner is None:
            max_steiner = self.n_terminals - 2

        self.compute_manhattan_mst()
        best_overall_weight = self.mst_weight
        
        steiner_added = []
        
        for s_idx in range(max_steiner):
            # 1. Generate targeted candidates (The "Kick")
            candidates = self._get_mst_corner_candidates()
            if len(candidates) == 0:
                break

            best_gain = 0.0
            best_cand = None
            
            # 2. Test only these O(N) candidates
            for cand in candidates:
                # Temporarily add point
                original_points = self.points.copy()
                self.points = np.vstack([self.points, cand])
                
                new_weight = self.compute_manhattan_mst()
                gain = best_overall_weight - new_weight
                
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_cand = cand
                
                # Restore
                self.points = original_points
                self.compute_manhattan_mst() # Restore MST metadata
            
            if best_cand is not None:
                self.points = np.vstack([self.points, best_cand])
                best_overall_weight -= best_gain
                self.compute_manhattan_mst() # Refresh edges for next iteration
            else:
                # No corner provides an improvement
                break

        # Final cleanup
        self.prune_redundant()

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": s_idx + 1
        }
