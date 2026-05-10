"""
Greedy Hanan Steiner Solver
===========================
Iteratively adds the single best point from the Hanan Grid.
"""

import numpy as np
from typing import Dict, Any, Optional
from .manhattan_base_solver import ManhattanBaseSolver


class HananGreedySolver(ManhattanBaseSolver):
    """
    Greedy Iterated 1-Steiner specialized for the Hanan Grid.
    """

    def solve(
        self,
        max_steiner: Optional[int] = None
    ) -> Dict[str, Any]:
        
        # 1. Generate Hanan Grid (all candidate locations)
        candidates = self.get_hanan_grid()
        if len(candidates) == 0:
            return {"mst_weight": self.compute_manhattan_mst(), "steiner_points": [], "iterations": 0}

        self.compute_manhattan_mst()
        best_overall_weight = self.mst_weight
        
        if max_steiner is None:
            max_steiner = self.n_terminals - 2

        steiner_added = []
        
        for s_idx in range(max_steiner):
            best_gain = 0.0
            best_cand_idx = -1
            
            # Test every point in the Hanan Grid
            for i, cand in enumerate(candidates):
                # Skip if already added
                if any(np.allclose(cand, s) for s in steiner_added):
                    continue
                    
                # Temporarily add point
                self.points = np.vstack([self.terminals, np.array(steiner_added + [cand])])
                new_weight = self.compute_manhattan_mst()
                
                gain = best_overall_weight - new_weight
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_cand_idx = i
            
            if best_cand_idx != -1:
                steiner_added.append(candidates[best_cand_idx])
                best_overall_weight -= best_gain
            else:
                # No single point improves the Manhattan tree anymore
                break

        self.points = np.vstack([self.terminals, np.array(steiner_added)]) if steiner_added else self.terminals
        self.compute_manhattan_mst()
        self.prune_redundant()

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": s_idx + 1
        }
