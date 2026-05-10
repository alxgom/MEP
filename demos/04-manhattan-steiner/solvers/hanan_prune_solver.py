"""
Hanan-to-Prune Steiner Solver
=============================
Adds the entire Hanan Grid and prunes redundant nodes.
"""

import numpy as np
from typing import Dict, Any
from .manhattan_base_solver import ManhattanBaseSolver


class HananPruneSolver(ManhattanBaseSolver):
    """
    Floods the space with all Hanan intersections and prunes the Manhattan MST.
    """

    def solve(self, **kwargs) -> Dict[str, Any]:
        # 1. Generate Hanan Grid (all candidate locations)
        candidates = self.get_hanan_grid()
        
        # 2. Add all points at once
        if len(candidates) > 0:
            self.points = np.vstack([self.terminals, candidates])
            
        # 3. Compute massive MST
        self.compute_manhattan_mst()
        
        # 4. Aggressively prune until stability
        # Degree-1 and degree-2 Steiner points are redundant in Manhattan space
        while True:
            removed = self.prune_redundant()
            if removed == 0:
                break

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": 1
        }
