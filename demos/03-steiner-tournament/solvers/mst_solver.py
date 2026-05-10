"""
Pure MST Solver (Baseline)
==========================
Computes the Minimum Spanning Tree using only the terminal points.
No Steiner points are added.
"""

from typing import Dict, Any
from .base_solver import BaseSteinerSolver


class PureMSTSolver(BaseSteinerSolver):
    """
    Zero-Steiner-point baseline. Just Kruskal's on terminals.
    """

    def solve(self, **kwargs) -> Dict[str, Any]:
        # No Steiner points added to self.points (already contains terminals)
        weight = self.compute_mst()
        
        return {
            "mst_weight": weight,
            "steiner_points": [],
            "iterations": 0,
            "max_120_deviation": 0.0
        }
