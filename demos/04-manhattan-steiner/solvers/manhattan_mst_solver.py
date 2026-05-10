"""
Manhattan MST Solver (Baseline)
===============================
Computes the Minimum Spanning Tree using only terminal points and Manhattan distance.
"""

from typing import Dict, Any
from .manhattan_base_solver import ManhattanBaseSolver


class ManhattanMSTSolver(ManhattanBaseSolver):
    """
    Zero-Steiner-point baseline for Manhattan space.
    """

    def solve(self, **kwargs) -> Dict[str, Any]:
        weight = self.compute_manhattan_mst()
        
        return {
            "mst_weight": weight,
            "steiner_points": [],
            "iterations": 0
        }
