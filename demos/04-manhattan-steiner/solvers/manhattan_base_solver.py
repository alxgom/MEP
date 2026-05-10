"""
Manhattan Base Solver Framework
===============================
Provides the mathematical core for Rectilinear Steiner Tree optimization.
Uses Cityblock (L1) distance and Hanan Grid generation.
"""

import math
import random
import itertools
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cityblock


class ManhattanBaseSolver:
    """
    Base class for Manhattan/Rectilinear routing.
    """

    def __init__(
        self,
        terminals: List[Tuple[float, float]],
        seed: Optional[int] = None,
    ):
        self.rng = random.Random(seed)
        self.terminals = np.array(terminals, dtype=float)
        self.n_terminals = len(terminals)
        
        if self.n_terminals > 0:
            self.min_coords = np.min(self.terminals, axis=0)
            self.max_coords = np.max(self.terminals, axis=0)
        else:
            self.min_coords = self.max_coords = np.array([0.0, 0.0])
            
        self.bb_diag = cityblock(self.min_coords, self.max_coords)
        
        # Current state: terminals + Steiner points
        self.points = self.terminals.copy()
        
        self.mst_edges: List[Tuple[int, int]] = []
        self.mst_weight: float = 0.0
        self.adj: Dict[int, List[int]] = {}

    def get_hanan_grid(self) -> np.ndarray:
        """
        Generate the Hanan Grid: Cartesian product of all terminal X and Y coordinates.
        According to Hanan (1966), an optimal RST exists on this grid.
        """
        if self.n_terminals == 0:
            return np.array([])
            
        unique_x = np.unique(self.terminals[:, 0])
        unique_y = np.unique(self.terminals[:, 1])
        
        # Create all (x, y) combinations
        grid = np.array(list(itertools.product(unique_x, unique_y)))
        
        # Remove points that are already terminals to avoid duplicates
        # We use a tolerance for floating point comparison
        def is_terminal(p):
            return any(np.allclose(p, t) for t in self.terminals)
            
        mask = [not is_terminal(p) for p in grid]
        return grid[mask]

    def compute_manhattan_mst(self) -> float:
        """
        Compute MST over current points using Manhattan (L1) distance.
        """
        n = len(self.points)
        if n < 2:
            self.mst_edges, self.mst_weight, self.adj = [], 0.0, {i: [] for i in range(n)}
            return 0.0
            
        # 1. Vectorized Manhattan Distance Matrix
        # |x1-x2| + |y1-y2|
        diff = np.abs(self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :])
        dist_matrix = np.sum(diff, axis=-1)
        
        # 2. SciPy MST
        csr_dist = csr_matrix(dist_matrix)
        mst_sparse = minimum_spanning_tree(csr_dist)
        
        u_indices, v_indices = mst_sparse.nonzero()
        weights = mst_sparse.data
        
        self.mst_edges = [(int(u), int(v)) for u, v in zip(u_indices, v_indices)]
        self.mst_weight = float(np.sum(weights))
        
        # 3. Adjacency
        self.adj = {i: [] for i in range(n)}
        for u, v in self.mst_edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            
        return self.mst_weight

    def prune_redundant(self) -> int:
        """
        Remove Steiner points with degree <= 2.
        In Rectilinear space, degree-2 nodes are 'elbows' that don't reduce length.
        """
        if len(self.points) <= self.n_terminals:
            return 0
            
        to_remove = []
        for i in range(len(self.points) - 1, self.n_terminals - 1, -1):
            deg = len(self.adj.get(i, []))
            if deg <= 2:
                to_remove.append(i)
        
        if not to_remove:
            return 0
            
        keep_mask = np.ones(len(self.points), dtype=bool)
        keep_mask[to_remove] = False
        
        self.points = self.points[keep_mask]
        self.compute_manhattan_mst()
        return len(to_remove)

    def solve(self, **kwargs) -> Dict[str, Any]:
        """To be implemented by subclasses."""
        raise NotImplementedError
