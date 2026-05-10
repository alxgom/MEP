"""
Soft-Physics Manhattan Solver
==============================
Uses an Lp norm (p=1.1) to create smooth gradients that approximate Manhattan space.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .manhattan_base_solver import ManhattanBaseSolver


class SoftPhysicsManhattanSolver(ManhattanBaseSolver):
    """
    Continuous physics solver using Lp-norm gradient descent.
    """

    def _compute_lp_mst(self, p: float) -> float:
        """Compute MST using Lp-norm distance."""
        n = len(self.points)
        if n < 2: return 0.0
        
        # distance = (|dx|^p + |dy|^p)^(1/p)
        diff = np.abs(self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :])
        dist_matrix = np.power(np.sum(diff**p, axis=-1), 1/p)
        
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        csr_dist = csr_matrix(dist_matrix)
        mst_sparse = minimum_spanning_tree(csr_dist)
        
        u_indices, v_indices = mst_sparse.nonzero()
        self.mst_edges = [(int(u), int(v)) for u, v in zip(u_indices, v_indices)]
        self.mst_weight = float(np.sum(mst_sparse.data))
        
        self.adj = {i: [] for i in range(n)}
        for u, v in self.mst_edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
        return self.mst_weight

    def _compute_lp_gradient(self, idx: int, p: float) -> np.ndarray:
        """
        Gradient of the Lp-norm distance sum.
        d/dx (|dx|^p + |dy|^p)^(1/p)
        """
        neighbors = self.adj.get(idx, [])
        if not neighbors: return np.array([0.0, 0.0])
        
        pt = self.points[idx]
        grad = np.array([0.0, 0.0])
        
        for nb_idx in neighbors:
            nb = self.points[nb_idx]
            dvec = pt - nb
            # Lp distance
            dist_p = np.sum(np.abs(dvec)**p)**(1/p)
            
            if dist_p > 1e-10:
                # Component-wise derivative
                # d/dx dist_p = (1/p) * (|dx|^p + |dy|^p)^(1/p - 1) * p * |dx|^(p-1) * sign(dx)
                #             = dist_p^(1-p) * |dx|^(p-1) * sign(dx)
                common = np.power(dist_p, 1-p)
                g_x = common * np.power(np.abs(dvec[0]), p-1) * np.sign(dvec[0])
                g_y = common * np.power(np.abs(dvec[1]), p-1) * np.sign(dvec[1])
                grad += np.array([g_x, g_y])
                
        return grad

    def solve(
        self,
        p: float = 1.1,
        max_iterations: int = 400,
        initial_lr: float = 0.05
    ) -> Dict[str, Any]:
        
        # 1. Initialize random Steiner points
        n_steiner = max(0, self.n_terminals - 2)
        if n_steiner > 0:
            new_pts = [
                [self.rng.uniform(self.min_coords[0], self.max_coords[0]),
                 self.rng.uniform(self.min_coords[1], self.max_coords[1])]
                for _ in range(n_steiner)
            ]
            self.points = np.vstack([self.terminals, np.array(new_pts)])
        
        self.velocities = np.zeros_like(self.points)
        
        # 2. Physics loop
        for i in range(max_iterations):
            self._compute_lp_mst(p)
            
            # Physics step
            forces = np.zeros_like(self.points)
            for j in range(self.n_terminals, len(self.points)):
                forces[j] = -self._compute_lp_gradient(j, p) # Move against gradient
            
            lr = initial_lr * (1.0 - i/max_iterations) # Simple quenching
            self.points[self.n_terminals:] += forces[self.n_terminals:] * lr
            
            # Maintenance
            # (Note: simpler merging for this experiment)
            # self.merge_points() 
        
        # 3. Final Manhattan Snap
        self.compute_manhattan_mst()
        self.prune_redundant()

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "iterations": max_iterations
        }
