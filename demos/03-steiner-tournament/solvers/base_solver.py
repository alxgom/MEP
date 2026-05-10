"""
Base Solver Framework for Steiner Tree Tournament
=================================================
Provides common physics-based optimization logic (gradient descent, MST, annihilation).
Optimized with NumPy and SciPy. (Reverted to high-travel baseline).
"""

import math
import random
import itertools
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class BaseSteinerSolver:
    """
    Base class providing the mathematical core for all Steiner solvers.
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
            self.min_x, self.max_x = self.min_coords[0], self.max_coords[0]
            self.min_y, self.max_y = self.min_coords[1], self.max_coords[1]
        else:
            self.min_x = self.max_x = self.min_y = self.max_y = 0.0
            
        self.bb_diag = math.hypot(self.max_x - self.min_x, self.max_y - self.min_y)
        
        # Current state: terminals are always at the beginning of the list
        self.points = self.terminals.copy()
        self.velocities = np.zeros_like(self.points)
        
        self.mst_edges: List[Tuple[int, int]] = []
        self.mst_weight: float = 0.0
        self.adj: Dict[int, List[int]] = {}
        self.iteration = 0

    def compute_mst(self) -> float:
        """Compute MST over all current points using SciPy."""
        n = len(self.points)
        if n < 2:
            self.mst_edges, self.mst_weight, self.adj = [], 0.0, {i: [] for i in range(n)}
            return 0.0
            
        diff = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        csr_dist = csr_matrix(dist_matrix)
        mst_sparse = minimum_spanning_tree(csr_dist)
        
        u_indices, v_indices = mst_sparse.nonzero()
        weights = mst_sparse.data
        
        self.mst_edges = [(int(u), int(v)) for u, v in zip(u_indices, v_indices)]
        self.mst_weight = float(np.sum(weights))
        
        self.adj = {i: [] for i in range(n)}
        for u, v in self.mst_edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            
        return self.mst_weight

    def merge_points(self, threshold: Optional[float] = None) -> int:
        if threshold is None:
            threshold = 0.005 * self.bb_diag
            
        n = len(self.points)
        if n <= self.n_terminals: return 0
            
        diff = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        to_remove = set()
        steiner_indices = list(range(self.n_terminals, n))
        
        for i, j in itertools.combinations(steiner_indices, 2):
            if i in to_remove or j in to_remove: continue
            if dist_matrix[i, j] < threshold:
                self.points[i] = (self.points[i] + self.points[j]) / 2
                self.velocities[i] = (self.velocities[i] + self.velocities[j]) / 2
                to_remove.add(j)
        
        for i in steiner_indices:
            if i in to_remove: continue
            if np.any(dist_matrix[i, :self.n_terminals] < threshold):
                to_remove.add(i)

        if not to_remove: return 0

        keep_mask = np.ones(len(self.points), dtype=bool)
        keep_mask[list(to_remove)] = False
        
        self.points = self.points[keep_mask]
        self.velocities = self.velocities[keep_mask]
        
        self.compute_mst()
        return len(to_remove)

    def prune_redundant(self, force_threshold: float = 0.01, unconditional: bool = False) -> int:
        if len(self.points) <= self.n_terminals: return 0
            
        to_remove = []
        for i in range(len(self.points) - 1, self.n_terminals - 1, -1):
            deg = len(self.adj.get(i, []))
            if deg <= 2:
                if unconditional:
                    to_remove.append(i)
                else:
                    fx, fy = self._compute_point_gradient(i)
                    if math.hypot(fx, fy) < force_threshold:
                        to_remove.append(i)
        
        if not to_remove: return 0
            
        keep_mask = np.ones(len(self.points), dtype=bool)
        keep_mask[to_remove] = False
        
        self.points = self.points[keep_mask]
        self.velocities = self.velocities[keep_mask]
        
        self.compute_mst()
        return len(to_remove)

    def _compute_point_gradient(self, idx: int) -> Tuple[float, float]:
        neighbors = self.adj.get(idx, [])
        if not neighbors: return 0.0, 0.0
        p = self.points[idx]
        nbs = self.points[neighbors]
        diff = nbs - p
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        safe_mask = dists.flatten() > 1e-10
        if not np.any(safe_mask): return 0.0, 0.0
        unit_vectors = diff[safe_mask] / dists[safe_mask]
        force = np.sum(unit_vectors, axis=0)
        return float(force[0]), float(force[1])

    def apply_physics_step(self, learning_rate: float = 0.05, friction: float = 0.6) -> float:
        n = len(self.points)
        if n <= self.n_terminals: return 0.0
            
        forces = np.zeros_like(self.points)
        for i in range(self.n_terminals, n):
            fx, fy = self._compute_point_gradient(i)
            forces[i] = [fx, fy]
            
        steiner_slice = slice(self.n_terminals, n)
        self.velocities[steiner_slice] = self.velocities[steiner_slice] * friction + forces[steiner_slice] * learning_rate
        
        limit = 0.02 * self.bb_diag
        v_mags = np.linalg.norm(self.velocities[steiner_slice], axis=1, keepdims=True)
        overspeed = v_mags.flatten() > limit
        if np.any(overspeed):
            self.velocities[steiner_slice][overspeed] *= limit / v_mags[overspeed]
            
        self.points[steiner_slice] += self.velocities[steiner_slice]
        max_force = np.max(np.linalg.norm(forces[steiner_slice], axis=1)) if n > self.n_terminals else 0.0
        return float(max_force)

    def final_cleanup(self, iterations: int = 100):
        for _ in range(iterations):
            self.compute_mst()
            if len(self.points) <= self.n_terminals: break
            self.apply_physics_step(learning_rate=0.02, friction=0.8)
            self.merge_points(threshold=0.002 * self.bb_diag)
            self.prune_redundant(unconditional=True)
        self.compute_mst()

    def get_max_angle_deviation(self) -> float:
        max_dev = 0.0
        for i in range(self.n_terminals, len(self.points)):
            nb_indices = self.adj.get(i, [])
            if len(nb_indices) < 2: continue
            p = self.points[i]
            nbs = self.points[nb_indices]
            angles = np.sort(np.arctan2(nbs[:, 1] - p[1], nbs[:, 0] - p[0]))
            diffs = np.diff(angles)
            diffs = np.append(diffs, (2 * np.pi - (angles[-1] - angles[0])))
            devs = np.abs(np.degrees(diffs) - 120.0)
            max_dev = max(max_dev, np.max(devs))
        return max_dev

    def solve(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
