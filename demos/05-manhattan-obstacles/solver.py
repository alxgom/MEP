"""
Obstacle-Aware Manhattan Steiner Solver
=======================================
Finds the shortest rectilinear network connecting terminals while avoiding obstacles.
Optimized for pre-computed environment data.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class ObstacleSteinerSolver:
    def __init__(self, env):
        self.env = env
        self.terminals = env.terminal_node_indices
        self.active_nodes = list(self.terminals)
        self.mst_edges = []
        self.mst_weight = 0.0

    @classmethod
    def from_precomputed(cls, terminal_indices, dist_matrix, predecessors, n_nodes, node_map, nodes, obstacles):
        """Worker-friendly initialization to avoid full object overhead."""
        instance = cls.__new__(cls)
        class MockEnv:
            def __init__(self, ti, dm, pred, nn, nm, nd, obs):
                self.terminal_node_indices = ti
                self.dist_matrix = dm
                self.predecessors = pred
                self.n_nodes = nn
                self.node_map = nm
                self.nodes = nd
                self.obstacles = obs
            def get_path(self, s, e):
                path = []
                curr = e
                while curr != s:
                    if curr == -9999 or curr < 0: return []
                    path.append(curr)
                    curr = self.predecessors[s, curr]
                path.append(s)
                return path[::-1]
        
        instance.env = MockEnv(terminal_indices, dist_matrix, predecessors, n_nodes, node_map, nodes, obstacles)
        instance.terminals = terminal_indices
        instance.active_nodes = list(terminal_indices)
        instance.mst_edges = []
        instance.mst_weight = 0.0
        return instance

    def _compute_geodesic_mst(self, node_indices: List[int]) -> float:
        k = len(node_indices)
        if k < 2: return 0.0
        sub_dist = self.env.dist_matrix[np.ix_(node_indices, node_indices)]
        if np.any(np.isinf(sub_dist)): return 1e9
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        self.mst_edges = [(node_indices[u], node_indices[v]) for u, v in zip(u_sub, v_sub)]
        self.mst_weight = float(np.sum(mst_sparse.data))
        return self.mst_weight

    def _get_mst_corner_candidates(self) -> np.ndarray:
        candidates = []
        for u, v in self.mst_edges:
            p1, p2 = self.env.nodes[u], self.env.nodes[v]
            if not np.isclose(p1[0], p2[0]) and not np.isclose(p1[1], p2[1]):
                c1, c2 = (float(p1[0]), float(p2[1])), (float(p2[0]), float(p1[1]))
                for c in [c1, c2]:
                    if c in self.env.node_map:
                        idx = self.env.node_map[c]
                        if idx not in self.active_nodes: candidates.append(idx)
        return np.unique(np.array(candidates)) if candidates else np.array([])

    def solve_mst(self) -> Dict[str, Any]:
        self.active_nodes = list(self.terminals)
        self._compute_geodesic_mst(self.active_nodes)
        full_segments = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1): full_segments.append((path[i], path[i+1]))
        return {"weight": self.mst_weight, "steiner_indices": [], "segments": full_segments}

    def solve_greedy(self, max_steiner: int = 15) -> Dict[str, Any]:
        self.active_nodes = list(self.terminals)
        best_w = self._compute_geodesic_mst(self.active_nodes)
        candidates = [i for i in range(self.env.n_nodes) if i not in self.terminals]
        for _ in range(max_steiner):
            best_gain, best_cand = 0.0, -1
            for cand in candidates:
                if cand in self.active_nodes: continue
                w = self._compute_geodesic_mst(self.active_nodes + [cand])
                gain = best_w - w
                if gain > best_gain + 1e-6: best_gain, best_cand = gain, cand
            if best_cand != -1:
                self.active_nodes.append(best_cand); best_w -= best_gain
            else: break
        self._compute_geodesic_mst(self.active_nodes)
        full_segments = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1): full_segments.append((path[i], path[i+1]))
        return {"weight": self.mst_weight, "steiner_indices": [i for i in self.active_nodes if i not in self.terminals], "segments": full_segments}

    def solve_fast_corner(self, max_steiner: int = 25, stochastic: bool = False, temperature: float = 0.5) -> Dict[str, Any]:
        self.active_nodes = list(self.terminals)
        self._compute_geodesic_mst(self.active_nodes)
        for _ in range(max_steiner):
            candidates = list(self._get_mst_corner_candidates())
            if not candidates: break
            current_w = self.mst_weight
            gains = np.array([max(0, current_w - self._compute_geodesic_mst(self.active_nodes + [c])) for c in candidates])
            best_cand = -1
            if stochastic:
                valid = np.where(gains > 1e-6)[0]
                if len(valid) == 0: break
                top_k = min(3, len(valid))
                top_idx = np.argsort(gains[valid])[-top_k:]
                best_cand = candidates[valid[np.random.choice(top_idx)]]
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] > 1e-6: best_cand = candidates[best_idx]
            if best_cand != -1:
                self.active_nodes.append(best_cand); self._compute_geodesic_mst(self.active_nodes)
            else: break
        full_segments = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1): full_segments.append((path[i], path[i+1]))
        return {"weight": self.mst_weight, "steiner_indices": [i for i in self.active_nodes if i not in self.terminals], "segments": full_segments}

    def solve_prune(self) -> Dict[str, Any]:
        self.active_nodes = list(range(self.env.n_nodes))
        self._compute_geodesic_mst(self.active_nodes)
        while True:
            to_remove = [i for i in self.active_nodes if i not in self.terminals and len([e for e in self.mst_edges if i in e]) <= 2]
            if not to_remove: break
            self.active_nodes = [i for i in self.active_nodes if i not in to_remove]
            self._compute_geodesic_mst(self.active_nodes)
        full_segments = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1): full_segments.append((path[i], path[i+1]))
        return {"weight": self.mst_weight, "steiner_indices": [i for i in self.active_nodes if i not in self.terminals], "segments": full_segments}

    def solve_monte_carlo(self, population_size: int = 5, generations: int = 15) -> Dict[str, Any]:
        best_nodes, best_w = list(self.terminals), self._compute_geodesic_mst(list(self.terminals))
        all_indices = list(range(self.env.n_nodes))
        population = [list(self.terminals) + np.random.choice(all_indices, min(len(self.terminals), 15), replace=False).tolist() for _ in range(population_size)]
        for _ in range(generations):
            weights = [self._compute_geodesic_mst(nodes) for nodes in population]
            for w, nodes in zip(weights, population):
                if w < best_w: best_w, best_nodes = w, list(nodes)
            sorted_pop = [p for _, p in sorted(zip(weights, population))]
            new_pop = sorted_pop[:2]
            while len(new_pop) < population_size:
                child = list(new_pop[np.random.randint(0, len(new_pop))])
                s_idx = [i for i in range(len(child)) if child[i] not in self.terminals]
                if s_idx: child[np.random.choice(s_idx)] = np.random.choice(all_indices)
                new_pop.append(child)
            population = new_pop
        self.active_nodes = best_nodes
        self._compute_geodesic_mst(self.active_nodes)
        while True:
            to_remove = [i for i in self.active_nodes if i not in self.terminals and len([e for e in self.mst_edges if i in e]) <= 2]
            if not to_remove: break
            self.active_nodes = [i for i in self.active_nodes if i not in to_remove]; self._compute_geodesic_mst(self.active_nodes)
        full_segments = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1): full_segments.append((path[i], path[i+1]))
        return {"weight": self.mst_weight, "steiner_indices": [i for i in self.active_nodes if i not in self.terminals], "segments": full_segments}
