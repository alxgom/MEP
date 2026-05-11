"""
Sequential Multi-Net Steiner Solver (v4.4)
=========================================
Supports:
- Hyper-Aggressive Negotiated Congestion (Node + Edge)
- Forest-Based Hybrid Rip-up (Component-to-Component Shortest Path)
- Global Permutation Search
"""

import numpy as np
import itertools
import copy
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path


class MultiNetSolver:
    def __init__(self, env):
        self.env = env # MultiNetEnvironment instance

    def _get_net_bounding_box_area(self, name):
        pts = self.env.nets[name]
        min_c, max_c = np.min(pts, axis=0), np.max(pts, axis=0)
        return (max_c[0] - min_c[0]) * (max_c[1] - min_c[1])

    def _compute_geodesic_mst(self, node_indices: List[int]) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute MST using the environment's current distance matrix."""
        k = len(node_indices)
        if k < 2: return 0.0, []
        
        sub_dist = self.env.dist_matrix[np.ix_(node_indices, node_indices)]
        if np.any(np.isinf(sub_dist)):
            return 1e9, [] # Disconnected
            
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        
        edges = [(node_indices[u], node_indices[v]) for u, v in zip(u_sub, v_sub)]
        weight = float(np.sum(mst_sparse.data))
        return weight, edges

    def _solve_single_net_stochastic(self, net_name, max_steiner=10, top_k=3):
        """Greedy 1-Steiner with Fast Corner Kick."""
        terminal_indices = self.env.terminal_indices[net_name]
        active_nodes = list(terminal_indices)
        
        current_w, current_edges = self._compute_geodesic_mst(active_nodes)
        
        for _ in range(max_steiner):
            best_gain, best_cand = 0, -1
            candidates = set()
            for u, v in current_edges:
                p1, p2 = self.env.nodes[u], self.env.nodes[v]
                if not np.isclose(p1[0], p2[0]) and not np.isclose(p1[1], p2[1]):
                    # Possible Hanan corners
                    c1, c2 = (float(p1[0]), float(p2[1])), (float(p2[0]), float(p1[1]))
                    for c in [c1, c2]:
                        idx = self.env.node_map.get(c)
                        if idx is not None and idx not in active_nodes:
                            candidates.add(idx)
            
            cand_list = list(candidates)
            if not cand_list: break
            
            gains = []
            for cand in cand_list:
                active_nodes.append(cand)
                w_new, _ = self._compute_geodesic_mst(active_nodes)
                active_nodes.pop()
                gains.append(max(0, current_w - w_new))
            
            gains = np.array(gains)
            valid = np.where(gains > 1e-6)[0]
            if len(valid) == 0: break
            
            k = min(top_k, len(valid))
            top_indices = np.argsort(gains[valid])[-k:]
            choice = np.random.choice(valid[top_indices])
            
            active_nodes.append(cand_list[choice])
            current_w, current_edges = self._compute_geodesic_mst(active_nodes)
            
        segments = []
        for u, v in current_edges:
            path = self.env.get_path_nodes(u, v)
            for i in range(len(path)-1):
                segments.append(tuple(sorted((path[i], path[i+1]))))
        return current_w, segments

    def solve_negotiated(self, max_iters=25):
        """Pathfinder-style Negotiated Congestion."""
        net_names = list(self.env.nets.keys())
        all_results = {}
        
        for e in self.env.history_penalties: self.env.history_penalties[e] = 0.0
        
        for i in range(max_iters):
            all_results = {}
            for e in self.env.edge_congestion: self.env.edge_congestion[e] = 0
            for n in self.env.node_congestion: self.env.node_congestion[n] = 0
            
            for name in net_names:
                self.env.rebuild(mode="Negotiated")
                w, segs = self._solve_single_net_stochastic(name)
                all_results[name] = {"weight": w, "segments": segs, "failed": w >= 1e8}
                for u, v in segs:
                    self.env.edge_congestion[tuple(sorted((u, v)))] += 1
                    self.env.node_congestion[u] += 1
                    self.env.node_congestion[v] += 1
            
            total_collisions = sum(1 for e, c in self.env.edge_congestion.items() if c > 1)
            total_collisions += sum(1 for n, c in self.env.node_congestion.items() if c > 1)
            if total_collisions == 0: break
            
            for e, c in self.env.edge_congestion.items():
                if c > 1: self.env.history_penalties[e] += 15.0
                
        fails = sum(1 for r in all_results.values() if r["failed"])
        return all_results, sum(r["weight"] for r in all_results.values() if not r["failed"]), total_collisions + fails

    def _find_connected_components(self, nodes: List[int], segments: List[Tuple[int, int]]) -> List[List[int]]:
        adj = {n: [] for n in nodes}
        for u, v in segments:
            if u in adj and v in adj:
                adj[u].append(v); adj[v].append(u)
        
        visited = set()
        components = []
        for n in nodes:
            if n not in visited:
                comp = []
                q = [n]
                visited.add(n)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for neighbor in adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append(comp)
        return components

    def solve_hybrid_ripup(self):
        """
        1. Run Negotiated pass for global coordination.
        2. Identify and DELETE shared segments (collisions).
        3. Keep surviving islands (components).
        4. Re-route each net (Smallest BB first) to heal the breaks.
        """
        # Phase 1: Negotiated draft (High precision, low iters)
        init_res, _, _ = self.solve_negotiated(max_iters=12)
        
        # Phase 2: Detect Global Collisions
        all_segs = []
        for r in init_res.values(): all_segs.extend(r["segments"])
        from collections import Counter
        edge_counts = Counter(all_segs)
        bad_segs = {s for s, c in edge_counts.items() if c > 1}
        
        # Phase 3: Identify Components for every net
        net_names = list(self.env.nets.keys())
        net_islands = {}
        for name in net_names:
            # Keep only segments that are 100% clean
            clean_segs = [s for s in init_res[name]["segments"] if s not in bad_segs]
            # Nodes are terminals + nodes in clean segments
            nodes = list(set(self.env.terminal_indices[name]) | {pt for s in clean_segs for pt in s})
            islands = self._find_connected_components(nodes, clean_segs)
            net_islands[name] = {"islands": islands, "area": self._get_net_bounding_box_area(name), "clean_segs": clean_segs}

        # Phase 4: Sequential Reconnection (Smallest BB First)
        sorted_nets = sorted(net_names, key=lambda n: net_islands[n]["area"])
        self.env.reset_locks()
        self.env.rebuild(mode="Hard_Lock")
        
        final_results = {}
        for name in sorted_nets:
            islands = net_islands[name]["islands"]
            current_segs = set(net_islands[name]["clean_segs"])
            
            # Forest Stitching: connect islands until they form 1 component
            while len(islands) > 1:
                best_d, best_u, best_v, best_idx_i, best_idx_j = float('inf'), -1, -1, -1, -1
                
                # Search all island-to-island bridges
                for i in range(len(islands)):
                    for j in range(i+1, len(islands)):
                        sub_dist = self.env.dist_matrix[np.ix_(islands[i], islands[j])]
                        if np.any(sub_dist < best_d):
                            r, c = np.unravel_index(np.argmin(sub_dist), sub_dist.shape)
                            if sub_dist[r, c] < best_d:
                                best_d = sub_dist[r, c]
                                best_u, best_v = islands[i][r], islands[j][c]
                                best_idx_i, best_idx_j = i, j
                
                if best_d >= 1e8: break # Blocked
                
                # Build bridge
                path = self.env.get_path_nodes(best_u, best_v)
                for k in range(len(path)-1):
                    current_segs.add(tuple(sorted((path[k], path[k+1]))))
                
                # Merge islands
                new_island = list(set(islands[best_idx_i]) | set(islands[best_idx_j]) | set(path))
                islands = [islands[k] for k in range(len(islands)) if k not in [best_idx_i, best_idx_j]]
                islands.append(new_island)

            # Verification
            final_nodes = list(set(self.env.terminal_indices[name]) | {pt for s in current_segs for pt in s})
            final_comps = self._find_connected_components(final_nodes, list(current_segs))
            
            if len(final_comps) > 1:
                final_results[name] = {"weight": 0, "segments": [], "failed": True}
            else:
                final_results[name] = {"weight": sum(self.env.base_weights[s] for s in current_segs),
                                      "segments": list(current_segs), "failed": False}
                self.env.lock_path(list(current_segs))
                
        return final_results, sum(r["weight"] for r in final_results.values() if not r["failed"]), sum(1 for r in final_results.values() if r["failed"])

    def solve_best_permutation(self):
        net_names = list(self.env.nets.keys())
        perms = list(itertools.permutations(net_names))
        if len(perms) > 24: perms = perms[:24]
        
        best_results, min_fails, min_total = None, float('inf'), float('inf')
        
        for p in perms:
            self.env.reset_locks()
            self.env.rebuild(mode="Hard_Lock")
            res, tot, fails = {}, 0, 0
            for name in p:
                w, segs = self._solve_single_net_stochastic(name)
                if w >= 1e8:
                    res[name] = {"weight": 0, "segments": [], "failed": True}; fails += 1
                else:
                    res[name] = {"weight": w, "segments": segs, "failed": False}; tot += w
                    self.env.lock_path(segs)
            if fails < min_fails or (fails == min_fails and tot < min_total):
                min_fails, min_total, best_results = fails, tot, res
        return best_results, min_total, min_fails
