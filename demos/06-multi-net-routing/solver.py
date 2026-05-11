"""
Sequential Multi-Net Steiner Solver (v4.8)
=========================================
Supports:
- Hyper-Aggressive Stochastic Negotiated Congestion (Node + Edge)
- Forest-Based Surgical Rip-up (Dual Node-Edge Separation + Ownership)
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

    def solve_negotiated(self, max_iters=30, congestion_base=500.0):
        """Pathfinder-style Negotiated Congestion."""
        net_names = list(self.env.nets.keys())
        all_results = {}
        
        for e in self.env.history_penalties: self.env.history_penalties[e] = 0.0
        
        for i in range(max_iters):
            all_results = {}
            self.env.edge_owners.clear()
            self.env.node_owners.clear()
            
            for name in net_names:
                self.env.rebuild(mode="Negotiated", active_net=name, congestion_base=congestion_base)
                w, segs = self._solve_single_net_stochastic(name)
                all_results[name] = {"weight": w, "segments": segs, "failed": w >= 1e8}
                self.env.add_usage(name, segs)
            
            # Unified Issue Detection
            issues = self._count_geometric_issues(all_results)
            if issues == 0: break
            
            # Apply penalties for collisions
            for e, owners in self.env.edge_owners.items():
                if len(owners) > 1: self.env.history_penalties[e] += 15.0
            # Note: Node-based history penalties could be added here if needed
                
        issues = self._count_geometric_issues(all_results)
        return all_results, sum(r["weight"] for r in all_results.values() if not r["failed"]), issues

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

    def _count_geometric_issues(self, results):
        """Rigorous post-solve check for any shared nodes or edges between nets."""
        edge_usage = {} 
        node_usage = {} 
        for name, data in results.items():
            if data.get("failed"): continue
            for u, v in data["segments"]:
                e = tuple(sorted((u, v)))
                edge_usage.setdefault(e, []).append(name)
                node_usage.setdefault(u, []).append(name)
                node_usage.setdefault(v, []).append(name)
        collisions = sum(1 for nets in edge_usage.values() if len(set(nets)) > 1)
        collisions += sum(1 for nets in node_usage.values() if len(set(nets)) > 1)
        fails = sum(1 for r in results.values() if r.get("failed"))
        return collisions + fails

    def solve_hybrid_ripup(self):
        """
        Forest-Based Surgical Rip-Up v4.7:
        1. Run Ideal Draft (respecting other terminals).
        2. Detect all shared nodes/edges.
        3. Resolve ownership surgically.
        4. Re-route sequentially to heal the breaks.
        """
        net_names = list(self.env.nets.keys())
        self.env.reset_locks()
        
        # 1. Draft (Ideal / Greedy)
        draft_res = {}
        for name in net_names:
            self.env.rebuild(mode="Hard_Lock", active_net=name)
            _, segs = self._solve_single_net_stochastic(name, top_k=3)
            draft_res[name] = segs
            
        return self._surgical_ripup_and_reconnect(draft_res)

    def solve_negotiated_hybrid(self):
        """
        Negotiated Hybrid v4.8:
        1. Run Coordinated Negotiated Congestion (Strong) to get a base draft.
        2. Perform Surgical Rip-Up on remaining collisions (fallback).
        3. Heal breaks using Hard Locks.
        """
        # 1. Coordinated Draft
        draft_res_full, total_w, issues = self.solve_negotiated(max_iters=30, congestion_base=500.0)
        
        # If negotiation resolved everything (no collisions, no failures), return it immediately
        if issues == 0:
            return draft_res_full, total_w, 0

        # Otherwise, use the negotiated results as a starting point for rip-up
        draft_res = {name: data["segments"] for name, data in draft_res_full.items()}
        return self._surgical_ripup_and_reconnect(draft_res)

    def _surgical_ripup_and_reconnect(self, draft_res: Dict[str, List[Tuple[int, int]]]):
        """Helper to perform ownership-based rip-up and reconnection."""
        net_names = list(self.env.nets.keys())
        net_areas = {name: self._get_net_bounding_box_area(name) for name in net_names}
        
        # 1. Collision Detection
        edge_owners, node_owners = {}, {}
        for name, segs in draft_res.items():
            for u, v in segs:
                e = tuple(sorted((u, v)))
                edge_owners.setdefault(e, set()).add(name)
                node_owners.setdefault(u, set()).add(name)
                node_owners.setdefault(v, set()).add(name)
        
        # 2. Surgical Resolution (Ownership Arbitration)
        net_to_ripped_edges = {name: set() for name in net_names}
        net_to_ripped_nodes = {name: set() for name in net_names}

        for e, owners in edge_owners.items():
            if len(owners) > 1:
                # Smallest area net wins the contested edge
                sorted_owners = sorted(list(owners), key=lambda o: net_areas[o])
                winner = sorted_owners[0]
                for loser in sorted_owners[1:]: net_to_ripped_edges[loser].add(e)

        for n, owners in node_owners.items():
            if len(owners) > 1:
                # Terminal owners always win over Steiner owners
                terminal_owners = [o for o in owners if n in self.env.terminal_indices[o]]
                winner = sorted(terminal_owners, key=lambda o: net_areas[o])[0] if terminal_owners else sorted(list(owners), key=lambda o: net_areas[o])[0]
                for loser in owners:
                    if loser != winner: net_to_ripped_nodes[loser].add(n)

        # 3. Component Extraction & Steiner Pruning
        net_data = {}
        for name in net_names:
            clean_segs = set()
            for s in draft_res[name]:
                if s not in net_to_ripped_edges[name] and s[0] not in net_to_ripped_nodes[name] and s[1] not in net_to_ripped_nodes[name]:
                    clean_segs.add(s)
            
            terminals = set(self.env.terminal_indices[name])
            
            # Iterative Pruning (Remove dead-end Steiner branches)
            changed = True
            while changed:
                changed = False
                adj = {}
                for u, v in clean_segs:
                    adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
                for n, neighbors in adj.items():
                    if len(neighbors) == 1 and n not in terminals:
                        neighbor = neighbors.pop()
                        clean_segs.discard(tuple(sorted((n, neighbor))))
                        changed = True
            
            final_clean = list(clean_segs)
            # nodes = terminals + all nodes in clean segments
            nodes = list(terminals | {pt for s in final_clean for pt in s})
            
            # Find all physical components
            all_islands = self._find_connected_components(nodes, final_clean)
            
            # CRITICAL: Keep only islands that contain at least one terminal.
            # Discard isolated Steiner clusters to avoid "orphan node" artifacts.
            terminal_islands = []
            for isl in all_islands:
                if any(n in terminals for n in isl):
                    terminal_islands.append(isl)
            
            net_data[name] = {"islands": terminal_islands, "clean_segs": final_clean, "area": net_areas[name]}

        # 4. Sequential Reconnection
        sorted_nets = sorted(net_names, key=lambda n: net_data[n]["area"])
        self.env.reset_locks()
        final_results = {}
        
        for name in sorted_nets:
            self.env.rebuild(mode="Hard_Lock", active_net=name)
            islands = net_data[name]["islands"]
            current_segs = set(net_data[name]["clean_segs"])
            terminals = set(self.env.terminal_indices[name])

            # Bridge islands until fully connected
            while len(islands) > 1:
                best_d, best_u, best_v, b_i, b_j = float('inf'), -1, -1, -1, -1
                for i in range(len(islands)):
                    for j in range(i+1, len(islands)):
                        sub = self.env.dist_matrix[np.ix_(islands[i], islands[j])]
                        min_idx = np.argmin(sub)
                        r, c = np.unravel_index(min_idx, sub.shape)
                        if sub[r, c] < best_d:
                            best_d, best_u, best_v, b_i, b_j = sub[r, c], islands[i][r], islands[j][c], i, j
                
                if best_d >= 1e8: break # Disconnected
                
                # Add bridge path
                path = self.env.get_path_nodes(best_u, best_v)
                for k in range(len(path)-1):
                    current_segs.add(tuple(sorted((path[k], path[k+1]))))
                
                # Robust Multi-Island Merging
                path_nodes = set(path)
                new_island = path_nodes.copy()
                remaining_islands = []
                for isl in islands:
                    if any(n in path_nodes for n in isl):
                        new_island.update(isl)
                    else:
                        remaining_islands.append(isl)
                islands = remaining_islands + [list(new_island)]

            # Final Cleanup & Pruning of the healed net
            changed = True
            while changed:
                changed = False
                adj = {}
                for u, v in current_segs:
                    adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
                for n, neighbors in adj.items():
                    if len(neighbors) == 1 and n not in terminals:
                        neighbor = neighbors.pop()
                        current_segs.discard(tuple(sorted((n, neighbor))))
                        changed = True

            final_nodes = list(terminals | {pt for s in current_segs for pt in s})
            final_results[name] = {
                "weight": sum(self.env.base_weights[s] for s in current_segs),
                "segments": list(current_segs),
                "failed": len(self._find_connected_components(final_nodes, list(current_segs))) > 1
            }
            if not final_results[name]["failed"]:
                self.env.lock_path(list(current_segs))
                
        return final_results, sum(r["weight"] for r in final_results.values() if not r["failed"]), self._count_geometric_issues(final_results)

    def solve_best_permutation(self):
        net_names = list(self.env.nets.keys())
        perms = list(itertools.permutations(net_names))
        if len(perms) > 24: perms = perms[:24]
        best_results, min_fails, min_total = None, float('inf'), float('inf')
        for p in perms:
            self.env.reset_locks()
            res, tot, fails = {}, 0, 0
            for name in p:
                self.env.rebuild(mode="Hard_Lock", active_net=name)
                w, segs = self._solve_single_net_stochastic(name)
                if w >= 1e8:
                    res[name] = {"weight": 0, "segments": [], "failed": True}; fails += 1
                else:
                    res[name] = {"weight": w, "segments": segs, "failed": False}; tot += w
                    self.env.lock_path(segs)
            if fails < min_fails or (fails == min_fails and tot < min_total):
                min_fails, min_total, best_results = fails, tot, res
        issues = self._count_geometric_issues(best_results) if best_results else 0
        return best_results, min_total, issues
