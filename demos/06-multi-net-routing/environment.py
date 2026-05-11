"""
Multi-Net Manhattan Environment (v4.4)
======================================
Supports:
- Expanded Hanan Grid (Detour Padding)
- Intermediate Grid Lines (Refinement)
- Hyper-Aggressive Negotiated Congestion
- Hard Node/Edge Locking
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path


class MultiNetEnvironment:
    def __init__(self, nets: Dict[str, np.ndarray]):
        self.nets = nets
        self.locked_edges = set() 
        self.locked_nodes = set()
        
        # 1. Global Hanan Grid + Detour Padding + REFINEMENT
        all_points = np.vstack(list(nets.values()))
        ux_raw = np.sort(np.unique(all_points[:, 0]))
        uy_raw = np.sort(np.unique(all_points[:, 1]))
        
        # Add midpoints (Grid Refinement) to allow "side-streets"
        ux_mid = (ux_raw[:-1] + ux_raw[1:]) / 2
        uy_mid = (uy_raw[:-1] + uy_raw[1:]) / 2
        
        # Add boundary padding
        pad = 80
        self.ux = np.sort(np.unique(np.concatenate([ux_raw, ux_mid, [ux_raw[0]-pad, ux_raw[-1]+pad]])))
        self.uy = np.sort(np.unique(np.concatenate([uy_raw, uy_mid, [uy_raw[0]-pad, uy_raw[-1]+pad]])))

        
        # 2. Nodes
        self.nodes = np.array([[x, y] for x in self.ux for y in self.uy])
        self.n_nodes = len(self.nodes)
        self.node_map = {(float(n[0]), float(n[1])): i for i, n in enumerate(self.nodes)}
        
        # 3. Base Adjacency (Distances)
        self.base_weights = {} 
        self.edge_congestion = {} # (u,v) -> int
        self.node_congestion = {} # node_idx -> int
        self.history_penalties = {} 
        
        # 4. Indices
        self.terminal_indices = {}
        self.all_terminal_nodes = set()
        for name, pts in nets.items():
            indices = [self.node_map[(float(p[0]), float(p[1]))] for p in pts]
            self.terminal_indices[name] = indices
            self.all_terminal_nodes.update(indices)

        self._init_topology()
        self.rebuild()

    def _init_topology(self):
        """Initialize the map with physical distances."""
        for i in range(self.n_nodes):
            self.node_congestion[i] = 0
            
        for y in self.uy:
            for i in range(len(self.ux) - 1):
                u, v = self.node_map[(float(self.ux[i]), float(y))], self.node_map[(float(self.ux[i+1]), float(y))]
                d = self.ux[i+1] - self.ux[i]
                edge = tuple(sorted((u, v)))
                self.base_weights[edge] = d
                self.edge_congestion[edge] = 0
                self.history_penalties[edge] = 0.0
        
        for x in self.ux:
            for i in range(len(self.uy) - 1):
                u, v = self.node_map[(float(x), float(self.uy[i]))], self.node_map[(float(x), float(self.uy[i+1]))]
                d = self.uy[i+1] - self.uy[i]
                edge = tuple(sorted((u, v)))
                self.base_weights[edge] = d
                self.edge_congestion[edge] = 0
                self.history_penalties[edge] = 0.0

    def rebuild(self, mode="Hard_Lock"):
        adj = lil_matrix((self.n_nodes, self.n_nodes))
        
        for edge, d in self.base_weights.items():
            u, v = edge
            if mode == "Hard_Lock":
                if edge not in self.locked_edges and u not in self.locked_nodes and v not in self.locked_nodes:
                    adj[u, v] = d; adj[v, u] = d
            else:
                # Negotiated Logic
                pres_e = self.edge_congestion[edge]
                pres_n = self.node_congestion[u] + self.node_congestion[v]
                hist = self.history_penalties[edge]
                term_penalty = 10.0 if (u in self.all_terminal_nodes or v in self.all_terminal_nodes) else 1.0
                penalty = (1.0 + hist) * (500.0 ** (pres_e + pres_n)) * term_penalty
                cost = d * penalty
                adj[u, v] = cost; adj[v, u] = cost
                
        self.adj_matrix = adj.tocsr()
        self.dist_matrix, self.predecessors = shortest_path(
            self.adj_matrix, directed=False, return_predecessors=True
        )

    def reset_locks(self):
        self.locked_edges.clear()
        self.locked_nodes.clear()
        for k in self.edge_congestion: self.edge_congestion[k] = 0
        for k in self.node_congestion: self.node_congestion[k] = 0

    def lock_path(self, segments: List[Tuple[int, int]]):
        for edge in segments:
            e = tuple(sorted(edge))
            self.locked_edges.add(e)
            self.locked_nodes.add(e[0]); self.locked_nodes.add(e[1])
        self.rebuild(mode="Hard_Lock")

    def get_path_nodes(self, start: int, end: int) -> List[int]:
        path = []
        curr = end
        while curr != start:
            if curr < 0: return []
            path.append(curr)
            curr = self.predecessors[start, curr]
        path.append(start)
        return path[::-1]
