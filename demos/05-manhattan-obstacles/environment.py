"""
Manhattan Obstacle Environment
==============================
Manages rectangular obstacles and generates the constrained Hanan Grid.
Provides geodesic distance calculations using SciPy graph solvers.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Obstacle:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.min_x = min(x1, x2)
        self.max_x = max(x1, x2)
        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)

    def contains(self, x: float, y: float, margin: float = 1e-5) -> bool:
        """Check if a point is strictly inside the obstacle."""
        return (self.min_x + margin < x < self.max_x - margin and
                self.min_y + margin < y < self.max_y - margin)

    def intersects_segment(self, p1: np.ndarray, p2: np.ndarray, margin: float = 1e-5) -> bool:
        """Check if an orthogonal segment (H or V) passes through the obstacle."""
        # Only works for horizontal or vertical segments (which Hanan Grid edges are)
        is_h = np.isclose(p1[1], p2[1])
        is_v = np.isclose(p1[0], p2[0])
        
        if is_h:
            y = p1[1]
            if self.min_y + margin < y < self.max_y - margin:
                # Potential intersection. Check x-overlap.
                seg_min_x = min(p1[0], p2[0])
                seg_max_x = max(p1[0], p2[0])
                # Overlap exists if max of mins < min of maxs
                overlap_min = max(seg_min_x, self.min_x)
                overlap_max = min(seg_max_x, self.max_x)
                return overlap_max > overlap_min + margin
        elif is_v:
            x = p1[0]
            if self.min_x + margin < x < self.max_x - margin:
                # Potential intersection. Check y-overlap.
                seg_min_y = min(p1[1], p2[1])
                seg_max_y = max(p1[1], p2[1])
                overlap_min = max(seg_min_y, self.min_y)
                overlap_max = min(seg_max_y, self.max_y)
                return overlap_max > overlap_min + margin
        return False


class GridEnvironment:
    def __init__(self, terminals: np.ndarray, obstacles: List[Obstacle]):
        self.terminals = terminals
        self.obstacles = obstacles
        
        # 1. Generate Hanan Grid Lines
        unique_x = np.sort(np.unique(np.concatenate([
            terminals[:, 0],
            [o.min_x for o in obstacles],
            [o.max_x for o in obstacles]
        ])))
        unique_y = np.sort(np.unique(np.concatenate([
            terminals[:, 1],
            [o.min_y for o in obstacles],
            [o.max_y for o in obstacles]
        ])))
        
        # 2. Generate Nodes (Filter those inside obstacles)
        raw_nodes = []
        for x in unique_x:
            for y in unique_y:
                if not any(o.contains(x, y) for o in obstacles):
                    raw_nodes.append([x, y])
        self.nodes = np.array(raw_nodes)
        self.n_nodes = len(self.nodes)
        
        # Mapping for fast lookup
        self.node_map = {(float(n[0]), float(n[1])): i for i, n in enumerate(self.nodes)}
        
        # Identify terminal indices in the grid
        self.terminal_node_indices = []
        for t in terminals:
            key = (float(t[0]), float(t[1]))
            if key in self.node_map:
                self.terminal_node_indices.append(self.node_map[key])
        
        # 3. Build Adjacency Matrix (Only orthogonal adjacent edges)
        self.adj_matrix = self._build_adj(unique_x, unique_y)
        
        # 4. Compute APSP (All-Pairs Shortest Path)
        self.dist_matrix, self.predecessors = shortest_path(
            self.adj_matrix, directed=False, return_predecessors=True
        )

    def _build_adj(self, ux, uy):
        """Connect neighboring grid nodes if segment is not blocked."""
        row, col, data = [], [], []
        
        # Horizontal connections
        for y in uy:
            row_nodes = []
            for x in ux:
                key = (float(x), float(y))
                if key in self.node_map:
                    row_nodes.append(self.node_map[key])
                else:
                    row_nodes.append(-1)
            
            for i in range(len(row_nodes) - 1):
                u, v = row_nodes[i], row_nodes[i+1]
                if u != -1 and v != -1:
                    p1, p2 = self.nodes[u], self.nodes[v]
                    if not any(o.intersects_segment(p1, p2) for o in self.obstacles):
                        d = abs(p1[0] - p2[0])
                        row.extend([u, v]); col.extend([v, u]); data.extend([d, d])
        
        # Vertical connections
        for x in ux:
            col_nodes = []
            for y in uy:
                key = (float(x), float(y))
                if key in self.node_map:
                    col_nodes.append(self.node_map[key])
                else:
                    col_nodes.append(-1)
            
            for i in range(len(col_nodes) - 1):
                u, v = col_nodes[i], col_nodes[i+1]
                if u != -1 and v != -1:
                    p1, p2 = self.nodes[u], self.nodes[v]
                    if not any(o.intersects_segment(p1, p2) for o in self.obstacles):
                        d = abs(p1[1] - p2[1])
                        row.extend([u, v]); col.extend([v, u]); data.extend([d, d])
                        
        return csr_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))

    def get_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Backtrace path from APSP predecessor matrix."""
        path = []
        curr = end_idx
        while curr != start_idx:
            if curr == -9999 or curr < 0: return [] # No path
            path.append(curr)
            curr = self.predecessors[start_idx, curr]
        path.append(start_idx)
        return path[::-1]
