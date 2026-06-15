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


class EscapeGraphEnvironment:
    def __init__(self, terminals: np.ndarray, obstacles: List[Obstacle]):
        self.terminals = terminals
        self.obstacles = obstacles

        # 1. Padded Bounding Box: Min/Max of all terminals and obstacles combined, padded by 80 units
        xs = [t[0] for t in terminals] + [o.min_x for o in obstacles] + [o.max_x for o in obstacles]
        ys = [t[1] for t in terminals] + [o.min_y for o in obstacles] + [o.max_y for o in obstacles]
        bbox_min_x = min(xs) - 80.0
        bbox_max_x = max(xs) + 80.0
        bbox_min_y = min(ys) - 80.0
        bbox_max_y = max(ys) + 80.0

        # 2. Interest Points: Union of terminals and all obstacle corners, excluding any points strictly inside obstacles
        interest_points = set()
        for t in terminals:
            interest_points.add((float(t[0]), float(t[1])))
        for o in obstacles:
            corners = [
                (o.min_x, o.min_y),
                (o.min_x, o.max_y),
                (o.max_x, o.min_y),
                (o.max_x, o.max_y)
            ]
            for c in corners:
                interest_points.add((float(c[0]), float(c[1])))

        filtered_interest_points = []
        for p in interest_points:
            if not any(o.contains(p[0], p[1]) for o in obstacles):
                filtered_interest_points.append(p)

        # 3. Ray Projections: From every interest point, project 4 orthogonal rays
        h_segments = []
        v_segments = []
        segment_endpoints = set()

        for x, y in filtered_interest_points:
            # East (+x)
            x_term_e = bbox_max_x
            for o in obstacles:
                if o.min_y < y < o.max_y and o.min_x >= x:
                    if o.min_x < x_term_e:
                        x_term_e = o.min_x
            h_segments.append((float(y), float(x), float(x_term_e)))
            segment_endpoints.add((float(x), float(y)))
            segment_endpoints.add((float(x_term_e), float(y)))

            # West (-x)
            x_term_w = bbox_min_x
            for o in obstacles:
                if o.min_y < y < o.max_y and o.max_x <= x:
                    if o.max_x > x_term_w:
                        x_term_w = o.max_x
            h_segments.append((float(y), float(x_term_w), float(x)))
            segment_endpoints.add((float(x), float(y)))
            segment_endpoints.add((float(x_term_w), float(y)))

            # North (+y)
            y_term_n = bbox_max_y
            for o in obstacles:
                if o.min_x < x < o.max_x and o.min_y >= y:
                    if o.min_y < y_term_n:
                        y_term_n = o.min_y
            v_segments.append((float(x), float(y), float(y_term_n)))
            segment_endpoints.add((float(x), float(y)))
            segment_endpoints.add((float(x), float(y_term_n)))

            # South (-y)
            y_term_s = bbox_min_y
            for o in obstacles:
                if o.min_x < x < o.max_x and o.max_y <= y:
                    if o.max_y > y_term_s:
                        y_term_s = o.max_y
            v_segments.append((float(x), float(y_term_s), float(y)))
            segment_endpoints.add((float(x), float(y)))
            segment_endpoints.add((float(x), float(y_term_s)))

        # 4. All-Pair Segment Intersections: Find all intersection points between all horizontal and vertical ray segments
        intersection_nodes = set()
        for y, h_x1, h_x2 in h_segments:
            for x, v_y1, v_y2 in v_segments:
                if (h_x1 - 1e-9 <= x <= h_x2 + 1e-9) and (v_y1 - 1e-9 <= y <= v_y2 + 1e-9):
                    intersection_nodes.add((float(x), float(y)))

        # 5. Combined Nodes: Union the segment endpoints and all intersection points
        combined_nodes = segment_endpoints.union(intersection_nodes)

        # Filter out any nodes strictly inside obstacles to form the node set
        cleaned_nodes = []
        for p in combined_nodes:
            if not any(o.contains(p[0], p[1]) for o in obstacles):
                cleaned_nodes.append(p)

        self.nodes = np.array(cleaned_nodes)
        self.n_nodes = len(self.nodes)
        self.node_map = {(float(n[0]), float(n[1])): i for i, n in enumerate(self.nodes)}

        # Identify terminal node indices
        self.terminal_node_indices = []
        for t in terminals:
            key = (float(t[0]), float(t[1]))
            if key in self.node_map:
                self.terminal_node_indices.append(self.node_map[key])

        # 6. Edge Connection: Connect consecutive nodes along all segments
        edges_dict = {}

        # Horizontal segments
        for y, x1, x2 in h_segments:
            if abs(x1 - x2) < 1e-9:
                continue
            seg_nodes = []
            for (nx, ny), idx in self.node_map.items():
                if abs(ny - y) < 1e-9 and (x1 - 1e-9 <= nx <= x2 + 1e-9):
                    seg_nodes.append((nx, idx))
            seg_nodes.sort(key=lambda item: item[0])
            for i in range(len(seg_nodes) - 1):
                idx1, idx2 = seg_nodes[i][1], seg_nodes[i+1][1]
                p1, p2 = self.nodes[idx1], self.nodes[idx2]
                midpoint = (p1 + p2) / 2.0
                if not any(o.intersects_segment(p1, p2) or o.contains(midpoint[0], midpoint[1]) for o in obstacles):
                    d = abs(p1[0] - p2[0])
                    u, v = min(idx1, idx2), max(idx1, idx2)
                    edges_dict[(u, v)] = d

        # Vertical segments
        for x, y1, y2 in v_segments:
            if abs(y1 - y2) < 1e-9:
                continue
            seg_nodes = []
            for (nx, ny), idx in self.node_map.items():
                if abs(nx - x) < 1e-9 and (y1 - 1e-9 <= ny <= y2 + 1e-9):
                    seg_nodes.append((ny, idx))
            seg_nodes.sort(key=lambda item: item[0])
            for i in range(len(seg_nodes) - 1):
                idx1, idx2 = seg_nodes[i][1], seg_nodes[i+1][1]
                p1, p2 = self.nodes[idx1], self.nodes[idx2]
                midpoint = (p1 + p2) / 2.0
                if not any(o.intersects_segment(p1, p2) or o.contains(midpoint[0], midpoint[1]) for o in obstacles):
                    d = abs(p1[1] - p2[1])
                    u, v = min(idx1, idx2), max(idx1, idx2)
                    edges_dict[(u, v)] = d

        # Build CSR adjacency matrix
        row, col, data = [], [], []
        for (u, v), d in edges_dict.items():
            row.extend([u, v])
            col.extend([v, u])
            data.extend([d, d])

        self.adj_matrix = csr_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))

        # 7. APSP: Compute All-Pairs Shortest Path
        self.dist_matrix, self.predecessors = shortest_path(
            self.adj_matrix, directed=False, return_predecessors=True
        )

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

