import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Polygon
from environment import NonOrthogonalEnvironment

def estimate_turns(v_coords: np.ndarray, incoming_dir: Optional[str], target_coords: np.ndarray) -> int:
    """
    Admissible and consistent turn estimation heuristic.
    Estimates the minimum number of 90-degree turns needed to reach target_coords from v_coords
    given the current incoming_dir.
    """
    dx = target_coords[0] - v_coords[0]
    dy = target_coords[1] - v_coords[1]
    
    if abs(dx) < 1e-7 and abs(dy) < 1e-7:
        return 0
        
    if abs(dx) < 1e-7:
        req_dir = 'N' if dy > 0 else 'S'
        if incoming_dir is None:
            return 0
        return 0 if incoming_dir == req_dir else 1
        
    if abs(dy) < 1e-7:
        req_dir = 'E' if dx > 0 else 'W'
        if incoming_dir is None:
            return 0
        return 0 if incoming_dir == req_dir else 1
        
    # Both dx and dy are non-zero (target is diagonal to us)
    req_x = 'E' if dx > 0 else 'W'
    req_y = 'N' if dy > 0 else 'S'
    if incoming_dir is None:
        return 1
    if incoming_dir in (req_x, req_y):
        return 1
    else:
        return 2

def state_expanded_astar(
    env: NonOrthogonalEnvironment,
    start_idx: int,
    target_idx: int,
    C_bend: float
) -> Tuple[Optional[List[int]], float]:
    """
    State-expanded A* pathfinder.
    Finds a turn-penalized path from start_idx to target_idx.
    State representation: (node_index, incoming_direction)
    """
    start_coords = env.nodes[start_idx]
    target_coords = env.nodes[target_idx]
    
    # Priority Queue elements: (f_score, g_score, counter, curr_node, incoming_dir)
    pq = []
    counter = 0
    
    # Initial state
    h_start = (abs(start_coords[0] - target_coords[0]) + 
               abs(start_coords[1] - target_coords[1]) + 
               C_bend * estimate_turns(start_coords, None, target_coords))
    heapq.heappush(pq, (h_start, 0.0, counter, start_idx, None))
    counter += 1
    
    g_scores = {(start_idx, None): 0.0}
    came_from = {}
    visited = set()
    best_target_state = None
    
    while pq:
        f, g, _, u, u_dir = heapq.heappop(pq)
        
        state = (u, u_dir)
        if state in visited:
            continue
        visited.add(state)
        
        if u == target_idx:
            best_target_state = state
            break
            
        # Explore neighbors
        for v, dist, edge_dir in env.adj.get(u, []):
            penalty = 0.0
            if u_dir is not None and u_dir != edge_dir:
                penalty = C_bend
                
            new_g = g + dist + penalty
            next_state = (v, edge_dir)
            
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                v_coords = env.nodes[v]
                h = (abs(v_coords[0] - target_coords[0]) + 
                     abs(v_coords[1] - target_coords[1]) + 
                     C_bend * estimate_turns(v_coords, edge_dir, target_coords))
                f_new = new_g + h
                came_from[next_state] = state
                heapq.heappush(pq, (f_new, new_g, counter, v, edge_dir))
                counter += 1
                
    if best_target_state is None:
        return None, float('inf')
        
    # Reconstruct path
    path = []
    curr = best_target_state
    while curr in came_from:
        path.append(curr[0])
        curr = came_from[curr]
    path.append(start_idx)
    path.reverse()
    
    # Calculate physical path length
    path_len = 0.0
    for i in range(len(path) - 1):
        u_pt = env.nodes[path[i]]
        v_pt = env.nodes[path[i+1]]
        path_len += np.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        
    return path, path_len

class DSU:
    """Disjoint Set Union for Kruskal's algorithm."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
    def find(self, i):
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def calculate_tree_turns(segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> int:
    """
    Computes the turn count of a Steiner tree.
    Groups segments by collinearity and returns (number of maximal straight segments) - 1.
    """
    horiz = {}
    vert = {}
    
    for p1, p2 in segments:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dy) < 1e-7:
            y = round(p1[1], 7)
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            if y not in horiz:
                horiz[y] = []
            horiz[y].append([x1, x2])
        elif abs(dx) < 1e-7:
            x = round(p1[0], 7)
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
            if x not in vert:
                vert[x] = []
            vert[x].append([y1, y2])
            
    def merge_intervals(intervals):
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            if current[0] <= prev[1] + 1e-7:
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(current)
        return len(merged)
        
    m = 0
    for y in horiz:
        m += merge_intervals(horiz[y])
    for x in vert:
        m += merge_intervals(vert[x])
        
    return max(0, m - 1)

def merge_collinear_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Merges touching/overlapping collinear segments."""
    horiz = {}
    vert = {}
    for p1, p2 in segments:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dy) < 1e-7:
            y = round(p1[1], 7)
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            if abs(x1 - x2) < 1e-7:
                continue
            if y not in horiz:
                horiz[y] = []
            horiz[y].append([x1, x2])
        elif abs(dx) < 1e-7:
            x = round(p1[0], 7)
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
            if abs(y1 - y2) < 1e-7:
                continue
            if x not in vert:
                vert[x] = []
            vert[x].append([y1, y2])
            
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = [[it[0], it[1]] for it in intervals]
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            if current[0] <= prev[1] + 1e-7:
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(current)
        return merged

    merged_segments = []
    for y, intervals in horiz.items():
        for x1, x2 in merge_intervals(intervals):
            merged_segments.append(((x1, y), (x2, y)))
    for x, intervals in vert.items():
        for y1, y2 in merge_intervals(intervals):
            merged_segments.append(((x, y1), (x, y2)))
            
    return merged_segments

def is_valid_segment(p1: Tuple[float, float], p2: Tuple[float, float], room: Polygon, obstacles: List[Polygon]) -> bool:
    """Checks if a segment is inside the room and does not cross obstacle interiors."""
    if np.hypot(p2[0]-p1[0], p2[1]-p1[1]) < 1e-7:
        return False
    line = LineString([p1, p2])
    if line.difference(room).length > 1e-7:
        return False
    for obs in obstacles:
        obs_interior = obs.buffer(-1e-5)
        if obs_interior.is_empty:
            if line.intersection(obs).length > 1e-7:
                return False
        else:
            if line.intersects(obs_interior):
                return False
    return True

class BendAwareKMBSolver:
    """
    Kou-Markowsky-Berman (KMB) algorithm using turn-penalized shortest paths.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        num_t = len(terminals)
        
        # Step 1: Compute state-expanded A* shortest paths between all terminal pairs
        paths = {}
        closure_costs = {}
        for i in range(num_t):
            for j in range(i+1, num_t):
                u, v = terminals[i], terminals[j]
                # In A*, path cost is length + C_bend * turns
                path, path_len = state_expanded_astar(self.env, u, v, self.C_bend)
                if path is None:
                    raise ValueError(f"No path found between terminals {u} and {v}.")
                
                turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                cost = path_len + self.C_bend * turns
                
                paths[(u, v)] = path
                paths[(v, u)] = path[::-1]
                closure_costs[(i, j)] = cost
                closure_costs[(j, i)] = cost
                
        # Step 2: Build complete metric closure graph and compute its MST
        sub_dist = np.zeros((num_t, num_t))
        for i in range(num_t):
            for j in range(i+1, num_t):
                cost = closure_costs[(i, j)]
                sub_dist[i, j] = cost
                sub_dist[j, i] = cost
                
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        terminal_mst_edges = [(terminals[u], terminals[v]) for u, v in zip(u_sub, v_sub)]
        
        # Step 3: Expand MST edges to grid paths and collect traversed grid edges
        subgraph_edges = set()
        for u_term, v_term in terminal_mst_edges:
            path = paths[(u_term, v_term)]
            for i in range(len(path) - 1):
                node_a, node_b = path[i], path[i+1]
                edge = (min(node_a, node_b), max(node_a, node_b))
                subgraph_edges.add(edge)
                
        # Step 4: Re-MST of the expanded subgraph using physical lengths
        subgraph_nodes = set()
        for u, v in subgraph_edges:
            subgraph_nodes.add(u)
            subgraph_nodes.add(v)
            
        sorted_sub_edges = []
        for u, v in subgraph_edges:
            pu, pv = self.env.nodes[u], self.env.nodes[v]
            w = np.hypot(pu[0] - pv[0], pu[1] - pv[1])
            sorted_sub_edges.append((w, u, v))
        sorted_sub_edges.sort(key=lambda item: item[0])
        
        dsu = DSU(subgraph_nodes)
        tree_edges = []
        for w, u, v in sorted_sub_edges:
            if dsu.union(u, v):
                tree_edges.append((u, v))
                
        # Step 5: Prune non-terminal leaves
        adj = {node: set() for node in subgraph_nodes}
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        terminal_set = set(terminals)
        while True:
            leaves = [node for node, nbrs in adj.items() if len(nbrs) <= 1 and node not in terminal_set]
            if not leaves:
                break
            for leaf in leaves:
                for nbr in adj[leaf]:
                    adj[nbr].remove(leaf)
                del adj[leaf]
                
        # Format final segment coordinates
        final_segments = []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                           (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
                                           
        merged_segs = merge_collinear_segments(final_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        
        return {
            "weight": total_len,
            "segments": merged_segs,
            "turns": turns
        }

class TurnCleanupSolver:
    """
    Solves routing using distance-based KMB (C_bend = 0) and applies an
    iterative segment-shifting post-processing pass to optimize turns.
    """
    def __init__(self, env: NonOrthogonalEnvironment):
        self.env = env
        
    def solve(self) -> Dict[str, Any]:
        # Solve with C_bend = 0
        base_solver = BendAwareKMBSolver(self.env, C_bend=0.0)
        base_res = base_solver.solve()
        
        # Optimize turns using segment shifting
        optimized_segs = self.optimize_turns(base_res["segments"])
        
        final_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in optimized_segs)
        final_turns = calculate_tree_turns(optimized_segs)
        
        return {
            "weight": final_len,
            "segments": optimized_segs,
            "turns": final_turns
        }
        
    def optimize_turns(
        self, segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        current_segs = merge_collinear_segments(segments)
        room = self.env.room
        obstacles = self.env.obstacles
        
        improved = True
        while improved:
            improved = False
            current_turns = calculate_tree_turns(current_segs)
            current_length = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in current_segs)
            
            for idx, seg in enumerate(current_segs):
                p1, p2 = seg
                is_horiz = abs(p1[1] - p2[1]) < 1e-7
                
                if is_horiz:
                    y = p1[1]
                    x1, x2 = p1[0], p2[0]
                    # Identify all vertical segments connected to this segment
                    candidates = []
                    connected_indices = []
                    for jdx, other in enumerate(current_segs):
                        if idx == jdx:
                            continue
                        op1, op2 = other
                        is_other_vert = abs(op1[0] - op2[0]) < 1e-7
                        if is_other_vert:
                            ox = op1[0]
                            oy1, oy2 = op1[1], op2[1]
                            if (x1 - 1e-7 <= ox <= x2 + 1e-7) and (abs(oy1 - y) < 1e-7 or abs(oy2 - y) < 1e-7):
                                connected_indices.append(jdx)
                                other_y = oy2 if abs(oy1 - y) < 1e-7 else oy1
                                candidates.append(other_y)
                                
                    # Try shifting horizontal segment vertically
                    for y_new in set(candidates):
                        if abs(y_new - y) < 1e-7:
                            continue
                        shifted_horiz = ((x1, y_new), (x2, y_new))
                        if not is_valid_segment(shifted_horiz[0], shifted_horiz[1], room, obstacles):
                            continue
                            
                        new_segs = []
                        for jdx, other in enumerate(current_segs):
                            if jdx == idx or jdx in connected_indices:
                                continue
                            new_segs.append(other)
                        new_segs.append(shifted_horiz)
                        
                        valid_shift = True
                        for jdx in connected_indices:
                            op1, op2 = current_segs[jdx]
                            ox = op1[0]
                            oy1, oy2 = op1[1], op2[1]
                            other_y = oy2 if abs(oy1 - y) < 1e-7 else oy1
                            
                            if abs(other_y - y_new) > 1e-7:
                                new_vert = ((ox, min(other_y, y_new)), (ox, max(other_y, y_new)))
                                if not is_valid_segment(new_vert[0], new_vert[1], room, obstacles):
                                    valid_shift = False
                                    break
                                new_segs.append(new_vert)
                                
                        if not valid_shift:
                            continue
                            
                        temp_segs = merge_collinear_segments(new_segs)
                        new_turns = calculate_tree_turns(temp_segs)
                        new_length = sum(np.hypot(tp2[0]-tp1[0], tp2[1]-tp1[1]) for tp1, tp2 in temp_segs)
                        
                        # Accept if turn count decreases or length decreases while keeping turn count
                        if new_turns < current_turns or (new_turns == current_turns and new_length < current_length - 1e-5):
                            current_segs = temp_segs
                            improved = True
                            break
                    if improved:
                        break
                else:
                    x = p1[0]
                    y1, y2 = p1[1], p2[1]
                    # Identify all horizontal segments connected to this segment
                    candidates = []
                    connected_indices = []
                    for jdx, other in enumerate(current_segs):
                        if idx == jdx:
                            continue
                        op1, op2 = other
                        is_other_horiz = abs(op1[1] - op2[1]) < 1e-7
                        if is_other_horiz:
                            oy = op1[1]
                            ox1, ox2 = op1[0], op2[0]
                            if (y1 - 1e-7 <= oy <= y2 + 1e-7) and (abs(ox1 - x) < 1e-7 or abs(ox2 - x) < 1e-7):
                                connected_indices.append(jdx)
                                other_x = ox2 if abs(ox1 - x) < 1e-7 else ox1
                                candidates.append(other_x)
                                
                    # Try shifting vertical segment horizontally
                    for x_new in set(candidates):
                        if abs(x_new - x) < 1e-7:
                            continue
                        shifted_vert = ((x_new, y1), (x_new, y2))
                        if not is_valid_segment(shifted_vert[0], shifted_vert[1], room, obstacles):
                            continue
                            
                        new_segs = []
                        for jdx, other in enumerate(current_segs):
                            if jdx == idx or jdx in connected_indices:
                                continue
                            new_segs.append(other)
                        new_segs.append(shifted_vert)
                        
                        valid_shift = True
                        for jdx in connected_indices:
                            op1, op2 = current_segs[jdx]
                            oy = op1[1]
                            ox1, ox2 = op1[0], op2[0]
                            other_x = ox2 if abs(ox1 - x) < 1e-7 else ox1
                            
                            if abs(other_x - x_new) > 1e-7:
                                new_horiz = ((min(other_x, x_new), oy), (max(other_x, x_new), oy))
                                if not is_valid_segment(new_horiz[0], new_horiz[1], room, obstacles):
                                    valid_shift = False
                                    break
                                new_segs.append(new_horiz)
                                
                        if not valid_shift:
                            continue
                            
                        temp_segs = merge_collinear_segments(new_segs)
                        new_turns = calculate_tree_turns(temp_segs)
                        new_length = sum(np.hypot(tp2[0]-tp1[0], tp2[1]-tp1[1]) for tp1, tp2 in temp_segs)
                        
                        if new_turns < current_turns or (new_turns == current_turns and new_length < current_length - 1e-5):
                            current_segs = temp_segs
                            improved = True
                            break
                    if improved:
                        break
                        
        return current_segs
