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
    C_bend: float,
    edge_weights: Optional[Dict[Tuple[int, int], float]] = None
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
            if edge_weights is not None:
                dist = edge_weights.get((min(u, v), max(u, v)), dist)
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

def dual_graph_astar(
    env: NonOrthogonalEnvironment,
    start_idx: int,
    target_idx: int,
    C_bend: float,
    edge_weights: Optional[Dict[Tuple[int, int], float]] = None
) -> Tuple[Optional[List[int]], float]:
    """
    Shortest path A* search on the explicit Directed Dual Graph states.
    States are directed edges (u, v) representing traversal from u to v.
    """
    if start_idx == target_idx:
        return [start_idx], 0.0
        
    # Priority Queue elements: (f_score, g_score, counter, state)
    # state is (u, v)
    pq = []
    counter = 0
    
    g_scores = {}
    came_from = {}
    visited = set()
    target_coords = env.nodes[target_idx]
    
    # Initialize start states: all (start_idx, v) for neighbors v of start_idx
    for v, dist, edge_dir in env.adj.get(start_idx, []):
        if edge_weights is not None:
            dist = edge_weights.get((min(start_idx, v), max(start_idx, v)), dist)
        state = (start_idx, v)
        g_scores[state] = dist
        v_coords = env.nodes[v]
        h = (abs(v_coords[0] - target_coords[0]) + 
             abs(v_coords[1] - target_coords[1]) + 
             C_bend * estimate_turns(v_coords, edge_dir, target_coords))
        heapq.heappush(pq, (dist + h, dist, counter, state))
        counter += 1
        
    best_target_state = None
    
    while pq:
        f, g, _, state = heapq.heappop(pq)
        
        if state in visited:
            continue
        visited.add(state)
        
        u, v = state
        if v == target_idx:
            best_target_state = state
            break
            
        pu = env.nodes[u]
        pv = env.nodes[v]
        dx_uv = pv[0] - pu[0]
        dy_uv = pv[1] - pu[1]
        if abs(dx_uv) > abs(dy_uv):
            dir_uv = 'E' if dx_uv > 0 else 'W'
        else:
            dir_uv = 'N' if dy_uv > 0 else 'S'
            
        for w, dist, dir_vw in env.adj.get(v, []):
            if w == u:
                continue
            if edge_weights is not None:
                dist = edge_weights.get((min(v, w), max(v, w)), dist)
                
            penalty = C_bend if dir_uv != dir_vw else 0.0
            new_g = g + dist + penalty
            next_state = (v, w)
            
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                w_coords = env.nodes[w]
                h = (abs(w_coords[0] - target_coords[0]) + 
                     abs(w_coords[1] - target_coords[1]) + 
                     C_bend * estimate_turns(w_coords, dir_vw, target_coords))
                came_from[next_state] = state
                heapq.heappush(pq, (new_g + h, new_g, counter, next_state))
                counter += 1
                
    if best_target_state is None:
        return None, float('inf')
        
    # Reconstruct path of node indices
    path = [best_target_state[1], best_target_state[0]]
    curr = best_target_state
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr[0])
    path.reverse()
    
    # Calculate physical length
    path_len = 0.0
    for i in range(len(path) - 1):
        u_pt = env.nodes[path[i]]
        v_pt = env.nodes[path[i+1]]
        path_len += np.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        
    return path, path_len

def dual_graph_gbfs(
    env: NonOrthogonalEnvironment,
    start_idx: int,
    target_idx: int,
    C_bend: float,
    edge_weights: Optional[Dict[Tuple[int, int], float]] = None
) -> Tuple[Optional[List[int]], float]:
    """
    Greedy Best-First Search (GBFS) on the Directed Dual Graph states.
    Priority queue is sorted purely by the heuristic estimate h(n) to target.
    """
    if start_idx == target_idx:
        return [start_idx], 0.0
        
    # Priority Queue elements: (h_score, counter, g_score, state)
    # state is (u, v)
    pq = []
    counter = 0
    
    g_scores = {}
    came_from = {}
    visited = set()
    target_coords = env.nodes[target_idx]
    
    # Initialize start states: all (start_idx, v) for neighbors v of start_idx
    for v, dist, edge_dir in env.adj.get(start_idx, []):
        if edge_weights is not None:
            dist = edge_weights.get((min(start_idx, v), max(start_idx, v)), dist)
        state = (start_idx, v)
        g_scores[state] = dist
        v_coords = env.nodes[v]
        h = (abs(v_coords[0] - target_coords[0]) + 
             abs(v_coords[1] - target_coords[1]) + 
             C_bend * estimate_turns(v_coords, edge_dir, target_coords))
        heapq.heappush(pq, (h, counter, dist, state))
        counter += 1
        
    best_target_state = None
    
    while pq:
        h, _, g, state = heapq.heappop(pq)
        
        if state in visited:
            continue
        visited.add(state)
        
        u, v = state
        if v == target_idx:
            best_target_state = state
            break
            
        pu = env.nodes[u]
        pv = env.nodes[v]
        dx_uv = pv[0] - pu[0]
        dy_uv = pv[1] - pu[1]
        if abs(dx_uv) > abs(dy_uv):
            dir_uv = 'E' if dx_uv > 0 else 'W'
        else:
            dir_uv = 'N' if dy_uv > 0 else 'S'
            
        for w, dist, dir_vw in env.adj.get(v, []):
            if w == u:
                continue
            if edge_weights is not None:
                dist = edge_weights.get((min(v, w), max(v, w)), dist)
                
            penalty = C_bend if dir_uv != dir_vw else 0.0
            new_g = g + dist + penalty
            next_state = (v, w)
            
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                w_coords = env.nodes[w]
                next_h = (abs(w_coords[0] - target_coords[0]) + 
                          abs(w_coords[1] - target_coords[1]) + 
                          C_bend * estimate_turns(w_coords, dir_vw, target_coords))
                came_from[next_state] = state
                heapq.heappush(pq, (next_h, counter, new_g, next_state))
                counter += 1
                
    if best_target_state is None:
        return None, float('inf')
        
    # Reconstruct path of node indices
    path = [best_target_state[1], best_target_state[0]]
    curr = best_target_state
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr[0])
    path.reverse()
    
    # Calculate physical length
    path_len = 0.0
    for i in range(len(path) - 1):
        u_pt = env.nodes[path[i]]
        v_pt = env.nodes[path[i+1]]
        path_len += np.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        
    return path, path_len

def h_to_tree(v_idx: int, incoming_dir: Optional[str], target_nodes: set, env: NonOrthogonalEnvironment, C_bend: float) -> float:
    """Computes minimum turn-penalized Manhattan distance from a node to a set of target nodes."""
    v_coords = env.nodes[v_idx]
    best_h = float('inf')
    for t_idx in target_nodes:
        t_coords = env.nodes[t_idx]
        h_val = (abs(v_coords[0] - t_coords[0]) + 
                 abs(v_coords[1] - t_coords[1]) + 
                 C_bend * estimate_turns(v_coords, incoming_dir, t_coords))
        if h_val < best_h:
            best_h = h_val
    return best_h

def state_expanded_astar_to_tree(
    env: NonOrthogonalEnvironment,
    start_idx: int,
    target_nodes: set,
    C_bend: float
) -> Tuple[Optional[List[int]], float]:
    """State-expanded A* searching from start_idx to any node in target_nodes."""
    if start_idx in target_nodes:
        return [start_idx], 0.0
        
    target_set = set(target_nodes)
    pq = []
    counter = 0
    
    h_start = h_to_tree(start_idx, None, target_set, env, C_bend)
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
        
        if u in target_set:
            best_target_state = state
            break
            
        for v, dist, edge_dir in env.adj.get(u, []):
            penalty = 0.0
            if u_dir is not None and u_dir != edge_dir:
                penalty = C_bend
            new_g = g + dist + penalty
            next_state = (v, edge_dir)
            
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                h = h_to_tree(v, edge_dir, target_set, env, C_bend)
                f_new = new_g + h
                came_from[next_state] = state
                heapq.heappush(pq, (f_new, new_g, counter, v, edge_dir))
                counter += 1
                
    if best_target_state is None:
        return None, float('inf')
        
    path = []
    curr = best_target_state
    while curr in came_from:
        path.append(curr[0])
        curr = came_from[curr]
    path.append(start_idx)
    path.reverse()
    
    path_len = 0.0
    for i in range(len(path) - 1):
        u_pt = env.nodes[path[i]]
        v_pt = env.nodes[path[i+1]]
        path_len += np.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        
    return path, path_len

def dual_graph_astar_to_tree(
    env: NonOrthogonalEnvironment,
    start_idx: int,
    target_nodes: set,
    C_bend: float
) -> Tuple[Optional[List[int]], float]:
    """A* search on Directed Dual Graph states from start_idx to any node in target_nodes."""
    if start_idx in target_nodes:
        return [start_idx], 0.0
        
    target_set = set(target_nodes)
    pq = []
    counter = 0
    
    g_scores = {}
    came_from = {}
    visited = set()
    
    for v, dist, edge_dir in env.adj.get(start_idx, []):
        state = (start_idx, v)
        g_scores[state] = dist
        h = h_to_tree(v, edge_dir, target_set, env, C_bend)
        heapq.heappush(pq, (dist + h, dist, counter, state))
        counter += 1
        
    best_target_state = None
    
    while pq:
        f, g, _, state = heapq.heappop(pq)
        
        if state in visited:
            continue
        visited.add(state)
        
        u, v = state
        if v in target_set:
            best_target_state = state
            break
            
        pu = env.nodes[u]
        pv = env.nodes[v]
        dx_uv = pv[0] - pu[0]
        dy_uv = pv[1] - pu[1]
        if abs(dx_uv) > abs(dy_uv):
            dir_uv = 'E' if dx_uv > 0 else 'W'
        else:
            dir_uv = 'N' if dy_uv > 0 else 'S'
            
        for w, dist, dir_vw in env.adj.get(v, []):
            if w == u:
                continue
            penalty = C_bend if dir_uv != dir_vw else 0.0
            new_g = g + dist + penalty
            next_state = (v, w)
            
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                h = h_to_tree(w, dir_vw, target_set, env, C_bend)
                came_from[next_state] = state
                heapq.heappush(pq, (new_g + h, new_g, counter, next_state))
                counter += 1
                
    if best_target_state is None:
        return None, float('inf')
        
    path = [best_target_state[1], best_target_state[0]]
    curr = best_target_state
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr[0])
    path.reverse()
    
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
        terminals = [tuple(self.env.nodes[t]) for t in self.env.terminal_indices]
        
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
                    x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                    
                    candidates = set()
                    for other in current_segs:
                        op1, op2 = other
                        if abs(op1[1] - op2[1]) < 1e-7:
                            candidates.add(op1[1])
                        else:
                            candidates.add(op1[1])
                            candidates.add(op2[1])
                            
                    for y_new in candidates:
                        if abs(y_new - y) < 1e-7:
                            continue
                            
                        shifted_seg = ((x1, y_new), (x2, y_new))
                        if not is_valid_segment(shifted_seg[0], shifted_seg[1], room, obstacles):
                            continue
                            
                        new_segs = []
                        valid_shift = True
                        terms_on_seg = [t for t in terminals if abs(t[1] - y) < 1e-7 and x1 - 1e-7 <= t[0] <= x2 + 1e-7]
                        connected_vert_xs = set()
                        
                        for jdx, other in enumerate(current_segs):
                            if jdx == idx:
                                continue
                            op1, op2 = other
                            is_other_vert = abs(op1[0] - op2[0]) < 1e-7
                            
                            if is_other_vert:
                                ox = op1[0]
                                oy1, oy2 = min(op1[1], op2[1]), max(op1[1], op2[1])
                                
                                if (x1 - 1e-7 <= ox <= x2 + 1e-7) and (oy1 - 1e-7 <= y <= oy2 + 1e-7):
                                    connected_vert_xs.add(ox)
                                    new_y1 = min(oy1, y_new)
                                    new_y2 = max(oy2, y_new)
                                    new_v = ((ox, new_y1), (ox, new_y2))
                                    if not is_valid_segment(new_v[0], new_v[1], room, obstacles):
                                        valid_shift = False
                                        break
                                    new_segs.append(new_v)
                                else:
                                    new_segs.append(other)
                            else:
                                new_segs.append(other)
                                
                        if not valid_shift:
                            continue
                            
                        for t in terms_on_seg:
                            tx, ty = t
                            if tx not in connected_vert_xs:
                                connector = ((tx, min(ty, y_new)), (tx, max(ty, y_new)))
                                if not is_valid_segment(connector[0], connector[1], room, obstacles):
                                    valid_shift = False
                                    break
                                new_segs.append(connector)
                                
                        if not valid_shift:
                            continue
                            
                        new_segs.append(shifted_seg)
                        temp_segs = merge_collinear_segments(new_segs)
                        new_turns = calculate_tree_turns(temp_segs)
                        new_length = sum(np.hypot(tp2[0]-tp1[0], tp2[1]-tp1[1]) for tp1, tp2 in temp_segs)
                        
                        if new_turns < current_turns or (new_turns == current_turns and new_length < current_length - 1e-5):
                            current_segs = temp_segs
                            improved = True
                            break
                    if improved:
                        break
                else:
                    x = p1[0]
                    y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                    
                    candidates = set()
                    for other in current_segs:
                        op1, op2 = other
                        if abs(op1[1] - op2[1]) < 1e-7:
                            candidates.add(op1[0])
                            candidates.add(op2[0])
                        else:
                            candidates.add(op1[0])
                            
                    for x_new in candidates:
                        if abs(x_new - x) < 1e-7:
                            continue
                            
                        shifted_seg = ((x_new, y1), (x_new, y2))
                        if not is_valid_segment(shifted_seg[0], shifted_seg[1], room, obstacles):
                            continue
                            
                        new_segs = []
                        valid_shift = True
                        terms_on_seg = [t for t in terminals if abs(t[0] - x) < 1e-7 and y1 - 1e-7 <= t[1] <= y2 + 1e-7]
                        connected_horiz_ys = set()
                        
                        for jdx, other in enumerate(current_segs):
                            if jdx == idx:
                                continue
                            op1, op2 = other
                            is_other_horiz = abs(op1[1] - op2[1]) < 1e-7
                            
                            if is_other_horiz:
                                oy = op1[1]
                                ox1, ox2 = min(op1[0], op2[0]), max(op1[0], op2[0])
                                
                                if (y1 - 1e-7 <= oy <= y2 + 1e-7) and (ox1 - 1e-7 <= x <= ox2 + 1e-7):
                                    connected_horiz_ys.add(oy)
                                    new_x1 = min(ox1, x_new)
                                    new_x2 = max(ox2, x_new)
                                    new_h = ((new_x1, oy), (new_x2, oy))
                                    if not is_valid_segment(new_h[0], new_h[1], room, obstacles):
                                        valid_shift = False
                                        break
                                    new_segs.append(new_h)
                                else:
                                    new_segs.append(other)
                            else:
                                new_segs.append(other)
                                
                        if not valid_shift:
                            continue
                            
                        for t in terms_on_seg:
                            tx, ty = t
                            if ty not in connected_horiz_ys:
                                connector = ((min(tx, x_new), ty), (max(tx, x_new), ty))
                                if not is_valid_segment(connector[0], connector[1], room, obstacles):
                                    valid_shift = False
                                    break
                                new_segs.append(connector)
                                
                        if not valid_shift:
                            continue
                            
                        new_segs.append(shifted_seg)
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

class BendAwareFastCornerSolver:
    """
    Constructive Steiner tree solver that greedily inserts corner candidates
    that minimize the turn-penalized MST cost.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float, max_steiner: int = 5):
        self.env = env
        self.C_bend = C_bend
        self.max_steiner = max_steiner
        
    def _evaluate_cost(self, nodes_list: List[int]) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        num_nodes = len(nodes_list)
        sub_dist = np.zeros((num_nodes, num_nodes))
        paths = {}
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                u, v = nodes_list[i], nodes_list[j]
                path, path_len = state_expanded_astar(self.env, u, v, self.C_bend)
                if path is None:
                    return float('inf'), []
                turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                cost = path_len + self.C_bend * turns
                sub_dist[i, j] = cost
                sub_dist[j, i] = cost
                paths[(u, v)] = path
                
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        
        subgraph_edges = set()
        for u_idx, v_idx in zip(u_sub, v_sub):
            u_node, v_node = nodes_list[u_idx], nodes_list[v_idx]
            path = paths.get((u_node, v_node)) or paths.get((v_node, u_node))[::-1]
            for i in range(len(path) - 1):
                subgraph_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                
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
                
        adj = {node: set() for node in subgraph_nodes}
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        terminal_set = set(self.env.terminal_indices)
        while True:
            leaves = [node for node, nbrs in adj.items() if len(nbrs) <= 1 and node not in terminal_set]
            if not leaves:
                break
            for leaf in leaves:
                for nbr in adj[leaf]:
                    adj[nbr].remove(leaf)
                del adj[leaf]
                
        final_segments = []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                           (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
                                           
        merged_segs = merge_collinear_segments(final_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return total_len + self.C_bend * turns, merged_segs

    def solve(self) -> Dict[str, Any]:
        active_nodes = list(self.env.terminal_indices)
        current_cost, current_segs = self._evaluate_cost(active_nodes)
        
        for _ in range(self.max_steiner):
            candidates = [i for i in range(len(self.env.nodes)) if i not in active_nodes]
            if not candidates:
                break
                
            best_cand = -1
            best_cost = current_cost
            best_segs = current_segs
            
            for cand in candidates:
                cost, segs = self._evaluate_cost(active_nodes + [cand])
                if cost < best_cost - 1e-5:
                    best_cost = cost
                    best_cand = cand
                    best_segs = segs
                    
            if best_cand != -1:
                active_nodes.append(best_cand)
                current_cost = best_cost
                current_segs = best_segs
            else:
                break
                
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in current_segs)
        turns = calculate_tree_turns(current_segs)
        return {
            "weight": total_len,
            "segments": current_segs,
            "turns": turns
        }

class BendAwareStochasticKMBSolver:
    """
    Stochastic KMB solver.
    Runs KMB multiple times, each time perturbing the edge lengths with random noise
    to explore alternative topologies, and selects the best result based on the
    true, unperturbed turn-penalized cost.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float, n_trials: int = 15, perturbation: float = 0.15, seed: Optional[int] = 42):
        self.env = env
        self.C_bend = C_bend
        self.n_trials = n_trials
        self.perturbation = perturbation
        self.rng = np.random.default_rng(seed)
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        num_t = len(terminals)
        
        best_weight = float('inf')
        best_segs = []
        best_turns = 0
        
        def run_kmb_trial(edge_weights):
            paths = {}
            closure_costs = {}
            for i in range(num_t):
                for j in range(i+1, num_t):
                    u, v = terminals[i], terminals[j]
                    path, path_len = state_expanded_astar(self.env, u, v, self.C_bend, edge_weights)
                    if path is None:
                        return None
                    
                    turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                    cost = path_len + self.C_bend * turns
                    paths[(u, v)] = path
                    paths[(v, u)] = path[::-1]
                    closure_costs[(i, j)] = cost
                    closure_costs[(j, i)] = cost
                    
            sub_dist = np.zeros((num_t, num_t))
            for i in range(num_t):
                for j in range(i+1, num_t):
                    sub_dist[i, j] = closure_costs[(i, j)]
                    sub_dist[j, i] = closure_costs[(i, j)]
                    
            mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
            u_sub, v_sub = mst_sparse.nonzero()
            
            subgraph_edges = set()
            for u_idx, v_idx in zip(u_sub, v_sub):
                u_node, v_node = terminals[u_idx], terminals[v_idx]
                path = paths[(u_node, v_node)]
                for i in range(len(path) - 1):
                    subgraph_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                    
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
                    
            final_segments = []
            for u, nbrs in adj.items():
                for v in nbrs:
                    if u < v:
                        final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                               (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
            return merge_collinear_segments(final_segments)

        base_segs = run_kmb_trial(None)
        if base_segs is not None:
            base_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in base_segs)
            base_turns = calculate_tree_turns(base_segs)
            best_weight = base_len + self.C_bend * base_turns
            best_segs = base_segs
            best_turns = base_turns
            
        unique_edges = []
        for u, neighbors in self.env.adj.items():
            for v, dist, _ in neighbors:
                if u < v:
                    unique_edges.append((u, v, dist))
                    
        for _ in range(self.n_trials):
            perturbed_weights = {}
            for u, v, dist in unique_edges:
                factor = 1.0 + self.rng.uniform(-self.perturbation, self.perturbation)
                perturbed_weights[(u, v)] = dist * factor
                
            trial_segs = run_kmb_trial(perturbed_weights)
            if trial_segs is None:
                continue
                
            trial_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in trial_segs)
            trial_turns = calculate_tree_turns(trial_segs)
            trial_cost = trial_len + self.C_bend * trial_turns
            
            if trial_cost < best_weight:
                best_weight = trial_cost
                best_segs = trial_segs
                best_turns = trial_turns
                
        actual_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in best_segs)
        return {
            "weight": actual_len,
            "segments": best_segs,
            "turns": best_turns
        }

class BendAwarePruneSolver:
    """
    Top-Down Dense Grid Pruning.
    Starts with all grid nodes active, and iteratively removes non-terminal nodes
    with degree <= 2 in the turn-penalized MST to simplify the tree structure.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def _evaluate_cost(self, nodes_list: List[int]) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        num_nodes = len(nodes_list)
        if num_nodes < 2:
            return float('inf'), []
            
        sub_dist = np.zeros((num_nodes, num_nodes))
        paths = {}
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                u, v = nodes_list[i], nodes_list[j]
                path, path_len = state_expanded_astar(self.env, u, v, self.C_bend)
                if path is None:
                    return float('inf'), []
                turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                cost = path_len + self.C_bend * turns
                sub_dist[i, j] = cost
                sub_dist[j, i] = cost
                paths[(u, v)] = path
                
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        
        subgraph_edges = set()
        for u_idx, v_idx in zip(u_sub, v_sub):
            u_node, v_node = nodes_list[u_idx], nodes_list[v_idx]
            path = paths.get((u_node, v_node)) or paths.get((v_node, u_node))[::-1]
            for i in range(len(path) - 1):
                subgraph_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                
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
                
        adj = {node: set() for node in subgraph_nodes}
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        terminal_set = set(self.env.terminal_indices)
        while True:
            leaves = [node for node, nbrs in adj.items() if len(nbrs) <= 1 and node not in terminal_set]
            if not leaves:
                break
            for leaf in leaves:
                for nbr in adj[leaf]:
                    adj[nbr].remove(leaf)
                del adj[leaf]
                
        final_segments = []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                           (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
                                           
        merged_segs = merge_collinear_segments(final_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return total_len + self.C_bend * turns, merged_segs

    def solve(self) -> Dict[str, Any]:
        active_nodes = list(range(len(self.env.nodes)))
        current_cost, current_segs = self._evaluate_cost(active_nodes)
        
        terminal_set = set(self.env.terminal_indices)
        
        while True:
            deg_dict = {}
            for p1, p2 in current_segs:
                u_idx = self.env.node_map.get((round(p1[0], 7), round(p1[1], 7)))
                v_idx = self.env.node_map.get((round(p2[0], 7), round(p2[1], 7)))
                if u_idx is not None and v_idx is not None:
                    deg_dict[u_idx] = deg_dict.get(u_idx, 0) + 1
                    deg_dict[v_idx] = deg_dict.get(v_idx, 0) + 1
                    
            prunables = [node for node in active_nodes if node not in terminal_set and deg_dict.get(node, 0) <= 2]
            if not prunables:
                break
                
            pruned_any = False
            for node in prunables:
                temp_nodes = [n for n in active_nodes if n != node]
                cost, segs = self._evaluate_cost(temp_nodes)
                if cost <= current_cost + 1e-5:
                    active_nodes = temp_nodes
                    current_cost = cost
                    current_segs = segs
                    pruned_any = True
                    break
                    
            if not pruned_any:
                break
                
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in current_segs)
        turns = calculate_tree_turns(current_segs)
        return {
            "weight": total_len,
            "segments": current_segs,
            "turns": turns
        }

class BendAwareDualGraphKMBSolver:
    """
    Kou-Markowsky-Berman (KMB) algorithm using dual-graph A* shortest paths.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        num_t = len(terminals)
        
        # Step 1: Compute dual graph A* shortest paths between all terminal pairs
        paths = {}
        closure_costs = {}
        for i in range(num_t):
            for j in range(i+1, num_t):
                u, v = terminals[i], terminals[j]
                path, path_len = dual_graph_astar(self.env, u, v, self.C_bend)
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

class BendAwareDualGraphGBFSSolver:
    """
    Kou-Markowsky-Berman (KMB) algorithm using dual-graph GBFS shortest paths.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        num_t = len(terminals)
        
        # Step 1: Compute dual graph GBFS shortest paths between all terminal pairs
        paths = {}
        closure_costs = {}
        for i in range(num_t):
            for j in range(i+1, num_t):
                u, v = terminals[i], terminals[j]
                path, path_len = dual_graph_gbfs(self.env, u, v, self.C_bend)
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

class BendAwareDualGraphFastCornerSolver:
    """
    Constructive Steiner tree solver that greedily inserts corner candidates
    that minimize the turn-penalized MST cost, using dual-graph A*.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float, max_steiner: int = 5):
        self.env = env
        self.C_bend = C_bend
        self.max_steiner = max_steiner
        
    def _evaluate_cost(self, nodes_list: List[int]) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        num_nodes = len(nodes_list)
        sub_dist = np.zeros((num_nodes, num_nodes))
        paths = {}
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                u, v = nodes_list[i], nodes_list[j]
                path, path_len = dual_graph_astar(self.env, u, v, self.C_bend)
                if path is None:
                    return float('inf'), []
                turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                cost = path_len + self.C_bend * turns
                sub_dist[i, j] = cost
                sub_dist[j, i] = cost
                paths[(u, v)] = path
                
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        
        subgraph_edges = set()
        for u_idx, v_idx in zip(u_sub, v_sub):
            u_node, v_node = nodes_list[u_idx], nodes_list[v_idx]
            path = paths.get((u_node, v_node)) or paths.get((v_node, u_node))[::-1]
            for i in range(len(path) - 1):
                subgraph_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                
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
                
        adj = {node: set() for node in subgraph_nodes}
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        terminal_set = set(self.env.terminal_indices)
        while True:
            leaves = [node for node, nbrs in adj.items() if len(nbrs) <= 1 and node not in terminal_set]
            if not leaves:
                break
            for leaf in leaves:
                for nbr in adj[leaf]:
                    adj[nbr].remove(leaf)
                del adj[leaf]
                
        final_segments = []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                           (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
                                           
        merged_segs = merge_collinear_segments(final_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return total_len + self.C_bend * turns, merged_segs

    def solve(self) -> Dict[str, Any]:
        active_nodes = list(self.env.terminal_indices)
        current_cost, current_segs = self._evaluate_cost(active_nodes)
        
        for _ in range(self.max_steiner):
            candidates = [i for i in range(len(self.env.nodes)) if i not in active_nodes]
            if not candidates:
                break
                
            best_cand = -1
            best_cost = current_cost
            best_segs = current_segs
            
            for cand in candidates:
                cost, segs = self._evaluate_cost(active_nodes + [cand])
                if cost < best_cost - 1e-5:
                    best_cost = cost
                    best_cand = cand
                    best_segs = segs
                    
            if best_cand != -1:
                active_nodes.append(best_cand)
                current_cost = best_cost
                current_segs = best_segs
            else:
                break
                
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in current_segs)
        turns = calculate_tree_turns(current_segs)
        return {
            "weight": total_len,
            "segments": current_segs,
            "turns": turns
        }

class BendAwareDualGraphGBFSFastCornerSolver:
    """
    Constructive Steiner tree solver that greedily inserts corner candidates
    that minimize the turn-penalized MST cost, using dual-graph GBFS.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float, max_steiner: int = 5):
        self.env = env
        self.C_bend = C_bend
        self.max_steiner = max_steiner
        
    def _evaluate_cost(self, nodes_list: List[int]) -> Tuple[float, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        num_nodes = len(nodes_list)
        sub_dist = np.zeros((num_nodes, num_nodes))
        paths = {}
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                u, v = nodes_list[i], nodes_list[j]
                path, path_len = dual_graph_gbfs(self.env, u, v, self.C_bend)
                if path is None:
                    return float('inf'), []
                turns = calculate_tree_turns([(tuple(self.env.nodes[path[k]]), tuple(self.env.nodes[path[k+1]])) for k in range(len(path)-1)])
                cost = path_len + self.C_bend * turns
                sub_dist[i, j] = cost
                sub_dist[j, i] = cost
                paths[(u, v)] = path
                
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        
        subgraph_edges = set()
        for u_idx, v_idx in zip(u_sub, v_sub):
            u_node, v_node = nodes_list[u_idx], nodes_list[v_idx]
            path = paths.get((u_node, v_node)) or paths.get((v_node, u_node))[::-1]
            for i in range(len(path) - 1):
                subgraph_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                
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
                
        adj = {node: set() for node in subgraph_nodes}
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        terminal_set = set(self.env.terminal_indices)
        while True:
            leaves = [node for node, nbrs in adj.items() if len(nbrs) <= 1 and node not in terminal_set]
            if not leaves:
                break
            for leaf in leaves:
                for nbr in adj[leaf]:
                    adj[nbr].remove(leaf)
                del adj[leaf]
                
        final_segments = []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    final_segments.append(((float(self.env.nodes[u][0]), float(self.env.nodes[u][1])),
                                           (float(self.env.nodes[v][0]), float(self.env.nodes[v][1]))))
                                           
        merged_segs = merge_collinear_segments(final_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return total_len + self.C_bend * turns, merged_segs

    def solve(self) -> Dict[str, Any]:
        active_nodes = list(self.env.terminal_indices)
        current_cost, current_segs = self._evaluate_cost(active_nodes)
        
        for _ in range(self.max_steiner):
            candidates = [i for i in range(len(self.env.nodes)) if i not in active_nodes]
            if not candidates:
                break
                
            best_cand = -1
            best_cost = current_cost
            best_segs = current_segs
            
            for cand in candidates:
                cost, segs = self._evaluate_cost(active_nodes + [cand])
                if cost < best_cost - 1e-5:
                    best_cost = cost
                    best_cand = cand
                    best_segs = segs
                    
            if best_cand != -1:
                active_nodes.append(best_cand)
                current_cost = best_cost
                current_segs = best_segs
            else:
                break
                
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in current_segs)
        turns = calculate_tree_turns(current_segs)
        return {
            "weight": total_len,
            "segments": current_segs,
            "turns": turns
        }

class StateExpandedSequentialFastCornerSolver:
    """
    Sequential Steiner tree solver that connects each terminal in turn
    to the growing tree using state-expanded A* to tree.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        if not terminals:
            return {"weight": 0.0, "segments": [], "turns": 0}
            
        tree_nodes = {terminals[0]}
        raw_segments = []
        
        for t_i in terminals[1:]:
            path, _ = state_expanded_astar_to_tree(self.env, t_i, tree_nodes, self.C_bend)
            if path is None:
                raise ValueError(f"No path found connecting terminal {t_i} to the growing tree.")
            for i in range(len(path) - 1):
                node_a, node_b = path[i], path[i+1]
                raw_segments.append(((float(self.env.nodes[node_a][0]), float(self.env.nodes[node_a][1])),
                                     (float(self.env.nodes[node_b][0]), float(self.env.nodes[node_b][1]))))
            tree_nodes.update(path)
            
        merged_segs = merge_collinear_segments(raw_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return {
            "weight": total_len,
            "segments": merged_segs,
            "turns": turns
        }

class DualGraphSequentialFastCornerSolver:
    """
    Sequential Steiner tree solver that connects each terminal in turn
    to the growing tree using dual-graph A* to tree.
    """
    def __init__(self, env: NonOrthogonalEnvironment, C_bend: float):
        self.env = env
        self.C_bend = C_bend
        
    def solve(self) -> Dict[str, Any]:
        terminals = self.env.terminal_indices
        if not terminals:
            return {"weight": 0.0, "segments": [], "turns": 0}
            
        tree_nodes = {terminals[0]}
        raw_segments = []
        
        for t_i in terminals[1:]:
            path, _ = dual_graph_astar_to_tree(self.env, t_i, tree_nodes, self.C_bend)
            if path is None:
                raise ValueError(f"No path found connecting terminal {t_i} to the growing tree.")
            for i in range(len(path) - 1):
                node_a, node_b = path[i], path[i+1]
                raw_segments.append(((float(self.env.nodes[node_a][0]), float(self.env.nodes[node_a][1])),
                                     (float(self.env.nodes[node_b][0]), float(self.env.nodes[node_b][1]))))
            tree_nodes.update(path)
            
        merged_segs = merge_collinear_segments(raw_segments)
        total_len = sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in merged_segs)
        turns = calculate_tree_turns(merged_segs)
        return {
            "weight": total_len,
            "segments": merged_segs,
            "turns": turns
        }
