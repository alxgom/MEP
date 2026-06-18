import numpy as np
from typing import List, Tuple
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

class NonOrthogonalEnvironment:
    """
    Generalized Orthogonal Grid Environment.
    Supports arbitrary non-orthogonal / rotated obstacles and rooms using Shapely.
    Constructs a boundary-conforming orthogonal grid by projecting rays from
    interest points and finding their intersections.
    """
    def __init__(self, room: Polygon, obstacles: List[Polygon], terminals: List[Tuple[float, float]]):
        self.room = room
        self.obstacles = obstacles
        self.terminals = terminals
        
        # 1. Collect Interest Points
        interest_points = set()
        for t in terminals:
            interest_points.add((float(t[0]), float(t[1])))
            
        # Vertices of the room boundary
        for pt in room.exterior.coords:
            interest_points.add((float(pt[0]), float(pt[1])))
        for interior in room.interiors:
            for pt in interior.coords:
                interest_points.add((float(pt[0]), float(pt[1])))
                
        # Vertices of all obstacle boundaries
        for obs in obstacles:
            for pt in obs.exterior.coords:
                interest_points.add((float(pt[0]), float(pt[1])))
            for interior in obs.interiors:
                for pt in interior.coords:
                    interest_points.add((float(pt[0]), float(pt[1])))
                    
        # Round and deduplicate interest points to avoid floating-point duplicates
        unique_interests = []
        seen_interests = set()
        for pt in interest_points:
            key = (round(pt[0], 7), round(pt[1], 7))
            if key not in seen_interests:
                seen_interests.add(key)
                unique_interests.append(pt)
                
        # Bounding box limits for ray tracing
        min_x, min_y, max_x, max_y = room.bounds
        dx = max_x - min_x
        dy = max_y - min_y
        max_x_bound = max_x + 0.1 * dx + 10.0
        min_x_bound = min_x - 0.1 * dx - 10.0
        max_y_bound = max_y + 0.1 * dy + 10.0
        min_y_bound = min_y - 0.1 * dy - 10.0
        
        # Boundary elements to intersect rays with
        boundaries = [room.boundary] + [obs.boundary for obs in obstacles]
        
        def extract_points(geom):
            pts = []
            if geom.is_empty:
                return pts
            if geom.geom_type == 'Point':
                pts.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                for p in geom.geoms:
                    pts.append((p.x, p.y))
            elif geom.geom_type in ('LineString', 'LinearRing'):
                for coord in geom.coords:
                    pts.append(coord)
            elif geom.geom_type in ('MultiLineString', 'GeometryCollection'):
                for g in geom.geoms:
                    pts.extend(extract_points(g))
            return pts
            
        # 2. Project Horizontal and Vertical Rays from Interest Points
        h_segments = [] # List of (y, x_min, x_max)
        v_segments = [] # List of (x, y_min, y_max)
        
        for x, y in unique_interests:
            # +x Ray (East)
            ray_e = LineString([(x, y), (max_x_bound, y)])
            pts_e = []
            for b in boundaries:
                inter = ray_e.intersection(b)
                if not inter.is_empty:
                    pts_e.extend(extract_points(inter))
            valid_e = [pt for pt in pts_e if pt[0] > x + 1e-7]
            if valid_e:
                closest_e = min(valid_e, key=lambda pt: pt[0])
                h_segments.append((y, x, closest_e[0]))
                
            # -x Ray (West)
            ray_w = LineString([(x, y), (min_x_bound, y)])
            pts_w = []
            for b in boundaries:
                inter = ray_w.intersection(b)
                if not inter.is_empty:
                    pts_w.extend(extract_points(inter))
            valid_w = [pt for pt in pts_w if pt[0] < x - 1e-7]
            if valid_w:
                closest_w = max(valid_w, key=lambda pt: pt[0])
                h_segments.append((y, closest_w[0], x))
                
            # +y Ray (North)
            ray_n = LineString([(x, y), (x, max_y_bound)])
            pts_n = []
            for b in boundaries:
                inter = ray_n.intersection(b)
                if not inter.is_empty:
                    pts_n.extend(extract_points(inter))
            valid_n = [pt for pt in pts_n if pt[1] > y + 1e-7]
            if valid_n:
                closest_n = min(valid_n, key=lambda pt: pt[1])
                v_segments.append((x, y, closest_n[1]))
                
            # -y Ray (South)
            ray_s = LineString([(x, y), (x, min_y_bound)])
            pts_s = []
            for b in boundaries:
                inter = ray_s.intersection(b)
                if not inter.is_empty:
                    pts_s.extend(extract_points(inter))
            valid_s = [pt for pt in pts_s if pt[1] < y - 1e-7]
            if valid_s:
                closest_s = max(valid_s, key=lambda pt: pt[1])
                v_segments.append((x, closest_s[1], y))
                
        # 3. Compute all ray-ray intersection points
        ray_intersections = set()
        for y_h, h_x1, h_x2 in h_segments:
            for x_v, v_y1, v_y2 in v_segments:
                if (h_x1 - 1e-7 <= x_v <= h_x2 + 1e-7) and (v_y1 - 1e-7 <= y_h <= v_y2 + 1e-7):
                    ray_intersections.add((float(x_v), float(y_h)))
                    
        # 4. Gather and filter candidate nodes
        all_raw_nodes = []
        # Add interest points
        all_raw_nodes.extend(unique_interests)
        # Add ray endpoints
        for y, x1, x2 in h_segments:
            all_raw_nodes.append((x1, y))
            all_raw_nodes.append((x2, y))
        for x, y1, y2 in v_segments:
            all_raw_nodes.append((x, y1))
            all_raw_nodes.append((x, y2))
        # Add ray-ray intersections
        all_raw_nodes.extend(ray_intersections)
        
        # Deduplicate nodes using coordinate rounding
        unique_nodes = []
        seen_nodes = set()
        for pt in all_raw_nodes:
            key = (round(pt[0], 7), round(pt[1], 7))
            if key not in seen_nodes:
                seen_nodes.add(key)
                unique_nodes.append(pt)
                
        # Filter nodes: must be inside the room and not strictly inside any obstacle
        filtered_nodes = []
        for p in unique_nodes:
            pt = Point(p)
            if room.distance(pt) > 1e-7:
                continue
            inside_obs = False
            for obs in obstacles:
                if obs.contains(pt):
                    inside_obs = True
                    break
            if not inside_obs:
                filtered_nodes.append(p)
                
        # Guarantee all terminals are in the final node set
        for t in terminals:
            found = False
            for fn in filtered_nodes:
                if np.hypot(fn[0] - t[0], fn[1] - t[1]) < 1e-7:
                    found = True
                    break
            if not found:
                filtered_nodes.append((float(t[0]), float(t[1])))
                
        self.nodes = np.array(filtered_nodes)
        self.node_map = {(float(n[0]), float(n[1])): i for i, n in enumerate(self.nodes)}
        
        # Locate terminal node indices
        self.terminal_indices = []
        for t in terminals:
            key = (float(t[0]), float(t[1]))
            best_idx = None
            best_dist = 1e9
            for k, idx in self.node_map.items():
                d = np.hypot(k[0] - t[0], k[1] - t[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_dist < 1e-5:
                self.terminal_indices.append(best_idx)
            else:
                raise ValueError(f"Terminal {t} not found in the grid map.")
                
        # 5. Connect adjacent nodes along horizontal and vertical ray segments
        edges_dict = {}
        
        def is_valid_segment(p1, p2):
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
            
        # Connect along horizontal segments
        for y, x_min, x_max in h_segments:
            seg_nodes = []
            for (nx, ny), idx in self.node_map.items():
                if abs(ny - y) < 1e-7 and (x_min - 1e-7 <= nx <= x_max + 1e-7):
                    seg_nodes.append((nx, idx))
            seg_nodes.sort(key=lambda item: item[0])
            for i in range(len(seg_nodes) - 1):
                idx1, idx2 = seg_nodes[i][1], seg_nodes[i+1][1]
                p1, p2 = self.nodes[idx1], self.nodes[idx2]
                if is_valid_segment(p1, p2):
                    d = abs(p1[0] - p2[0])
                    u, v = min(idx1, idx2), max(idx1, idx2)
                    edges_dict[(u, v)] = d
                    
        # Connect along vertical segments
        for x, y_min, y_max in v_segments:
            seg_nodes = []
            for (nx, ny), idx in self.node_map.items():
                if abs(nx - x) < 1e-7 and (y_min - 1e-7 <= ny <= y_max + 1e-7):
                    seg_nodes.append((ny, idx))
            seg_nodes.sort(key=lambda item: item[0])
            for i in range(len(seg_nodes) - 1):
                idx1, idx2 = seg_nodes[i][1], seg_nodes[i+1][1]
                p1, p2 = self.nodes[idx1], self.nodes[idx2]
                if is_valid_segment(p1, p2):
                    d = abs(p1[1] - p2[1])
                    u, v = min(idx1, idx2), max(idx1, idx2)
                    edges_dict[(u, v)] = d
                    
        # Build adjacency list: u -> list of (v, weight, direction)
        self.adj = {i: [] for i in range(len(self.nodes))}
        for (u, v), d in edges_dict.items():
            pu, pv = self.nodes[u], self.nodes[v]
            dx = pv[0] - pu[0]
            dy = pv[1] - pu[1]
            if abs(dx) > abs(dy):
                dir_uv = 'E' if dx > 0 else 'W'
                dir_vu = 'W' if dx > 0 else 'E'
            else:
                dir_uv = 'N' if dy > 0 else 'S'
                dir_vu = 'S' if dy > 0 else 'N'
            self.adj[u].append((v, d, dir_uv))
            self.adj[v].append((u, d, dir_vu))
