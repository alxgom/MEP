import sys
import os
import math
import time
import itertools
import heapq
import numpy as np
import pygame
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale
from shapely.prepared import prep as shapely_prep
from scipy.spatial import cKDTree

# Add relative paths to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '08-bend-aware-non-orthogonal')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import generative_layout
from solver import state_expanded_astar, calculate_tree_turns

# Constants
SCALE_TO_MM    = 1000.0
WALL_THICKNESS = 150.0
GRID_SPACING   = 200    # mm — regular routing grid resolution
FPS            = 20
C_BEND         = 4000.0  # Turn penalty in mm

# Graph types
GRAPH_TYPES = [
    "Regular 200mm Grid",
    "Hannan Grid (numpy)",
    "Hannan + Shifted Nodes",
]

ROUTING_STRATEGIES = [
    "Greedy (Dual-Sort)",
    "First Fit",
    "Best Fit",
    "Negotiated Congestion"
]
routing_strategy_idx = 1

AUTO_PLACEMENT_MODES = [
    "Manual",
    "Proximity (Option 1)",
    "Topological Fields (Option 2)"
]
auto_placement_mode_idx = 0
show_heatmap = False

# Pygame Window Config
WINDOW_WIDTH, WINDOW_HEIGHT = 1250, 850
CANVAS_LEFT = 280
CANVAS_TOP = 40
CANVAS_W = WINDOW_WIDTH - CANVAS_LEFT - 40
CANVAS_H = WINDOW_HEIGHT - CANVAS_TOP - 40

# Color Scheme
COLOR_BG = (28, 28, 36)
COLOR_PANEL = (35, 35, 45)
COLOR_ROOM = (45, 52, 54)
COLOR_ROOM_COVERED = (52, 73, 94) # Slate Blue for false ceilings
COLOR_HALLWAY = (57, 72, 85)
COLOR_WALL = (99, 110, 114)
COLOR_COLUMN = (15, 76, 129)
COLOR_SHAFT = (231, 76, 60)
COLOR_SHAFT_BG = (231, 76, 60, 40)
COLOR_DOOR = (220, 220, 220)
COLOR_MACHINE = (127, 140, 141)
COLOR_MACHINE_HOVER = (149, 165, 166)
COLOR_TEXT = (236, 240, 241)
COLOR_MUTED = (149, 165, 166)

# Route Colors
ROUTE_COLORS = {
    "Shaft": (46, 204, 113),      # Green
    "Kitchen": (241, 196, 15),    # Yellow
    "Bathroom": (52, 152, 219),   # Blue
    "Bathroom 1": (52, 152, 219), # Blue
    "Bathroom 2": (155, 89, 182), # Purple
    "Toilet": (230, 126, 34),     # Orange
    "Washroom": (26, 188, 156)    # Turquoise
}

SCALE_PX_PER_MM = min(CANVAS_W / 15000.0, CANVAS_H / 11000.0)
OFFSET_X = CANVAS_LEFT + (CANVAS_W - 15000.0 * SCALE_PX_PER_MM) / 2
OFFSET_Y = CANVAS_TOP + (CANVAS_H - 11000.0 * SCALE_PX_PER_MM) / 2

def to_screen(x, y):
    sx = OFFSET_X + x * SCALE_PX_PER_MM
    sy = OFFSET_Y + (11000.0 - y) * SCALE_PX_PER_MM
    return int(sx), int(sy)

def to_mm(sx, sy):
    x = (sx - OFFSET_X) / SCALE_PX_PER_MM
    y = 11000.0 - (sy - OFFSET_Y) / SCALE_PX_PER_MM
    return x, y

class EnvView:
    def __init__(self, nodes, adj):
        self.nodes = nodes
        self.adj   = adj

def snap_to_integer_grid(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        ext = [(round(x), round(y)) for x, y in geom.exterior.coords]
        ints = [[(round(x), round(y)) for x, y in interior.coords]
                for interior in geom.interiors]
        return Polygon(ext, ints)
    elif geom.geom_type == 'LineString':
        return LineString([(round(x), round(y)) for x, y in geom.coords])
    elif geom.geom_type in ('MultiLineString', 'MultiPolygon', 'GeometryCollection'):
        return unary_union([snap_to_integer_grid(g) for g in geom.geoms])
    return geom

# Global layout variables
rooms = []
columns = []
shafts = []
doors = []
walls = []
wall_polys = []
routing_region_base = None
shaft_extraction = None
terminals        = {}
wet_room_names   = []
machine_cx    = 0.0
machine_cy    = 0.0
machine_angle = 0

# Grid cache
graph_type_idx  = 1
grid_nodes      = None
grid_adj_base   = None
grid_edge_list  = None
grid_edge_coords= None
grid_kd         = None
current_env     = None
_bnd_segs       = None
show_grid_graph = False

# Auto-placement cache
ap_scores = {}
ap_fields = {}

def get_representative_point(poly):
    centroid = poly.centroid
    if poly.contains(centroid):
        return (round(centroid.x), round(centroid.y))
    rep = poly.representative_point()
    return (round(rep.x), round(rep.y))

# NUMPY HELPERS FOR HANNAN RAY CASTING
def get_segment_segment_intersections_np(segs_a, segs_b):
    A1 = segs_a[:, 0:2]
    A2 = segs_a[:, 2:4]
    B1 = segs_b[:, 0:2]
    B2 = segs_b[:, 2:4]
    
    dA = A2 - A1
    dB = B2 - B1
    
    T = dB[:, None, 0] * dA[None, :, 1] - dB[:, None, 1] * dA[None, :, 0]
    
    mask_T = np.abs(T) > 1e-8
    
    denom = np.where(mask_T, T, 1.0)
    
    num_u = dA[None, :, 0] * (A1[None, :, 1] - B1[:, None, 1]) - dA[None, :, 1] * (A1[None, :, 0] - B1[:, None, 0])
    num_t = dB[:, None, 0] * (A1[None, :, 1] - B1[:, None, 1]) - dB[:, None, 1] * (A1[None, :, 0] - B1[:, None, 0])
    
    u = num_u / denom
    t = num_t / denom
    
    valid = mask_T & (u >= 0.0) & (u <= 1.0) & (t >= 0.0) & (t <= 1.0)
    
    pts = B1[:, None, :] + u[:, :, None] * dB[:, None, :]
    
    idx_b, idx_a = np.where(valid)
    return pts[idx_b, idx_a]

def generate_hannan_nodes_vectorized(walls_list, columns_list, shafts_list, machine_pins=None):
    t0 = time.perf_counter()
    
    all_segs = []
    for w in walls_list:
        coords = list(w.coords)
        for i in range(len(coords) - 1):
            all_segs.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
            
    for col in columns_list:
        coords = list(col.exterior.coords)
        for i in range(len(coords) - 1):
            all_segs.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
            
    for shaft in shafts_list:
        coords = list(shaft.exterior.coords)
        for i in range(len(coords) - 1):
            all_segs.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
            
    if machine_pins:
        pts = [machine_pins[k] for k in ("tl", "tr", "br", "bl")]
        for i in range(4):
            p1, p2 = pts[i], pts[(i+1)%4]
            all_segs.append((p1[0], p1[1], p2[0], p2[1]))
            
    segs_np = np.array(all_segs, dtype=np.float32)
    if len(segs_np) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0.0
        
    t_interest = time.perf_counter()
    x_coords = np.concatenate([segs_np[:, 0], segs_np[:, 2]])
    y_coords = np.concatenate([segs_np[:, 1], segs_np[:, 3]])
    
    x_coords = np.unique(np.round(x_coords))
    y_coords = np.unique(np.round(y_coords))
    
    t_bnd = time.perf_counter()
    min_x, max_x = 0.0, 15000.0
    min_y, max_y = 0.0, 11000.0
    
    t_rays = time.perf_counter()
    horiz_rays = []
    for y in y_coords:
        if min_y <= y <= max_y:
            horiz_rays.append((min_x, y, max_x, y))
            
    vert_rays = []
    for x in x_coords:
        if min_x <= x <= max_x:
            vert_rays.append((x, min_y, x, max_y))
            
    hrays_np = np.array(horiz_rays, dtype=np.float32)
    vrays_np = np.array(vert_rays, dtype=np.float32)
    
    t_cross = time.perf_counter()
    grid_pts = []
    if len(hrays_np) > 0 and len(vrays_np) > 0:
        grid_pts = get_segment_segment_intersections_np(hrays_np, vrays_np)
        
    t_filter = time.perf_counter()
    if len(grid_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), 0.0
        
    grid_pts = np.unique(np.round(grid_pts), axis=0)
    
    if routing_region_base:
        prep_region = shapely_prep(routing_region_base)
        inside_mask = [prep_region.contains(Point(pt[0], pt[1])) for pt in grid_pts]
        filtered_pts = grid_pts[inside_mask]
    else:
        filtered_pts = grid_pts
        
    elapsed = (time.perf_counter() - t0) * 1000.0
    return filtered_pts, elapsed

def add_door_points_to_grid(nodes_np):
    if len(doors) == 0:
        return nodes_np
    door_pts = []
    for d in doors:
        # Doors are dictionaries of segment endpoints
        # Let's compute door center point
        cx = 0.5 * (d["d1"][0] + d["d2"][0])
        cy = 0.5 * (d["d1"][1] + d["d2"][1])
        door_pts.append((cx, cy))
        
    door_np = np.array(door_pts, dtype=np.float32)
    combined = np.concatenate([nodes_np, door_np], axis=0)
    return np.unique(np.round(combined), axis=0)

def build_hannan_grid(machine_pins=None, shift_walls=False):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, _bnd_segs
    
    nodes_np, _ = generate_hannan_nodes_vectorized(walls, columns, shafts, machine_pins)
    
    if shift_walls:
        shifted_nodes = []
        for w in walls:
            coords = list(w.coords)
            for i in range(len(coords) - 1):
                p1, p2 = np.array(coords[i]), np.array(coords[i+1])
                d = p2 - p1
                L = np.linalg.norm(d)
                if L > 1e-5:
                    d_norm = d / L
                    normal = np.array([-d_norm[1], d_norm[0]])
                    # Shift 100mm in both normal directions
                    for side in (-100.0, 100.0):
                        offset = normal * side
                        p1_off = p1 + offset
                        p2_off = p2 + offset
                        for t in np.linspace(0, 1, int(L / 400.0) + 2):
                            pt = p1_off + t * (p2_off - p1_off)
                            shifted_nodes.append((round(pt[0]), round(pt[1])))
                            
        if shifted_nodes:
            shift_np = np.array(shifted_nodes, dtype=np.float32)
            if routing_region_base:
                prep_region = shapely_prep(routing_region_base)
                inside = [prep_region.contains(Point(pt[0], pt[1])) for pt in shift_np]
                shift_np = shift_np[inside]
            if len(shift_np) > 0:
                nodes_np = np.concatenate([nodes_np, shift_np], axis=0)
                nodes_np = np.unique(np.round(nodes_np), axis=0)
                
    nodes_np = add_door_points_to_grid(nodes_np)
    
    # Pre-calculate boundary segments for fast vectorized ray-casting line-of-sight
    bnd_segs_list = []
    for w in walls:
        coords = list(w.coords)
        for i in range(len(coords) - 1):
            bnd_segs_list.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
    for col in columns:
        coords = list(col.exterior.coords)
        for i in range(len(coords) - 1):
            bnd_segs_list.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
    for shaft in shafts:
        coords = list(shaft.exterior.coords)
        for i in range(len(coords) - 1):
            bnd_segs_list.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
            
    _bnd_segs = np.array(bnd_segs_list, dtype=np.float32)
    
    grid_nodes = nodes_np
    grid_kd = cKDTree(grid_nodes)
    
    # Adjacency compilation
    t0 = time.perf_counter()
    N = len(grid_nodes)
    adj = {i: [] for i in range(N)}
    
    x_coords = grid_nodes[:, 0]
    y_coords = grid_nodes[:, 1]
    
    # Sort for horizontal adjacency
    h_idx = np.lexsort((x_coords, y_coords))
    sorted_hx = x_coords[h_idx]
    sorted_hy = y_coords[h_idx]
    
    # Sort for vertical adjacency
    v_idx = np.lexsort((y_coords, x_coords))
    sorted_vx = x_coords[v_idx]
    sorted_vy = y_coords[v_idx]
    
    edges_candidates = []
    
    # Horizontal candidates
    for i in range(N - 1):
        idx1, idx2 = h_idx[i], h_idx[i+1]
        if abs(sorted_hy[i] - sorted_hy[i+1]) < 1e-3:
            edges_candidates.append((idx1, idx2, 0))
            
    # Vertical candidates
    for i in range(N - 1):
        idx1, idx2 = v_idx[i], v_idx[i+1]
        if abs(sorted_vx[i] - sorted_vx[i+1]) < 1e-3:
            edges_candidates.append((idx1, idx2, 1))
            
    # Filter candidates with boundary segments
    valid_edges = []
    for u, v, direction in edges_candidates:
        p1 = grid_nodes[u]
        p2 = grid_nodes[v]
        dist = np.linalg.norm(p2 - p1)
        if dist > 3000.0:
            continue
            
        mid = 0.5 * (p1 + p2)
        if routing_region_base and not routing_region_base.contains(Point(mid[0], mid[1])):
            continue
            
        edge_line = LineString([p1, p2])
        collides = False
        for w in walls:
            if edge_line.crosses(w) or edge_line.within(w):
                intersect = edge_line.intersection(w)
                if intersect.geom_type == 'Point':
                    continue
                collides = True
                break
        if collides:
            continue
            
        for col in columns:
            if edge_line.crosses(col) or edge_line.within(col):
                collides = True
                break
        if collides:
            continue
        for shaft in shafts:
            if edge_line.crosses(shaft) or edge_line.within(shaft):
                collides = True
                break
        if collides:
            continue
            
        valid_edges.append((u, v, dist, direction))
        
    for u, v, dist, direction in valid_edges:
        adj[u].append((v, dist, direction))
        adj[v].append((u, dist, direction))
        
    grid_adj_base = adj
    grid_edge_list = valid_edges
    grid_edge_coords = np.zeros((len(valid_edges), 4), dtype=np.float32)
    for idx, (u, v, dist, direction) in enumerate(valid_edges):
        p1, p2 = grid_nodes[u], grid_nodes[v]
        grid_edge_coords[idx] = [p1[0], p1[1], p2[0], p2[1]]

def update_dynamic_env(machine_poly):
    global current_env
    t_start = time.perf_counter()
    N = len(grid_nodes)
    blocked_nodes = set()
    blocked_edges = set()
    
    prep_m = shapely_prep(machine_poly)
    for i in range(N):
        pt = grid_nodes[i]
        p_obj = Point(pt[0], pt[1])
        if prep_m.contains(p_obj) or machine_poly.touches(p_obj):
            blocked_nodes.add(i)
            
    for idx, (u, v, dist, direction) in enumerate(grid_edge_list):
        p1, p2 = grid_nodes[u], grid_nodes[v]
        edge_line = LineString([p1, p2])
        if prep_m.intersects(edge_line):
            intersect = machine_poly.intersection(edge_line)
            if intersect.geom_type == 'Point':
                continue
            blocked_edges.add((min(u, v), max(u, v)))
            
    filtered_adj = {i: [] for i in range(N)}
    for u in range(N):
        if u in blocked_nodes:
            continue
        for v, dist, direction in grid_adj_base.get(u, []):
            if v in blocked_nodes:
                continue
            edge = (min(u, v), max(u, v))
            if edge in blocked_edges:
                continue
            filtered_adj[u].append((v, dist, direction))
            
    current_env = EnvView(grid_nodes, filtered_adj)
    print(f"Grid update: {(time.perf_counter() - t_start)*1000:.1f} ms  (blocked nodes={len(blocked_nodes)}, edges={len(blocked_edges)})")

def build_regular_grid(machine_poly):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd
    t0 = time.perf_counter()
    
    nodes_list = []
    x_steps = np.arange(0, 15000 + 1, GRID_SPACING)
    y_steps = np.arange(0, 11000 + 1, GRID_SPACING)
    
    for y in y_steps:
        for x in x_steps:
            nodes_list.append((float(x), float(y)))
            
    pts_np = np.array(nodes_list, dtype=np.float32)
    
    if routing_region_base:
        prep_region = shapely_prep(routing_region_base)
        inside_mask = [prep_region.contains(Point(pt[0], pt[1])) for pt in pts_np]
        pts_np = pts_np[inside_mask]
        
    pts_np = add_door_points_to_grid(pts_np)
    
    grid_nodes = pts_np
    grid_kd = cKDTree(grid_nodes)
    
    N = len(grid_nodes)
    adj = {i: [] for i in range(N)}
    
    pairs = grid_kd.query_pairs(GRID_SPACING + 5)
    valid_edges = []
    for u, v in pairs:
        p1, p2 = grid_nodes[u], grid_nodes[v]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        
        is_h = dx > 10.0 and dy < 10.0
        is_v = dy > 10.0 and dx < 10.0
        if not (is_h or is_v):
            continue
            
        edge_line = LineString([p1, p2])
        collides = False
        
        for w in walls:
            if edge_line.crosses(w) or edge_line.within(w):
                collides = True
                break
        if collides: continue
        for col in columns:
            if edge_line.crosses(col) or edge_line.within(col):
                collides = True
                break
        if collides: continue
        for shaft in shafts:
            if edge_line.crosses(shaft) or edge_line.within(shaft):
                collides = True
                break
        if collides: continue
        
        direction = 0 if is_h else 1
        dist = np.linalg.norm(p2 - p1)
        valid_edges.append((u, v, dist, direction))
        
    for u, v, dist, direction in valid_edges:
        adj[u].append((v, dist, direction))
        adj[v].append((u, dist, direction))
        
    grid_adj_base = adj
    grid_edge_list = valid_edges
    grid_edge_coords = np.zeros((len(valid_edges), 4), dtype=np.float32)
    for idx, (u, v, dist, direction) in enumerate(valid_edges):
        p1, p2 = grid_nodes[u], grid_nodes[v]
        grid_edge_coords[idx] = [p1[0], p1[1], p2[0], p2[1]]

def build_grid(machine_pins=None):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env
    if graph_type_idx == 0:
        global_pins = machine_pins if machine_pins else get_machine_pins(machine_cx, machine_cy, machine_angle)
        machine_poly = Polygon([
            global_pins["tl"], global_pins["tr"], global_pins["br"], global_pins["bl"]
        ])
        build_regular_grid(machine_poly)
    elif graph_type_idx == 1:
        build_hannan_grid(machine_pins=machine_pins, shift_walls=False)
    else:
        build_hannan_grid(machine_pins=machine_pins, shift_walls=True)

# Machine representation helper
def get_machine_pins(cx, cy, angle_deg):
    w, h = 700.0, 600.0 # 70cm x 60cm
    local_pins = {
        "left_mid": (-w/2, 0.0),
        "right_mid": (w/2, 0.0),
        "tl": (-w/2, h/2),
        "tr": (w/2, h/2),
        "bl": (-w/2, -h/2),
        "br": (w/2, -h/2)
    }
    
    rad = math.radians(angle_deg)
    global_pins = {}
    for name, (px, py) in local_pins.items():
        gx = cx + px * math.cos(rad) - py * math.sin(rad)
        gy = cy + px * math.sin(rad) + py * math.cos(rad)
        global_pins[name] = (round(gx), round(gy))
        
    # Rotate and translate corners
    corners = {
        "c_tl": (-w/2,  h/2),
        "c_tr": ( w/2,  h/2),
        "c_br": ( w/2, -h/2),
        "c_bl": (-w/2, -h/2)
    }
    global_corners = {}
    for name, (px, py) in corners.items():
        gx = cx + px * math.cos(rad) - py * math.sin(rad)
        gy = cy + px * math.sin(rad) + py * math.cos(rad)
        global_corners[name] = (round(gx), round(gy))
        
    return {**global_pins, **global_corners}

def snap_pins_to_graph(global_pins):
    if grid_kd is None:
        return {}
    pin_pts = np.array([list(pt) for name, pt in global_pins.items() if not name.startswith("c_")], dtype=float)
    names = [name for name in global_pins.keys() if not name.startswith("c_")]
    _, idxs = grid_kd.query(pin_pts)
    return {name: int(idx) for name, idx in zip(names, idxs)}

# ──────────────────────────────────────────────────────────────────────────
# ROUTING UTILITIES AND CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────
DIR_RIGHT, DIR_LEFT, DIR_UP, DIR_DOWN = 0, 1, 2, 3
DIR_REV = {DIR_RIGHT: DIR_LEFT, DIR_LEFT: DIR_RIGHT, DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP}

def get_outward_vector(pin_name, machine_angle):
    rad = math.radians(machine_angle)
    is_left = pin_name in ('left_mid', 'tl', 'bl')
    local_normal = (-1.0, 0.0) if is_left else (1.0, 0.0)
    
    gx = local_normal[0] * math.cos(rad) - local_normal[1] * math.sin(rad)
    gy = local_normal[0] * math.sin(rad) + local_normal[1] * math.cos(rad)
    
    if abs(gx) > abs(gy):
        return DIR_RIGHT if gx > 0 else DIR_LEFT
    else:
        return DIR_UP if gy > 0 else DIR_DOWN

def block_path_and_node(path, pin_node_idx, accumulated_weights, env, block_nodes=True):
    for i in range(len(path) - 1):
        e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
        accumulated_weights[e] = 1e9
        
    if block_nodes:
        for idx, u in enumerate(path):
            if u == pin_node_idx or idx == 0:
                continue
            if u in env.adj:
                for nbr, dist, direction in env.adj[u]:
                    edge = (min(u, nbr), max(u, nbr))
                    accumulated_weights[edge] = 1e9

def run_super_sink_astar(env, start_node_idx, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    if not target_pin_names:
        return None, 0.0, None
        
    super_sink_idx = len(env.nodes)
    
    search_nodes = np.zeros((super_sink_idx + 1, 2), dtype=np.float32)
    search_nodes[:super_sink_idx] = env.nodes
    search_nodes[super_sink_idx] = global_pins["left_mid"]
    
    search_adj = {i: list(env.adj[i]) for i in env.adj}
    search_adj[super_sink_idx] = []
    
    pin_name_by_idx = {}
    
    for pin_name in target_pin_names:
        pin_idx = pin_node_map[pin_name]
        pin_name_by_idx[pin_idx] = pin_name
        
        allowed_out = get_outward_vector(pin_name, machine_angle)
        inward_dir = DIR_REV[allowed_out]
        
        search_adj[pin_idx].append((super_sink_idx, 0.0, inward_dir))
        search_adj[super_sink_idx].append((pin_idx, 0.0, allowed_out))
        
    search_env = EnvView(search_nodes, search_adj)
    
    try:
        path, path_len = state_expanded_astar(search_env, start_node_idx, super_sink_idx, C_bend=C_bend, edge_weights=edge_weights)
    except Exception as e:
        print(f"Super Sink A* error: {e}")
        return None, 0.0, None
        
    if path is None or len(path) < 2:
        return None, 0.0, None
        
    chosen_pin_idx = path[-2]
    chosen_pin_name = pin_name_by_idx.get(chosen_pin_idx, target_pin_names[0])
    path_without_sink = path[:-1]
    
    return path_without_sink, path_len, chosen_pin_name

def get_all_terminal_node_indices(pin_node_map, shaft_node_idx):
    terminal_nodes = {}
    terminal_nodes["Shaft"] = shaft_node_idx
    for name, pt in terminals.items():
        _, node_idx = grid_kd.query(pt)
        terminal_nodes[name] = int(node_idx)
    return terminal_nodes

def count_segment_crossings(routes):
    crossings = 0
    all_segs = []
    for name, segs in routes:
        for p1, p2 in segs:
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            all_segs.append((name, (x1, y1, x2, y2)))
            
    for i in range(len(all_segs)):
        name1, (ax1, ay1, ax2, ay2) = all_segs[i]
        is_horiz1 = abs(ay1 - ay2) < 1e-7
        
        for j in range(i + 1, len(all_segs)):
            name2, (bx1, by1, bx2, by2) = all_segs[j]
            if name1 == name2:
                continue
            is_horiz2 = abs(by1 - by2) < 1e-7
            
            if is_horiz1 != is_horiz2:
                if is_horiz1:
                    if ax1 <= bx1 <= ax2 and by1 <= ay1 <= by2:
                        crossings += 1
                else:
                    if bx1 <= ax1 <= bx2 and ay1 <= by1 <= ay2:
                        crossings += 1
            else:
                if is_horiz1:
                    if abs(ay1 - by1) < 1e-7:
                        if max(ax1, bx1) <= min(ax2, bx2):
                            crossings += 1
                else:
                    if abs(ax1 - bx1) < 1e-7:
                        if max(ay1, by1) <= min(ay2, by2):
                            crossings += 1
    return crossings

def run_sequential_routing(perm, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, shaft_path, block_nodes):
    accumulated_weights = {}
    exhaust_node_idx = pin_node_map[chosen_exhaust_pin]
    block_path_and_node(shaft_path, exhaust_node_idx, accumulated_weights, current_env, block_nodes=block_nodes)

    kitchen_pin_name = "right_mid" if chosen_exhaust_pin == "left_mid" else "left_mid"
    routes = []
    
    # Shaft segments
    shaft_segs = []
    for i in range(len(shaft_path) - 1):
        p1 = current_env.nodes[shaft_path[i]]
        p2 = current_env.nodes[shaft_path[i+1]]
        shaft_segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    routes.append(("Shaft", shaft_segs))
    
    total_nodes = len(shaft_path)
    
    # Pre-calculate terminal node indices
    terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)

    # Helper to block other terminals temporarily
    def get_weights_blocking_other_terminals(curr_room, base_weights):
        w = base_weights.copy()
        for r_name, t_node_idx in terminal_nodes.items():
            if r_name == curr_room:
                continue
            if t_node_idx in current_env.adj:
                for nbr, dist, direction in current_env.adj[t_node_idx]:
                    edge = (min(t_node_idx, nbr), max(t_node_idx, nbr))
                    w[edge] = 1e9
        return w

    # 1. Route Kitchen (Fixed position right after Shaft)
    kitchen_pt = terminals.get("Kitchen")
    if kitchen_pt:
        _, kitchen_node_idx = grid_kd.query(kitchen_pt)
        kitchen_node_idx = int(kitchen_node_idx)
        
        current_weights = get_weights_blocking_other_terminals("Kitchen", accumulated_weights)
        kitchen_path, _, _ = run_super_sink_astar(
            current_env,
            kitchen_node_idx,
            [kitchen_pin_name],
            pin_node_map,
            global_pins,
            machine_angle,
            C_BEND,
            edge_weights=current_weights
        )
        if kitchen_path is None:
            return False, None, "No path to Kitchen", 0
            
        kitchen_segs = []
        for i in range(len(kitchen_path) - 1):
            p1 = current_env.nodes[kitchen_path[i]]
            p2 = current_env.nodes[kitchen_path[i+1]]
            kitchen_segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
        routes.append(("Kitchen", kitchen_segs))
        total_nodes += len(kitchen_path)
        
        kitchen_pin_node_idx = pin_node_map[kitchen_pin_name]
        block_path_and_node(kitchen_path, kitchen_pin_node_idx, accumulated_weights, current_env, block_nodes=block_nodes)

    # 2. Route small duct rooms in perm order
    available_small_pins = ["tl", "tr", "bl", "br"]
    for room_name in perm:
        if not available_small_pins:
            return False, None, f"No port for {room_name}", 0
        room_pt = terminals[room_name]
        _, room_node_idx = grid_kd.query(room_pt)
        room_node_idx = int(room_node_idx)
        
        current_weights = get_weights_blocking_other_terminals(room_name, accumulated_weights)
        room_path, _, chosen_small_pin = run_super_sink_astar(
            current_env,
            room_node_idx,
            available_small_pins,
            pin_node_map,
            global_pins,
            machine_angle,
            C_BEND,
            edge_weights=current_weights
        )
        if room_path is None:
            return False, None, f"No path to {room_name}", 0
            
        room_segs = []
        for i in range(len(room_path) - 1):
            p1 = current_env.nodes[room_path[i]]
            p2 = current_env.nodes[room_path[i+1]]
            room_segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
        routes.append((room_name, room_segs))
        total_nodes += len(room_path)
        
        chosen_pin_node_idx = pin_node_map[chosen_small_pin]
        block_path_and_node(room_path, chosen_pin_node_idx, accumulated_weights, current_env, block_nodes=block_nodes)
        available_small_pins.remove(chosen_small_pin)
        
    return True, routes, "Success", total_nodes

def get_solution_score(routes, crossings):
    total_len = 0.0
    total_turns = 0
    for name, segs in routes:
        total_len += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
        total_turns += calculate_tree_turns(segs)
    score = int(total_len) + int(C_BEND) * total_turns + int(5 * C_BEND) * crossings
    return score

# ──────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL DISTANCE FIELDS AUTO-PLACEMENT ALGORITHMS
# ──────────────────────────────────────────────────────────────────────────
def is_machine_placement_valid(cx, cy, angle):
    global_pins = get_machine_pins(cx, cy, angle)
    machine_poly = Polygon([
        global_pins["c_tl"], global_pins["c_tr"], global_pins["c_br"], global_pins["c_bl"]
    ])
    
    if not routing_region_base or not routing_region_base.contains(Point(cx, cy)):
        return False
    if not routing_region_base.contains(machine_poly):
        return False
    if any(machine_poly.intersects(col) for col in columns):
        return False
    return True

def compute_dijkstra_distance_field(start_node_idx, env):
    distances = {n: 1e9 for n in env.adj}
    distances[start_node_idx] = 0.0
    pq = [(0.0, start_node_idx)]
    
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]:
            continue
        for v, edge_dist, direction in env.adj.get(u, []):
            new_dist = dist + edge_dist
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances

def get_auto_placement_scores(env, pin_node_map, shaft_node_idx):
    terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)
    
    weights = {
        "Shaft": 2.5,
        "Kitchen": 1.5,
        "Bathroom": 1.0,
        "Bathroom 1": 1.0,
        "Bathroom 2": 1.0,
        "Toilet": 1.0,
        "Washroom": 1.0
    }
    
    distance_fields = {}
    for name, node_idx in terminal_nodes.items():
        distance_fields[name] = compute_dijkstra_distance_field(node_idx, env)
        
    node_scores = {}
    for n in env.adj:
        total_score = 0.0
        reachable = True
        for name, field in distance_fields.items():
            dist = field.get(n, 1e9)
            if dist >= 1e8:
                reachable = False
                break
            w = weights.get(name, 1.0)
            total_score += w * dist
            
        if reachable:
            node_scores[n] = total_score
            
    return node_scores, distance_fields

def run_auto_placement():
    global machine_cx, machine_cy, machine_angle, ap_scores, ap_fields
    if grid_nodes is None:
        return
        
    dummy_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    pin_node_map = snap_pins_to_graph(dummy_pins)
    
    if not pin_node_map or not shaft_extraction:
        return
        
    rep_pt = shaft_extraction.representative_point()
    shaft_center = (round(rep_pt.x), round(rep_pt.y))
    _, shaft_node_idx = grid_kd.query(shaft_center)
    shaft_node_idx = int(shaft_node_idx)
    
    # ── Option 1: Proximity to Shaft ──
    if auto_placement_mode_idx == 1:
        best_room_name = None
        min_d = 1e9
        for name, pt in terminals.items():
            d = math.hypot(pt[0] - shaft_center[0], pt[1] - shaft_center[1])
            if d < min_d:
                min_d = d
                best_room_name = name
                
        if best_room_name:
            t_pt = terminals[best_room_name]
            sorted_nodes = sorted(
                range(len(grid_nodes)),
                key=lambda idx: math.hypot(grid_nodes[idx][0] - t_pt[0], grid_nodes[idx][1] - t_pt[1])
            )
            for n_idx in sorted_nodes:
                pt = grid_nodes[n_idx]
                for rot in [0, 90, 180, 270]:
                    if is_machine_placement_valid(pt[0], pt[1], rot):
                        machine_cx, machine_cy = pt[0], pt[1]
                        machine_angle = rot
                        ap_scores = {}
                        ap_fields = {}
                        return
                        
    # ── Option 2: Topological Distance Fields ──
    elif auto_placement_mode_idx == 2:
        t0 = time.perf_counter()
        
        node_scores, distance_fields = get_auto_placement_scores(current_env, pin_node_map, shaft_node_idx)
        ap_scores = node_scores
        ap_fields = distance_fields
        
        if not node_scores:
            return
            
        sorted_nodes = sorted(node_scores.keys(), key=lambda n: node_scores[n])
        
        for n_idx in sorted_nodes:
            n_x, n_y = grid_nodes[n_idx][0], grid_nodes[n_idx][1]
            
            best_rot = None
            min_rot_score = 1e18
            
            for rot in [0, 90, 180, 270]:
                if is_machine_placement_valid(n_x, n_y, rot):
                    global_pins = get_machine_pins(n_x, n_y, rot)
                    pin_nodes = {}
                    for pin_name, pt in global_pins.items():
                        if pin_name.startswith("c_"): continue
                        _, p_idx = grid_kd.query(pt)
                        pin_nodes[pin_name] = int(p_idx)
                        
                    d_left = distance_fields["Shaft"].get(pin_nodes["left_mid"], 1e9)
                    d_right = distance_fields["Shaft"].get(pin_nodes["right_mid"], 1e9)
                    if d_left < d_right:
                        chosen_exhaust = "left_mid"
                        kitchen_pin = "right_mid"
                        shaft_dist = d_left
                    else:
                        chosen_exhaust = "right_mid"
                        kitchen_pin = "left_mid"
                        shaft_dist = d_right
                        
                    kitchen_dist = 0.0
                    if "Kitchen" in distance_fields:
                        kitchen_dist = distance_fields["Kitchen"].get(pin_nodes[kitchen_pin], 1e9)
                        
                    small_pins = ["tl", "tr", "bl", "br"]
                    room_dists = 0.0
                    remaining_rooms = [r for r in wet_room_names if r != "Kitchen"]
                    used_pins = set()
                    
                    for r_name in remaining_rooms:
                        best_d = 1e9
                        best_p = None
                        for p in small_pins:
                            if p in used_pins:
                                continue
                            d = distance_fields[r_name].get(pin_nodes[p], 1e9)
                            if d < best_d:
                                best_d = d
                                best_p = p
                        if best_p is not None:
                            used_pins.add(best_p)
                            room_dists += best_d
                        else:
                            room_dists += 1e9
                            
                    rot_score = 2.5 * shaft_dist + 1.5 * kitchen_dist + room_dists
                    if rot_score < min_rot_score:
                        min_rot_score = rot_score
                        best_rot = rot
                        
            if best_rot is not None:
                machine_cx, machine_cy = n_x, n_y
                machine_angle = best_rot
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[Auto-Placement] Solved position ({machine_cx}, {machine_cy}) at rotation {machine_angle} in {elapsed_ms:.2f}ms")
                return

# ──────────────────────────────────────────────────────────────────────────
# MAIN SOLVER WRAPPER
# ──────────────────────────────────────────────────────────────────────────
def solve_ventilation_routing():
    t_start = time.perf_counter()
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)

    machine_poly = Polygon([
        global_pins["c_tl"], global_pins["c_tr"], global_pins["c_br"], global_pins["c_bl"]
    ])

    m_center = Point(machine_cx, machine_cy)
    if not routing_region_base or not routing_region_base.contains(m_center):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: Machine outside region", elapsed_ms, 0

    if any(machine_poly.intersects(col) for col in columns):
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: Machine collides with column", elapsed_ms, 0

    if grid_nodes is None:
        return None, "Building grid… press Space to retry", 0.0, 0

    if graph_type_idx == 0:
        update_dynamic_env(machine_poly)
    else:
        build_hannan_grid(machine_pins=global_pins, shift_walls=(graph_type_idx == 2))
        update_dynamic_env(machine_poly)

    pin_node_map = snap_pins_to_graph(global_pins)
    if not pin_node_map or not shaft_extraction:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: Missing pins or shaft", elapsed_ms, 0

    rep_pt = shaft_extraction.representative_point()
    shaft_center = (round(rep_pt.x), round(rep_pt.y))
    _, shaft_node_idx = grid_kd.query(shaft_center)
    shaft_node_idx = int(shaft_node_idx)

    # Pre-calculate terminal node indices
    terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)
    
    # Block other terminals for Shaft search
    shaft_weights = {}
    for r_name, t_node_idx in terminal_nodes.items():
        if r_name == "Shaft":
            continue
        if t_node_idx in current_env.adj:
            for nbr, dist, direction in current_env.adj[t_node_idx]:
                edge = (min(t_node_idx, nbr), max(t_node_idx, nbr))
                shaft_weights[edge] = 1e9

    # 1. Route Shaft via Super Sink
    shaft_path, _, chosen_exhaust_pin = run_super_sink_astar(
        current_env,
        shaft_node_idx,
        ["left_mid", "right_mid"],
        pin_node_map,
        global_pins,
        machine_angle,
        C_BEND,
        edge_weights=shaft_weights
    )

    if shaft_path is None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: No path to shaft", elapsed_ms, 0

    # 2. Backtracking search over permutations of the small duct rooms
    from itertools import permutations
    other_rooms = sorted(
        [name for name in terminals.keys() if name != "Kitchen" and any(w in name for w in ["Bathroom", "Toilet", "Washroom"])],
        key=lambda name: math.hypot(terminals[name][0] - machine_cx, terminals[name][1] - machine_cy)
    )
    
    if routing_strategy_idx == 0:
        close_to_far = tuple(other_rooms)
        far_to_close = tuple(reversed(other_rooms))
        all_perms = [close_to_far, far_to_close]
    else:
        all_perms = list(permutations(other_rooms))

    best_routes = None
    best_crossings = 1e9
    best_score = 1e18
    best_total_nodes = 0
    perm_attempts = 0

    if routing_strategy_idx == 3:
        # ── Strategy 3: Negotiated Congestion ──
        nets_list = ["Shaft", "Kitchen"] + other_rooms
        current_paths = {}
        current_pins = {}
        
        P_present = 20000.0
        P_history = 4000.0
        history_congestion = {}
        node_history_congestion = {}
        
        for iteration in range(20):
            perm_attempts += 1
            
            for net_name in nets_list:
                if net_name == "Shaft":
                    start_node_idx = shaft_node_idx
                    targets = ["left_mid", "right_mid"]
                elif net_name == "Kitchen":
                    kitchen_pt = terminals.get("Kitchen")
                    if not kitchen_pt:
                        continue
                    _, kitchen_node_idx = grid_kd.query(kitchen_pt)
                    start_node_idx = int(kitchen_node_idx)
                    shaft_pin = current_pins.get("Shaft", "left_mid")
                    kitchen_pin_name = "right_mid" if shaft_pin == "left_mid" else "left_mid"
                    targets = [kitchen_pin_name]
                else:
                    room_pt = terminals[net_name]
                    _, room_node_idx = grid_kd.query(room_pt)
                    start_node_idx = int(room_node_idx)
                    used_small_pins = [current_pins[n] for n in other_rooms if n != net_name and n in current_pins]
                    targets = [p for p in ["tl", "tr", "bl", "br"] if p not in used_small_pins]
                    if not targets:
                        targets = ["tl"]
                
                current_paths[net_name] = None
                
                edge_usage = {}
                node_usage = {}
                for other_name, path in current_paths.items():
                    if path is None:
                        continue
                    for u in path:
                        node_usage[u] = node_usage.get(u, 0) + 1
                    for k in range(len(path) - 1):
                        e = (min(path[k], path[k+1]), max(path[k], path[k+1]))
                        edge_usage[e] = edge_usage.get(e, 0) + 1
                        
                current_weights = {}
                terminal_nodes = get_all_terminal_node_indices(pin_node_map, shaft_node_idx)
                for r_name, t_node_idx in terminal_nodes.items():
                    if r_name == net_name:
                        continue
                    if t_node_idx in current_env.adj:
                        for nbr, dist, direction in current_env.adj[t_node_idx]:
                            edge = (min(t_node_idx, nbr), max(t_node_idx, nbr))
                            current_weights[edge] = 1e9
                            
                for u in current_env.adj:
                    for v, dist, direction in current_env.adj[u]:
                        edge = (min(u, v), max(u, v))
                        if edge in current_weights and current_weights[edge] >= 1e9:
                            continue
                        
                        pres = edge_usage.get(edge, 0)
                        hist = history_congestion.get(edge, 0.0)
                        
                        node_pres = max(node_usage.get(u, 0), node_usage.get(v, 0))
                        node_hist = max(node_history_congestion.get(u, 0.0), node_history_congestion.get(v, 0.0))
                        
                        congestion_weight = (pres * P_present) + hist + (node_pres * 20000.0) + node_hist
                        current_weights[edge] = dist + congestion_weight
                        
                path, _, chosen_pin = run_super_sink_astar(
                    current_env,
                    start_node_idx,
                    targets,
                    pin_node_map,
                    global_pins,
                    machine_angle,
                    C_BEND,
                    edge_weights=current_weights
                )
                
                if path is not None:
                    current_paths[net_name] = path
                    current_pins[net_name] = chosen_pin
                    
            routes_cand = []
            success = True
            total_nodes_cand = 0
            for name in nets_list:
                path = current_paths.get(name)
                if path is None:
                    success = False
                    break
                segs = []
                for k in range(len(path) - 1):
                    p1 = current_env.nodes[path[k]]
                    p2 = current_env.nodes[path[k+1]]
                    segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
                routes_cand.append((name, segs))
                total_nodes_cand += len(path)
                
            if success:
                crossings = count_segment_crossings(routes_cand)
                score = get_solution_score(routes_cand, crossings)
                
                if score < best_score:
                    best_score = score
                    best_crossings = crossings
                    best_routes = routes_cand
                    best_total_nodes = total_nodes_cand
                    
                if crossings == 0:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    status_text = f"Success: Routed all (tried {perm_attempts} iters, 0 crossings) in {elapsed_ms:.1f}ms"
                    return routes_cand, status_text, elapsed_ms, total_nodes_cand
                    
                edge_counts = {}
                node_counts = {}
                for name, path in current_paths.items():
                    if path is None:
                        continue
                    for u in path:
                        node_counts[u] = node_counts.get(u, 0) + 1
                    for k in range(len(path) - 1):
                        edge = (min(path[k], path[k+1]), max(path[k], path[k+1]))
                        edge_counts[edge] = edge_counts.get(edge, 0) + 1
                        
                for edge, count in edge_counts.items():
                    if count > 1:
                        history_congestion[edge] = history_congestion.get(edge, 0.0) + P_history
                for node, count in node_counts.items():
                    if count > 1:
                        node_history_congestion[node] = node_history_congestion.get(node, 0.0) + 4000.0
                        
        if best_routes is not None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            status_text = f"Success: Routed all (tried {perm_attempts} iters, {best_crossings} crossings) in {elapsed_ms:.1f}ms"
            return best_routes, status_text, elapsed_ms, best_total_nodes
        else:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return None, f"Routing Blocked (tried {perm_attempts} iters) in {elapsed_ms:.1f}ms", elapsed_ms, 0

    # Pass 1: Try collision-free solutions (block_nodes=True)
    for perm in all_perms:
        perm_attempts += 1
        success, routes_cand, status_cand, total_nodes_cand = run_sequential_routing(
            perm, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, shaft_path, block_nodes=True
        )
        if success:
            crossings = count_segment_crossings(routes_cand)
            score = get_solution_score(routes_cand, crossings)
            if crossings == 0:
                if routing_strategy_idx == 1:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    status_text = f"Success: Routed all (tried {perm_attempts} perms, 0 crossings) in {elapsed_ms:.1f}ms"
                    return routes_cand, status_text, elapsed_ms, total_nodes_cand
                else:
                    if score < best_score:
                        best_score = score
                        best_crossings = crossings
                        best_routes = routes_cand
                        best_total_nodes = total_nodes_cand
            else:
                if score < best_score:
                    best_score = score
                    best_crossings = crossings
                    best_routes = routes_cand
                    best_total_nodes = total_nodes_cand

    # Pass 2: Fall back to allowing crossings (block_nodes=False) if no collision-free solution was found
    if best_routes is None or best_crossings > 0:
        for perm in all_perms:
            perm_attempts += 1
            success, routes_cand, status_cand, total_nodes_cand = run_sequential_routing(
                perm, pin_node_map, global_pins, shaft_node_idx, chosen_exhaust_pin, shaft_path, block_nodes=False
            )
            if success:
                crossings = count_segment_crossings(routes_cand)
                score = get_solution_score(routes_cand, crossings)
                if score < best_score:
                    best_score = score
                    best_crossings = crossings
                    best_routes = routes_cand
                    best_total_nodes = total_nodes_cand
                    if crossings == 0 and routing_strategy_idx == 1:
                        break

    if best_routes is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        status_text = f"Success: Routed all (tried {perm_attempts} perms, {best_crossings} crossings) in {elapsed_ms:.1f}ms"
        return best_routes, status_text, elapsed_ms, best_total_nodes
    else:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, f"Routing Blocked (tried {perm_attempts} perms) in {elapsed_ms:.1f}ms", elapsed_ms, 0

# ──────────────────────────────────────────────────────────────────────────
# DWELLING AND ROOM GENERATORS
# ──────────────────────────────────────────────────────────────────────────
def generate_new_dwelling():
    global rooms, columns, shafts, doors, walls, wall_polys, routing_region_base, shaft_extraction, terminals, wet_room_names
    global machine_cx, machine_cy, machine_angle, _bnd_segs
    
    rooms_m = generative_layout.generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    rooms = []
    covered_names = ["Hallway", "Kitchen", "Bathroom", "Bathroom 1", "Bathroom 2", "Toilet", "Washroom", "Bedroom 1"]
    for r in rooms_m:
        scaled_poly = snap_to_integer_grid(shapely_scale(r.polygon, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        room_scaled = generative_layout.Room(scaled_poly, r.name)
        room_scaled.has_cover = any(cn in r.name for cn in covered_names)
        rooms.append(room_scaled)
        
    shafts_m = generative_layout.generate_mep_shafts(rooms_m)
    shafts = [snap_to_integer_grid(shapely_scale(s, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for s in shafts_m]

    columns_m = generative_layout.generate_structural_grid(unary_union([r.polygon for r in rooms_m]), spacing=4.0)
    columns = []
    for col in columns_m:
        col_scaled = snap_to_integer_grid(shapely_scale(col, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        if any(col_scaled.intersects(s) for s in shafts):
            continue
        columns.append(col_scaled)
    
    doors_m = generative_layout.find_door_openings(rooms_m)
    doors = []
    for d in doors_m:
        d_scaled = {
            "d1": (round(d["d1"][0] * SCALE_TO_MM), round(d["d1"][1] * SCALE_TO_MM)),
            "d2": (round(d["d2"][0] * SCALE_TO_MM), round(d["d2"][1] * SCALE_TO_MM)),
            "swing_dir": d["swing_dir"],
            "width": d["width"] * SCALE_TO_MM,
            "is_entrance": d.get("is_entrance", False)
        }
        doors.append(d_scaled)
        
    entrance = generative_layout.find_entrance_door(rooms_m, unary_union([r.polygon for r in rooms_m]))
    if entrance:
        doors.append({
            "d1": (round(entrance["d1"][0] * SCALE_TO_MM), round(entrance["d1"][1] * SCALE_TO_MM)),
            "d2": (round(entrance["d2"][0] * SCALE_TO_MM), round(entrance["d2"][1] * SCALE_TO_MM)),
            "swing_dir": entrance["swing_dir"],
            "width": entrance["width"] * SCALE_TO_MM,
            "is_entrance": True
        })

    # Extract wall line centerlines and subtract columns/shafts
    walls_m = []
    for i in range(len(rooms_m)):
        for j in range(i + 1, len(rooms_m)):
            shared = rooms_m[i].polygon.intersection(rooms_m[j].polygon)
            if isinstance(shared, LineString):
                walls_m.append(shared)
            elif hasattr(shared, 'geoms'):
                for g in shared.geoms:
                    if isinstance(g, LineString):
                        walls_m.append(g)
                        
    raw_walls = [snap_to_integer_grid(shapely_scale(w, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for w in walls_m]
    walls = []
    for rw in raw_walls:
        w_cut = rw
        for col in columns:
            w_cut = w_cut.difference(col)
        for s in shafts:
            w_cut = w_cut.difference(s)
        if w_cut.is_empty:
            continue
        if w_cut.geom_type == 'LineString':
            walls.append(w_cut)
        elif hasattr(w_cut, 'geoms'):
            for g in w_cut.geoms:
                if g.geom_type == 'LineString':
                    walls.append(g)
    
    wall_polys = []
    for w in walls:
        wp = w.buffer(WALL_THICKNESS / 2 - 0.1)
        for col in columns:
            wp = wp.difference(col)
        for s in shafts:
            wp = wp.difference(s)
        if not wp.is_empty:
            wall_polys.append(wp)
    
    routing_region_m = unary_union([r.polygon for r in rooms_m if any(cn in r.name for cn in covered_names)])
    routing_region_base = snap_to_integer_grid(shapely_scale(routing_region_m, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
    
    for col in columns:
        routing_region_base = routing_region_base.difference(col)
    for shaft in shafts:
        routing_region_base = routing_region_base.difference(shaft)
        
    shaft_extraction = shafts[0] if shafts else None
    
    wet_rooms = [r for r in rooms if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    terminals = {}
    for r in wet_rooms:
        t_pt = get_representative_point(r.polygon)
        terminals[r.name] = t_pt
        
    best_room = None
    best_dist = 1e9
    if shaft_extraction:
        rep_pt = shaft_extraction.representative_point()
        sx, sy = rep_pt.x, rep_pt.y
        for r in wet_rooms:
            if any(w in r.name for w in ["Bathroom", "Washroom"]):
                rx, ry = r.polygon.centroid.x, r.polygon.centroid.y
                dist = abs(sx - rx) + abs(sy - ry)
                if dist < best_dist:
                    best_dist = dist
                    best_room = r
                    
    if best_room:
        machine_cx, machine_cy = get_representative_point(best_room.polygon)
    else:
        machine_cx, machine_cy = 7500.0, 5500.0
        
    machine_angle = 0
    wet_room_names = list(terminals.keys())
    _bnd_segs = None
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

def draw_distance_heatmap(screen, node_scores):
    if not node_scores:
        return
    min_s = min(node_scores.values())
    max_s = max(node_scores.values())
    diff = max_s - min_s if max_s > min_s else 1.0
    
    for node_idx, score in node_scores.items():
        if node_idx >= len(current_env.nodes):
            continue
        pt = current_env.nodes[node_idx]
        px, py = to_screen(pt[0], pt[1])
        t = (score - min_s) / diff
        r = int(255 * t)
        g = int(255 * (1.0 - t))
        b = 0
        pygame.draw.circle(screen, (r, g, b), (px, py), 4)

def main():
    global machine_cx, machine_cy, machine_angle, show_grid_graph, graph_type_idx, routing_strategy_idx
    global auto_placement_mode_idx, show_heatmap
    
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Integrated Auto-Placement & Ventilation Router (Demo 10.5)")
    clock = pygame.time.Clock()
    
    font_title = pygame.font.SysFont("Outfit", 20, bold=True)
    font_bold = pygame.font.SysFont("Outfit", 15, bold=True)
    font_small = pygame.font.SysFont("Outfit", 13)
    
    generate_new_dwelling()
    
    dragging = False
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    
    routes = []
    status = "Initial"
    elapsed_ms = 0.0
    total_nodes = 0
    
    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
    
    running = True
    while running:
        if auto_placement_mode_idx > 0 and not dragging:
            run_auto_placement()
            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    m_poly = Polygon([g_pins["c_tl"], g_pins["c_tr"], g_pins["c_br"], g_pins["c_bl"]])
                    world_x, world_y = to_mm(mx, my)
                    p_obj = Point(world_x, world_y)
                    
                    if m_poly.contains(p_obj) or m_poly.distance(p_obj) < 200.0:
                        dragging = True
                        auto_placement_mode_idx = 0
                        drag_offset_x = world_x - machine_cx
                        drag_offset_y = world_y - machine_cy
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    wx, wy = to_mm(mx, my)
                    machine_cx = wx - drag_offset_x
                    machine_cy = wy - drag_offset_y
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    generate_new_dwelling()
                    if auto_placement_mode_idx > 0:
                        run_auto_placement()
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_r:
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle + 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_g:
                    show_grid_graph = not show_grid_graph
                    
                elif event.key == pygame.K_c:
                    routing_strategy_idx = (routing_strategy_idx + 1) % len(ROUTING_STRATEGIES)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_TAB:
                    graph_type_idx = (graph_type_idx + 1) % len(GRAPH_TYPES)
                    g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    build_grid(machine_pins=g_pins)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_a:
                    if auto_placement_mode_idx > 0:
                        auto_placement_mode_idx = 0
                    else:
                        auto_placement_mode_idx = 2
                    if auto_placement_mode_idx > 0:
                        run_auto_placement()
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_p:
                    auto_placement_mode_idx = (auto_placement_mode_idx + 1) % len(AUTO_PLACEMENT_MODES)
                    if auto_placement_mode_idx > 0:
                        run_auto_placement()
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_v:
                    show_heatmap = not show_heatmap
                    
        # ── RENDERING ────────────────────────────────────────────────────────
        screen.fill(COLOR_BG)
        
        for room in rooms:
            if not hasattr(room, 'polygon') or room.polygon.is_empty:
                continue
            coords = list(room.polygon.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            color = COLOR_ROOM_COVERED if room.has_cover else COLOR_ROOM
            pygame.draw.polygon(screen, color, screen_coords)
            pygame.draw.polygon(screen, COLOR_WALL, screen_coords, 1)
            
        for d in doors:
            # Draw door line segment
            sp1 = to_screen(d["d1"][0], d["d1"][1])
            sp2 = to_screen(d["d2"][0], d["d2"][1])
            pygame.draw.line(screen, COLOR_DOOR, sp1, sp2, 4)
            
        for col_poly in columns:
            coords = list(col_poly.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            pygame.draw.polygon(screen, COLOR_COLUMN, screen_coords)
            
        for s_poly in shafts:
            coords = list(s_poly.exterior.coords)
            screen_coords = [to_screen(x, y) for x, y in coords]
            pygame.draw.polygon(screen, COLOR_SHAFT, screen_coords)
            
        if show_grid_graph and current_env is not None:
            for u in current_env.adj:
                for v, dist, direction in current_env.adj[u]:
                    p1 = current_env.nodes[u]
                    p2 = current_env.nodes[v]
                    sp1 = to_screen(p1[0], p1[1])
                    sp2 = to_screen(p2[0], p2[1])
                    pygame.draw.line(screen, (75, 75, 90), sp1, sp2, 1)
                    
        if show_heatmap and auto_placement_mode_idx == 2 and ap_scores:
            draw_distance_heatmap(screen, ap_scores)
            
        for r_name, pt in terminals.items():
            s_pt = to_screen(pt[0], pt[1])
            pygame.draw.circle(screen, (255, 255, 255), s_pt, 7)
            c_core = ROUTE_COLORS.get(r_name, (255, 255, 255))
            pygame.draw.circle(screen, c_core, s_pt, 5)
            lbl_name = r_name.replace("Bathroom", "Bath").replace("Washroom", "Wash")
            text_surf = font_small.render(lbl_name, True, COLOR_TEXT)
            screen.blit(text_surf, (s_pt[0] - text_surf.get_width()//2, s_pt[1] + 10))
            
        if shaft_extraction:
            s_rep = get_representative_point(shaft_extraction)
            s_pt = to_screen(s_rep[0], s_rep[1])
            pygame.draw.circle(screen, (255, 255, 255), s_pt, 8)
            pygame.draw.circle(screen, (231, 76, 60), s_pt, 6)
            
        if routes:
            for name, segs in routes:
                c = ROUTE_COLORS.get(name, COLOR_TEXT)
                width = 5 if name in ("Shaft", "Kitchen") else 3
                for p1, p2 in segs:
                    sp1 = to_screen(p1[0], p1[1])
                    sp2 = to_screen(p2[0], p2[1])
                    pygame.draw.line(screen, c, sp1, sp2, width)
                    
        # Bounding box coordinates for machine
        g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        m_screen_pts = [
            to_screen(g_pins["c_tl"][0], g_pins["c_tl"][1]),
            to_screen(g_pins["c_tr"][0], g_pins["c_tr"][1]),
            to_screen(g_pins["c_br"][0], g_pins["c_br"][1]),
            to_screen(g_pins["c_bl"][0], g_pins["c_bl"][1])
        ]
        
        base_color = (230, 126, 34) if auto_placement_mode_idx > 0 else (127, 140, 141)
        pygame.draw.polygon(screen, base_color, m_screen_pts)
        pygame.draw.polygon(screen, (255, 255, 255), m_screen_pts, 2)
        
        for pin_name in ["tl", "tr", "bl", "br", "left_mid", "right_mid"]:
            pt = g_pins[pin_name]
            sp = to_screen(pt[0], pt[1])
            is_large = pin_name in ("left_mid", "right_mid")
            color = (241, 196, 15) if is_large else (230, 126, 34)
            size = 5 if is_large else 4
            pygame.draw.circle(screen, color, sp, size)
            pygame.draw.circle(screen, (255, 255, 255), sp, size, 1)
            
        if auto_placement_mode_idx == 2 and ap_fields:
            s_rep = get_representative_point(shaft_extraction)
            _, p_left_idx = grid_kd.query(g_pins["left_mid"])
            _, p_right_idx = grid_kd.query(g_pins["right_mid"])
            d_left = ap_fields["Shaft"].get(int(p_left_idx), 1e9)
            d_right = ap_fields["Shaft"].get(int(p_right_idx), 1e9)
            exhaust_pt = g_pins["left_mid"] if d_left < d_right else g_pins["right_mid"]
            kitchen_pin = "right_mid" if d_left < d_right else "left_mid"
            
            sp_port = to_screen(exhaust_pt[0], exhaust_pt[1])
            sp_term = to_screen(s_rep[0], s_rep[1])
            pygame.draw.line(screen, (46, 204, 113), sp_port, sp_term, 2)
            
            if "Kitchen" in terminals:
                k_term = terminals["Kitchen"]
                k_port = g_pins[kitchen_pin]
                pygame.draw.line(screen, (241, 196, 15), to_screen(k_port[0], k_port[1]), to_screen(k_term[0], k_term[1]), 2)
                
            small_pins = ["tl", "tr", "bl", "br"]
            remaining_rooms = [r for r in wet_room_names if r != "Kitchen"]
            used_pins = set()
            for r_name in remaining_rooms:
                term_pt = terminals[r_name]
                best_d = 1e9
                best_p = None
                for p in small_pins:
                    if p in used_pins: continue
                    _, p_idx = grid_kd.query(g_pins[p])
                    d = ap_fields[r_name].get(int(p_idx), 1e9)
                    if d < best_d:
                        best_d = d
                        best_p = p
                if best_p:
                    used_pins.add(best_p)
                    port_pt = g_pins[best_p]
                    pygame.draw.line(screen, ROUTE_COLORS.get(r_name, COLOR_TEXT), to_screen(port_pt[0], port_pt[1]), to_screen(term_pt[0], term_pt[1]), 1)

        # ── SIDEBAR PANEL ──
        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, CANVAS_LEFT - 10, WINDOW_HEIGHT))
        pygame.draw.line(screen, COLOR_WALL, (CANVAS_LEFT - 10, 0), (CANVAS_LEFT - 10, WINDOW_HEIGHT), 2)
        
        title_surf = font_title.render("Auto-Placement visualizer", True, COLOR_TEXT)
        screen.blit(title_surf, (20, 20))
        sub_surf = font_small.render("Vents & Extraction Router Dashboard", True, COLOR_MUTED)
        screen.blit(sub_surf, (20, 42))
        
        # 1. Auto-placement State Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 75, CANVAS_LEFT - 40, 95), border_radius=6)
        lbl_ap_title = font_bold.render("AUTO-PLACEMENT STATE", True, (230, 126, 34))
        screen.blit(lbl_ap_title, (25, 85))
        mode_text = AUTO_PLACEMENT_MODES[auto_placement_mode_idx]
        lbl_ap_mode = font_bold.render(f"Mode: {mode_text}", True, COLOR_TEXT)
        screen.blit(lbl_ap_mode, (25, 105))
        lbl_ap_keys = font_small.render("[A] Toggle Auto-Placement | [P] Cycle Mode", True, COLOR_MUTED)
        screen.blit(lbl_ap_keys, (25, 125))
        h_text = "Enabled" if show_heatmap else "Disabled"
        lbl_ap_heatmap = font_small.render(f"[V] Toggle Heatmap: {h_text}", True, COLOR_MUTED)
        screen.blit(lbl_ap_heatmap, (25, 145))
        
        # 2. Solver Config Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 185, CANVAS_LEFT - 40, 120), border_radius=6)
        lbl_solv_title = font_bold.render("ROUTING PATH SOLVER", True, (52, 152, 219))
        screen.blit(lbl_solv_title, (25, 195))
        lbl_strat = font_small.render(f"Strategy: {ROUTING_STRATEGIES[routing_strategy_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_strat, (25, 215))
        lbl_graph = font_small.render(f"Grid type: {GRAPH_TYPES[graph_type_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_graph, (25, 235))
        lbl_keys = font_small.render("[C] Cycle Strategy | [Tab] Cycle Grid", True, COLOR_MUTED)
        screen.blit(lbl_keys, (25, 255))
        lbl_gkey = font_small.render("[G] Toggle Grid Mesh Lines", True, COLOR_MUTED)
        screen.blit(lbl_gkey, (25, 275))
        lbl_skey = font_small.render("[Space] Gen New Apartment Dwelling", True, COLOR_MUTED)
        screen.blit(lbl_skey, (25, 290))
        
        # 3. Placement Info Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 320, CANVAS_LEFT - 40, 80), border_radius=6)
        lbl_pos_title = font_bold.render("MACHINE PLACEMENT", True, (46, 204, 113))
        screen.blit(lbl_pos_title, (25, 330))
        lbl_coord = font_small.render(f"Position: ({int(machine_cx)}, {int(machine_cy)}) mm", True, COLOR_TEXT)
        screen.blit(lbl_coord, (25, 350))
        lbl_rot = font_small.render(f"Rotation: {machine_angle}°", True, COLOR_TEXT)
        screen.blit(lbl_rot, (25, 370))
        
        # 4. KPI Metrics Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 415, CANVAS_LEFT - 40, 115), border_radius=6)
        lbl_kpi_title = font_bold.render("ROUTING RUNTIME KPIs", True, (241, 196, 15))
        screen.blit(lbl_kpi_title, (25, 425))
        
        total_len_mm = 0.0
        total_turns_count = 0
        if routes:
            for name, segs in routes:
                total_len_mm += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
                total_turns_count += calculate_tree_turns(segs)
                
        lbl_len = font_small.render(f"Total Duct Length: {total_len_mm/1000.0:.2f} m", True, COLOR_TEXT)
        screen.blit(lbl_len, (25, 445))
        lbl_turns = font_small.render(f"Total Bends/Turns: {total_turns_count}", True, COLOR_TEXT)
        screen.blit(lbl_turns, (25, 465))
        crossings_count = count_segment_crossings(routes) if routes else 0
        lbl_cross = font_small.render(f"Duct Crossings: {crossings_count}", True, COLOR_TEXT)
        screen.blit(lbl_cross, (25, 485))
        lbl_score = font_small.render(f"Total Cost Score: {get_solution_score(routes, crossings_count) if routes else 0}", True, COLOR_TEXT)
        screen.blit(lbl_score, (25, 505))
        
        # 5. Status Box
        pygame.draw.rect(screen, (40, 45, 55), (15, 545, CANVAS_LEFT - 40, 260), border_radius=6)
        lbl_status_title = font_bold.render("SOLVER EXECUTION STATUS", True, (155, 89, 182))
        screen.blit(lbl_status_title, (25, 555))
        
        words = status.split()
        lines = []
        curr_line = ""
        for w in words:
            if len(curr_line + " " + w) > 28:
                lines.append(curr_line)
                curr_line = w
            else:
                curr_line = (curr_line + " " + w).strip()
        if curr_line:
            lines.append(curr_line)
            
        y_off = 575
        for ln in lines:
            lbl_line = font_small.render(ln, True, COLOR_TEXT)
            screen.blit(lbl_line, (25, y_off))
            y_off += 18
            
        lbl_runtime = font_small.render(f"Pathfinder time: {elapsed_ms:.1f} ms", True, COLOR_TEXT)
        screen.blit(lbl_runtime, (25, 730))
        lbl_nodes = font_small.render(f"Total routed nodes: {total_nodes}", True, COLOR_MUTED)
        screen.blit(lbl_nodes, (25, 750))
        lbl_fps = font_small.render(f"Render engine: Pygame ({clock.get_fps():.0f} FPS)", True, COLOR_MUTED)
        screen.blit(lbl_fps, (25, 770))
        
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
