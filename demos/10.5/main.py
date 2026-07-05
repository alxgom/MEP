import sys
import os
import math
import time
import itertools
import heapq
from collections import deque
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
    "Negotiated Congestion",
    "Negotiated Congestion (Favour Large)"
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
WINDOW_WIDTH, WINDOW_HEIGHT = 1490, 850
CANVAS_LEFT = 280
CANVAS_TOP = 40
PANEL_W = 240          # right-side plot panel
CANVAS_W = WINDOW_WIDTH - CANVAS_LEFT - PANEL_W - 10
CANVAS_H = WINDOW_HEIGHT - CANVAS_TOP - 40

# Color Scheme
COLOR_BG = (28, 28, 36)
COLOR_PANEL = (35, 35, 45)
COLOR_PLOT_BG = (22, 22, 30)
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

# History buffers for the right-side plots
HIST_MAXLEN = 400
hist_length = deque(maxlen=HIST_MAXLEN)        # total duct length in metres
hist_score  = deque(maxlen=HIST_MAXLEN)         # weighted cost score
hist_turns  = deque(maxlen=HIST_MAXLEN)         # number of turns
hist_turns_per_len = deque(maxlen=HIST_MAXLEN)  # turns / meter of routed length
hist_ap_idx = None                              # sample index of last auto-placement
hist_event_markers = []                         # list of (index, label, color) tuples
weight_mode_idx = 0                             # 0 for Default, 1 for Equal Weights
heatmap_scale_mode = 0                          # 0 for Linear (75% Saturation), 1 for Log Scale

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
base_regular_env = None
base_regular_kd = None

def get_representative_point(poly):
    centroid = poly.centroid
    if poly.contains(centroid):
        return (round(centroid.x), round(centroid.y))
    rep = poly.representative_point()
    return (round(rep.x), round(rep.y))

def _extract_bnd_segs(region):
    segs = []
    def _add_ring(coords):
        c = list(coords)
        for i in range(len(c)-1):
            segs.append([c[i][0], c[i][1], c[i+1][0], c[i+1][1]])
    def _add_poly(poly):
        _add_ring(poly.exterior.coords)
        for interior in poly.interiors:
            _add_ring(interior.coords)
    if region.geom_type == 'Polygon':
        _add_poly(region)
    elif region.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for g in region.geoms:
            if g.geom_type == 'Polygon':
                _add_poly(g)
    return np.array(segs, dtype=np.float64) if segs else np.empty((0,4), dtype=np.float64)

def _cast_rays_numpy(interest_pts_arr, bnd, eps=0.5):
    h_segs, v_segs = [], []
    dx_s = bnd[:, 2] - bnd[:, 0]
    dy_s = bnd[:, 3] - bnd[:, 1]

    for x0, y0 in interest_pts_arr:
        nh = np.abs(dy_s) > eps
        if np.any(nh):
            t_h  = (y0 - bnd[nh, 1]) / dy_s[nh]
            ok_h = (t_h >= -eps) & (t_h <= 1.0 + eps)
            x_i  = bnd[nh, 0] + t_h * dx_s[nh]

            east = x_i[ok_h & (x_i > x0 + eps)]
            if len(east): h_segs.append((y0, x0, float(east.min())))

            west = x_i[ok_h & (x_i < x0 - eps)]
            if len(west): h_segs.append((y0, float(west.max()), x0))

        nv = np.abs(dx_s) > eps
        if np.any(nv):
            t_v  = (x0 - bnd[nv, 0]) / dx_s[nv]
            ok_v = (t_v >= -eps) & (t_v <= 1.0 + eps)
            y_i  = bnd[nv, 1] + t_v * dy_s[nv]

            north = y_i[ok_v & (y_i > y0 + eps)]
            if len(north): v_segs.append((x0, y0, float(north.min())))

            south = y_i[ok_v & (y_i < y0 - eps)]
            if len(south): v_segs.append((x0, float(south.max()), y0))

    return h_segs, v_segs

def _ray_ray_intersections_numpy(h_segs, v_segs, eps=0.5):
    if not h_segs or not v_segs:
        return []
    h = np.array(h_segs, dtype=np.float64)
    v = np.array(v_segs, dtype=np.float64)

    y_h  = h[:, 0:1]
    x1_h = h[:, 1:2]
    x2_h = h[:, 2:3]
    x_v  = v[:, 0:1].T
    y1_v = v[:, 1:2].T
    y2_v = v[:, 2:3].T

    cross = ((x_v >= x1_h - eps) & (x_v <= x2_h + eps) &
             (y_h >= y1_v - eps) & (y_h <= y2_v + eps))

    hi, vi = np.where(cross)
    return [(float(v[j, 0]), float(h[i, 0])) for i, j in zip(hi, vi)]

def _commit_grid(nodes_arr, valid_edges):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}

    if shaft_extraction is not None and len(nodes_arr) > 0:
        rep_pt = shaft_extraction.representative_point()
        sc = (round(rep_pt.x), round(rep_pt.y))
        
        minx_s, miny_s, maxx_s, maxy_s = shaft_extraction.bounds
        cx_s = (minx_s + maxx_s) / 2
        cy_s = (miny_s + maxy_s) / 2
        
        offset_val = 100.0
        face_pts = [
            (maxx_s + offset_val, cy_s, 'W'),
            (minx_s - offset_val, cy_s, 'E'),
            (cx_s, maxy_s + offset_val, 'S'),
            (cx_s, miny_s - offset_val, 'N')
        ]
        
        shaft_idx = len(nodes_arr)
        nodes_arr = np.vstack([nodes_arr, [sc[0], sc[1]]])
        
        connected_any = False
        for px, py, d in face_pts:
            diffs = np.hypot(nodes_arr[:-1, 0] - px, nodes_arr[:-1, 1] - py)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 10.0:
                w = float(np.hypot(sc[0] - nodes_arr[min_idx, 0], sc[1] - nodes_arr[min_idx, 1]))
                valid_edges.append((min_idx, shaft_idx, w, d))
                connected_any = True
                
        if not connected_any:
            diffs = np.hypot(nodes_arr[:-1, 0] - sc[0], nodes_arr[:-1, 1] - sc[1])
            min_idx = np.argmin(diffs)
            dx = sc[0] - nodes_arr[min_idx, 0]
            dy = sc[1] - nodes_arr[min_idx, 1]
            if abs(dx) > abs(dy):
                d = 'E' if dx > 0 else 'W'
            else:
                d = 'N' if dy > 0 else 'S'
            w = float(np.hypot(dx, dy))
            valid_edges.append((min_idx, shaft_idx, w, d))

    pin_indices = {}
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    for name, pt in global_pins.items():
        diffs = np.hypot(nodes_arr[:, 0] - pt[0], nodes_arr[:, 1] - pt[1])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 1.0:
            pin_indices[int(min_idx)] = name

    left_dir = 'W'
    right_dir = 'E'
    if machine_angle == 90:
        left_dir = 'S'
        right_dir = 'N'
    elif machine_angle == 180:
        left_dir = 'E'
        right_dir = 'W'
    elif machine_angle == 270:
        left_dir = 'N'
        right_dir = 'S'

    filtered_edges = []
    for u, v, w, d in valid_edges:
        keep = True
        if u in pin_indices:
            pin_name = pin_indices[u]
            is_left = pin_name in ('left_mid', 'tl', 'bl')
            allowed = left_dir if is_left else right_dir
            if d != allowed:
                keep = False
        if v in pin_indices:
            pin_name = pin_indices[v]
            is_left = pin_name in ('left_mid', 'tl', 'bl')
            allowed = left_dir if is_left else right_dir
            if DIR_REV[d] != allowed:
                keep = False
        if keep:
            filtered_edges.append((u, v, w, d))
            
    valid_edges = filtered_edges

    adj = {i: [] for i in range(len(nodes_arr))}
    for u, v, w, d in valid_edges:
        adj[u].append((v, w, d))
        adj[v].append((u, w, DIR_REV[d]))

    nodes_f32 = nodes_arr.astype(np.float32)
    if valid_edges:
        ec = np.array([[nodes_f32[u,0], nodes_f32[u,1],
                        nodes_f32[v,0], nodes_f32[v,1]]
                       for u, v, w, d in valid_edges], dtype=np.float32)
    else:
        ec = np.empty((0,4), dtype=np.float32)

    grid_nodes       = nodes_f32
    grid_adj_base    = adj
    grid_edge_list   = valid_edges
    grid_edge_coords = ec
    grid_kd          = cKDTree(grid_nodes)
    current_env      = EnvView(grid_nodes, adj)

def _wall_filter(raw_edges, nodes_arr):
    wall_bounds = [wp.bounds for wp in wall_polys]
    valid = []
    for u, v, w, d in raw_edges:
        pu = nodes_arr[u]; pv = nodes_arr[v]
        ex1, ey1 = float(min(pu[0], pv[0])), float(min(pu[1], pv[1]))
        ex2, ey2 = float(max(pu[0], pv[0])), float(max(pu[1], pv[1]))
        
        line = None
        blocked = False
        for idx, wp in enumerate(wall_polys):
            wx1, wy1, wx2, wy2 = wall_bounds[idx]
            if not (ex2 >= wx1 - 1.0 and ex1 <= wx2 + 1.0 and ey2 >= wy1 - 1.0 and ey1 <= wy2 + 1.0):
                continue
            if line is None:
                line = LineString([(float(pu[0]), float(pu[1])), (float(pv[0]), float(pv[1]))])
            if line.intersects(wp):
                inter = line.intersection(wp)
                if not inter.is_empty and inter.length > WALL_THICKNESS + 1:
                    blocked = True; break
        if not blocked:
            valid.append((u, v, w, d))
    return valid

def build_regular_grid():
    if routing_region_base is None:
        return
    t0 = time.perf_counter()

    bx1, by1, bx2, by2 = routing_region_base.bounds
    xs = np.arange(int(bx1 // GRID_SPACING) * GRID_SPACING,
                   int(bx2 // GRID_SPACING + 1) * GRID_SPACING + 1,
                   GRID_SPACING, dtype=np.int32)
    ys = np.arange(int(by1 // GRID_SPACING) * GRID_SPACING,
                   int(by2 // GRID_SPACING + 1) * GRID_SPACING + 1,
                   GRID_SPACING, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys)
    cands  = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.int32)

    preg  = shapely_prep(routing_region_base)
    valid = np.array([preg.contains(Point(int(x), int(y))) for x, y in cands], dtype=bool)
    nodes_arr = cands[valid]
    t1 = time.perf_counter()

    node_map  = {(int(p[0]), int(p[1])): i for i, p in enumerate(nodes_arr)}
    raw_edges = []
    for i, (x, y) in enumerate(nodes_arr):
        e = (int(x) + GRID_SPACING, int(y))
        n = (int(x), int(y) + GRID_SPACING)
        if e in node_map: raw_edges.append((i, node_map[e], GRID_SPACING, 'E'))
        if n in node_map: raw_edges.append((i, node_map[n], GRID_SPACING, 'N'))
    t2 = time.perf_counter()

    valid_edges = _wall_filter(raw_edges, nodes_arr)
    t3 = time.perf_counter()

    _commit_grid(nodes_arr, valid_edges)

def build_base_regular_grid():
    global base_regular_env, base_regular_kd
    if routing_region_base is None:
        return
    t0 = time.perf_counter()

    bx1, by1, bx2, by2 = routing_region_base.bounds
    xs = np.arange(int(bx1 // GRID_SPACING) * GRID_SPACING,
                   int(bx2 // GRID_SPACING + 1) * GRID_SPACING + 1,
                   GRID_SPACING, dtype=np.int32)
    ys = np.arange(int(by1 // GRID_SPACING) * GRID_SPACING,
                   int(by2 // GRID_SPACING + 1) * GRID_SPACING + 1,
                   GRID_SPACING, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys)
    cands  = np.column_stack([xv.ravel(), yv.ravel()]).astype(np.int32)

    preg  = shapely_prep(routing_region_base)
    valid = np.array([preg.contains(Point(int(x), int(y))) for x, y in cands], dtype=bool)
    nodes_arr = cands[valid]

    node_map  = {(int(p[0]), int(p[1])): i for i, p in enumerate(nodes_arr)}
    raw_edges = []
    for i, (x, y) in enumerate(nodes_arr):
        e = (int(x) + GRID_SPACING, int(y))
        n = (int(x), int(y) + GRID_SPACING)
        if e in node_map: raw_edges.append((i, node_map[e], GRID_SPACING, 'E'))
        if n in node_map: raw_edges.append((i, node_map[n], GRID_SPACING, 'N'))

    valid_edges = _wall_filter(raw_edges, nodes_arr)
    
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}
    adj = {i: [] for i in range(len(nodes_arr))}
    for u, v, w, d in valid_edges:
        adj[u].append((v, w, d))
        adj[v].append((u, w, DIR_REV[d]))
        
    base_regular_env = EnvView(nodes_arr.astype(np.float32), adj)
    base_regular_kd = cKDTree(base_regular_env.nodes)
    print(f"[Base Regular Grid] Built {len(nodes_arr)} nodes in {(time.perf_counter() - t0)*1000:.1f}ms")

def update_dynamic_env(machine_poly):
    global current_env
    if grid_nodes is None:
        current_env = None
        return

    t0 = time.perf_counter()
    mx1, my1, mx2, my2 = machine_poly.bounds
    prm = shapely_prep(machine_poly)

    nx, ny = grid_nodes[:, 0], grid_nodes[:, 1]
    node_bbox = (nx >= mx1) & (nx <= mx2) & (ny >= my1) & (ny <= my2)
    blocked_nodes = set()
    for ni in np.where(node_bbox)[0]:
        if prm.contains(Point(float(grid_nodes[ni, 0]), float(grid_nodes[ni, 1]))):
            blocked_nodes.add(int(ni))

    ec = grid_edge_coords
    seg_x1 = np.minimum(ec[:, 0], ec[:, 2])
    seg_x2 = np.maximum(ec[:, 0], ec[:, 2])
    seg_y1 = np.minimum(ec[:, 1], ec[:, 3])
    seg_y2 = np.maximum(ec[:, 1], ec[:, 3])
    cand_edges = np.where(
        (seg_x2 >= mx1) & (seg_x1 <= mx2) &
        (seg_y2 >= my1) & (seg_y1 <= my2)
    )[0]

    blocked_edges = set()
    for ei in cand_edges:
        u, v, w, d = grid_edge_list[ei]
        if u in blocked_nodes or v in blocked_nodes:
            blocked_edges.add(ei)
        elif machine_poly.intersects(LineString([
                (float(grid_nodes[u, 0]), float(grid_nodes[u, 1])),
                (float(grid_nodes[v, 0]), float(grid_nodes[v, 1]))
            ])):
            blocked_edges.add(ei)

    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}
    filtered_adj = {i: [] for i in range(len(grid_nodes))}
    for ei, (u, v, w, d) in enumerate(grid_edge_list):
        if ei in blocked_edges or u in blocked_nodes or v in blocked_nodes:
            continue
        filtered_adj[u].append((v, w, d))
        filtered_adj[v].append((u, w, DIR_REV[d]))

    current_env = EnvView(grid_nodes, filtered_adj)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"Grid update: {ms:.1f} ms  (blocked nodes={len(blocked_nodes)}, edges={len(blocked_edges)})")

def build_hannan_grid(machine_pins=None, shift_walls=False):
    global _bnd_segs
    if routing_region_base is None:
        return
    t0 = time.perf_counter()

    interest = set()

    def _add_poly_verts(poly):
        coords = list(poly.exterior.coords) if hasattr(poly, 'exterior') else []
        for pt in coords:
            interest.add((round(float(pt[0])), round(float(pt[1]))))
        if hasattr(poly, 'interiors'):
            for interior in poly.interiors:
                for pt in interior.coords:
                    interest.add((round(float(pt[0])), round(float(pt[1]))))

    def _add_region_verts(region):
        if region.geom_type == 'Polygon': _add_poly_verts(region)
        elif region.geom_type in ('MultiPolygon', 'GeometryCollection'):
            for g in region.geoms:
                if g.geom_type == 'Polygon': _add_poly_verts(g)

    _add_region_verts(routing_region_base)
    
    offset_val = 100.0
    for r in rooms:
        if r.has_cover:
            sh_room = r.polygon.buffer(-offset_val, join_style=2)
            if not sh_room.is_empty:
                _add_poly_verts(sh_room)

    for col in columns:
        sh_col = col.buffer(offset_val, join_style=2)
        _add_poly_verts(sh_col)

    for shaft in shafts:
        minx_s, miny_s, maxx_s, maxy_s = shaft.bounds
        cx_s = (minx_s + maxx_s) / 2
        cy_s = (miny_s + maxy_s) / 2
        
        offset_pts = [
            (maxx_s + offset_val, cy_s),
            (minx_s - offset_val, cy_s),
            (cx_s, maxy_s + offset_val),
            (cx_s, miny_s - offset_val)
        ]
        for px, py in offset_pts:
            interest.add((round(px), round(py)))

    for pt in terminals.values():
        interest.add((round(pt[0]), round(pt[1])))
    if shaft_extraction:
        interest.add((round(shaft_extraction.centroid.x),
                      round(shaft_extraction.centroid.y)))

    for d in doors:
        cx = (d["d1"][0] + d["d2"][0]) // 2
        cy = (d["d1"][1] + d["d2"][1]) // 2
        interest.add((round(cx), round(cy)))

    if machine_pins:
        for pt in machine_pins.values():
            interest.add((round(pt[0]), round(pt[1])))

    if shift_walls:
        SHIFT = int(WALL_THICKNESS / 2) + 1
        for wall in walls:
            coords = list(wall.coords)
            for i in range(len(coords)-1):
                x1, y1 = round(coords[i][0]),   round(coords[i][1])
                x2, y2 = round(coords[i+1][0]), round(coords[i+1][1])
                length = math.hypot(x2-x1, y2-y1)
                if length < 1: continue
                nx, ny = -(y2-y1)/length, (x2-x1)/length
                mx, my = (x1+x2)//2, (y1+y2)//2
                interest.add((round(mx + nx*SHIFT), round(my + ny*SHIFT)))
                interest.add((round(mx - nx*SHIFT), round(my - ny*SHIFT)))

    interest_arr = np.array(list(interest), dtype=np.float64)
    t1 = time.perf_counter()

    if _bnd_segs is None:
        _bnd_segs = _extract_bnd_segs(routing_region_base)
    t2 = time.perf_counter()

    h_segs, v_segs = _cast_rays_numpy(interest_arr, _bnd_segs)
    t3 = time.perf_counter()

    inter_pts = _ray_ray_intersections_numpy(h_segs, v_segs)
    t4 = time.perf_counter()

    cand_set = set((round(x), round(y)) for x,y in interest)
    for y, x1, x2 in h_segs:
        cand_set.add((round(x1), round(y))); cand_set.add((round(x2), round(y)))
    for x, y1, y2 in v_segs:
        cand_set.add((round(x), round(y1))); cand_set.add((round(x), round(y2)))
    cand_set.update((round(x), round(y)) for x,y in inter_pts)

    preg = shapely_prep(routing_region_base)
    valid_pts = [p for p in cand_set if preg.contains(Point(p[0], p[1]))]
    nodes_arr = np.array(valid_pts, dtype=np.float32)
    node_map  = {p: i for i, p in enumerate(valid_pts)}
    t5 = time.perf_counter()

    raw_edges = []
    seen_edges = set()
    nodes_np = nodes_arr

    for y, x1, x2 in h_segs:
        mask = (np.abs(nodes_np[:,1] - y) < 1.0) & \
               (nodes_np[:,0] >= x1 - 1) & (nodes_np[:,0] <= x2 + 1)
        idxs = np.where(mask)[0]
        if len(idxs) < 2: continue
        order = idxs[np.argsort(nodes_np[idxs, 0])]
        for k in range(len(order)-1):
            u_i, v_i = int(order[k]), int(order[k+1])
            key = (min(u_i,v_i), max(u_i,v_i))
            if key not in seen_edges:
                seen_edges.add(key)
                w = float(abs(nodes_np[v_i,0] - nodes_np[u_i,0]))
                d = 'E' if nodes_np[v_i,0] > nodes_np[u_i,0] else 'W'
                raw_edges.append((u_i, v_i, w, d))

    for x, y1, y2 in v_segs:
        mask = (np.abs(nodes_np[:,0] - x) < 1.0) & \
               (nodes_np[:,1] >= y1 - 1) & (nodes_np[:,1] <= y2 + 1)
        idxs = np.where(mask)[0]
        if len(idxs) < 2: continue
        order = idxs[np.argsort(nodes_np[idxs, 1])]
        for k in range(len(order)-1):
            u_i, v_i = int(order[k]), int(order[k+1])
            key = (min(u_i,v_i), max(u_i,v_i))
            if key not in seen_edges:
                seen_edges.add(key)
                w = float(abs(nodes_np[v_i,1] - nodes_np[u_i,1]))
                d = 'N' if nodes_np[v_i,1] > nodes_np[u_i,1] else 'S'
                raw_edges.append((u_i, v_i, w, d))

    t6 = time.perf_counter()

    valid_edges = _wall_filter(raw_edges, nodes_arr)
    t7 = time.perf_counter()

    _commit_grid(nodes_arr, valid_edges)

def build_grid(machine_pins=None):
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env
    if graph_type_idx == 0:
        build_regular_grid()
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

def run_super_sink_astar(env, start_node_indices, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    if isinstance(start_node_indices, (int, np.integer)):
        start_node_indices = [start_node_indices]
    if not target_pin_names or not start_node_indices:
        return None, 0.0, None
        
    num_nodes = len(env.nodes)
    super_source_idx = num_nodes
    super_sink_idx = num_nodes + 1
    
    search_nodes = np.zeros((num_nodes + 2, 2), dtype=np.float32)
    search_nodes[:num_nodes] = env.nodes
    search_nodes[super_source_idx] = env.nodes[start_node_indices[0]]
    search_nodes[super_sink_idx] = global_pins["left_mid"]
    
    search_adj = {i: list(env.adj[i]) for i in env.adj}
    search_adj[super_source_idx] = []
    search_adj[super_sink_idx] = []
    
    # Link super_source to all starting boundary nodes with 0 cost
    for start_node in start_node_indices:
        search_adj[super_source_idx].append((start_node, 0.0, None))
        
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
        path, path_len = state_expanded_astar(search_env, super_source_idx, super_sink_idx, C_bend=C_bend, edge_weights=edge_weights)
    except Exception as e:
        print(f"Super Source/Sink A* error: {e}")
        return None, 0.0, None
        
    if path is None or len(path) < 3:
        return None, 0.0, None
        
    chosen_pin_idx = path[-2]
    chosen_pin_name = pin_name_by_idx.get(chosen_pin_idx, target_pin_names[0])
    path_without_virtual = path[1:-1]
    
    return path_without_virtual, path_len, chosen_pin_name

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
    if shaft_path and shaft_extraction:
        s_rep = get_representative_point(shaft_extraction)
        p_start = current_env.nodes[shaft_path[0]]
        shaft_segs.append(((float(s_rep[0]), float(s_rep[1])), (float(p_start[0]), float(p_start[1]))))
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
    # Must not intersect wall lines
    if any(machine_poly.intersects(w) for w in walls):
        return False
    # Must not intersect columns
    if any(machine_poly.intersects(col) for col in columns):
        return False
    # Must not intersect shafts
    if any(machine_poly.intersects(s) for s in shafts):
        return False
    return True

def compute_dijkstra_distance_field(start_nodes, env):
    if isinstance(start_nodes, (int, np.integer)):
        start_nodes = [start_nodes]
    distances = {n: 1e9 for n in env.adj}
    pq = []
    for n in start_nodes:
        distances[n] = 0.0
        heapq.heappush(pq, (0.0, n))
    
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

def get_placement_weights():
    if weight_mode_idx == 1:
        return {
            "Shaft": 1.0,
            "Kitchen": 1.0,
            "Bathroom": 1.0,
            "Bathroom 1": 1.0,
            "Bathroom 2": 1.0,
            "Toilet": 1.0,
            "Washroom": 1.0
        }
    else:
        return {
            "Shaft": 2.5,
            "Kitchen": 1.5,
            "Bathroom": 1.0,
            "Bathroom 1": 1.0,
            "Bathroom 2": 1.0,
            "Toilet": 1.0,
            "Washroom": 1.0
        }

def get_auto_placement_scores(env, shaft_boundary_nodes):
    terminal_nodes = {}
    for name, pt in terminals.items():
        _, node_idx = base_regular_kd.query(pt)
        terminal_nodes[name] = int(node_idx)
    
    weights = get_placement_weights()
    
    distance_fields = {}
    distance_fields["Shaft"] = compute_dijkstra_distance_field(shaft_boundary_nodes, env)
    for name, node_idx in terminal_nodes.items():
        distance_fields[name] = compute_dijkstra_distance_field(node_idx, env)
        
    node_scores = {}
    for n in range(len(env.nodes)):
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
    if base_regular_env is None or not shaft_extraction:
        return
        
    rep_pt = shaft_extraction.representative_point()
    shaft_center = (round(rep_pt.x), round(rep_pt.y))
    _, shaft_node_idx = base_regular_kd.query(shaft_center)
    shaft_node_idx = int(shaft_node_idx)
    
    shaft_boundary_nodes = [i for i, pt in enumerate(base_regular_env.nodes) 
                            if Point(pt[0], pt[1]).distance(shaft_extraction) < 500.0]
    if not shaft_boundary_nodes:
        shaft_boundary_nodes = [shaft_node_idx]
    
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
                range(len(base_regular_env.nodes)),
                key=lambda idx: math.hypot(base_regular_env.nodes[idx][0] - t_pt[0], base_regular_env.nodes[idx][1] - t_pt[1])
            )
            for n_idx in sorted_nodes:
                pt = base_regular_env.nodes[n_idx]
                for rot in [0, 90, 180, 270]:
                    if is_machine_placement_valid(pt[0], pt[1], rot):
                        machine_cx, machine_cy = pt[0], pt[1]
                        machine_angle = rot
                        ap_scores = {}
                        ap_fields = {}
                        pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                        build_grid(machine_pins=pins)
                        return
                        
    # ── Option 2: Topological Distance Fields ──
    elif auto_placement_mode_idx == 2:
        t0 = time.perf_counter()
        
        node_scores, distance_fields = get_auto_placement_scores(base_regular_env, shaft_boundary_nodes)
        ap_scores = node_scores
        ap_fields = distance_fields
        
        if not node_scores:
            return
            
        sorted_nodes = sorted(node_scores.keys(), key=lambda n: node_scores[n])
        
        for n_idx in sorted_nodes:
            n_x, n_y = base_regular_env.nodes[n_idx][0], base_regular_env.nodes[n_idx][1]
            
            best_rot = None
            min_rot_score = 1e18
            
            for rot in [0, 90, 180, 270]:
                if is_machine_placement_valid(n_x, n_y, rot):
                    global_pins = get_machine_pins(n_x, n_y, rot)
                    pin_nodes = {}
                    for pin_name, pt in global_pins.items():
                        if pin_name.startswith("c_"): continue
                        _, p_idx = base_regular_kd.query(pt)
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
                            
                    w = get_placement_weights()
                    rot_score = w["Shaft"] * shaft_dist + w["Kitchen"] * kitchen_dist + room_dists
                    if rot_score < min_rot_score:
                        min_rot_score = rot_score
                        best_rot = rot
                        
            if best_rot is not None:
                machine_cx, machine_cy = n_x, n_y
                machine_angle = best_rot
                pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                build_grid(machine_pins=pins)
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
    
    shaft_boundary_nodes = [i for i, pt in enumerate(current_env.nodes) 
                            if Point(pt[0], pt[1]).distance(shaft_extraction) < 500.0]
    if not shaft_boundary_nodes:
        shaft_boundary_nodes = [shaft_node_idx]

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

    # 1. Route Shaft via Super Source/Sink
    shaft_path, _, chosen_exhaust_pin = run_super_sink_astar(
        current_env,
        shaft_boundary_nodes,
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

    if routing_strategy_idx in (3, 4):
        # ── Strategy 3 & 4: Negotiated Congestion ──
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
                    start_nodes = shaft_boundary_nodes
                    targets = ["left_mid", "right_mid"]
                else:
                    if net_name == "Kitchen":
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
                    start_nodes = [start_node_idx]
                
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
                        if routing_strategy_idx == 4 and net_name in ("Shaft", "Kitchen"):
                            congestion_weight *= 0.35
                        current_weights[edge] = dist + congestion_weight
                        
                path, _, chosen_pin = run_super_sink_astar(
                    current_env,
                    start_nodes,
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
                if name == "Shaft" and path and shaft_extraction:
                    s_rep = get_representative_point(shaft_extraction)
                    p_start = current_env.nodes[path[0]]
                    segs.append(((float(s_rep[0]), float(s_rep[1])), (float(p_start[0]), float(p_start[1]))))
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
    build_base_regular_grid()
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

def get_turbo_color(t):
    # Clamp t to [0.0, 1.0]
    t = max(0.0, min(1.0, t))
    # Control points mapping Turbo color distribution
    points = [
        (0.0, (48, 18, 59)),
        (0.15, (70, 107, 227)),
        (0.35, (40, 188, 235)),
        (0.5, (50, 240, 150)),
        (0.65, (195, 230, 45)),
        (0.8, (250, 112, 32)),
        (1.0, (122, 4, 3))
    ]
    
    for i in range(len(points) - 1):
        t1, c1 = points[i]
        t2, c2 = points[i+1]
        if t1 <= t <= t2:
            factor = (t - t1) / (t2 - t1)
            r = int(c1[0] + factor * (c2[0] - c1[0]))
            g = int(c1[1] + factor * (c2[1] - c1[1]))
            b = int(c1[2] + factor * (c2[2] - c1[2]))
            return (r, g, b)
    return (122, 4, 3)

def draw_colorbar(screen, node_scores):
    if not node_scores:
        return
        
    cb_x = WINDOW_WIDTH - 85
    cb_y = CANVAS_TOP + 40
    cb_w = 20
    cb_h = 250
    
    # Border
    pygame.draw.rect(screen, (255, 255, 255), (cb_x - 1, cb_y - 1, cb_w + 2, cb_h + 2), 1)
    
    # Fill gradient: High cost at the top (y=0, t=1.0), Low cost at the bottom (y=cb_h, t=0.0)
    for y in range(cb_h):
        t = 1.0 - (y / cb_h)
        if heatmap_scale_mode == 0:
            t_sat = min(1.0, t / 0.75)
            c = get_turbo_color(t_sat)
        else:
            c = get_turbo_color(t)
        pygame.draw.line(screen, c, (cb_x, cb_y + y), (cb_x + cb_w - 1, cb_y + y))
        
    font_lbl = pygame.font.SysFont("Outfit", 14, bold=True)
    
    lbl_high = font_lbl.render("Max Cost (Red)", True, (231, 76, 60))
    screen.blit(lbl_high, (cb_x - lbl_high.get_width() - 8, cb_y))
    
    lbl_low = font_lbl.render("Min Cost (Blue)", True, (52, 152, 219))
    screen.blit(lbl_low, (cb_x - lbl_low.get_width() - 8, cb_y + cb_h - 12))
    
    lbl_title = font_lbl.render("COST HEATMAP", True, COLOR_TEXT)
    screen.blit(lbl_title, (cb_x + cb_w//2 - lbl_title.get_width()//2, cb_y - 20))

def draw_distance_heatmap(screen, node_scores):
    if not node_scores or base_regular_env is None:
        return
    min_s = min(node_scores.values())
    max_s = max(node_scores.values())
    diff = max_s - min_s if max_s > min_s else 1.0
    bg = COLOR_BG
    alpha = 0.60  # blend fraction: 0 = invisible, 1 = full colour

    # Precalculate log divisor if mode is 1
    if heatmap_scale_mode == 1:
        min_s_safe = max(1.0, min_s)
        max_ratio = max_s / min_s_safe
        max_log = math.log(max_ratio) if max_ratio > 1.0 else 1.0

    for node_idx, score in node_scores.items():
        if node_idx >= len(base_regular_env.nodes):
            continue
        pt = base_regular_env.nodes[node_idx]
        px, py = to_screen(pt[0], pt[1])
        
        if heatmap_scale_mode == 0:
            t = (score - min_s) / diff
            t_sat = min(1.0, t / 0.75)
        else:
            s_norm = score / max(1.0, min_s)
            val_log = math.log(max(1.0, s_norm))
            t_sat = val_log / max_log if max_log > 0 else 0.0
            
        tr, tg, tb = get_turbo_color(t_sat)
        # blend toward background for a faint underlay
        c = (
            int(bg[0] + alpha * (tr - bg[0])),
            int(bg[1] + alpha * (tg - bg[1])),
            int(bg[2] + alpha * (tb - bg[2])),
        )
        pygame.draw.circle(screen, c, (px, py), 2)

def record_history(routes, crossings_count):
    """Append one sample to the history buffers (called after every successful solve)."""
    length_m = 0.0
    turns = 0
    if routes:
        for _, segs in routes:
            length_m += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
            turns += calculate_tree_turns(segs)
    length_m /= 1000.0  # mm → m
    score = get_solution_score(routes, crossings_count) if routes else 0
    turns_per_m = turns / length_m if length_m > 0 else 0.0
    
    hist_length.append(length_m)
    hist_score.append(score)
    hist_turns.append(turns)
    hist_turns_per_len.append(turns_per_m)

def draw_plots(screen, font_small, font_bold):
    """Draw Length, Score, Turns, and Turns/Length sparklines in the right panel."""
    px = WINDOW_WIDTH - PANEL_W + 8
    pw = PANEL_W - 24
    ph = 145   # height of each chart area
    gap = 20   # gap between the plots
    titles   = ["DUCT LENGTH  (m)", "COST SCORE", "TURNS", "TURNS / METRE"]
    buffers  = [hist_length, hist_score, hist_turns, hist_turns_per_len]
    colors   = [(46, 204, 113), (241, 196, 15), (155, 89, 182), (26, 188, 156)]
    y_starts = [60, 60 + ph + gap, 60 + 2 * (ph + gap), 60 + 3 * (ph + gap)]

    for title, buf, col, py in zip(titles, buffers, colors, y_starts):
        # Background
        pygame.draw.rect(screen, COLOR_PLOT_BG, (px - 4, py, pw + 8, ph), border_radius=6)
        pygame.draw.rect(screen, (55, 55, 70),  (px - 4, py, pw + 8, ph), 1, border_radius=6)

        # Title
        lbl = font_bold.render(title, True, col)
        screen.blit(lbl, (px, py + 6))

        chart_y  = py + 26
        chart_h  = ph - 42
        chart_w  = pw
        n = len(buf)

        if n < 2:
            lbl_wait = font_small.render("Move machine to trace…", True, COLOR_MUTED)
            screen.blit(lbl_wait, (px, chart_y + chart_h // 2 - 8))
            continue

        vals  = list(buf)
        initial_val = vals[0]
        
        # Zero-based scaling
        lo    = 0.0
        hi    = max(vals)
        span  = hi - lo if hi > lo else 1.0

        def sx(i):  return px + int(i / (n - 1) * chart_w)
        def sy(v):  return chart_y + chart_h - int((v - lo) / span * chart_h)

        # Draw start baseline (dashed line at vals[0])
        ref_y = sy(initial_val)
        for dash_x in range(px, px + chart_w, 8):
            pygame.draw.line(screen, (80, 80, 100),
                             (dash_x, ref_y), (min(dash_x + 4, px + chart_w), ref_y))

        # Highlight minimum values reached
        min_val = min(vals)
        min_idx = vals.index(min_val)
        min_y = sy(min_val)
        
        # Draw horizontal dotted line at minimum
        for dash_x in range(px, px + chart_w, 6):
            pygame.draw.line(screen, (231, 76, 60),
                             (dash_x, min_y), (min(dash_x + 3, px + chart_w), min_y))

        # Draw vertical event markers (strategy changes, weight mode changes, etc.)
        for idx, label, m_col in hist_event_markers:
            rel = idx - (HIST_MAXLEN - n)
            if 0 <= rel < n:
                mx_px = sx(rel)
                for dash_y in range(chart_y, chart_y + chart_h, 8):
                    pygame.draw.line(screen, m_col, (mx_px, dash_y), (mx_px, min(dash_y + 4, chart_y + chart_h)), 1)
                lbl_ev = font_small.render(label, True, m_col)
                screen.blit(lbl_ev, (mx_px - lbl_ev.get_width() // 2, py + 16))

        # Sparkline
        pts = [(sx(i), sy(v)) for i, v in enumerate(vals)]
        if len(pts) >= 2:
            pygame.draw.lines(screen, col, False, pts, 2)

        # Draw minimum dot
        pygame.draw.circle(screen, (231, 76, 60), (sx(min_idx), min_y), 4)

        # Current value dot
        pygame.draw.circle(screen, (255, 255, 255), pts[-1], 4)
        pygame.draw.circle(screen, col, pts[-1], 3)

        # Percentages compared to initial state
        def get_pct_str(v):
            if abs(initial_val) < 1e-5:
                return "0.0%"
            pct = (v / initial_val) * 100.0
            return f"{pct:.1f}%"

        cur_val = vals[-1]
        lbl_cur = font_small.render(f"Cur: {cur_val:.1f} ({get_pct_str(cur_val)})", True, (255, 255, 255))
        lbl_min = font_small.render(f"Min: {min_val:.1f} ({get_pct_str(min_val)})", True, (231, 76, 60))
        lbl_hi  = font_small.render(f"Max: {hi:.1f}", True, COLOR_MUTED)

        screen.blit(lbl_cur, (px, chart_y + chart_h + 2))
        screen.blit(lbl_min, (px + chart_w - lbl_min.get_width(), chart_y + chart_h + 2))
        screen.blit(lbl_hi,  (px + chart_w - lbl_hi.get_width(), chart_y - 2))

def main():
    global machine_cx, machine_cy, machine_angle, show_grid_graph, graph_type_idx, routing_strategy_idx
    global auto_placement_mode_idx, show_heatmap, hist_ap_idx, weight_mode_idx, ap_scores, ap_fields, heatmap_scale_mode
    
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Integrated Auto-Placement & Ventilation Router (Demo 10.5)")
    clock = pygame.time.Clock()
    
    font_title = pygame.font.SysFont("Outfit", 24, bold=True)
    font_bold = pygame.font.SysFont("Outfit", 18, bold=True)
    font_small = pygame.font.SysFont("Outfit", 15)
    
    generate_new_dwelling()
    
    dragging = False
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    
    routes = []
    status = "Initial"
    elapsed_ms = 0.0
    total_nodes = 0
    
    needs_auto_placement = (auto_placement_mode_idx > 0)
    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
    
    running = True
    while running:
        if needs_auto_placement:
            needs_auto_placement = False
            run_auto_placement()
            routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
            if routes and not status.startswith("Blocked"):
                crossings_c = count_segment_crossings(routes)
                record_history(routes, crossings_c)
                hist_event_markers.append((len(hist_length) - 1, "Auto", (230, 126, 34)))

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
                elif event.button == 4: # Scroll Up (CCW)
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle + 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                elif event.button == 5: # Scroll Down (CW)
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle - 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                        
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
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    generate_new_dwelling()
                    hist_length.clear()
                    hist_score.clear()
                    hist_turns.clear()
                    hist_turns_per_len.clear()
                    hist_event_markers.clear()
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                    
                elif event.key == pygame.K_r:
                    auto_placement_mode_idx = 0
                    machine_angle = (machine_angle + 90) % 360
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"Rot:{machine_angle}", (46, 204, 113)))
                    
                elif event.key == pygame.K_g:
                    show_grid_graph = not show_grid_graph
                    
                elif event.key == pygame.K_c:
                    routing_strategy_idx = (routing_strategy_idx + 1) % len(ROUTING_STRATEGIES)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"Strat:{routing_strategy_idx}", (52, 152, 219)))
                    
                elif event.key == pygame.K_TAB:
                    graph_type_idx = (graph_type_idx + 1) % len(GRAPH_TYPES)
                    g_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    build_grid(machine_pins=g_pins)
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"Grid:{graph_type_idx}", (155, 89, 182)))
                    
                elif event.key == pygame.K_a:
                    if auto_placement_mode_idx > 0:
                        auto_placement_mode_idx = 0
                    else:
                        auto_placement_mode_idx = 2
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_p:
                    auto_placement_mode_idx = (auto_placement_mode_idx + 1) % len(AUTO_PLACEMENT_MODES)
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    
                elif event.key == pygame.K_v:
                    show_heatmap = not show_heatmap
                    
                elif event.key == pygame.K_h:
                    heatmap_scale_mode = (heatmap_scale_mode + 1) % 2
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"H:{'Log' if heatmap_scale_mode==1 else 'Lin'}", (150, 150, 150)))
                    
                elif event.key == pygame.K_w:
                    weight_mode_idx = (weight_mode_idx + 1) % 2
                    if auto_placement_mode_idx > 0:
                        needs_auto_placement = True
                    else:
                        if base_regular_env is not None and shaft_extraction is not None:
                            rep_pt = shaft_extraction.representative_point()
                            shaft_center = (round(rep_pt.x), round(rep_pt.y))
                            _, shaft_node_idx = base_regular_kd.query(shaft_center)
                            shaft_boundary_nodes = [i for i, pt in enumerate(base_regular_env.nodes) 
                                                    if Point(pt[0], pt[1]).distance(shaft_extraction) < 500.0]
                            if not shaft_boundary_nodes:
                                shaft_boundary_nodes = [int(shaft_node_idx)]
                            node_scores, distance_fields = get_auto_placement_scores(base_regular_env, shaft_boundary_nodes)
                            ap_scores = node_scores
                            ap_fields = distance_fields
                    routes, status, elapsed_ms, total_nodes = solve_ventilation_routing()
                    if routes and not status.startswith("Blocked"):
                        crossings_c = count_segment_crossings(routes)
                        record_history(routes, crossings_c)
                        hist_event_markers.append((len(hist_length) - 1, f"W:{'Eq' if weight_mode_idx==1 else 'Def'}", (241, 196, 15)))
                    
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
                    
        if show_heatmap and ap_scores:
            draw_distance_heatmap(screen, ap_scores)
            draw_colorbar(screen, ap_scores)
            
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
        pygame.draw.rect(screen, (40, 45, 55), (15, 75, CANVAS_LEFT - 40, 135), border_radius=6)
        lbl_ap_title = font_bold.render("AUTO-PLACEMENT STATE", True, (230, 126, 34))
        screen.blit(lbl_ap_title, (25, 85))
        mode_text = AUTO_PLACEMENT_MODES[auto_placement_mode_idx]
        lbl_ap_mode = font_bold.render(f"Mode: {mode_text}", True, COLOR_TEXT)
        screen.blit(lbl_ap_mode, (25, 105))
        lbl_ap_keys = font_small.render("[A] Auto-Placement | [P] Cycle Mode", True, COLOR_MUTED)
        screen.blit(lbl_ap_keys, (25, 125))
        h_text = "Disabled"
        if show_heatmap:
            h_text = "Linear (75%)" if heatmap_scale_mode == 0 else "Log Scale"
        lbl_ap_heatmap = font_small.render(f"[V] Heatmap: {h_text}", True, COLOR_MUTED)
        screen.blit(lbl_ap_heatmap, (25, 145))
        lbl_ap_scale = font_small.render("[H] Toggle Heatmap Scale", True, COLOR_MUTED)
        screen.blit(lbl_ap_scale, (25, 160))
        w_text = "Default" if weight_mode_idx == 0 else "Equal (1.0)"
        lbl_ap_weights = font_small.render(f"[W] Placement Weights: {w_text}", True, COLOR_MUTED)
        screen.blit(lbl_ap_weights, (25, 180))
        
        # 2. Solver Config Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 220, CANVAS_LEFT - 40, 120), border_radius=6)
        lbl_solv_title = font_bold.render("ROUTING PATH SOLVER", True, (52, 152, 219))
        screen.blit(lbl_solv_title, (25, 230))
        lbl_strat = font_small.render(f"Strategy: {ROUTING_STRATEGIES[routing_strategy_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_strat, (25, 250))
        lbl_graph = font_small.render(f"Grid type: {GRAPH_TYPES[graph_type_idx]}", True, COLOR_TEXT)
        screen.blit(lbl_graph, (25, 270))
        lbl_keys = font_small.render("[C] Cycle Strategy | [Tab] Cycle Grid", True, COLOR_MUTED)
        screen.blit(lbl_keys, (25, 290))
        lbl_gkey = font_small.render("[G] Toggle Grid Mesh Lines", True, COLOR_MUTED)
        screen.blit(lbl_gkey, (25, 310))
        lbl_skey = font_small.render("[Space] Gen New Apartment Dwelling", True, COLOR_MUTED)
        screen.blit(lbl_skey, (25, 325))
        
        # 3. Placement Info Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 350, CANVAS_LEFT - 40, 80), border_radius=6)
        lbl_pos_title = font_bold.render("MACHINE PLACEMENT", True, (46, 204, 113))
        screen.blit(lbl_pos_title, (25, 360))
        lbl_coord = font_small.render(f"Position: ({int(machine_cx)}, {int(machine_cy)}) mm", True, COLOR_TEXT)
        screen.blit(lbl_coord, (25, 380))
        lbl_rot = font_small.render(f"Rotation: {machine_angle}°", True, COLOR_TEXT)
        screen.blit(lbl_rot, (25, 400))
        
        # 4. KPI Metrics Card
        pygame.draw.rect(screen, (40, 45, 55), (15, 440, CANVAS_LEFT - 40, 115), border_radius=6)
        lbl_kpi_title = font_bold.render("ROUTING RUNTIME KPIs", True, (241, 196, 15))
        screen.blit(lbl_kpi_title, (25, 450))
        
        total_len_mm = 0.0
        total_turns_count = 0
        if routes:
            for name, segs in routes:
                total_len_mm += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs)
                total_turns_count += calculate_tree_turns(segs)
                
        lbl_len = font_small.render(f"Total Duct Length: {total_len_mm/1000.0:.2f} m", True, COLOR_TEXT)
        screen.blit(lbl_len, (25, 470))
        lbl_turns = font_small.render(f"Total Turns: {total_turns_count}", True, COLOR_TEXT)
        screen.blit(lbl_turns, (25, 490))
        crossings_count = count_segment_crossings(routes) if routes else 0
        lbl_cross = font_small.render(f"Duct Crossings: {crossings_count}", True, COLOR_TEXT)
        screen.blit(lbl_cross, (25, 510))
        lbl_score = font_small.render(f"Total Cost Score: {get_solution_score(routes, crossings_count) if routes else 0}", True, COLOR_TEXT)
        screen.blit(lbl_score, (25, 530))
        
        # 5. Status Box
        pygame.draw.rect(screen, (40, 45, 55), (15, 565, CANVAS_LEFT - 40, 240), border_radius=6)
        lbl_status_title = font_bold.render("SOLVER EXECUTION STATUS", True, (155, 89, 182))
        screen.blit(lbl_status_title, (25, 575))
        
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
            
        y_off = 595
        for ln in lines:
            lbl_line = font_small.render(ln, True, COLOR_TEXT)
            screen.blit(lbl_line, (25, y_off))
            y_off += 18
            
        lbl_runtime = font_small.render(f"Pathfinder time: {elapsed_ms:.1f} ms", True, COLOR_TEXT)
        screen.blit(lbl_runtime, (25, 745))
        lbl_nodes = font_small.render(f"Total routed nodes: {total_nodes}", True, COLOR_MUTED)
        screen.blit(lbl_nodes, (25, 765))
        lbl_fps = font_small.render(f"Render engine: Pygame ({clock.get_fps():.0f} FPS)", True, COLOR_MUTED)
        screen.blit(lbl_fps, (25, 785))
        
        # ── RIGHT PANEL: plots ──────────────────────────────────────────────
        panel_x = WINDOW_WIDTH - PANEL_W
        pygame.draw.rect(screen, COLOR_PANEL, (panel_x, 0, PANEL_W, WINDOW_HEIGHT))
        pygame.draw.line(screen, (55, 55, 70), (panel_x, 0), (panel_x, WINDOW_HEIGHT))
        lbl_panel = font_bold.render("PLACEMENT EXPLORER", True, COLOR_MUTED)
        screen.blit(lbl_panel, (panel_x + PANEL_W // 2 - lbl_panel.get_width() // 2, 8))
        draw_plots(screen, font_small, font_bold)

        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
