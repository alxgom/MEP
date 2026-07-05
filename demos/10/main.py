import sys
import os
import math
import time
import itertools
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

# Graph types that can be cycled via Tab key
GRAPH_TYPES = [
    "Regular 200mm Grid",
    "Hannan Grid (numpy)",
    "Hannan + Shifted Nodes",
]

ROUTING_STRATEGIES = [
    "Greedy Sequential",
    "First Fit",
    "Best Fit"
]
routing_strategy_idx = 1

# Pygame Window Config
WINDOW_WIDTH, WINDOW_HEIGHT = 1250, 850
CANVAS_LEFT = 280
CANVAS_TOP = 40
CANVAS_W = WINDOW_WIDTH - CANVAS_LEFT - 40
CANVAS_H = WINDOW_HEIGHT - CANVAS_TOP - 40

# Color Scheme (Premium Slate Dark mode)
COLOR_BG = (28, 28, 36)
COLOR_PANEL = (35, 35, 45)
COLOR_ROOM = (45, 52, 54)
COLOR_ROOM_COVERED = (52, 73, 94) # Slate Blue for false ceilings
COLOR_HALLWAY = (57, 72, 85)
COLOR_WALL = (99, 110, 114)
COLOR_COLUMN = (15, 76, 129) # Solid Blue-gray
COLOR_SHAFT = (231, 76, 60) # Red
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

# Coordinate Mapping helpers (Millimeters to Screen pixels)
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

# ── Lightweight env view for the grid visualiser ─────────────────────────────
class EnvView:
    """Thin wrapper with nodes + filtered adjacency."""
    def __init__(self, nodes, adj):
        self.nodes = nodes
        self.adj   = adj

def snap_to_integer_grid(geom):
    """Snap Shapely geometry coordinates to nearest integer (mm)."""
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
wall_polys = []          # Pre-buffered wall polygons
routing_region_base = None
shaft_extraction = None
terminals        = {}
wet_room_names   = []
machine_cx    = 0.0
machine_cy    = 0.0
machine_angle = 0

# Grid cache ── rebuilt per dwelling (and per drag for Hannan) ──────────────────
graph_type_idx  = 1      # 0=Regular, 1=Hannan, 2=Hannan+Shifted
grid_nodes      = None   # np.ndarray (N, 2) float32 mm coordinates
grid_adj_base   = None   # dict i → [(j, weight, dir), …] (no machine blocking)
grid_edge_list  = None   # list of (u, v, weight, dir) unique edges
grid_edge_coords= None   # np.ndarray (E, 4) float32 for bbox filter
grid_kd         = None   # cKDTree for O(log N) nearest-node queries
current_env     = None   # EnvView — updated every drag event
_bnd_segs       = None   # numpy (S,4) boundary segments — rebuilt per dwelling
show_grid_graph = False

def get_representative_point(poly):
    centroid = poly.centroid
    if poly.contains(centroid):
        return (round(centroid.x), round(centroid.y))
    rep = poly.representative_point()
    return (round(rep.x), round(rep.y))

# ──────────────────────────────────────────────────────────────────────────
# NUMPY HELPERS FOR FAST HANNAN RAY CASTING
# ──────────────────────────────────────────────────────────────────────────

def _extract_bnd_segs(region):
    """Extract all boundary line segments from a (Multi)Polygon as numpy (S,4) array.
    Works with polygon holes (column/shaft cutouts already subtracted into region)."""
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
    """
    For each interest point cast 4 axis-aligned rays, find closest boundary crossing.
    All intersection math is vectorised numpy over the full segment array — no Shapely calls.

    interest_pts_arr: (N, 2) float64
    bnd: (S, 4) float64  [x1,y1,x2,y2]
    Returns: h_segs list[(y, x_min, x_max)], v_segs list[(x, y_min, y_max)]
    """
    h_segs, v_segs = [], []
    dx_s = bnd[:, 2] - bnd[:, 0]   # (S,)
    dy_s = bnd[:, 3] - bnd[:, 1]   # (S,)

    for x0, y0 in interest_pts_arr:
        # ─ Horizontal rays (intersect non-horizontal segments) ─────
        nh = np.abs(dy_s) > eps
        if np.any(nh):
            t_h  = (y0 - bnd[nh, 1]) / dy_s[nh]          # param along segment
            ok_h = (t_h >= -eps) & (t_h <= 1.0 + eps)
            x_i  = bnd[nh, 0] + t_h * dx_s[nh]           # x at crossing

            east = x_i[ok_h & (x_i > x0 + eps)]
            if len(east): h_segs.append((y0, x0, float(east.min())))

            west = x_i[ok_h & (x_i < x0 - eps)]
            if len(west): h_segs.append((y0, float(west.max()), x0))

        # ─ Vertical rays (intersect non-vertical segments) ──────
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
    """
    Compute all (x_v, y_h) crossing points where a horizontal and vertical
    scanline segment overlap. Fully vectorised — no Python loop over pairs.
    Returns list of (x, y) float tuples.
    """
    if not h_segs or not v_segs:
        return []
    h = np.array(h_segs, dtype=np.float64)   # (H, 3): [y, x_min, x_max]
    v = np.array(v_segs, dtype=np.float64)   # (V, 3): [x, y_min, y_max]

    y_h  = h[:, 0:1]     # (H,1)
    x1_h = h[:, 1:2]     # (H,1)
    x2_h = h[:, 2:3]     # (H,1)
    x_v  = v[:, 0:1].T   # (1,V)
    y1_v = v[:, 1:2].T   # (1,V)
    y2_v = v[:, 2:3].T   # (1,V)

    cross = ((x_v >= x1_h - eps) & (x_v <= x2_h + eps) &
             (y_h >= y1_v - eps) & (y_h <= y2_v + eps))  # (H,V) bool

    hi, vi = np.where(cross)
    return [(float(v[j, 0]), float(h[i, 0])) for i, j in zip(hi, vi)]


# ──────────────────────────────────────────────────────────────────────────
# GRID BUILDERS
# ──────────────────────────────────────────────────────────────────────────

def _commit_grid(nodes_arr, valid_edges):
    """
    Shared finalisation step: build adj dict, edge-coord array, KD-tree,
    and update all global grid variables.
    """
    global grid_nodes, grid_adj_base, grid_edge_list, grid_edge_coords, grid_kd, current_env
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}

    # ── Connect extraction shaft to all valid face-normal offset nodes ──
    if shaft_extraction is not None and len(nodes_arr) > 0:
        rep_pt = shaft_extraction.representative_point()
        sc = (round(rep_pt.x), round(rep_pt.y))
        
        # Bounding box of the extraction shaft
        minx_s, miny_s, maxx_s, maxy_s = shaft_extraction.bounds
        cx_s = (minx_s + maxx_s) / 2
        cy_s = (miny_s + maxy_s) / 2
        
        # The 4 face-normal offset points and the direction of entry to shaft
        offset_val = 100.0
        face_pts = [
            (maxx_s + offset_val, cy_s, 'W'),  # East face node (enters West to shaft)
            (minx_s - offset_val, cy_s, 'E'),  # West face node (enters East to shaft)
            (cx_s, maxy_s + offset_val, 'S'),  # North face node (enters South to shaft)
            (cx_s, miny_s - offset_val, 'N')   # South face node (enters North to shaft)
        ]
        
        # Append shaft node
        shaft_idx = len(nodes_arr)
        nodes_arr = np.vstack([nodes_arr, [sc[0], sc[1]]])
        
        # Connect to any face-normal offset node that is close to the grid nodes
        connected_any = False
        for px, py, d in face_pts:
            diffs = np.hypot(nodes_arr[:-1, 0] - px, nodes_arr[:-1, 1] - py)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 10.0:  # within 10mm
                w = float(np.hypot(sc[0] - nodes_arr[min_idx, 0], sc[1] - nodes_arr[min_idx, 1]))
                valid_edges.append((min_idx, shaft_idx, w, d))
                connected_any = True
                
        # Fallback: if no face-normal nodes are in the grid, connect to the single closest node
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

    # ── Map pin coordinates to node indices to filter machine face-edges ──
    pin_indices = {}
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    for name, pt in global_pins.items():
        diffs = np.hypot(nodes_arr[:, 0] - pt[0], nodes_arr[:, 1] - pt[1])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 1.0:  # exact match within 1mm
            pin_indices[int(min_idx)] = name

    # Allowed direction based on machine_angle
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

    # Filter edges so that machine pins only connect outward normal to their face
    filtered_edges = []
    for u, v, w, d in valid_edges:
        keep = True
        
        # Check u as pin
        if u in pin_indices:
            pin_name = pin_indices[u]
            is_left = pin_name in ('left_mid', 'tl', 'bl')
            allowed = left_dir if is_left else right_dir
            if d != allowed:
                keep = False
                
        # Check v as pin
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
    """Remove edges that run parallel inside walls (length of overlap > WALL_THICKNESS).
    Shared by both grid builders."""
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
            # Bounding box overlap check
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
    """
    Regular GRID_SPACING-mm orthogonal grid (built once per dwelling, no machine).
    Dynamic machine update is handled by update_dynamic_env() on every drag.
    """
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
    print(f"[Regular] {len(nodes_arr)} nodes, {len(valid_edges)} edges | "
          f"pts={1000*(t1-t0):.0f}ms edges={1000*(t2-t1):.0f}ms "
          f"wfilt={1000*(t3-t2):.0f}ms total={1000*(t3-t0):.0f}ms")


def build_hannan_grid(machine_pins=None, shift_walls=False):
    """
    Hannan (boundary-conforming) grid using numpy ray casting.
    machine_pins: dict {name: (x,y)} — pin positions become actual nodes.
    shift_walls:  add corridor nodes offset ±WALL_THICKNESS from wall centerlines.
    Total build time: ~15–30 ms (vs 1100 ms with Shapely rays).
    """
    global _bnd_segs
    if routing_region_base is None:
        return
    t0 = time.perf_counter()

    # ── 1. Interest points ────────────────────────────────────────────────────
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
    
    # ── Offset corridors (mimics regulations, staying 10cm away from walls/obstacles) ──
    offset_val = 100.0  # 10cm offset
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
        
        # 4 face-normal offset points (10cm from the face centerlines)
        offset_pts = [
            (maxx_s + offset_val, cy_s),  # East
            (minx_s - offset_val, cy_s),  # West
            (cx_s, maxy_s + offset_val),  # North
            (cx_s, miny_s - offset_val)   # South
        ]
        for px, py in offset_pts:
            interest.add((round(px), round(py)))

    # Terminal centroids
    for pt in terminals.values():
        interest.add((round(pt[0]), round(pt[1])))
    if shaft_extraction:
        interest.add((round(shaft_extraction.centroid.x),
                      round(shaft_extraction.centroid.y)))

    # Door midpoints/centroids to preserve hallway/room routing continuity
    for d in doors:
        cx = (d["d1"][0] + d["d2"][0]) // 2
        cy = (d["d1"][1] + d["d2"][1]) // 2
        interest.add((round(cx), round(cy)))

    # Machine pins — key insight: pins ARE nodes (not just snapped)
    if machine_pins:
        for pt in machine_pins.values():
            interest.add((round(pt[0]), round(pt[1])))

    # Optional: wall-corridor shifted nodes (add nodes on each side of every wall)
    if shift_walls:
        SHIFT = int(WALL_THICKNESS / 2) + 1
        for wall in walls:
            coords = list(wall.coords)
            for i in range(len(coords)-1):
                x1, y1 = round(coords[i][0]),   round(coords[i][1])
                x2, y2 = round(coords[i+1][0]), round(coords[i+1][1])
                length = math.hypot(x2-x1, y2-y1)
                if length < 1: continue
                nx, ny = -(y2-y1)/length, (x2-x1)/length  # outward normal
                mx, my = (x1+x2)//2, (y1+y2)//2
                interest.add((round(mx + nx*SHIFT), round(my + ny*SHIFT)))
                interest.add((round(mx - nx*SHIFT), round(my - ny*SHIFT)))

    interest_arr = np.array(list(interest), dtype=np.float64)
    t1 = time.perf_counter()

    # ── 2. Boundary segments (cache per dwelling) ────────────────────────
    if _bnd_segs is None:
        _bnd_segs = _extract_bnd_segs(routing_region_base)
    t2 = time.perf_counter()

    # ── 3. Ray casting — all numpy, no Shapely ───────────────────────────
    h_segs, v_segs = _cast_rays_numpy(interest_arr, _bnd_segs)
    t3 = time.perf_counter()

    # ── 4. Ray-ray intersections — numpy broadcast ───────────────────────
    inter_pts = _ray_ray_intersections_numpy(h_segs, v_segs)
    t4 = time.perf_counter()

    # ── 5. Candidate nodes ───────────────────────────────────────────────
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

    # ── 6. Edge build along scanlines (numpy per-segment filter) ───────────
    raw_edges = []
    seen_edges = set()
    nodes_np = nodes_arr  # (N,2)

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

    # ── 7. Wall filter (shared helper) ────────────────────────────────
    valid_edges = _wall_filter(raw_edges, nodes_arr)
    t7 = time.perf_counter()

    _commit_grid(nodes_arr, valid_edges)
    t8 = time.perf_counter()

    label = GRAPH_TYPES[graph_type_idx]
    print(f"[{label}] {len(nodes_arr)} nodes, {len(valid_edges)} edges | "
          f"interest={1000*(t1-t0):.0f}ms bnd={1000*(t2-t1):.0f}ms "
          f"rays={1000*(t3-t2):.0f}ms cross={1000*(t4-t3):.0f}ms "
          f"filter={1000*(t5-t4):.0f}ms edges={1000*(t6-t5):.0f}ms "
          f"wfilt={1000*(t7-t6):.0f}ms total={1000*(t8-t0):.0f}ms")


def build_grid(machine_pins=None):
    """
    Dispatch to the correct builder based on graph_type_idx.
    machine_pins: dict {name: (x,y)} — for Hannan grids, pins become actual nodes.
    """
    if graph_type_idx == 0:
        build_regular_grid()          # machine not included; update_dynamic_env() handles per-drag
    elif graph_type_idx == 1:
        build_hannan_grid(machine_pins=machine_pins, shift_walls=False)
    else:  # 2
        build_hannan_grid(machine_pins=machine_pins, shift_walls=True)

def generate_new_dwelling():
    global rooms, columns, shafts, doors, walls, wall_polys, routing_region_base, shaft_extraction, terminals, wet_room_names, machine_cx, machine_cy, machine_angle
    
    rooms_m = generative_layout.generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    # Scale all to integer millimeters
    rooms = []
    covered_names = ["Hallway", "Kitchen", "Bathroom", "Bathroom 1", "Bathroom 2", "Toilet", "Washroom", "Bedroom 1"]
    for r in rooms_m:
        scaled_poly = snap_to_integer_grid(shapely_scale(r.polygon, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        room_scaled = generative_layout.Room(scaled_poly, r.name)
        room_scaled.has_cover = any(cn in r.name for cn in covered_names)
        rooms.append(room_scaled)
        
    footprint = unary_union([r.polygon for r in rooms])
    
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
    
    # Pre-buffer wall polygons and subtract columns/shafts so they do not cap inside them
    wall_polys = []
    for w in walls:
        wp = w.buffer(WALL_THICKNESS / 2 - 0.1)
        for col in columns:
            wp = wp.difference(col)
        for s in shafts:
            wp = wp.difference(s)
        if not wp.is_empty:
            wall_polys.append(wp)
    
    # Navigable space (covered rooms unioned)
    routing_region_m = unary_union([r.polygon for r in rooms_m if any(cn in r.name for cn in covered_names)])
    routing_region_base = snap_to_integer_grid(shapely_scale(routing_region_m, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
    
    # Subtract columns and shafts
    for col in columns:
        routing_region_base = routing_region_base.difference(col)
    for shaft in shafts:
        routing_region_base = routing_region_base.difference(shaft)
        
    # Pre-assigned extraction shaft (always index 0)
    shaft_extraction = shafts[0] if shafts else None
    
    # Centroids/representative points of wet rooms
    wet_rooms = [r for r in rooms if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    terminals = {}
    for r in wet_rooms:
        t_pt = get_representative_point(r.polygon)
        terminals[r.name] = t_pt
        
    # Initial machine placement: placed on a Bathroom or Washroom closest to the extraction shaft
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
    _bnd_segs = None  # invalidate boundary-seg cache for new dwelling
    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
    build_grid(machine_pins=pins)

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
        
    return global_pins

def snap_pins_to_graph(global_pins):
    """Map each machine pin to its nearest grid node via O(log N) KD-tree lookup."""
    if grid_kd is None:
        return {}
    pin_pts = np.array([list(pt) for pt in global_pins.values()], dtype=float)
    _, idxs = grid_kd.query(pin_pts)
    return {name: int(idx) for name, idx in zip(global_pins.keys(), idxs)}

def update_dynamic_env(machine_poly):
    """
    Per-drag update (~1-5 ms):
      1. numpy bbox pre-filter to find nodes + edges near the machine
      2. Shapely check only for the small candidate set
      3. Rebuild adj without blocked nodes/edges
    """
    global current_env
    if grid_nodes is None:
        current_env = None
        return

    t0 = time.perf_counter()
    mx1, my1, mx2, my2 = machine_poly.bounds
    prm = shapely_prep(machine_poly)

    # ── Blocked nodes (inside machine footprint) ───────────────────────────
    nx, ny = grid_nodes[:, 0], grid_nodes[:, 1]
    node_bbox = (nx >= mx1) & (nx <= mx2) & (ny >= my1) & (ny <= my2)
    blocked_nodes = set()
    for ni in np.where(node_bbox)[0]:
        if prm.contains(Point(float(grid_nodes[ni, 0]), float(grid_nodes[ni, 1]))):
            blocked_nodes.add(int(ni))

    # ── Blocked edges (cross machine bbox) ─────────────────────────────
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

    # ── Build filtered adj ────────────────────────────────────────────────
    if not blocked_nodes and not blocked_edges:
        current_env = EnvView(grid_nodes, grid_adj_base)
        return

    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}
    filtered_adj = {i: [] for i in range(len(grid_nodes))}
    for ei, (u, v, w, d) in enumerate(grid_edge_list):
        if ei in blocked_edges or u in blocked_nodes or v in blocked_nodes:
            continue
        filtered_adj[u].append((v, w, d))
        filtered_adj[v].append((u, w, DIR_REV[d]))

    current_env = EnvView(grid_nodes, filtered_adj)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"Grid update: {ms:.1f} ms  "
          f"(blocked nodes={len(blocked_nodes)}, edges={len(blocked_edges)})")

def block_path_and_node(path, chosen_pin_node_idx, accumulated_weights, env, block_nodes=False):
    # Block all incident edges to the chosen pin node
    if chosen_pin_node_idx in env.adj:
        for nbr, dist, direction in env.adj[chosen_pin_node_idx]:
            edge = (min(chosen_pin_node_idx, nbr), max(chosen_pin_node_idx, nbr))
            accumulated_weights[edge] = 1e9
    # Block all incident edges to the terminal node (start of the path)
    if len(path) > 0:
        start_node_idx = path[0]
        if start_node_idx in env.adj:
            for nbr, dist, direction in env.adj[start_node_idx]:
                edge = (min(start_node_idx, nbr), max(start_node_idx, nbr))
                accumulated_weights[edge] = 1e9
    # Block all edges along the path
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = (min(u, v), max(u, v))
        accumulated_weights[edge] = 1e9
    # Block intermediate nodes along the path (preventing crossings)
    if block_nodes and len(path) > 2:
        for u in path[1:-1]:
            if u in env.adj:
                for nbr, dist, direction in env.adj[u]:
                    edge = (min(u, nbr), max(u, nbr))
                    accumulated_weights[edge] = 1e9

def run_super_sink_astar(env, start_node_idx, target_pin_names, pin_node_map, global_pins, machine_angle, C_bend, edge_weights=None):
    DIR_REV = {'E': 'W', 'N': 'S', 'W': 'E', 'S': 'N'}
    
    # 1. Allowed outward direction based on machine_angle
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
        
    # 2. Append the virtual Super Sink node at machine center
    super_sink_idx = len(env.nodes)
    search_nodes = np.vstack([env.nodes, [machine_cx, machine_cy]])
    
    # 3. Shallow copy of adjacency dict
    search_adj = {k: list(v) for k, v in env.adj.items()}
    search_adj[super_sink_idx] = []
    
    # 4. Connect pin nodes to super sink
    pin_name_by_idx = {}
    for pin_name in target_pin_names:
        if pin_name not in pin_node_map:
            continue
        pin_idx = pin_node_map[pin_name]
        pin_name_by_idx[pin_idx] = pin_name
        
        is_left = pin_name in ('left_mid', 'tl', 'bl')
        allowed_out = left_dir if is_left else right_dir
        inward_dir = DIR_REV[allowed_out]
        
        # Edge pin -> sink (weight 0, inward direction)
        search_adj[pin_idx].append((super_sink_idx, 0.0, inward_dir))
        # Edge sink -> pin (weight 0, outward direction)
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

def count_segment_crossings(routes):
    """Counts the number of perpendicular crossings between different routed paths."""
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
                    if ax1 < bx1 < ax2 and by1 < ay1 < by2:
                        crossings += 1
                else:
                    if bx1 < ax1 < bx2 and ay1 < by1 < ay2:
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
    available_small_pins = ["tl", "tr", "bl", "br"]

    for room_name in perm:
        if room_name == "Kitchen":
            kitchen_pt = terminals.get("Kitchen")
            if not kitchen_pt:
                continue
            _, kitchen_node_idx = grid_kd.query(kitchen_pt)
            kitchen_node_idx = int(kitchen_node_idx)
            
            kitchen_path, _, _ = run_super_sink_astar(
                current_env,
                kitchen_node_idx,
                [kitchen_pin_name],
                pin_node_map,
                global_pins,
                machine_angle,
                C_BEND,
                edge_weights=accumulated_weights
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
        else:
            if not available_small_pins:
                return False, None, f"No port for {room_name}", 0
            room_pt = terminals[room_name]
            _, room_node_idx = grid_kd.query(room_pt)
            room_node_idx = int(room_node_idx)
            
            room_path, _, chosen_small_pin = run_super_sink_astar(
                current_env,
                room_node_idx,
                available_small_pins,
                pin_node_map,
                global_pins,
                machine_angle,
                C_BEND,
                edge_weights=accumulated_weights
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
    # C_bend is 4000.0, crossings are heavily penalized (100,000.0 each)
    score = total_len + 4000.0 * total_turns + 100000.0 * crossings
    return score

def solve_ventilation_routing():
    t_start = time.perf_counter()
    global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)

    machine_poly = Polygon([
        global_pins["tl"], global_pins["tr"], global_pins["br"], global_pins["bl"]
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

    pin_node_map = snap_pins_to_graph(global_pins)
    if not pin_node_map or not shaft_extraction:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: Missing pins or shaft", elapsed_ms, 0

    rep_pt = shaft_extraction.representative_point()
    shaft_center = (round(rep_pt.x), round(rep_pt.y))
    _, shaft_node_idx = grid_kd.query(shaft_center)
    shaft_node_idx = int(shaft_node_idx)

    # 1. Route Shaft via Super Sink
    shaft_path, _, chosen_exhaust_pin = run_super_sink_astar(
        current_env,
        shaft_node_idx,
        ["left_mid", "right_mid"],
        pin_node_map,
        global_pins,
        machine_angle,
        C_BEND
    )

    if shaft_path is None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, "Blocked: No path to shaft", elapsed_ms, 0

    # 2. Backtracking search over permutations of the rooms
    from itertools import permutations
    other_rooms = sorted(
        [name for name in terminals.keys() if name != "Kitchen" and any(w in name for w in ["Bathroom", "Toilet", "Washroom"])],
        key=lambda name: math.hypot(terminals[name][0] - machine_cx, terminals[name][1] - machine_cy)
    )
    all_rooms_to_permute = ["Kitchen"] + other_rooms
    
    if routing_strategy_idx == 0:
        # Greedy Sequential: only the default proximity order
        all_perms = [tuple(all_rooms_to_permute)]
    else:
        all_perms = list(permutations(all_rooms_to_permute))

    best_routes = None
    best_crossings = 1e9
    best_score = 1e18
    best_total_nodes = 0
    perm_attempts = 0

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
                if routing_strategy_idx == 1: # First Fit
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    status_text = f"Success: Routed all (tried {perm_attempts} perms, 0 crossings)"
                    return routes_cand, status_text, elapsed_ms, total_nodes_cand
                else: # Best Fit or Greedy
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

    # Return the best route found
    if best_routes is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        status_text = f"Success: Routed all (tried {perm_attempts} perms, {best_crossings} crossings)"
        return best_routes, status_text, elapsed_ms, best_total_nodes
    else:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return None, f"Routing Blocked (tried {perm_attempts} perms)", elapsed_ms, 0

def main():
    global machine_cx, machine_cy, machine_angle, show_grid_graph, graph_type_idx, routing_strategy_idx
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Integrated Ventilation Router - Interactive Pygame")
    clock = pygame.time.Clock()
    
    # Fonts
    font_large = pygame.font.SysFont("Consolas", 18, bold=True)
    font_medium = pygame.font.SysFont("Consolas", 14, bold=True)
    font_small = pygame.font.SysFont("Consolas", 11)
    font_kpi = pygame.font.SysFont("Consolas", 20, bold=True)
    font_tiny = pygame.font.SysFont("Consolas", 8, bold=True)
    
    # Initial setup
    generate_new_dwelling()
    
    # Solved state
    routes, status, elapsed_ms, eval_count = solve_ventilation_routing()
    
    # Drag state
    dragging = False
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    
    running = True
    while running:
        need_solve = False
        
        # Hover check
        mouse_pos = pygame.mouse.get_pos()
        mx_mm, my_mm = to_mm(mouse_pos[0], mouse_pos[1])
        
        # Build machine polygon to check mouse hover / drag trigger
        global_pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
        m_poly = Polygon([
            global_pins["tl"], global_pins["tr"], global_pins["br"], global_pins["bl"]
        ])
        
        is_hover = m_poly.contains(Point(mx_mm, my_mm))
        
        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    show_grid_graph = not show_grid_graph
                elif event.key == pygame.K_TAB:
                    graph_type_idx = (graph_type_idx + 1) % len(GRAPH_TYPES)
                    pins = get_machine_pins(machine_cx, machine_cy, machine_angle)
                    build_grid(machine_pins=pins)
                    routes, status, elapsed_ms, eval_count = solve_ventilation_routing()
                    print(f"Switched to: {GRAPH_TYPES[graph_type_idx]}")
                elif event.key == pygame.K_c:
                    routing_strategy_idx = (routing_strategy_idx + 1) % len(ROUTING_STRATEGIES)
                    need_solve = True
                    print(f"Switched Strategy to: {ROUTING_STRATEGIES[routing_strategy_idx]}")
                elif event.key == pygame.K_SPACE:
                    generate_new_dwelling()
                    need_solve = True
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    machine_angle = (machine_angle + 90) % 360
                    need_solve = True
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    machine_angle = (machine_angle - 90) % 360
                    need_solve = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    if is_hover:
                        dragging = True
                        drag_offset_x = mx_mm - machine_cx
                        drag_offset_y = my_mm - machine_cy
                elif event.button == 4: # Scroll Up (Rotate CCW)
                    machine_angle = (machine_angle + 90) % 360
                    need_solve = True
                elif event.button == 5: # Scroll Down (Rotate CW)
                    machine_angle = (machine_angle - 90) % 360
                    need_solve = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    # Check boundary box limits
                    machine_cx = max(500.0, min(14500.0, mx_mm - drag_offset_x))
                    machine_cy = max(500.0, min(10500.0, my_mm - drag_offset_y))
                    need_solve = True

        if need_solve:
            routes, status, elapsed_ms, eval_count = solve_ventilation_routing()

        # Rendering
        screen.fill(COLOR_BG)
        
        # 1. Draw Rooms
        for r in rooms:
            coords = r.polygon.exterior.coords
            scr_points = [to_screen(x, y) for x, y in coords]
            
            # Colors
            if r.is_hallway:
                color = COLOR_HALLWAY
            elif r.has_cover:
                color = COLOR_ROOM_COVERED
            else:
                color = COLOR_ROOM
                
            pygame.draw.polygon(screen, color, scr_points)
            
            # Label
            centroid = r.polygon.centroid
            sx, sy = to_screen(centroid.x, centroid.y)
            lbl = font_small.render(r.name, True, COLOR_TEXT)
            lbl_rect = lbl.get_rect(center=(sx, sy - 8))
            screen.blit(lbl, lbl_rect)
            
            # Area
            area_m2 = r.polygon.area / 1000000.0
            area_lbl = font_small.render(f"{area_m2:.1f} m²", True, COLOR_MUTED)
            area_rect = area_lbl.get_rect(center=(sx, sy + 6))
            screen.blit(area_lbl, area_rect)
            
            if r.has_cover:
                cover_lbl = font_small.render("[COVERED]", True, (26, 188, 156))
                cover_rect = cover_lbl.get_rect(center=(sx, sy + 18))
                screen.blit(cover_lbl, cover_rect)
                
        # 2. Draw Walls (Lines between rooms)
        for w in walls:
            for i in range(len(w.coords) - 1):
                p1 = to_screen(w.coords[i][0], w.coords[i][1])
                p2 = to_screen(w.coords[i+1][0], w.coords[i+1][1])
                pygame.draw.line(screen, COLOR_WALL, p1, p2, 3)
                
        # 3. Draw Columns
        for col in columns:
            coords = col.exterior.coords
            scr_points = [to_screen(x, y) for x, y in coords]
            pygame.draw.polygon(screen, COLOR_COLUMN, scr_points)
            
        # 4. Draw Shafts
        for shaft in shafts:
            coords = shaft.exterior.coords
            scr_points = [to_screen(x, y) for x, y in coords]
            is_extr = (shaft == shaft_extraction)
            color = COLOR_SHAFT if is_extr else COLOR_MUTED
            pygame.draw.polygon(screen, color, scr_points, 2 if not is_extr else 3)
            
            # Cross hatching inside extraction shaft
            if is_extr:
                s_minx, s_miny, s_maxx, s_maxy = shaft.bounds
                pygame.draw.line(screen, COLOR_SHAFT, to_screen(s_minx, s_miny), to_screen(s_maxx, s_maxy), 1)
                pygame.draw.line(screen, COLOR_SHAFT, to_screen(s_maxx, s_miny), to_screen(s_minx, s_maxy), 1)

        # Draw Grid Graph (Nodes & Edges) if toggled on
        if show_grid_graph and current_env:
            for u in current_env.adj:
                for v, weight, direction in current_env.adj[u]:
                    if u < v:
                        p1 = current_env.nodes[u]
                        p2 = current_env.nodes[v]
                        pygame.draw.line(screen, (75, 75, 95), to_screen(p1[0], p1[1]), to_screen(p2[0], p2[1]), 1)
            for p in current_env.nodes:
                pygame.draw.circle(screen, (100, 100, 120), to_screen(p[0], p[1]), 2)

        # 5. Draw Doors
        for d in doors:
            p1 = to_screen(d["d1"][0], d["d1"][1])
            p2 = to_screen(d["d2"][0], d["d2"][1])
            pygame.draw.line(screen, COLOR_DOOR, p1, p2, 4)

        # 6. Draw Solved Routes (Ventilation pipes)
        if routes:
            for name, segments in routes:
                color = ROUTE_COLORS.get(name, COLOR_TEXT)
                for p1, p2 in segments:
                    pygame.draw.line(screen, color, to_screen(p1[0], p1[1]), to_screen(p2[0], p2[1]), 5)

        # 7. Draw Terminals (centroids of wet rooms)
        for name, pt in terminals.items():
            cx, cy = to_screen(pt[0], pt[1])
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6)
            pygame.draw.circle(screen, COLOR_PANEL, (cx, cy), 4)

        # 8. Draw Extraction Machine
        scr_m_points = [to_screen(x, y) for x, y in m_poly.exterior.coords]
        m_color = COLOR_MACHINE_HOVER if (is_hover or dragging) else COLOR_MACHINE
        border_color = (46, 204, 113) if routes else (231, 76, 60) # Green if solved, red if blocked
        
        # Fill
        pygame.draw.polygon(screen, m_color, scr_m_points)
        # Outline
        pygame.draw.polygon(screen, border_color, scr_m_points, 3)
        
        # Draw 6 Pins
        pin_colors = {
            "left_mid": (231, 76, 60), "right_mid": (231, 76, 60), # Large Red
            "tl": (230, 126, 34), "tr": (230, 126, 34), "bl": (230, 126, 34), "br": (230, 126, 34) # Small Orange
        }
        for pin_name, pt in global_pins.items():
            px, py = to_screen(pt[0], pt[1])
            size = 12 if "mid" in pin_name else 8
            pygame.draw.rect(screen, pin_colors[pin_name], (px - size/2, py - size/2, size, size))
            pygame.draw.rect(screen, (255, 255, 255), (px - size/2, py - size/2, size, size), 1)

        # 9. Sidebar Panel (HUD)
        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, CANVAS_LEFT, WINDOW_HEIGHT))
        pygame.draw.line(screen, COLOR_WALL, (CANVAS_LEFT, 0), (CANVAS_LEFT, WINDOW_HEIGHT), 2)
        
        # Render Sidebar Info
        title = font_large.render("MEP VENTILATION ROUTER", True, COLOR_TEXT)
        screen.blit(title, (20, 30))
        subtitle = font_small.render("demo/10 - Pygame Edition", True, COLOR_MUTED)
        screen.blit(subtitle, (20, 52))
        
        # Status Box
        pygame.draw.rect(screen, (20, 20, 25), (20, 90, CANVAS_LEFT - 40, 75), 0, 4)
        status_lbl = font_medium.render("ROUTING STATUS:", True, COLOR_MUTED)
        screen.blit(status_lbl, (30, 93))
        
        color_status = (46, 204, 113) if routes else (231, 76, 60)
        status_txt = "SOLVED" if routes else "BLOCKED"
        status_txt_render = font_large.render(status_txt, True, color_status)
        screen.blit(status_txt_render, (30, 110))
        
        explanation_lbl = font_small.render(status, True, COLOR_TEXT)
        screen.blit(explanation_lbl, (30, 137))
        
        # KPI Cards (Length & Turns)
        total_length_m = 0.0
        total_turns = 0
        if routes:
            for name, segs in routes:
                total_length_m += sum(np.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in segs) / 1000.0
                total_turns += calculate_tree_turns(segs)
                
        # Card backgrounds
        pygame.draw.rect(screen, (20, 20, 25), (20, 180, 115, 55), 0, 4)
        pygame.draw.rect(screen, (20, 20, 25), (145, 180, 115, 55), 0, 4)
        
        # Card 1 (Total Length)
        lbl_card1 = font_tiny.render("TOTAL DUCT LENGTH", True, COLOR_MUTED)
        screen.blit(lbl_card1, (26, 187))
        val_card1 = font_kpi.render(f"{total_length_m:.1f} m", True, (46, 204, 113) if routes else COLOR_TEXT)
        screen.blit(val_card1, (26, 204))
        
        # Card 2 (Total Turns)
        lbl_card2 = font_tiny.render("TOTAL BENDS", True, COLOR_MUTED)
        screen.blit(lbl_card2, (151, 187))
        val_card2 = font_kpi.render(f"{total_turns}", True, (230, 126, 34) if routes else COLOR_TEXT)
        screen.blit(val_card2, (151, 204))
        
        # Execution Specs
        specs_y = 250
        n_nodes = len(grid_nodes) if grid_nodes is not None else 0
        n_edges = len(grid_edge_list) if grid_edge_list else 0
        specs = [
            ("Graph:",          GRAPH_TYPES[graph_type_idx]),
            ("Strategy:",       ROUTING_STRATEGIES[routing_strategy_idx]),
            ("Nodes / Edges:",  f"{n_nodes} / {n_edges}"),
            ("Update time:",    f"{elapsed_ms:.1f} ms"),
            ("Machine Angle:",  f"{machine_angle}°"),
            ("Position:",       f"({int(machine_cx)}, {int(machine_cy)}) mm")
        ]

        for label, val in specs:
            lbl_r = font_medium.render(label, True, COLOR_TEXT)
            col   = (46, 204, 113) if "ms" in val or "Hannan" in val else COLOR_MUTED
            val_r = font_medium.render(val, True, col)
            screen.blit(lbl_r, (20, specs_y))
            screen.blit(val_r, (170, specs_y))
            specs_y += 24
            
        # Legend Panel
        legend_y = specs_y + 30
        legend_lbl = font_large.render("LEGEND:", True, COLOR_TEXT)
        screen.blit(legend_lbl, (20, legend_y))
        legend_y += 25
        
        legend_items = [
            ("Extraction Shaft", COLOR_SHAFT),
            ("False Ceiling Cover", COLOR_ROOM_COVERED),
            ("Wet Room Centroids", (255, 255, 255)),
            ("Structural Columns", COLOR_COLUMN),
            ("Kitchen duct path", ROUTE_COLORS["Kitchen"]),
            ("Shaft exhaust path", ROUTE_COLORS["Shaft"]),
            ("Other wet room ducts", ROUTE_COLORS["Bathroom"])
        ]
        for name, color in legend_items:
            if "Ceiling" in name or "Column" in name:
                pygame.draw.rect(screen, color, (20, legend_y + 2, 12, 12))
            else:
                pygame.draw.circle(screen, color, (26, legend_y + 8), 6)
            
            lbl_item = font_small.render(name, True, COLOR_TEXT)
            screen.blit(lbl_item, (45, legend_y))
            legend_y += 22
            
        # Controls HUD
        controls_y = legend_y + 30
        controls_lbl = font_large.render("CONTROLS:", True, COLOR_TEXT)
        screen.blit(controls_lbl, (20, controls_y))
        controls_y += 25
        
        controls = [
            ("Click + Drag",        "Move Machine"),
            ("Scroll Wheel / A D",  "Rotate 90°"),
            ("Tab",                 "Cycle Graph Type"),
            ("C",                   "Cycle Routing Strategy"),
            ("G",                   "Toggle Grid Overlay"),
            ("Spacebar",            "New Random Layout"),
            ("Escape",              "Exit")
        ]
        for key, action in controls:
            lbl_key = font_medium.render(key, True, (241, 196, 15))
            lbl_act = font_small.render(action, True, COLOR_TEXT)
            screen.blit(lbl_key, (20, controls_y))
            screen.blit(lbl_act, (20, controls_y + 16))
            controls_y += 35
            
        # Draw boundary container box for layout canvas
        pygame.draw.rect(screen, COLOR_WALL, (CANVAS_LEFT, 0, WINDOW_WIDTH - CANVAS_LEFT, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
