import sys
import os
import math
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale

# Add relative paths to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '08-bend-aware-non-orthogonal')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

from environment import NonOrthogonalEnvironment
from solver import BendAwareDualGraphGBFSSolver
import generative_layout

class WallAwareEnvironment(NonOrthogonalEnvironment):
    """
    Wall-Aware routing environment running in millimeter integer space.
    Allows routing perpendicular to partition walls but blocks parallel inside-wall routing.
    """
    def __init__(self, room, obstacles, terminals, walls, wall_thickness=150): # 150mm wall thickness
        self.walls = walls
        self.wall_thickness = wall_thickness
        super().__init__(room, obstacles, terminals)
        
        print("Filtering integer routing grid for wall constraints...")
        filtered_adj = {i: [] for i in range(len(self.nodes))}
        removed_count = 0
        
        # Buffer walls in integer space
        wall_polys = [w.buffer(wall_thickness / 2 - 0.1) for w in walls]
        
        for u in self.adj:
            for v, weight, direction in self.adj[u]:
                if u < v:
                    p1 = self.nodes[u]
                    p2 = self.nodes[v]
                    line = LineString([p1, p2])
                    
                    is_invalid = False
                    for w_poly in wall_polys:
                        inter = line.intersection(w_poly)
                        if not inter.is_empty:
                            # Check if overlap length exceeds wall thickness
                            if inter.length > (wall_thickness + 1):
                                is_invalid = True
                                break
                                
                    if is_invalid:
                        removed_count += 1
                    else:
                        filtered_adj[u].append((v, weight, direction))
                        filtered_adj[v].append((u, weight, direction))
                        
        self.adj = filtered_adj
        print(f"Wall constraint filter completed. Removed {removed_count} parallel inside-wall edges.")

def snap_to_integer_grid(geom):
    """Snaps any Shapely geometry to the nearest integer grid (rounding coordinates)."""
    if geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        ext = [(round(x), round(y)) for x, y in geom.exterior.coords]
        ints = []
        for interior in geom.interiors:
            ints.append([(round(x), round(y)) for x, y in interior.coords])
        return Polygon(ext, ints)
    elif geom.geom_type == 'LineString':
        return LineString([(round(x), round(y)) for x, y in geom.coords])
    elif geom.geom_type == 'MultiLineString':
        return unary_union([snap_to_integer_grid(g) for g in geom.geoms])
    elif geom.geom_type == 'MultiPolygon':
        return unary_union([snap_to_integer_grid(g) for g in geom.geoms])
    return geom

def main():
    print("==================================================")
    print("DEMO 09: WALL-AWARE MILLIMETER ROUTING PIPELINE")
    print("==================================================")
    
    print("\n1. Generating floor plan layout in meters...")
    rooms_m = generative_layout.generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    # Scale factor from meters to millimeters
    SCALE_TO_MM = 1000.0
    
    print("\n2. Setting up ceiling covers (navigable zones) in meter space first...")
    covered_names = ["Hallway", "Kitchen", "Bathroom", "Bathroom 1", "Bathroom 2", "Toilet", "Washroom", "Bedroom 1"]
    for r in rooms_m:
        r.has_cover = any(cn in r.name for cn in covered_names)
        
    # CRITICAL: Union the covered rooms FIRST in floating-point meter space
    routing_region_m = unary_union([r.polygon for r in rooms_m if r.has_cover])
    
    # Extract wall line centerlines in meter space
    print("\n3. Extracting internal partition walls in meter space first...")
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
    
    print(f"\n4. Scaling all geometries to integer millimeters (scaling factor = {int(SCALE_TO_MM)})...")
    rooms = []
    for r in rooms_m:
        scaled_poly = snap_to_integer_grid(shapely_scale(r.polygon, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
        room_scaled = generative_layout.Room(scaled_poly, r.name)
        room_scaled.has_cover = r.has_cover
        rooms.append(room_scaled)
        
    all_polys = [r.polygon for r in rooms]
    footprint = unary_union(all_polys)
    
    # Scale the combined routing region to millimeters
    routing_region = snap_to_integer_grid(shapely_scale(routing_region_m, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0)))
    
    # Scale walls to millimeters
    walls = [snap_to_integer_grid(shapely_scale(w, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for w in walls_m]
    print(f"Scaled {len(walls)} partition wall segments to integer space.")
    
    # Scale obstacles (columns, shafts, doors)
    columns_m = generative_layout.generate_structural_grid(unary_union([r.polygon for r in rooms_m]), spacing=4.0)
    columns = [snap_to_integer_grid(shapely_scale(col, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for col in columns_m]
    
    shafts_m = generative_layout.generate_mep_shafts(rooms_m)
    shafts = [snap_to_integer_grid(shapely_scale(s, xfact=SCALE_TO_MM, yfact=SCALE_TO_MM, origin=(0,0))) for s in shafts_m]
    
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
        
    # Subtract columns and shafts directly from the navigable routing_region.
    # This turns structural columns and MEP shafts into physical "holes" in the ceiling zone,
    # geometrically preventing the router from generating paths that cross them.
    for col in columns:
        routing_region = routing_region.difference(col)
    for shaft in shafts:
        routing_region = routing_region.difference(shaft)
        
    print("\n5. Setting up wet room terminals (centroids)...")
    wet_rooms_m = [r for r in rooms_m if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    terminals = []
    terminal_room_map = {}
    for r in wet_rooms_m:
        centroid = r.polygon.centroid
        t_pt = (round(centroid.x * SCALE_TO_MM), round(centroid.y * SCALE_TO_MM))
        terminals.append(t_pt)
        terminal_room_map[t_pt] = r.name
    print(f"Terminals (in mm): {list(terminal_room_map.values())}")
    
    # Establish obstacles to pass to grid builder for secondary checks
    obstacles = [col for col in columns if col.intersects(routing_region)] + [s for s in shafts if s.intersects(routing_region)]
    
    print("\n6. Building Ray-Cast integer grid...")
    env = WallAwareEnvironment(routing_region, obstacles, terminals, walls, wall_thickness=150) # 150mm walls
    print(f"Final routing nodes: {len(env.nodes)}")
    
    print("\n7. Running solver (Bend-Aware Dual-Graph GBFS, C_bend=5000.0)...")
    solver = BendAwareDualGraphGBFSSolver(env, C_bend=5000.0)
    result = solver.solve()
    
    segments = result["segments"]
    print(f"Route solved successfully!")
    print(f"Total length: {result['weight'] / 1000.0:.2f} m ({result['weight']:.0f} mm)")
    print(f"Total turns: {result['turns']}")
    
    # Save SVG visualization locally inside demo/09
    output_filename = os.path.join(current_dir, "floorplan_ventilation.svg")
    save_mep_to_svg(rooms, columns, shafts, doors, terminals, segments, output_filename)
    print(f"\nSaved layout visualization to {output_filename}")

def save_mep_to_svg(rooms, columns, shafts, doors, terminals, segments, filename, wall_thickness=150):
    all_polys = [r.polygon for r in rooms]
    merged = unary_union(all_polys)
    minx, miny, maxx, maxy = merged.bounds
    
    scale = 40  
    
    def mm_to_px_x(val):
        return offset_x + (val / 1000.0) * scale
    def mm_to_px_y(val):
        return offset_y + ((maxy - val) / 1000.0) * scale
        
    svg_w = ((maxx - minx) / 1000.0 + 2) * scale
    svg_h = ((maxy - miny) / 1000.0 + 2) * scale
    offset_x = 1 * scale
    offset_y = 1 * scale
    
    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">')
    svg_lines.append('  <!-- Background -->')
    svg_lines.append(f'  <rect width="100%" height="100%" fill="#fbfcfc"/>')
    
    svg_lines.append('  <style>')
    svg_lines.append('    .wall { fill: #2c3e50; stroke: #1a252f; stroke-width: 1; }')
    svg_lines.append('    .room { fill: #ffffff; stroke: none; }')
    svg_lines.append('    .room-covered { fill: #e8f8f5; stroke: none; }')
    svg_lines.append('    .hallway { fill: #f8f9f9; stroke: none; }')
    svg_lines.append('    .hallway-covered { fill: #eaf2f8; stroke: none; }')
    svg_lines.append('    .column { fill: #2c3e50; stroke: #1a252f; stroke-width: 1; }')
    svg_lines.append('    .shaft { fill: #eaeded; stroke: #7f8c8d; stroke-dasharray: 2; stroke-width: 1; }')
    svg_lines.append('    .door-swing { stroke: #7f8c8d; stroke-width: 0.8; fill: none; stroke-dasharray: 2; }')
    svg_lines.append('    .door-leaf { stroke: #2c3e50; stroke-width: 2.5; fill: none; }')
    svg_lines.append('    .label { font-family: "Courier New", Courier, monospace; font-size: 13px; font-weight: bold; fill: #2c3e50; text-anchor: middle; }')
    svg_lines.append('    .area { font-family: "Courier New", Courier, monospace; font-size: 10px; fill: #7f8c8d; text-anchor: middle; }')
    svg_lines.append('    .grid { stroke: #eaeded; stroke-width: 0.5; }')
    svg_lines.append('    .scale-bar { stroke: #2c3e50; stroke-width: 3; }')
    svg_lines.append('    .shaft-cross { stroke: #7f8c8d; stroke-width: 1; }')
    svg_lines.append('    .vent-duct { stroke: #3498db; stroke-width: 5; fill: none; stroke-linecap: round; stroke-linejoin: round; stroke-opacity: 0.85; }')
    svg_lines.append('    .vent-terminal { fill: #e74c3c; stroke: #ffffff; stroke-width: 2; }')
    svg_lines.append('  </style>')
    
    for x in range(int(minx / 1000), int(maxx / 1000) + 1):
        x_pos = offset_x + x * scale
        svg_lines.append(f'  <line x1="{x_pos}" y1="{offset_y}" x2="{x_pos}" y2="{offset_y + ((maxy-miny)/1000.0)*scale}" class="grid"/>')
    for y in range(int(miny / 1000), int(maxy / 1000) + 1):
        y_pos = offset_y + y * scale
        svg_lines.append(f'  <line x1="{offset_x}" y1="{y_pos}" x2="{offset_x + ((maxx-minx)/1000.0)*scale}" y2="{y_pos}" class="grid"/>')

    # Draw Wall Base
    for room in rooms:
        coords = room.polygon.exterior.coords
        svg_points = " ".join([f"{mm_to_px_x(x)},{mm_to_px_y(y)}" for x, y in coords])
        svg_lines.append(f'  <polygon points="{svg_points}" class="wall"/>')
        
    # Draw Inner Rooms
    for room in rooms:
        room_poly = room.polygon
        for shaft in shafts:
            room_poly = room_poly.difference(shaft)
            
        inner_poly = room_poly.buffer(-wall_thickness)
        if inner_poly.is_empty:
            continue
            
        coords = inner_poly.exterior.coords
        svg_points = " ".join([f"{mm_to_px_x(x)},{mm_to_px_y(y)}" for x, y in coords])
        
        if room.is_hallway:
            cls = "hallway-covered" if room.has_cover else "hallway"
        else:
            cls = "room-covered" if room.has_cover else "room"
            
        svg_lines.append(f'  <polygon points="{svg_points}" class="{cls}"/>')
        
        # Room Label & Area
        centroid = inner_poly.centroid
        cx = mm_to_px_x(centroid.x)
        cy = mm_to_px_y(centroid.y)
        svg_lines.append(f'  <text x="{cx}" y="{cy - 5}" class="label">{room.name}</text>')
        area_m2 = room.polygon.area / 1000000.0
        svg_lines.append(f'  <text x="{cx}" y="{cy + 10}" class="area">{area_m2:.1f} m²</text>')
        if room.has_cover:
            svg_lines.append(f'  <text x="{cx}" y="{cy + 22}" font-family="monospace" font-size="8" fill="#16a085" text-anchor="middle">[FALSE CEILING]</text>')
        
    # Cut Door Openings & Draw Swings
    for door in doors:
        d1 = door["d1"]
        d2 = door["d2"]
        px, py = door["swing_dir"]
        
        leaf_end_x = d1[0] + door["width"] * px
        leaf_end_y = d1[1] + door["width"] * py
        
        svg_d1_x = mm_to_px_x(d1[0])
        svg_d1_y = mm_to_px_y(d1[1])
        svg_d2_x = mm_to_px_x(d2[0])
        svg_d2_y = mm_to_px_y(d2[1])
        svg_leaf_x = mm_to_px_x(leaf_end_x)
        svg_leaf_y = mm_to_px_y(leaf_end_y)
        
        svg_lines.append(f'  <!-- Door Opening -->')
        svg_lines.append(f'  <line x1="{svg_d1_x}" y1="{svg_d1_y}" x2="{svg_d2_x}" y2="{svg_d2_y}" stroke="#ffffff" stroke-width="4"/>')
        svg_lines.append(f'  <line x1="{svg_d1_x}" y1="{svg_d1_y}" x2="{svg_leaf_x}" y2="{svg_leaf_y}" class="door-leaf"/>')
        
        sweep = 0 if (px * (d2[1]-d1[1]) - py * (d2[0]-d1[0])) > 0 else 1
        r = (door["width"] / 1000.0) * scale
        svg_lines.append(f'  <path d="M {svg_leaf_x} {svg_leaf_y} A {r} {r} 0 0 {sweep} {svg_d2_x} {svg_d2_y}" class="door-swing"/>')
        
        if door.get("is_entrance", False):
            ox, oy = -px, -py
            lx = d1[0] + 0.5 * (d2[0] - d1[0]) + ox * 800
            ly = d1[1] + 0.5 * (d2[1] - d1[1]) + oy * 800
            svg_lx = mm_to_px_x(lx)
            svg_ly = mm_to_px_y(ly)
            svg_lines.append(f'  <text x="{svg_lx}" y="{svg_ly}" font-family="monospace" font-size="10" font-weight="bold" fill="#e74c3c" text-anchor="middle">ENTRANCE</text>')

    # Draw MEP Shafts
    for s in shafts:
        minx_s, miny_s, maxx_s, maxy_s = s.bounds
        sx = mm_to_px_x(minx_s)
        sy = mm_to_px_y(maxy_s)
        sw = ((maxx_s - minx_s) / 1000.0) * scale
        sh = ((maxy_s - miny_s) / 1000.0) * scale
        
        svg_lines.append(f'  <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" class="shaft"/>')
        svg_lines.append(f'  <line x1="{sx}" y1="{sy}" x2="{sx + sw}" y2="{sy + sh}" class="shaft-cross"/>')
        svg_lines.append(f'  <line x1="{sx + sw}" y1="{sy}" x2="{sx}" y2="{sy + sh}" class="shaft-cross"/>')
        svg_lines.append(f'  <text x="{sx + sw/2}" y="{sy + sh/2 + 3}" font-family="monospace" font-size="8" fill="#7f8c8d" text-anchor="middle">MEP</text>')

    # Draw Structural Columns
    for col in columns:
        minx_c, miny_c, maxx_c, maxy_c = col.bounds
        cx = mm_to_px_x(minx_c)
        cy = mm_to_px_y(maxy_c)
        cw = ((maxx_c - minx_c) / 1000.0) * scale
        ch = ((maxy_c - miny_c) / 1000.0) * scale
        svg_lines.append(f'  <rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" class="column"/>')

    # Draw Ventilation Ducts
    svg_lines.append('  <!-- Ventilation Duct Layout -->')
    for p1, p2 in segments:
        svg_x1 = mm_to_px_x(p1[0])
        svg_y1 = mm_to_px_y(p1[1])
        svg_x2 = mm_to_px_x(p2[0])
        svg_y2 = mm_to_px_y(p2[1])
        svg_lines.append(f'  <line x1="{svg_x1}" y1="{svg_y1}" x2="{svg_x2}" y2="{svg_y2}" class="vent-duct"/>')

    # Draw Terminals
    svg_lines.append('  <!-- Wet Room Terminals -->')
    for t in terminals:
        svg_tx = mm_to_px_x(t[0])
        svg_ty = mm_to_px_y(t[1])
        svg_lines.append(f'  <circle cx="{svg_tx}" cy="{svg_ty}" r="6" class="vent-terminal"/>')

    # Draw Scale Bar
    scale_y = svg_h - 40
    svg_lines.append(f'  <line x1="{offset_x}" y1="{scale_y}" x2="{offset_x + 5 * scale}" y2="{scale_y}" class="scale-bar"/>')
    svg_lines.append(f'  <text x="{offset_x + 2.5 * scale}" y="{scale_y - 10}" font-family="monospace" font-size="12" fill="#2c3e50" text-anchor="middle">5 Meters</text>')
    
    svg_lines.append('</svg>')
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

if __name__ == "__main__":
    main()
