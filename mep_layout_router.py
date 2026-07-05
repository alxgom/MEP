import sys
import os
import math
import random
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union

# Add the demo directory to sys.path to import solver and environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'demos', '08-bend-aware-non-orthogonal'))

from environment import NonOrthogonalEnvironment
from solver import BendAwareKMBSolver
import generative_layout

def main():
    print("Generating floor plan...")
    # Generate an 8-room layout (will contain Toilet, Washroom, Bathroom 1 & 2, Kitchen, Bedrooms, Living Room)
    rooms = generative_layout.generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    # Extract building footprint
    all_polys = [r.polygon for r in rooms]
    footprint = unary_union(all_polys)
    
    # Columns and shafts
    columns = generative_layout.generate_structural_grid(footprint, spacing=4.0)
    shafts = generative_layout.generate_mep_shafts(rooms)
    doors = generative_layout.find_door_openings(rooms)
    entrance = generative_layout.find_entrance_door(rooms, footprint)
    if entrance:
        doors.append(entrance)
        
    print("Configuring false ceiling covers (navigable routing zones)...")
    # All wet rooms (Toilet, Washroom, Bathrooms, Kitchen) and the Hallway get false ceilings
    # We also optionally include one bedroom to show routing through a dry room with ceiling cover
    covered_names = ["Hallway", "Kitchen", "Bathroom", "Bathroom 1", "Bathroom 2", "Toilet", "Washroom", "Bedroom 1"]
    
    covered_rooms = [r for r in rooms if any(cn in r.name for cn in covered_names)]
    for r in rooms:
        r.has_cover = any(cn in r.name for cn in covered_names)
        
    # The navigable routing area is the union of the covered rooms
    routing_region = unary_union([r.polygon for r in covered_rooms])
    
    print("Setting up terminals (wet room centroids)...")
    # Centroids of the wet rooms
    wet_rooms = [r for r in rooms if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    terminals = []
    terminal_room_map = {}
    for r in wet_rooms:
        centroid = r.polygon.centroid
        t_pt = (centroid.x, centroid.y)
        terminals.append(t_pt)
        terminal_room_map[t_pt] = r.name
        
    print(f"Terminals: {list(terminal_room_map.values())}")
    
    # Obstacles: Columns that intersect the routing region
    obstacles = [col for col in columns if col.intersects(routing_region)]
    
    print("Building routing environment grid...")
    env = NonOrthogonalEnvironment(routing_region, obstacles, terminals)
    print(f"Routing grid nodes: {len(env.nodes)}")
    
    print("Solving optimal ventilation routing tree (Bend-Aware KMB)...")
    # C_bend = 5.0 (penalizes sharp turns to optimize ventilation flow)
    solver = BendAwareKMBSolver(env, C_bend=5.0)
    result = solver.solve()
    
    segments = result["segments"]
    print(f"Route solved! Total length: {result['weight']:.2f} meters, Turns: {result['turns']}")
    
    # Save the floor plan overlaid with the ventilation ducts
    save_mep_to_svg(rooms, columns, shafts, doors, terminals, segments, "floorplan_ventilation.svg")

def save_mep_to_svg(rooms, columns, shafts, doors, terminals, segments, filename="floorplan_ventilation.svg", wall_thickness=0.15):
    all_polys = [r.polygon for r in rooms]
    merged = unary_union(all_polys)
    minx, miny, maxx, maxy = merged.bounds
    
    scale = 40  
    svg_w = (maxx - minx + 2) * scale
    svg_h = (maxy - miny + 2) * scale
    offset_x = 1 * scale
    offset_y = 1 * scale
    
    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">')
    svg_lines.append('  <!-- Background -->')
    svg_lines.append(f'  <rect width="100%" height="100%" fill="#fbfcfc"/>')
    
    svg_lines.append('  <style>')
    svg_lines.append('    .wall { fill: #2c3e50; stroke: #1a252f; stroke-width: 1; }')
    svg_lines.append('    .room { fill: #ffffff; stroke: none; }')
    svg_lines.append('    .room-covered { fill: #e8f8f5; stroke: none; } /* Soft pastel teal for false ceiling */')
    svg_lines.append('    .hallway { fill: #f8f9f9; stroke: none; }')
    svg_lines.append('    .hallway-covered { fill: #eaf2f8; stroke: none; } /* Light blue-gray for covered corridor */')
    svg_lines.append('    .column { fill: #2c3e50; stroke: #1a252f; stroke-width: 1; }')
    svg_lines.append('    .shaft { fill: #eaeded; stroke: #7f8c8d; stroke-dasharray: 2; stroke-width: 1; }')
    svg_lines.append('    .door-swing { stroke: #7f8c8d; stroke-width: 0.8; fill: none; stroke-dasharray: 2; }')
    svg_lines.append('    .door-leaf { stroke: #2c3e50; stroke-width: 2.5; fill: none; }')
    svg_lines.append('    .label { font-family: "Courier New", Courier, monospace; font-size: 13px; font-weight: bold; fill: #2c3e50; text-anchor: middle; }')
    svg_lines.append('    .area { font-family: "Courier New", Courier, monospace; font-size: 10px; fill: #7f8c8d; text-anchor: middle; }')
    svg_lines.append('    .grid { stroke: #eaeded; stroke-width: 0.5; }')
    svg_lines.append('    .scale-bar { stroke: #2c3e50; stroke-width: 3; }')
    svg_lines.append('    .shaft-cross { stroke: #7f8c8d; stroke-width: 1; }')
    svg_lines.append('    ')
    svg_lines.append('    /* Ventilation styles */')
    svg_lines.append('    .vent-duct { stroke: #3498db; stroke-width: 5; fill: none; stroke-linecap: round; stroke-linejoin: round; stroke-opacity: 0.85; }')
    svg_lines.append('    .vent-terminal { fill: #e74c3c; stroke: #ffffff; stroke-width: 2; }')
    svg_lines.append('  </style>')
    
    # Grid lines
    for x in range(int(minx), int(maxx) + 1):
        x_pos = offset_x + x * scale
        svg_lines.append(f'  <line x1="{x_pos}" y1="{offset_y}" x2="{x_pos}" y2="{offset_y + (maxy-miny)*scale}" class="grid"/>')
    for y in range(int(miny), int(maxy) + 1):
        y_pos = offset_y + y * scale
        svg_lines.append(f'  <line x1="{offset_x}" y1="{y_pos}" x2="{offset_x + (maxx-minx)*scale}" y2="{y_pos}" class="grid"/>')

    # Draw Wall Base
    for room in rooms:
        coords = room.polygon.exterior.coords
        svg_points = " ".join([f"{offset_x + x * scale},{offset_y + (maxy - y) * scale}" for x, y in coords])
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
        svg_points = " ".join([f"{offset_x + x * scale},{offset_y + (maxy - y) * scale}" for x, y in coords])
        
        # Color based on whether it has false ceiling covers
        if room.is_hallway:
            cls = "hallway-covered" if room.has_cover else "hallway"
        else:
            cls = "room-covered" if room.has_cover else "room"
            
        svg_lines.append(f'  <polygon points="{svg_points}" class="{cls}"/>')
        
        # Room Label & Area
        centroid = inner_poly.centroid
        cx = offset_x + centroid.x * scale
        cy = offset_y + (maxy - centroid.y) * scale
        svg_lines.append(f'  <text x="{cx}" y="{cy - 5}" class="label">{room.name}</text>')
        svg_lines.append(f'  <text x="{cx}" y="{cy + 10}" class="area">{room.polygon.area:.1f} m²</text>')
        if room.has_cover:
            svg_lines.append(f'  <text x="{cx}" y="{cy + 22}" font-family="monospace" font-size="8" fill="#16a085" text-anchor="middle">[FALSE CEILING]</text>')
        
    # Cut Door Openings & Draw Swings
    for door in doors:
        d1 = door["d1"]
        d2 = door["d2"]
        px, py = door["swing_dir"]
        
        leaf_end_x = d1[0] + door["width"] * px
        leaf_end_y = d1[1] + door["width"] * py
        
        svg_d1_x = offset_x + d1[0] * scale
        svg_d1_y = offset_y + (maxy - d1[1]) * scale
        svg_d2_x = offset_x + d2[0] * scale
        svg_d2_y = offset_y + (maxy - d2[1]) * scale
        svg_leaf_x = offset_x + leaf_end_x * scale
        svg_leaf_y = offset_y + (maxy - leaf_end_y) * scale
        
        svg_lines.append(f'  <!-- {"Entrance" if door.get("is_entrance", False) else "Door"} Opening -->')
        svg_lines.append(f'  <line x1="{svg_d1_x}" y1="{svg_d1_y}" x2="{svg_d2_x}" y2="{svg_d2_y}" stroke="#ffffff" stroke-width="4"/>')
        
        # Draw covered floor background back onto the threshold mask so it doesn't leave a white patch where there is a ceiling cover
        # We can draw the line with the corresponding color
        # (For simplicity, leaving it white mimics standard architectural cut layouts)
        
        svg_lines.append(f'  <line x1="{svg_d1_x}" y1="{svg_d1_y}" x2="{svg_leaf_x}" y2="{svg_leaf_y}" class="door-leaf"/>')
        
        sweep = 0 if (px * (d2[1]-d1[1]) - py * (d2[0]-d1[0])) > 0 else 1
        r = door["width"] * scale
        svg_lines.append(f'  <path d="M {svg_leaf_x} {svg_leaf_y} A {r} {r} 0 0 {sweep} {svg_d2_x} {svg_d2_y}" class="door-swing"/>')
        
        if door.get("is_entrance", False):
            ox, oy = -px, -py
            lx = d1[0] + 0.5 * (d2[0] - d1[0]) + ox * 0.8
            ly = d1[1] + 0.5 * (d2[1] - d1[1]) + oy * 0.8
            svg_lx = offset_x + lx * scale
            svg_ly = offset_y + (maxy - ly) * scale
            svg_lines.append(f'  <text x="{svg_lx}" y="{svg_ly}" font-family="monospace" font-size="10" font-weight="bold" fill="#e74c3c" text-anchor="middle">ENTRANCE</text>')

    # Draw MEP Shafts
    for s in shafts:
        minx_s, miny_s, maxx_s, maxy_s = s.bounds
        sx = offset_x + minx_s * scale
        sy = offset_y + (maxy - maxy_s) * scale
        sw = (maxx_s - minx_s) * scale
        sh = (maxy_s - miny_s) * scale
        
        svg_lines.append(f'  <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" class="shaft"/>')
        svg_lines.append(f'  <line x1="{sx}" y1="{sy}" x2="{sx + sw}" y2="{sy + sh}" class="shaft-cross"/>')
        svg_lines.append(f'  <line x1="{sx + sw}" y1="{sy}" x2="{sx}" y2="{sy + sh}" class="shaft-cross"/>')
        svg_lines.append(f'  <text x="{sx + sw/2}" y="{sy + sh/2 + 3}" font-family="monospace" font-size="8" fill="#7f8c8d" text-anchor="middle">MEP</text>')

    # Draw Structural Columns
    for col in columns:
        minx_c, miny_c, maxx_c, maxy_c = col.bounds
        cx = offset_x + minx_c * scale
        cy = offset_y + (maxy - maxy_c) * scale
        cw = (maxx_c - minx_c) * scale
        ch = (maxy_c - miny_c) * scale
        svg_lines.append(f'  <rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" class="column"/>')

    # Draw Ventilation Ducts (Steiner Tree route)
    svg_lines.append('  <!-- Ventilation Duct Layout -->')
    for p1, p2 in segments:
        svg_x1 = offset_x + p1[0] * scale
        svg_y1 = offset_y + (maxy - p1[1]) * scale
        svg_x2 = offset_x + p2[0] * scale
        svg_y2 = offset_y + (maxy - p2[1]) * scale
        svg_lines.append(f'  <line x1="{svg_x1}" y1="{svg_y1}" x2="{svg_x2}" y2="{svg_y2}" class="vent_duct" stroke="#3498db" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.85"/>')

    # Draw Terminals (dots at the wet room centroids)
    svg_lines.append('  <!-- Wet Room Terminals -->')
    for t in terminals:
        svg_tx = offset_x + t[0] * scale
        svg_ty = offset_y + (maxy - t[1]) * scale
        svg_lines.append(f'  <circle cx="{svg_tx}" cy="{svg_ty}" r="6" class="vent-terminal"/>')

    # Draw Scale Bar
    scale_y = svg_h - 40
    svg_lines.append(f'  <line x1="{offset_x}" y1="{scale_y}" x2="{offset_x + 5 * scale}" y2="{scale_y}" class="scale-bar"/>')
    svg_lines.append(f'  <text x="{offset_x + 2.5 * scale}" y="{scale_y - 10}" font-family="monospace" font-size="12" fill="#2c3e50" text-anchor="middle">5 Meters</text>')
    
    svg_lines.append('</svg>')
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"MEP Ventilation layout successfully saved to {filename}")

if __name__ == "__main__":
    main()
