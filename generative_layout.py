import random
import math
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import split, unary_union

class Room:
    def __init__(self, polygon, name):
        self.polygon = polygon
        self.name = name
        self.is_hallway = name == "Hallway"

def get_footprint(width=14.0, height=10.0):
    # Randomly choose between a rectangle and an L-shape footprint
    if random.choice([True, False]):
        # L-Shape footprint
        notch_w = random.uniform(3.0, 5.0)
        notch_h = random.uniform(3.0, 4.0)
        p = Polygon([
            (0, 0),
            (width, 0),
            (width, height - notch_h),
            (width - notch_w, height - notch_h),
            (width - notch_w, height),
            (0, height)
        ])
        return p
    else:
        # Standard rectangle
        return box(0, 0, width, height)

def split_polygon(poly, min_size=2.5, horizontal=False):
    # Get bounding box of the polygon
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    
    if width < min_size * 2 and height < min_size * 2:
        return [poly], []
        
    if horizontal:
        if height < min_size * 2:
            return [poly], []
        split_val = random.uniform(miny + min_size, maxy - min_size)
        split_line = LineString([(minx - 1, split_val), (maxx + 1, split_val)])
    else:
        if width < min_size * 2:
            return [poly], []
        split_val = random.uniform(minx + min_size, maxx - min_size)
        split_line = LineString([(split_val, miny - 1), (split_val, maxy + 1)])
        
    geoms = split(poly, split_line)
    rooms = list(geoms.geoms)
    return rooms, []

def get_balanced_room_names(n):
    if n <= 1:
        return ["Living Room"]
    if n == 2:
        return ["Bathroom", "Living Room"]
    if n == 3:
        return ["Bathroom", "Bedroom", "Living Room"]
        
    beds = ["Master Bedroom"]
    baths = ["Bathroom"]
    washrooms = []
    
    remaining = n - 4
    while remaining > 0:
        if len(beds) > len(baths):
            if "Toilet" not in baths:
                baths.append("Toilet")
            else:
                baths.append(f"Bathroom {len(baths)}")
        else:
            if len(washrooms) == 0:
                washrooms.append("Washroom")
            else:
                beds.append(f"Bedroom {len(beds)}")
        remaining -= 1
        
    all_names = []
    if "Toilet" in baths:
        all_names.append("Toilet")
    if washrooms:
        all_names.append("Washroom")
        
    other_baths = [b for b in baths if b != "Toilet"]
    other_baths.sort()
    all_names.extend(other_baths)
    
    all_names.append("Kitchen")
    
    other_beds = [b for b in beds if b != "Master Bedroom"]
    other_beds.sort()
    all_names.extend(other_beds)
    
    all_names.append("Master Bedroom")
    all_names.append("Living Room")
    
    return all_names

def generate_layout(width=15.0, height=11.0, num_rooms=8, min_room_size=2.4):
    footprint = get_footprint(width, height)
    minx, miny, maxx, maxy = footprint.bounds
    
    # 1. Define a central hallway spine (horizontal)
    hallway_h = 1.1
    # Place slightly offset from center for asymmetry (e.g. larger rooms on one side)
    hallway_y = miny + (maxy - miny) * 0.45
    
    # We create the hallway strip across the footprint
    hallway_box = box(minx - 1, hallway_y - hallway_h/2, maxx + 1, hallway_y + hallway_h/2)
    hallway_poly = hallway_box.intersection(footprint)
    
    # 2. Subtract hallway to get the separate room blocks (Top and Bottom zones)
    room_blocks_geom = footprint.difference(hallway_poly)
    
    room_blocks = []
    if isinstance(room_blocks_geom, Polygon):
        room_blocks.append(room_blocks_geom)
    elif hasattr(room_blocks_geom, 'geoms'):
        room_blocks.extend(room_blocks_geom.geoms)
        
    # 3. Subdivide room blocks vertically (perpendicular to the hallway)
    # This guarantees every room touches the central hallway spine!
    rooms_polys = []
    rooms_to_generate = num_rooms - 1  # 1 is the hallway itself
    
    total_area = sum(b.area for b in room_blocks)
    block_room_counts = []
    for b in room_blocks:
        count = max(1, round(rooms_to_generate * (b.area / total_area)))
        block_room_counts.append(count)
        
    # Adjust count to match exactly
    while sum(block_room_counts) < rooms_to_generate:
        block_room_counts[0] += 1
    while sum(block_room_counts) > rooms_to_generate:
        block_room_counts[0] -= 1
        
    for idx, block in enumerate(room_blocks):
        target_count = block_room_counts[idx]
        block_polys = [block]
        
        attempts = 0
        while len(block_polys) < target_count and attempts < 30:
            attempts += 1
            block_polys.sort(key=lambda p: p.area, reverse=True)
            largest = block_polys[0]
            
            # Split vertically (perpendicular to the hallway)
            split_result, _ = split_polygon(largest, min_size=min_room_size, horizontal=False)
            if len(split_result) == 1:
                continue
                
            block_polys.pop(0)
            block_polys.extend(split_result)
            
        rooms_polys.extend(block_polys)
        
    # Assign room names dynamically
    named_rooms = []
    named_rooms.append(Room(hallway_poly, "Hallway"))
    
    remaining_rooms = sorted(rooms_polys, key=lambda p: p.area)
    names = get_balanced_room_names(len(remaining_rooms))
    
    for i, poly in enumerate(remaining_rooms):
        if i < len(names):
            name = names[i]
        else:
            name = f"Room {i+1}"
        named_rooms.append(Room(poly, name))
        
    return named_rooms

def generate_structural_grid(footprint, spacing=4.5, col_size=0.35):
    minx, miny, maxx, maxy = footprint.bounds
    columns = []
    
    xs = []
    x = minx
    while x <= maxx:
        xs.append(x)
        x += spacing
    if abs(xs[-1] - maxx) > 1.0:
        xs.append(maxx)
        
    ys = []
    y = miny
    while y <= maxy:
        ys.append(y)
        y += spacing
    if abs(ys[-1] - maxy) > 1.0:
        ys.append(maxy)
        
    for cx in xs:
        for cy in ys:
            pt = Point(cx, cy)
            if footprint.exterior.distance(pt) < 0.1 or any(line.distance(pt) < 0.1 for line in [footprint.boundary]):
                col_box = box(cx - col_size/2, cy - col_size/2, cx + col_size/2, cy + col_size/2)
                if footprint.intersects(col_box):
                    columns.append(col_box)
    return columns

def generate_mep_shafts(rooms, shaft_size=0.7):
    shafts = []
    wet_rooms = [r for r in rooms if any(w in r.name for w in ["Kitchen", "Bathroom", "Toilet", "Washroom"])]
    
    for wr in wet_rooms:
        bounds = wr.polygon.bounds
        corners = [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3])
        ]
        
        for cx, cy in corners:
            pt = Point(cx, cy)
            touches_other = False
            for other in rooms:
                if other != wr and other.polygon.distance(pt) < 0.05:
                    touches_other = True
                    break
            
            if touches_other:
                shaft_box = box(cx - shaft_size/2, cy - shaft_size/2, cx + shaft_size/2, cy + shaft_size/2)
                if not any(shaft_box.intersects(s) for s in shafts):
                    shafts.append(shaft_box)
                    if len(shafts) >= 3:
                        return shafts
    return shafts

def can_have_door(r1, r2, has_hallway):
    if r1.is_hallway or r2.is_hallway:
        return True
        
    if r1.name == "Living Room" or r2.name == "Living Room":
        is_wet = lambda r: any(w in r.name for w in ["Bathroom", "Toilet"])
        if is_wet(r1) or is_wet(r2):
            return not has_hallway
        return True
        
    is_bath = lambda r: "Bathroom" in r.name
    is_bed = lambda r: "Bedroom" in r.name
    if (is_bath(r1) and is_bed(r2)) or (is_bath(r2) and is_bed(r1)):
        return True
        
    return False

def find_door_openings(rooms, door_width=0.85):
    doors = []
    has_hallway = any(r.is_hallway for r in rooms)
    connected = set()
    
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1 = rooms[i]
            r2 = rooms[j]
            
            if not can_have_door(r1, r2, has_hallway):
                continue
                
            shared_boundary = r1.polygon.intersection(r2.polygon)
            
            if isinstance(shared_boundary, (LineString, Polygon)) or hasattr(shared_boundary, 'geoms'):
                lines = []
                if isinstance(shared_boundary, LineString):
                    lines.append(shared_boundary)
                elif hasattr(shared_boundary, 'geoms'):
                    for g in shared_boundary.geoms:
                        if isinstance(g, LineString):
                            lines.append(g)
                            
                for line in lines:
                    if line.length > door_width + 0.6:
                        pair = tuple(sorted([r1.name, r2.name]))
                        if pair in connected:
                            continue
                            
                        coords = list(line.coords)
                        p1 = coords[0]
                        p2 = coords[-1]
                        
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dist = math.hypot(dx, dy)
                        
                        ux = dx / dist
                        uy = dy / dist
                        
                        mx = p1[0] + 0.5 * dx
                        my = p1[1] + 0.5 * dy
                        
                        d1_x = mx - (door_width / 2) * ux
                        d1_y = my - (door_width / 2) * uy
                        d2_x = mx + (door_width / 2) * ux
                        d2_y = my + (door_width / 2) * uy
                        
                        swing_into_r1 = not r1.is_hallway and r1.name != "Living Room"
                        
                        px = -uy
                        py = ux
                        
                        doors.append({
                            "line": LineString([(d1_x, d1_y), (d2_x, d2_y)]),
                            "d1": (d1_x, d1_y),
                            "d2": (d2_x, d2_y),
                            "swing_dir": (px, py) if swing_into_r1 else (-px, -py),
                            "width": door_width
                        })
                        connected.add(pair)
                        break
    return doors

def find_entrance_door(rooms, footprint, door_width=0.95):
    entrance_rooms = [r for r in rooms if r.name == "Hallway"]
    if not entrance_rooms:
        entrance_rooms = [r for r in rooms if r.name == "Living Room"]
        
    if not entrance_rooms:
        return None
        
    entrance_room = entrance_rooms[0]
    ext_wall = entrance_room.polygon.intersection(footprint.boundary)
    
    lines = []
    if isinstance(ext_wall, LineString):
        lines.append(ext_wall)
    elif hasattr(ext_wall, 'geoms'):
        for g in ext_wall.geoms:
            if isinstance(g, LineString):
                lines.append(g)
                
    best_line = None
    best_score = -1
    for line in lines:
        if line.length > door_width + 0.6:
            _, miny, _, _ = line.bounds
            score = line.length - miny * 0.5
            if score > best_score:
                best_score = score
                best_line = line
                
    if best_line is not None:
        coords = list(best_line.coords)
        p1 = coords[0]
        p2 = coords[-1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        
        ux = dx / dist
        uy = dy / dist
        
        mx = p1[0] + 0.5 * dx
        my = p1[1] + 0.5 * dy
        
        d1_x = mx - (door_width / 2) * ux
        d1_y = my - (door_width / 2) * uy
        d2_x = mx + (door_width / 2) * ux
        d2_y = my + (door_width / 2) * uy
        
        centroid = entrance_room.polygon.centroid
        cx = centroid.x - mx
        cy = centroid.y - my
        
        px1, py1 = -uy, ux
        px2, py2 = uy, -ux
        
        dot1 = px1 * cx + py1 * cy
        dot2 = px2 * cx + py2 * cy
        
        px, py = (px1, py1) if dot1 > dot2 else (px2, py2)
        
        return {
            "d1": (d1_x, d1_y),
            "d2": (d2_x, d2_y),
            "swing_dir": (px, py),
            "width": door_width,
            "is_entrance": True
        }
    return None

def save_to_svg(rooms, columns, shafts, doors, filename="floorplan.svg", wall_thickness=0.15):
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
    svg_lines.append('    .hallway { fill: #f8f9f9; stroke: none; }')
    svg_lines.append('    .column { fill: #2c3e50; stroke: #1a252f; stroke-width: 1; }')
    svg_lines.append('    .shaft { fill: #eaeded; stroke: #7f8c8d; stroke-dasharray: 2; stroke-width: 1; }')
    svg_lines.append('    .door-swing { stroke: #e74c3c; stroke-width: 1; fill: none; stroke-dasharray: 2; }')
    svg_lines.append('    .door-leaf { stroke: #2c3e50; stroke-width: 2.5; fill: none; }')
    svg_lines.append('    .label { font-family: "Courier New", Courier, monospace; font-size: 13px; font-weight: bold; fill: #2c3e50; text-anchor: middle; }')
    svg_lines.append('    .area { font-family: "Courier New", Courier, monospace; font-size: 10px; fill: #7f8c8d; text-anchor: middle; }')
    svg_lines.append('    .grid { stroke: #eaeded; stroke-width: 0.5; }')
    svg_lines.append('    .scale-bar { stroke: #2c3e50; stroke-width: 3; }')
    svg_lines.append('    .shaft-cross { stroke: #7f8c8d; stroke-width: 1; }')
    svg_lines.append('  </style>')
    
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
        cls = "hallway" if room.is_hallway else "room"
        svg_lines.append(f'  <polygon points="{svg_points}" class="{cls}"/>')
        
        # Room Label & Area
        centroid = inner_poly.centroid
        cx = offset_x + centroid.x * scale
        cy = offset_y + (maxy - centroid.y) * scale
        svg_lines.append(f'  <text x="{cx}" y="{cy - 5}" class="label">{room.name}</text>')
        svg_lines.append(f'  <text x="{cx}" y="{cy + 10}" class="area">{room.polygon.area:.1f} m²</text>')
        
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
        
    # Draw Scale Bar
    scale_y = svg_h - 40
    svg_lines.append(f'  <line x1="{offset_x}" y1="{scale_y}" x2="{offset_x + 5 * scale}" y2="{scale_y}" class="scale-bar"/>')
    svg_lines.append(f'  <text x="{offset_x + 2.5 * scale}" y="{scale_y - 10}" font-family="monospace" font-size="12" fill="#2c3e50" text-anchor="middle">5 Meters</text>')
    
    svg_lines.append('</svg>')
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"Layout successfully saved to {filename}")

if __name__ == "__main__":
    rooms = generate_layout(width=15.0, height=11.0, num_rooms=8)
    
    all_polys = [r.polygon for r in rooms]
    footprint = unary_union(all_polys)
    
    columns = generate_structural_grid(footprint, spacing=4.0)
    shafts = generate_mep_shafts(rooms)
    doors = find_door_openings(rooms)
    
    entrance = find_entrance_door(rooms, footprint)
    if entrance:
        doors.append(entrance)
        
    save_to_svg(rooms, columns, shafts, doors, "floorplan.svg")
