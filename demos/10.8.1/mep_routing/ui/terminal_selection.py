"""State transitions for preferred terminal selection tools."""

from __future__ import annotations

import math

import numpy as np


def map_preferred_points_to_nodes(preferences, candidate_nodes, nodes, tolerance_mm):
    """Map saved preferred points to unique nearby candidate node indices."""
    if not preferences or not candidate_nodes:
        return [], {}
    mapped_nodes = []
    mapped_preference_indices = {}
    candidate_array = np.asarray(nodes)[list(candidate_nodes)]
    for preference_index, preference_point in enumerate(preferences):
        deltas = candidate_array - np.asarray(preference_point, dtype=np.float32)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        nearest_position = int(np.argmin(distances))
        if float(distances[nearest_position]) > tolerance_mm:
            continue
        node_index = int(candidate_nodes[nearest_position])
        if node_index not in mapped_preference_indices:
            mapped_nodes.append(node_index)
            mapped_preference_indices[node_index] = preference_index
    return mapped_nodes, mapped_preference_indices


def find_room_candidate_node(world_point, route_names, room_contains_point, candidate_nodes_for_room, nodes):
    """Return the closest candidate node in the terminal room containing a point."""
    best = None
    best_distance = float("inf")
    for room_name in route_names:
        if not room_contains_point(room_name, world_point):
            continue
        for node_index in candidate_nodes_for_room(room_name):
            point = nodes[int(node_index)]
            distance = math.hypot(float(point[0]) - world_point[0], float(point[1]) - world_point[1])
            if distance < best_distance:
                best_distance = distance
                best = room_name, int(node_index)
    return best


def apply_preferred_terminal_point(preferences_by_room, world_point, remove, route_names, room_contains_point, candidate_nodes_for_room, nodes, tolerance_mm):
    """Add or remove the nearest valid terminal node preference in place."""
    hit = find_room_candidate_node(world_point, route_names, room_contains_point, candidate_nodes_for_room, nodes)
    if hit is None:
        return False, None
    room_name, node_index = hit
    candidate_nodes = candidate_nodes_for_room(room_name)
    _mapped_nodes, mapped_preference_indices = map_preferred_points_to_nodes(
        preferences_by_room.get(room_name, []), candidate_nodes, nodes, tolerance_mm
    )
    preferences = list(preferences_by_room.get(room_name, []))
    if remove:
        preference_index = mapped_preference_indices.get(node_index)
        if preference_index is None:
            return False, room_name
        del preferences[preference_index]
        if preferences:
            preferences_by_room[room_name] = preferences
        else:
            preferences_by_room.pop(room_name, None)
        return True, room_name
    if node_index in mapped_preference_indices:
        return False, room_name
    point = nodes[node_index]
    preferences.append((float(point[0]), float(point[1])))
    preferences_by_room[room_name] = preferences
    return True, room_name


def apply_preferred_terminal_area(preferences_by_room, preferred_areas, start_world, end_world, remove, route_names, candidate_nodes_for_room, nodes, tolerance_mm):
    """Add or remove preferences within a rectangular world-space selection."""
    if start_world is None or end_world is None:
        return False, None
    minx, maxx = sorted((float(start_world[0]), float(end_world[0])))
    miny, maxy = sorted((float(start_world[1]), float(end_world[1])))
    if maxx - minx < 1.0 or maxy - miny < 1.0:
        return False, None
    changed = False
    last_room = None
    for room_name in route_names:
        candidate_nodes = candidate_nodes_for_room(room_name)
        if not candidate_nodes:
            continue
        preferences = list(preferences_by_room.get(room_name, []))
        if remove:
            kept = [point for point in preferences if not (minx <= point[0] <= maxx and miny <= point[1] <= maxy)]
            if len(kept) == len(preferences):
                continue
            changed, last_room = True, room_name
            if kept:
                preferences_by_room[room_name] = kept
            else:
                preferences_by_room.pop(room_name, None)
            preferred_areas[:] = [
                area for area in preferred_areas if not (
                    area["room"] == room_name and not (
                        area["bounds"][2] < minx or area["bounds"][0] > maxx
                        or area["bounds"][3] < miny or area["bounds"][1] > maxy
                    )
                )
            ]
            continue
        _mapped_nodes, mapped_preference_indices = map_preferred_points_to_nodes(
            preferences, candidate_nodes, nodes, tolerance_mm
        )
        existing_nodes = set(mapped_preference_indices)
        added_for_room = False
        for node_index in candidate_nodes:
            node_index = int(node_index)
            if node_index in existing_nodes:
                continue
            point = nodes[node_index]
            if minx <= float(point[0]) <= maxx and miny <= float(point[1]) <= maxy:
                preferences.append((float(point[0]), float(point[1])))
                existing_nodes.add(node_index)
                changed, added_for_room, last_room = True, True, room_name
        if preferences:
            preferences_by_room[room_name] = preferences
        if added_for_room:
            preferred_areas.append({"room": room_name, "bounds": (minx, miny, maxx, maxy)})
    return changed, last_room


def draw_preferred_terminal_areas(screen, preferred_areas, selected_route_name, route_colors, candidate_nodes_for_room, nodes, world_to_screen, marker_side):
    """Draw selected terminal areas and their available routing nodes."""
    import pygame

    for area in preferred_areas:
        room_name = area["room"]
        if selected_route_name and room_name != selected_route_name:
            color, node_color = (86, 90, 94), (70, 74, 78)
        else:
            color = route_colors.get(room_name, (155, 89, 182))
            node_color = color
        minx, miny, maxx, maxy = area["bounds"]
        x1, y1 = world_to_screen(minx, miny)
        x2, y2 = world_to_screen(maxx, maxy)
        rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        if rect.width > 1 and rect.height > 1:
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            overlay.fill((color[0], color[1], color[2], 45))
            screen.blit(overlay, rect.topleft)
            dash = 8
            for dx in range(0, rect.width, dash * 2):
                pygame.draw.line(screen, color, (rect.left + dx, rect.top), (min(rect.left + dx + dash, rect.right), rect.top), 2)
                pygame.draw.line(screen, color, (rect.left + dx, rect.bottom), (min(rect.left + dx + dash, rect.right), rect.bottom), 2)
            for dy in range(0, rect.height, dash * 2):
                pygame.draw.line(screen, color, (rect.left, rect.top + dy), (rect.left, min(rect.top + dy + dash, rect.bottom)), 2)
                pygame.draw.line(screen, color, (rect.right, rect.top + dy), (rect.right, min(rect.top + dy + dash, rect.bottom)), 2)
        for node_index in candidate_nodes_for_room(room_name):
            point = nodes[int(node_index)]
            if minx <= float(point[0]) <= maxx and miny <= float(point[1]) <= maxy:
                node_rect = pygame.Rect(0, 0, marker_side, marker_side)
                node_rect.center = world_to_screen(float(point[0]), float(point[1]))
                pygame.draw.rect(screen, node_color, node_rect, 1)


def draw_preferred_terminal_markers(screen, route_names, preferences_by_room, selected_route_name, routes, route_colors, text_color, candidate_nodes_for_room, nodes, world_to_screen, marker_side, tolerance_mm):
    """Draw preferred candidate nodes, filled when they became routed starts."""
    import pygame

    routed_start_by_room = {
        route_name: segments[0][0]
        for route_name, segments in (routes or [])
        if segments
    }
    for room_name in route_names:
        candidate_nodes = candidate_nodes_for_room(room_name)
        _mapped_nodes, mapped_preference_indices = map_preferred_points_to_nodes(
            preferences_by_room.get(room_name, []), candidate_nodes, nodes, tolerance_mm
        )
        route_color = route_colors.get(room_name, text_color)
        muted = bool(selected_route_name and room_name != selected_route_name)
        for node_index in candidate_nodes:
            node_index = int(node_index)
            if node_index not in mapped_preference_indices:
                continue
            point = nodes[node_index]
            rect = pygame.Rect(0, 0, marker_side, marker_side)
            rect.center = world_to_screen(float(point[0]), float(point[1]))
            routed_start = routed_start_by_room.get(room_name)
            is_routed = routed_start is not None and abs(float(point[0]) - routed_start[0]) < 1e-6 and abs(float(point[1]) - routed_start[1]) < 1e-6
            if muted:
                fill, border = ((44, 48, 52) if is_routed else None), (86, 90, 94)
            else:
                fill, border = (route_color if is_routed else None), (255, 255, 255)
            if fill is not None:
                pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, border, rect, 2)


def draw_routed_terminal_endpoint_markers(screen, routes, terminal_route_names, selected_route_name, route_colors, text_color, world_to_screen, marker_side):
    """Draw the terminal endpoint of each completed terminal route."""
    import pygame

    for route_name, segments in routes or []:
        if route_name not in terminal_route_names or not segments:
            continue
        start = segments[0][0]
        rect = pygame.Rect(0, 0, marker_side, marker_side)
        rect.center = world_to_screen(float(start[0]), float(start[1]))
        if selected_route_name and route_name != selected_route_name:
            fill, border = (44, 48, 52), (86, 90, 94)
        else:
            fill, border = route_colors.get(route_name, text_color), (255, 255, 255)
        pygame.draw.rect(screen, fill, rect)
        pygame.draw.rect(screen, border, rect, 2)
