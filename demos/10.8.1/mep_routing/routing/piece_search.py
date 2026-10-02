"""Hard minimum-piece A* with saturated straight-run labels.

The two representations share physical-piece transitions and dominance rules.
State A* groups arrivals by node/direction; line-graph A* also retains the
incoming directed edge. No route geometry or Shapely work occurs in expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

from .clearance import weighted_edge_cost
from .metrics import segment_metric_dir
from .segments import add_port_stub_segment


@dataclass(frozen=True)
class PieceLengthPolicy:
    endpoint_min_mm: float
    internal_min_mm: float
    route_name: str = ""
    cap_mm: float = field(init=False)

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (
            self.endpoint_min_mm, self.internal_min_mm,
        )):
            raise ValueError("piece minima must be finite and nonnegative")
        object.__setattr__(self, "cap_mm", max(self.endpoint_min_mm, self.internal_min_mm))

    def advance(self, run, direction, length):
        """Close the preceding piece only on a physical direction change."""
        if length < 1.0:  # Match the existing physical-piece metric.
            return run
        incoming, achieved, first = run
        if achieved == 0:
            return direction, min(length, self.cap_mm), first
        if direction is not None and direction == incoming:
            return direction, min(achieved + length, self.cap_mm), first
        minimum = self.endpoint_min_mm if first else self.internal_min_mm
        if achieved + 1e-7 < minimum:
            return None
        return direction, min(length, self.cap_mm), False

    def finish(self, run):
        return run[1] == 0 or run[1] + 1e-7 >= self.endpoint_min_mm


@dataclass
class PieceSearchStats:
    expanded: int = 0
    generated: int = 0
    dominated: int = 0
    rejected_turns: int = 0
    rejected_goals: int = 0
    peak_queue: int = 0
    max_labels_per_state: int = 0


@dataclass(frozen=True)
class PieceSearchGraph:
    """Physical edge geometry reused by all route searches in one scene solve."""

    points: tuple
    adjacency: dict

    @classmethod
    def from_env(cls, env):
        points = tuple((float(point[0]), float(point[1])) for point in env.nodes)
        adjacency = {}
        for node, neighbours in env.adj.items():
            prepared = []
            for neighbour, distance, _direction in neighbours:
                start, end = points[node], points[int(neighbour)]
                dx, dy = end[0] - start[0], end[1] - start[1]
                direction = ("E" if dx > 0 else "W") if abs(dy) < 1e-7 else (
                    ("N" if dy > 0 else "S") if abs(dx) < 1e-7 else None
                )
                prepared.append((int(neighbour), float(distance), direction, math.hypot(dx, dy)))
            adjacency[int(node)] = tuple(prepared)
        return cls(points, adjacency)


@dataclass(slots=True)
class _Label:
    node: int
    previous: int
    run: tuple
    cost: float
    parent: int | None
    active: bool = True


def _steps(segments, route_name, start_index=0):
    return tuple(
        (segment_metric_dir(route_name, start_index + index, start, end),
         math.hypot(end[0] - start[0], end[1] - start[1]))
        for index, (start, end) in enumerate(segments)
    )


def _append_steps(policy, run, steps):
    for direction, length in steps:
        run = policy.advance(run, direction, length)
        if run is None:
            return None
    return run


def run_piece_length_astar(
    env, start_nodes, target_pin_names, pin_node_map, bend_cost, *,
    piece_policy, edge_weights=None, line_graph=False, source_segments=None,
    heuristic_mode=0, machine_center=(0.0, 0.0), estimate_turns_fn=None,
    stats=None,
    geometry=None,
):
    """Return the legacy path tuple, enforcing materialized centerline minima.

    Source adapters are supplied per start node. Target adapters are prepared
    once from the same port-stub materializer as the final route. All history
    is represented by a capped run and parent-label references. Dominance is
    valid for minimum-only rules; no maximum length or path-history exclusions
    are assumed.
    """
    from .search import _start_node_list, path_physical_length

    if piece_policy is None:
        raise ValueError("length-aware search requires a PieceLengthPolicy")
    stats = stats if stats is not None else PieceSearchStats()
    policy = piece_policy
    estimate_turns_fn = estimate_turns_fn or (lambda *_args: 0)
    targets = [item for pin in target_pin_names for item in pin_node_map.get(pin, ())]
    if not targets:
        return None, 0.0, None, None
    geometry = geometry if geometry is not None else PieceSearchGraph.from_env(env)
    points = geometry.points
    target_points = tuple(points[int(target["node_idx"])] for target in targets)
    cx, cy = machine_center
    radius = max(abs(x - cx) + abs(y - cy) for x, y in target_points)
    heuristic_cache = {}

    def heuristic(node, direction):
        if heuristic_mode == 3:
            return 0.0
        key = node, direction
        if key not in heuristic_cache:
            point = points[node]
            if heuristic_mode == 2:
                value = max(0.0, abs(point[0] - cx) + abs(point[1] - cy) - radius)
            else:
                estimate_bends = heuristic_mode == 0 and (direction is None or isinstance(direction, str))
                value = min(
                    abs(x - point[0]) + abs(y - point[1])
                    + (bend_cost * estimate_turns_fn(point, direction, (x, y)) if estimate_bends else 0.0)
                    for x, y in target_points
                )
            heuristic_cache[key] = value
        return heuristic_cache[key]
    by_node = {}
    for target in targets:
        segments = []
        # Standalone graph tests may have no physical port adapter.
        if "pin_point" in target and "access_point" in target:
            add_port_stub_segment(
                segments, target["pin"], target["node_idx"],
                {target["pin"]: target["pin_point"]}, env.nodes, target,
            )
        by_node.setdefault(int(target["node_idx"]), []).append(
            (target, _steps(segments, policy.route_name, start_index=1)),
        )

    labels, frontiers, queue = [], {}, []

    def push(node, previous, run, cost, parent):
        direction, length, first = run
        base = (previous if line_graph or not isinstance(direction, str) else -1, node, direction, first)
        frontier = frontiers.get(base, ())
        # More achieved length and no greater cost permits every continuation
        # of the weaker label, including endpoint acceptance.
        for index in frontier:
            old = labels[index]
            if old.cost <= cost and old.run[1] >= length:
                stats.dominated += 1
                return
        retained = []
        for index in frontier:
            old = labels[index]
            if cost <= old.cost and length >= old.run[1]:
                old.active = False
                stats.dominated += 1
            else:
                retained.append(index)
        index = len(labels)
        labels.append(_Label(node, previous, run, cost, parent))
        retained.append(index)
        frontiers[base] = retained
        # The relaxed heuristic does not depend on length or endpoint phase.
        heapq.heappush(queue, (cost + heuristic(node, direction), cost, index))
        stats.generated += 1
        stats.peak_queue = max(stats.peak_queue, len(queue))
        stats.max_labels_per_state = max(stats.max_labels_per_state, len(retained))

    for node in _start_node_list(start_nodes):
        node = int(node)
        prefix = (source_segments or {}).get(node, ())
        run = _append_steps(policy, (None, 0.0, True), _steps(prefix, policy.route_name))
        if run is not None:
            push(node, -1, run, 0.0, None)

    best_cost, best_index, best_target = float("inf"), None, None
    opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}
    while queue:
        priority, cost, index = heapq.heappop(queue)
        if priority >= best_cost:
            break
        label = labels[index]
        if not label.active:
            continue
        stats.expanded += 1
        node, incoming = label.node, label.run[0]
        for target, suffix in by_node.get(node, ()):
            finished = _append_steps(policy, label.run, suffix)
            if finished is None or not policy.finish(finished):
                stats.rejected_goals += 1
                continue
            goal_cost = cost + (bend_cost if incoming is not None and incoming != target["in_dir"] else 0.0)
            if goal_cost < best_cost:
                best_cost, best_index, best_target = goal_cost, index, target
        for neighbour, distance, direction, physical_length in geometry.adjacency.get(node, ()):
            if neighbour == label.previous or (incoming in opposite and opposite[incoming] == direction):
                continue  # Do not accumulate length by immediately doubling back.
            if direction is None:
                start, end = points[node], points[neighbour]
                direction = segment_metric_dir(policy.route_name, 0 if label.run[1] == 0 else 1, start, end)
            run = policy.advance(label.run, direction, physical_length)
            if run is None:
                stats.rejected_turns += 1
                continue
            edge_cost = weighted_edge_cost(edge_weights, node, neighbour, distance)
            turn_cost = bend_cost if incoming is not None and incoming != direction else 0.0
            push(neighbour, node, run, cost + edge_cost + turn_cost, index)

    if best_index is None:
        return None, 0.0, None, None
    path = []
    while best_index is not None:
        label = labels[best_index]
        path.append(label.node)
        best_index = label.parent
    path.reverse()
    return path, path_physical_length(env, path), best_target["pin"], best_target
