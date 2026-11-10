"""Experimental directed-Steiner DP with bounded straight-run histories.

Weighted graph directions remain cost labels. Physical vectors and distances
are independent feasibility labels, including the machine and grille adapters.
Lengths saturate at the minimum; this bounds history distinctions, not total
memory. Terminal subsets still give exponential growth in the room count.
"""

from dataclasses import dataclass
import heapq
import math

from mep_routing.routing import SteinerScore, SteinerTreeResult

from .fittings import FittingCounts, count_tree_fittings, junction_fitting_count


@dataclass
class LengthSteinerStats:
    state_count: int = 0
    subset_labels: int = 0
    branch_labels: int = 0
    rejected_physical_tree: bool = False
    fittings: FittingCounts | None = None
    fitting_weight: float = 0.0
    primary_cost: float | None = None
    search_cost: float | None = None


@dataclass(frozen=True)
class TreeLengthPolicy:
    minimum_mm: float
    split_at_tees: bool = False

    def __post_init__(self):
        if not math.isfinite(self.minimum_mm) or self.minimum_mm <= 0:
            raise ValueError("minimum length must be finite and positive")

    def advance(self, run, direction, length):
        previous, achieved = run
        if length <= 1e-7:
            return run
        if previous is not None and previous != direction:
            # A reversal cannot manufacture a long straight piece.
            if sum(a * b for a, b in zip(previous, direction)) < -0.999999:
                return None
            if achieved + 1e-7 < self.minimum_mm:
                return None
            achieved = 0.0
        return direction, round(min(self.minimum_mm, achieved + length), 9)

    def complete(self, run):
        return run[1] + 1e-7 >= self.minimum_mm


def segment_geometry(start, end):
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = math.hypot(dx, dy)
    direction = None if length <= 1e-7 else (round(dx / length, 9), round(dy / length, 9))
    return direction, length


def validate_tree_lengths(lines, root_point, terminal_points, policy):
    """Check the physical directed tree, merging collinear size/grid splits.

    Return False for a disconnected union, cycle, multiple physical parents or a
    short run. Tees include terminal takeoffs that leave a continuing trunk.
    """
    root = tuple(map(float, root_point[:2]))
    terminals = {tuple(map(float, point[:2])) for point in terminal_points}
    children, parents = {}, {}
    for start, end in lines:
        start, end = tuple(map(float, start[:2])), tuple(map(float, end[:2]))
        if math.dist(start, end) <= 1e-7:
            continue
        children.setdefault(start, set()).add(end)
        parents.setdefault(end, set()).add(start)
    if root in parents or any(len(value) > 1 for value in parents.values()):
        return False
    visited, reached = set(), set()
    stack = [(root, (None, 0.0))]
    while stack:
        node, run = stack.pop()
        if node in visited:
            return False
        visited.add(node)
        following = children.get(node, ())
        if node in terminals:
            if not policy.complete(run):
                return False
            reached.add(node)
        if policy.split_at_tees and len(following) + (node in terminals) > 1:
            if node != root and not policy.complete(run):
                return False
            run = (None, 0.0)
        for end in following:
            updated = policy.advance(run, *segment_geometry(node, end))
            if updated is None:
                return False
            stack.append((end, updated))
    return reached == terminals and set(children) <= visited and set(parents) <= visited


def rank_length_steiner_roots(prepared, roots, names, bend_penalty, policy, *, fitting_weight=0.0):
    """Score private outlet nodes against the full terminal set in one shared search."""
    return _search_length_steiner(
        prepared, roots, names, bend_penalty, policy, fitting_weight=fitting_weight, rank_only=True,
    )


def solve_length_steiner(prepared, root, names, bend_penalty, policy, *,
                         root_prefix=(), stats=None, fitting_weight=0.0):
    return _search_length_steiner(
        prepared, ((root, root_prefix),), names, bend_penalty, policy,
        stats=stats, fitting_weight=fitting_weight,
    )


def _search_length_steiner(prepared, roots, names, bend_penalty, policy, *,
                           stats=None, fitting_weight=0.0, rank_only=False):
    """Use the baseline's weighted arcs and terminal costs, adding feasibility.

    Tee resets are available exclusively to subset merges, never ordinary
    graph traversal. Reconstruct and independently validate the physical union;
    a rejected union is reported as blocked, with no unconstrained fallback.
    """
    stats = stats if stats is not None else LengthSteinerStats()
    if not math.isfinite(fitting_weight) or fitting_weight < 0:
        raise ValueError("fitting weight must be finite and nonnegative")
    if fitting_weight and not policy.split_at_tees:
        raise ValueError("fitting perturbation requires lengths at bends and tees")
    stats.fitting_weight = fitting_weight
    positions, graph = prepared.nodes, prepared.adjacency
    states, indices = [], {}
    reverse, forward, reset_indices = [], [], {}
    geometry = {
        (int(u), int(edge[0])): segment_geometry(positions[int(u)], positions[int(edge[0])])
        for u, edges in graph.items() for edge in edges
    }
    direction_bits = {None: 1}

    def direction_bit(direction):
        if direction not in direction_bits:
            direction_bits[direction] = 1 << len(direction_bits)
        return direction_bits[direction]

    def intern(state):
        index = indices.get(state)
        if index is None:
            index = len(states)
            indices[state] = index
            states.append(state)
            reverse.append([])
            forward.append([])
        return index

    starts = {}
    for root, prefix in roots:
        run = (None, 0.0)
        for start, end in prefix:
            run = policy.advance(run, *segment_geometry(start, end))
            if run is None:
                break
        if run is not None:
            index = intern((int(root), None, *run))
            starts[index] = (int(root), prefix)
    if not starts or not names:
        return () if rank_only else None

    # Enumerate reachable histories, including the starting states of branches.
    # Adding a reset here enumerates states only; it adds no traversable arc.
    current = 0
    while current < len(states):
        node, incoming, direction, length = states[current]
        if policy.split_at_tees and (policy.complete((direction, length)) or
                                    (current in starts and direction is None)):
            reset_indices[current] = intern((node, incoming, None, 0.0))
        for edge in graph.get(node, ()):
            nxt, cost_direction = int(edge[0]), str(edge[2])
            updated = policy.advance((direction, length), *geometry[(node, nxt)])
            if updated is None:
                continue
            following = intern((nxt, cost_direction, *updated))
            cost = float(edge[1]) + (
                float(bend_penalty) if incoming is not None and incoming != cost_direction else 0.0
            )
            if fitting_weight and geometry[(node, nxt)][1] > 1e-7 and direction is not None and direction != updated[0]:
                cost += fitting_weight
            reverse[following].append((current, cost, node, nxt))
            forward[current].append((following, cost, node, nxt, *geometry[(node, nxt)]))
        current += 1
    stats.state_count = len(states)
    at_node = {}
    for index, state in enumerate(states):
        at_node.setdefault(state[0], []).append(index)
    branch_starts = set(reset_indices.values())
    parent_directions = {}
    for index, reset in reset_indices.items():
        parent_directions.setdefault(reset, set()).add(states[index][2])
    forbidden_departures = {}
    for reset, directions in parent_directions.items():
        if len(directions) == 1 and None not in directions:
            direction = next(iter(directions))
            forbidden_departures[reset] = direction_bit(tuple(-value for value in direction))
    junction_charges = {}
    departures, local_endings = {}, {}
    for start in branch_starts:
        # Zero-length wall split arcs do not define distinct physical branches.
        # Preserve their baseline costs while finding the first real departure.
        zero_costs, zero_paths, queue = {start: 0.0}, {start: ()}, [(0.0, start)]
        outgoing, endings = [], []
        while queue:
            value, index = heapq.heappop(queue)
            if value != zero_costs[index]:
                continue
            prefix = zero_paths[index]
            endings.append((index, value, prefix))
            for following, cost, u, v, direction, length in forward[index]:
                path = (*prefix, (u, v))
                if length > 1e-7:
                    group = direction_bit(direction)
                    if not group & forbidden_departures.get(start, 0):
                        outgoing.append((group, following, value + cost, path))
                elif value + cost < zero_costs.get(following, math.inf):
                    zero_costs[following], zero_paths[following] = value + cost, path
                    heapq.heappush(queue, (value + cost, following))
        departures[start], local_endings[start] = outgoing, endings

    full = (1 << len(names)) - 1
    costs, choices = [dict() for _ in range(full + 1)], [dict() for _ in range(full + 1)]
    terminal_seeds = [dict() for _ in range(full + 1)]
    branch_costs = [dict() for _ in range(full + 1)]
    # Match the baseline terminal direction/cost convention exactly.
    from .directed_wall_graph import _direction

    for bit, name in enumerate(names):
        mask = 1 << bit
        for node in prepared.candidates[name]:
            wall = prepared.wall_points[(name, node)]
            connector = segment_geometry(positions[node], wall)
            cost_direction = _direction(positions[node], wall)
            for index in at_node.get(node, ()):
                _, incoming, direction, length = states[index]
                run = policy.advance((direction, length), *connector)
                # A grille exactly at a tee ends the incoming piece without
                # inventing a zero-length outgoing duct. Only merge-owned
                # branch starts receive this allowance.
                at_tee = index in branch_starts and connector[1] <= 1e-7
                if run is None or not (policy.complete(run) or at_tee):
                    continue
                value = connector[1] + (
                    float(bend_penalty) if connector[1] > 1e-7 and incoming is not None
                    and incoming != cost_direction else 0.0
                )
                if fitting_weight and connector[1] > 1e-7 and direction is not None and direction != connector[0]:
                    value += fitting_weight
                if value < costs[mask].get(index, math.inf):
                    costs[mask][index] = value
                    choices[mask][index] = ("terminal", name, node)
                    terminal_seeds[mask][index] = (value, choices[mask][index], direction_bit(connector[0]))

    straight_masks = set()
    if fitting_weight:
        for direction, bit in tuple(direction_bits.items()):
            if direction is not None:
                opposite = direction_bits.get(tuple(-value for value in direction))
                if opposite is not None:
                    straight_masks.add(bit | opposite)

    for mask in range(1, full + 1):
        merged_branches = {}
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and sub < other:
                left, right = costs[sub], costs[other]
                if policy.split_at_tees:
                    # Compute each physical branch start once. Many incoming
                    # length histories point to this same reset state.
                    for reset in branch_costs[sub].keys() & branch_costs[other].keys():
                        left_groups, right_groups = branch_costs[sub][reset], branch_costs[other][reset]
                        merged = merged_branches.setdefault(reset, {})
                        for lg, (lv, lc) in left_groups.items():
                            for rg, (rv, rc) in right_groups.items():
                                if lg & rg & ~1:
                                    continue
                                value = lv + rv
                                union = lg | rg
                                if value < merged.get(union, (math.inf, None))[0]:
                                    merged[union] = value, ("split", sub, other, lc, rc)
                    pairs = ()
                else:
                    pairs = ((index, index) for index in left.keys() & right.keys())
                for index, child in pairs:
                    value = left[child] + right[child]
                    if value < costs[mask].get(index, math.inf):
                        costs[mask][index] = value
                        choices[mask][index] = ("merge", sub, other, child)
            sub = (sub - 1) & mask
        if policy.split_at_tees:
            best_branches = {}
            for index, reset in reset_indices.items():
                direction = states[index][2]
                forbidden = direction_bit(tuple(-value for value in direction)) if direction is not None else 0
                key = reset, forbidden
                if key not in best_branches:
                    if fitting_weight:
                        candidates = []
                        for occupied, (value, choice) in merged_branches.get(reset, {}).items():
                            if occupied & forbidden:
                                continue
                            ports = occupied, forbidden
                            charge = junction_charges.get(ports)
                            if charge is None:
                                charge = fitting_weight * junction_fitting_count(occupied, forbidden, straight_masks)
                                junction_charges[ports] = charge
                            candidates.append((value + charge, choice))
                        best_branches[key] = min(candidates, key=lambda item: item[0], default=None)
                    else:
                        best_branches[key] = min(
                            (value for occupied, value in merged_branches.get(reset, {}).items()
                             if not occupied & forbidden), key=lambda item: item[0], default=None,
                        )
                branch = best_branches[key]
                if branch is not None and branch[0] < costs[mask].get(index, math.inf):
                    costs[mask][index] = branch[0]
                    choices[mask][index] = ("departures", branch[1])
        queue = [(value, index) for index, value in costs[mask].items()]
        heapq.heapify(queue)
        while queue:
            value, index = heapq.heappop(queue)
            if value != costs[mask].get(index):
                continue
            for previous, arc_cost, u, v in reverse[index]:
                candidate = value + arc_cost
                if candidate < costs[mask].get(previous, math.inf):
                    costs[mask][previous] = candidate
                    choices[mask][previous] = ("arc", index, u, v)
                    heapq.heappush(queue, (candidate, previous))
        if policy.split_at_tees:
            for start, outgoing in departures.items():
                groups = dict(merged_branches.get(start, {}))
                for group, following, arc_cost, path in outgoing:
                    value = arc_cost + costs[mask].get(following, math.inf)
                    if value < groups.get(group, (math.inf, None))[0]:
                        groups[group] = value, ("depart", following, path, None)
                for index, prefix_cost, path in local_endings[start]:
                    terminal = terminal_seeds[mask].get(index)
                    if terminal is not None:
                        value, choice, group = terminal
                        if group & forbidden_departures.get(start, 0):
                            continue
                        if value + prefix_cost < groups.get(group, (math.inf, None))[0]:
                            groups[group] = value + prefix_cost, ("depart", index, path, choice)
                if groups:
                    branch_costs[mask][start] = groups
    stats.subset_labels = sum(map(len, costs))
    stats.branch_labels = sum(len(groups) for subsets in branch_costs for groups in subsets.values())
    if rank_only:
        ranked = ((root, costs[full][index]) for index, (root, _) in starts.items() if index in costs[full])
        return tuple(sorted(ranked, key=lambda item: (item[1], item[0])))
    initial, (root, root_prefix) = next(iter(starts.items()))
    if initial not in costs[full]:
        return None
    arcs, selected = set(), {}
    pending = [(full, initial)]

    def expand_branch(mask, departure):
        if departure[0] == "split":
            expand_branch(departure[1], departure[3])
            expand_branch(departure[2], departure[4])
            return
        _, child, path, terminal = departure
        arcs.update(path)
        if terminal is None:
            pending.append((mask, child))
        else:
            selected[terminal[1]] = terminal[2]
    while pending:
        mask, index = pending.pop()
        choice = choices[mask][index]
        if choice[0] == "terminal":
            selected[choice[1]] = choice[2]
        elif choice[0] == "merge":
            pending.extend(((choice[1], choice[3]), (choice[2], choice[3])))
        elif choice[0] == "departures":
            expand_branch(mask, choice[1])
        else:
            arcs.add((choice[2], choice[3]))
            pending.append((mask, choice[1]))
    lines = [*root_prefix, *((positions[u], positions[v]) for u, v in arcs)]
    lines.extend((positions[node], prepared.wall_points[(name, node)]) for name, node in selected.items())
    physical_root = root_prefix[0][0] if root_prefix else positions[root]
    if not validate_tree_lengths(lines, physical_root,
                                 [prepared.wall_points[(name, node)] for name, node in selected.items()], policy):
        stats.rejected_physical_tree = True
        return None
    stats.fittings = count_tree_fittings(
        lines, [prepared.wall_points[(name, node)] for name, node in selected.items()],
    )
    prefix_charge = fitting_weight * count_tree_fittings(root_prefix).total if fitting_weight else 0.0
    stats.search_cost = float(costs[full][initial]) + prefix_charge
    stats.primary_cost = stats.search_cost - fitting_weight * stats.fittings.total
    directed = tuple(sorted(arcs))
    tree = SteinerTreeResult(
        frozenset({root} | {node for edge in directed for node in edge}), directed, directed,
        SteinerScore(len(directed), len(directed), 1.0, stats.search_cost),
    )
    return tree, selected
