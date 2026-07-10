from __future__ import annotations

import heapq
from numbers import Integral


def add_edge(graph, u, v, cap, cost, meta=None):
    fwd = {"to": v, "rev": len(graph[v]), "cap": cap, "orig_cap": cap, "cost": float(cost), "meta": meta}
    rev = {"to": u, "rev": len(graph[u]), "cap": 0, "orig_cap": 0, "cost": -float(cost), "meta": None}
    graph[u].append(fwd)
    graph[v].append(rev)


def min_cost_flow(graph, source, sink, flow_required):
    flow = 0
    cost = 0.0
    potentials = [0.0] * len(graph)

    while flow < flow_required:
        dist = [float("inf")] * len(graph)
        parent = [None] * len(graph)
        dist[source] = 0.0
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-9:
                continue
            for ei, edge in enumerate(graph[u]):
                if edge["cap"] <= 0:
                    continue
                v = edge["to"]
                nd = d + edge["cost"] + potentials[u] - potentials[v]
                if nd + 1e-9 < dist[v]:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    heapq.heappush(pq, (nd, v))

        if parent[sink] is None:
            break

        for i, d in enumerate(dist):
            if d < float("inf"):
                potentials[i] += d

        add = flow_required - flow
        v = sink
        while v != source:
            u, ei = parent[v]
            add = min(add, graph[u][ei]["cap"])
            v = u

        v = sink
        while v != source:
            u, ei = parent[v]
            edge = graph[u][ei]
            edge["cap"] -= add
            graph[v][edge["rev"]]["cap"] += add
            cost += add * edge["cost"]
            v = u

        flow += add

    return flow, cost


def positive_flow_edges(graph, u):
    return [
        edge
        for edge in graph[u]
        if edge["orig_cap"] > 0 and edge["orig_cap"] - edge["cap"] > 0
    ]


def trace_flow_path(graph, start_node, sink):
    states = []
    target = None
    u = start_node
    seen = set()

    while u != sink:
        if u in seen:
            return None, None
        seen.add(u)

        candidates = positive_flow_edges(graph, u)
        if not candidates:
            return None, None
        edge = candidates[0]
        edge["cap"] += 1

        meta = edge.get("meta")
        if meta:
            if meta[0] == "state":
                states.append((meta[1], meta[2]))
            elif meta[0] == "target":
                target = meta[1]
        u = edge["to"]

    if not states or target is None:
        return None, None
    path = [states[0][0]]
    path.extend(v for _, v in states)
    return path, target


def source_start_nodes(source_spec, kd):
    if isinstance(source_spec, (list, tuple, set)):
        values = list(source_spec)
        if not values:
            return []
        if isinstance(values[0], Integral):
            return [int(v) for v in values]
    _, start_idx = kd.query(source_spec)
    return [int(start_idx)]
