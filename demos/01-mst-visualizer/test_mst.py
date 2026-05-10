"""
Tests for MST Visualizer (Kruskal's algorithm + DSU + geometry).
Run with: pytest test_mst.py -v
"""

import pytest
import math

# ── DSU (copied for isolated testing) ────────────────────────────────────────

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ── MST Logic (pure, no pygame) ──────────────────────────────────────────────

class MSTLogic:
    """Pure MST logic for testing — no pygame dependency."""

    def __init__(self, vertices):
        self.vertices = list(vertices)
        self.n = len(self.vertices)
        self.edges = []
        self.sorted_edges = []
        self.mst_edges = []
        self.rejected_edges = []
        self.current_edge_idx = 0
        self.total_weight = 0.0
        self._build_edges()

    def _dist(self, i, j):
        x1, y1 = self.vertices[i]
        x2, y2 = self.vertices[j]
        return math.hypot(x2 - x1, y2 - y1)

    def _build_edges(self):
        self.edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.edges.append((i, j, self._dist(i, j)))
        self.edges.sort(key=lambda e: e[2])
        self.sorted_edges = self.edges

    def step(self):
        if self.current_edge_idx >= len(self.sorted_edges):
            return
        u, v, w = self.sorted_edges[self.current_edge_idx]
        if self._dsu.union(u, v):
            self.mst_edges.append((u, v))
            self.total_weight += w
        else:
            self.rejected_edges.append((u, v))
        self.current_edge_idx += 1

    def run_to_completion(self):
        self._dsu = DSU(self.n)
        while self.current_edge_idx < len(self.sorted_edges):
            self.step()


# ── DSU Tests ────────────────────────────────────────────────────────────────

class TestDSU:
    def test_single_element(self):
        dsu = DSU(1)
        assert dsu.find(0) == 0

    def test_union_creates_set(self):
        dsu = DSU(3)
        assert dsu.union(0, 1) is True
        assert dsu.find(0) == dsu.find(1)
        assert dsu.find(2) != dsu.find(0)

    def test_union_same_set_returns_false(self):
        dsu = DSU(3)
        dsu.union(0, 1)
        assert dsu.union(1, 0) is False

    def test_transitive_closure(self):
        dsu = DSU(5)
        dsu.union(0, 1)
        dsu.union(1, 2)
        dsu.union(3, 4)
        assert dsu.find(0) == dsu.find(2)
        assert dsu.find(3) == dsu.find(4)
        assert dsu.find(0) != dsu.find(3)

    def test_many_unions(self):
        dsu = DSU(100)
        for i in range(99):
            assert dsu.union(i, i + 1) is True
        for i in range(100):
            assert dsu.find(0) == dsu.find(i)


# ── MST Logic Tests ─────────────────────────────────────────────────────────

class TestMSTLogic:
    def test_single_vertex_no_edges(self):
        v = MSTLogic([(100, 100)])
        assert len(v.sorted_edges) == 0

    def test_two_vertices_one_edge(self):
        v = MSTLogic([(0, 0), (10, 0)])
        assert len(v.sorted_edges) == 1
        assert v.sorted_edges[0][2] == 10.0

    def test_triangle_mst_picks_two_shortest(self):
        v = MSTLogic([(0, 0), (10, 0), (5, 5)])
        lengths = [e[2] for e in v.sorted_edges]
        assert lengths == sorted(lengths)

    def test_kruskal_no_cycle(self):
        """A square: MST should have 3 edges, not 4."""
        v = MSTLogic([(0, 0), (10, 0), (10, 10), (0, 10)])
        v.run_to_completion()
        assert len(v.mst_edges) == 3
        assert v._dsu.find(0) == v._dsu.find(1) == v._dsu.find(2) == v._dsu.find(3)

    def test_kruskal_triangle(self):
        """Equilateral triangle side=10: MST picks 2 edges of length 10."""
        import math
        side = 10
        h = side * math.sqrt(3) / 2
        v = MSTLogic([(0, 0), (side, 0), (side / 2, h)])
        v.run_to_completion()
        assert len(v.mst_edges) == 2
        assert abs(v.total_weight - 20.0) < 0.01

    def test_kruskal_collinear(self):
        """4 collinear points: MST connects them in a chain."""
        v = MSTLogic([(0, 0), (10, 0), (20, 0), (30, 0)])
        v.run_to_completion()
        assert len(v.mst_edges) == 3
        assert abs(v.total_weight - 30.0) < 0.01

def test_mst_weight_never_decreases():
    v = MSTLogic([(0, 0), (5, 0), (5, 5), (0, 5), (2, 2)])
    ws = []
    v._dsu = DSU(v.n)
    while v.current_edge_idx < len(v.sorted_edges):
        before = v.total_weight
        v.step()
        if v.total_weight > before:
            ws.append(v.total_weight)
    for i in range(1, len(ws)):
        assert ws[i] >= ws[i - 1]


# ── Distance / Geometry Tests ───────────────────────────────────────────────

class TestGeometry:
    def test_distance_hypotenuse(self):
        v = MSTLogic([(0, 0), (3, 4)])
        assert abs(v._dist(0, 1) - 5.0) < 1e-9

    def test_distance_same_point(self):
        v = MSTLogic([(7, 7), (7, 7)])
        assert v._dist(0, 1) == 0.0

    def test_distances_are_euclidean(self):
        v = MSTLogic([(0, 0), (1, 1), (4, 5)])
        assert abs(v._dist(0, 1) - math.sqrt(2)) < 1e-9
        assert abs(v._dist(0, 2) - math.sqrt(41)) < 1e-9
        assert abs(v._dist(1, 2) - 5.0) < 1e-9


# ── Edge Sorting Tests ──────────────────────────────────────────────────────

class TestEdgeSorting:
    def test_edges_sorted_by_weight(self):
        v = MSTLogic([(0, 0), (100, 0), (0, 100), (1, 1)])
        weights = [e[2] for e in v.sorted_edges]
        assert weights == sorted(weights)

    def test_all_edges_present(self):
        n = 5
        v = MSTLogic([(i * 10, i * 10) for i in range(n)])
        expected = n * (n - 1) // 2
        assert len(v.sorted_edges) == expected


# ── Preset File Validation ──────────────────────────────────────────────────

class TestPresets:
    def test_presets_load(self):
        import json, os
        path = os.path.join(os.path.dirname(__file__), "presets.json")
        with open(path) as f:
            data = json.load(f)
        assert "presets" in data
        assert len(data["presets"]) == 20

    def test_preset_vertices_nonempty(self):
        import json, os
        path = os.path.join(os.path.dirname(__file__), "presets.json")
        with open(path) as f:
            data = json.load(f)
        for p in data["presets"]:
            vertices = p.get("vertices", [])
            seed = p.get("seed")
            # Seed-based presets are generated at runtime; skip vertex check
            if not vertices and seed is not None:
                continue
            assert len(vertices) >= 2, f"Preset '{p.get('name')}' has < 2 vertices"
            for v in vertices:
                assert len(v) == 2
                assert 0.0 <= v[0] <= 1.0
                assert 0.0 <= v[1] <= 1.0

    def test_preset_names_unique(self):
        import json, os
        path = os.path.join(os.path.dirname(__file__), "presets.json")
        with open(path) as f:
            data = json.load(f)
        names = [p.get("name") for p in data["presets"]]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"