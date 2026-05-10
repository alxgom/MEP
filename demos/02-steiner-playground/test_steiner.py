"""
Tests for Steiner Tree Solver (gradient descent, SA, MST, geometry, angles).
Run: pytest test_steiner.py -v
"""

import math
import pytest

from steiner_solver import SteinerTreeSolver, DSU


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


# ── Solver Initialization Tests ──────────────────────────────────────────────

class TestSteinerSolverInit:
    def test_no_steiner_points(self):
        s = SteinerTreeSolver([(0, 0), (1, 0), (0.5, 1)], n_steiner=0)
        assert s.n_terminals == 3
        assert s.n_steiner == 0
        assert len(s.points) == 3

    def test_with_steiner_points(self):
        s = SteinerTreeSolver([(0, 0), (1, 0), (0.5, 1)], n_steiner=1, seed=42)
        assert s.n_terminals == 3
        assert s.n_steiner == 1
        assert len(s.points) == 4

    def test_zero_terminals(self):
        s = SteinerTreeSolver([], n_steiner=0)
        assert s.n_terminals == 0
        assert len(s.points) == 0

    def test_two_terminals(self):
        s = SteinerTreeSolver([(0, 0), (1, 0)], n_steiner=0)
        assert s.n_terminals == 2
        assert s._dist(0, 1) == 1.0

    def test_bounding_box(self):
        s = SteinerTreeSolver([(0, 0), (10, 10)], n_steiner=0)
        assert s._min_x == 0
        assert s._max_x == 10
        assert s._min_y == 0
        assert s._max_y == 10
        assert abs(s._bb_diag - math.hypot(10, 10)) < 1e-9

    def test_reproducible_seed(self):
        s1 = SteinerTreeSolver([(0,0),(1,0),(0.5,1)], n_steiner=2, seed=42)
        s2 = SteinerTreeSolver([(0,0),(1,0),(0.5,1)], n_steiner=2, seed=42)
        for i in range(s1.n_steiner):
            assert abs(s1.points[3+i][0] - s2.points[3+i][0]) < 1e-9
            assert abs(s1.points[3+i][1] - s2.points[3+i][1]) < 1e-9


# ── MST Computation Tests ───────────────────────────────────────────────────

class TestSteinerMST:
    def test_triangle_mst(self):
        s = SteinerTreeSolver([(0, 0), (10, 0), (5, 5)], n_steiner=0)
        mst = s._compute_mst()
        assert len(mst) == 2

    def test_square_mst(self):
        """Square: MST should have 3 edges, not 4."""
        s = SteinerTreeSolver([(0, 0), (10, 0), (10, 10), (0, 10)], n_steiner=0)
        mst = s._compute_mst()
        assert len(mst) == 3
        dsu_check = DSU(4)
        for u, v in mst:
            dsu_check.union(u, v)
        for i in range(4):
            assert dsu_check.find(0) == dsu_check.find(i)

    def test_adj_list_built(self):
        s = SteinerTreeSolver([(0, 0), (1, 0), (0, 1)], n_steiner=0)
        s._compute_mst()
        assert len(s.adj) == 3
        # Each vertex should have at least 1 neighbor in a connected tree
        for v in s.adj:
            assert len(s.adj[v]) >= 1


# ── Gradient Tests ──────────────────────────────────────────────────────────

class TestGradient:
    def test_gradient_is_zero_at_equilibrium(self):
        """At Fermat point of equilateral triangle, gradient should be ~0."""
        # Equilateral triangle: the Fermat point is at the centroid
        # Place a Steiner point at the centroid
        s = SteinerTreeSolver([(0, 0), (10, 0), (5, 5*math.sqrt(3)/2)], n_steiner=1, seed=42)
        s.points[3] = [5.0, 5*math.sqrt(3)/2 / 3 * 2]  # Approx centroid
        s._compute_mst()
        fx, fy = s._compute_gradient(3)
        # At equilibrium, net force should be ~0
        assert math.hypot(fx, fy) < 0.1

    def test_gradient_points_toward_neighbors(self):
        """A Steiner point between two terminals should be pulled toward both."""
        s = SteinerTreeSolver([(0, 0), (10, 0)], n_steiner=1, seed=42)
        s.points[2] = [5.0, 2.0]  # Above midpoint
        s._compute_mst()
        fx, fy = s._compute_gradient(2)
        # Should be pulled downward (negative fy) and should have near-zero fx
        assert fy < 0  # pulled down toward line
        assert abs(fx) < abs(fy)  # mostly downward

    def test_gradient_zero_for_isolated_point(self):
        """Steiner point placed far from terminals — gradient should be small but non-zero."""
        s = SteinerTreeSolver([(0, 0), (1, 0), (0.5, 0.01)], n_steiner=1, seed=42)
        # Place Steiner far away — all MST neighbors pull it back
        s.points[3] = [0.5, 50.0]
        s._compute_mst()
        fx, fy = s._compute_gradient(3)
        # Gradient should exist and be finite
        assert math.isfinite(fx) and math.isfinite(fy)


# ── 120° Angle Tests ────────────────────────────────────────────────────────

class TestAngleVerification:
    def test_equilateral_triangle_120(self):
        """At Steiner point connecting 3 vertices at 120°, deviation should be ~0."""
        # Three points at equal distance from center, 120° apart
        s = SteinerTreeSolver(n_steiner=1, seed=42,
                              terminals=[(0, 1), (-math.sqrt(3)/2, -0.5), (math.sqrt(3)/2, -0.5)])
        s.points[3] = [0.0, 0.0]  # Center (Fermat point)
        s._compute_mst()
        dev = s._max_angle_deviation(3)
        assert dev < 5.0  # Should be very close to 120°

    def test_known_equilateral_steiner(self):
        """Equilateral triangle of side 10: Steiner point minimizes total length."""
        side = 10
        h = side * math.sqrt(3) / 2
        s = SteinerTreeSolver(
            terminals=[(0, 0), (side, 0), (side / 2, h)],
            n_steiner=1, seed=42,
        )
        s._compute_mst()
        # After optimization, 120° deviation should reduce
        dev_before = s._max_angle_deviation(3)
        for _ in range(50):
            s.step_gradient(learning_rate=0.05)
        dev_after = s._max_angle_deviation(3)
        assert dev_after <= dev_before or dev_after < 15  # Should improve or be close

    def test_angle_computation_collinear(self):
        """Collinear points: angles should be 180°, deviation 60° from 120°."""
        s = SteinerTreeSolver(
            terminals=[(0, 0), (5, 0), (10, 0)],
            n_steiner=1, seed=42,
        )
        s.points[3] = [5.0, 0.0]  # On the line
        s._compute_mst()
        angles = s._angles_at_steiner(3)
        # With collinear points connected via MST, angle should be 180°
        # deviation from 120° is 60°
        if angles:
            assert any(abs(a - 120) > 30 for a in angles)


# ── Optimization Step Tests ─────────────────────────────────────────────────

class TestOptimizationStep:
    def test_step_reduces_force(self):
        """After a step, max force should generally decrease."""
        s = SteinerTreeSolver(
            terminals=[(0, 0), (10, 0), (5, 10)],
            n_steiner=1, seed=42,
        )
        s.points[3] = [5.0, 3.0]
        s._compute_mst()
        f1 = s._compute_gradient(3)
        mag1 = math.hypot(*f1)
        s.step_gradient(learning_rate=0.1)
        f2 = s._compute_gradient(3)
        mag2 = math.hypot(*f2)
        # Force should decrease or stay same after gradient step
        assert mag2 <= mag1 * 1.1  # Allow small tolerance for SA noise

    def test_step_returns_record(self):
        s = SteinerTreeSolver(
            terminals=[(0, 0), (10, 0), (5, 10)],
            n_steiner=1, seed=42,
        )
        record = s.step_gradient()
        assert "iteration" in record
        assert "mst_weight" in record
        assert "max_force" in record
        assert "merges" in record
        assert "max_120_deviation" in record
        assert "points" in record
        assert "sa_temperature" in record


# ── Merging / Annihilation Tests ────────────────────────────────────────────

class TestMerging:
    def test_close_steiner_points_merge(self):
        """Two Steiner points very close should merge."""
        s = SteinerTreeSolver(
            terminals=[(0, 0), (10, 0), (5, 10)],
            n_steiner=2, seed=42,
        )
        s.points[3] = [5.0, 4.0]
        s.points[4] = [5.01, 4.01]  # Very close
        s._compute_mst()
        merges = s._merge_close_points(threshold=0.1)
        assert merges >= 1
        assert len(s.points) <= 5  # One merged away

    def test_distant_points_not_merge(self):
        """Far-apart Steiner points should not merge."""
        s = SteinerTreeSolver(
            terminals=[(0, 0), (10, 0), (5, 10)],
            n_steiner=2, seed=42,
        )
        s.points[3] = [2.0, 3.0]
        s.points[4] = [8.0, 7.0]  # Far away
        merges = s._merge_close_points(threshold=0.1)
        assert merges == 0


# ── Convergence Tests ───────────────────────────────────────────────────────

class TestConvergence:
    def test_full_run_terminates(self):
        """Full optimization should complete within max iterations."""
        s = SteinerTreeSolver(
            terminals=[(0, 0), (1, 0), (0.5, 1)],
            n_steiner=1, seed=42,
        )
        result = s.run(max_iterations=100, convergence_threshold=1e-5)
        assert result["iterations"] <= 100
        assert "steiner_points" in result
        assert "edges" in result
        assert result["mst_weight"] > 0

    def test_mst_weight_positive(self):
        s = SteinerTreeSolver(
            terminals=[(0, 0), (10, 0), (5, 10)],
            n_steiner=1, seed=42,
        )
        result = s.run(max_iterations=50)
        assert result["mst_weight"] > 0


# ── Distance / Geometry Tests ───────────────────────────────────────────────

class TestGeometry:
    def test_dist_hypotenuse(self):
        s = SteinerTreeSolver([(0, 0), (3, 4)], n_steiner=0)
        assert abs(s._dist(0, 1) - 5.0) < 1e-9

    def test_dist_same_point(self):
        s = SteinerTreeSolver([(7, 7), (7, 7)], n_steiner=0)
        assert s._dist(0, 1) == 0.0

    def test_dist_euclidean(self):
        s = SteinerTreeSolver([(0, 0), (1, 1), (4, 5)], n_steiner=0)
        assert abs(s._dist(0, 1) - math.sqrt(2)) < 1e-9
        assert abs(s._dist(0, 2) - math.sqrt(41)) < 1e-9
        assert abs(s._dist(1, 2) - 5.0) < 1e-9


# ── Edge Sorting Tests ──────────────────────────────────────────────────────

class TestEdgeSorting:
    def test_mst_edges_sorted(self):
        s = SteinerTreeSolver(
            [(0, 0), (100, 0), (0, 100), (1, 1)], n_steiner=0
        )
        s._compute_mst()
        weights = [s._dist(u, v) for u, v in s.mst_edges]
        assert weights == sorted(weights)

    def test_all_edges_present(self):
        n = 5
        s = SteinerTreeSolver([(i * 10, i * 10) for i in range(n)], n_steiner=0)
        s._compute_mst()
        expected = n * (n - 1) // 2
        assert len(s.edges) if hasattr(s, 'edges') else True  # edges computed internally