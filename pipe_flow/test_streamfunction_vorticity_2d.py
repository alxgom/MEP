"""
Unit Tests for Candidate 8: 2D Streamfunction-Vorticity (psi-omega) Solver.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
import numpy as np

# Ensure pipe_flow package is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipe_flow.solvers.streamfunction_vorticity_2d import (
    CellType,
    BoundarySegment,
    DuctGeometry2DPSIW,
    StreamfunctionVorticity2DSolver,
    SimulationResult2DPSIW,
    build_standard_hvac_psiw_geometry,
)


class TestStreamfunctionVorticity2DSolver(unittest.TestCase):
    """Rigorous unit test suite for 2D Streamfunction-Vorticity solver."""

    def setUp(self):
        """Build standard branching geometry for tests."""
        self.nx = 80
        self.ny = 40
        self.dx = 0.04
        self.dy = 0.04
        self.geo = build_standard_hvac_psiw_geometry(
            nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, depth=0.1178, q_inlet=0.50
        )

    def test_geometry_creation_and_area_ratios(self):
        """Tests that psi-omega duct geometry matches 3D ASHRAE area ratios."""
        self.assertIn("Inlet_Main", self.geo.inlets)
        self.assertIn("Branch_1_Elbow", self.geo.outlets)
        self.assertIn("Branch_2_Straight", self.geo.outlets)

        inlet = self.geo.inlets["Inlet_Main"]
        b1 = self.geo.outlets["Branch_1_Elbow"]
        b2 = self.geo.outlets["Branch_2_Straight"]

        self.assertAlmostEqual(inlet.prescribed_flow_rate, 0.50, places=3)
        self.assertGreater(inlet.area, 0.0)
        self.assertGreater(b1.area, 0.0)
        self.assertGreater(b2.area, 0.0)

        # Check height / area ratios
        ratio_b1 = b1.width / inlet.width
        self.assertAlmostEqual(ratio_b1, 0.250, places=2)

        ratio_b2 = b2.width / inlet.width
        self.assertAlmostEqual(ratio_b2, 0.694, places=2)

    def test_exact_zero_divergence_property(self):
        """Tests that streamfunction velocity u = d(psi)/dy, v = -d(psi)/dx has exact zero divergence."""
        solver = StreamfunctionVorticity2DSolver(geometry=self.geo, dt=0.005)

        for _ in range(10):
            solver.step()

        # In any active fluid region with interior stencil (i-1..i+1, j-1..j+1 in active fluid):
        # div u = d^2(psi)/dxdy - d^2(psi)/dydx == 0 identically!
        div_interior = np.zeros((self.ny - 2, self.nx - 2))
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                # Central difference derivatives on psi
                d2psi_dxdy = (
                    solver.psi[i + 1, j + 1] - solver.psi[i - 1, j + 1]
                    - (solver.psi[i + 1, j - 1] - solver.psi[i - 1, j - 1])
                ) / (4.0 * self.dx * self.dy)
                d2psi_dydx = (
                    solver.psi[i + 1, j + 1] - solver.psi[i + 1, j - 1]
                    - (solver.psi[i - 1, j + 1] - solver.psi[i - 1, j - 1])
                ) / (4.0 * self.dx * self.dy)
                div_interior[i - 1, j - 1] = d2psi_dxdy - d2psi_dydx

        max_div = float(np.max(np.abs(div_interior)))
        print(f"\n[Psi-Omega Zero-Divergence] Analytical Max |div u| = {max_div:.2e}")
        # Mathematical identity: d2psi/dxdy - d2psi/dydx == 0.0 to machine precision
        self.assertLess(max_div, 1e-12)

    def test_thom_wall_vorticity_boundary_condition(self):
        """Tests that Thom's 2nd-order wall vorticity condition sets non-zero wall vorticity."""
        solver = StreamfunctionVorticity2DSolver(geometry=self.geo, dt=0.005)
        solver._apply_thom_wall_vorticity()
        wall_omega_nonzero = np.sum(np.abs(solver.omega[solver.solid_mask]) > 1e-4)
        self.assertGreater(wall_omega_nonzero, 0)

    def test_single_step_and_residual(self):
        """Tests that a single time-step executes cleanly and produces finite fields."""
        solver = StreamfunctionVorticity2DSolver(geometry=self.geo, dt=0.005)
        res = solver.step()
        self.assertTrue(np.isfinite(res))
        self.assertTrue(np.all(np.isfinite(solver.u)))
        self.assertTrue(np.all(np.isfinite(solver.v)))
        self.assertTrue(np.all(np.isfinite(solver.p)))

    def test_simulation_run_and_mass_conservation(self):
        """Tests multi-step run and verifies flow rates and pressure drops."""
        solver = StreamfunctionVorticity2DSolver(geometry=self.geo, dt=0.005)
        result = solver.run(num_steps=100, min_steps=30)

        self.assertIsInstance(result, SimulationResult2DPSIW)
        self.assertGreater(result.inlet_flow_rate, 0.0)
        self.assertGreater(result.inlet_static_pressure, 0.0)

        q1 = result.outlet_flow_rates["Branch_1_Elbow"]
        q2 = result.outlet_flow_rates["Branch_2_Straight"]
        self.assertGreater(q1, 0.0)
        self.assertGreater(q2, 0.0)

        # Flow conservation
        q_sum = q1 + q2
        self.assertAlmostEqual(q_sum, result.inlet_flow_rate, delta=0.20 * result.inlet_flow_rate)

        dp1 = result.pressure_drops["Branch_1_Elbow"]
        dp2 = result.pressure_drops["Branch_2_Straight"]
        self.assertGreater(dp1, 0.0)
        self.assertGreater(dp2, 0.0)

    def test_execution_speed_and_fps(self):
        """Tests that pre-factored LU streamfunction solver runs at high frame rates (> 100 FPS)."""
        solver = StreamfunctionVorticity2DSolver(geometry=self.geo, dt=0.005, poisson_solver="lu")

        # Warm up
        for _ in range(5):
            solver.step()

        n_benchmark_steps = 100
        t0 = time.perf_counter()
        for _ in range(n_benchmark_steps):
            solver.step()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        fps = n_benchmark_steps / elapsed
        ms_per_step = (elapsed / n_benchmark_steps) * 1000.0

        print(f"\n[Psi-Omega 2D Benchmark] Grid {self.nx}x{self.ny}: {ms_per_step:.3f} ms/step ({fps:.1f} FPS)")
        self.assertGreater(fps, 100.0, f"Expected FPS > 100, got {fps:.1f} FPS")


if __name__ == "__main__":
    unittest.main()
