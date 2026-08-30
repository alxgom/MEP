"""
Unit Tests for Candidate 7: 2D Lattice Boltzmann Method (LBM D2Q9 + Smagorinsky LES) Solver.
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

from pipe_flow.solvers.lbm_2d import (
    CellType,
    BoundarySegment,
    DuctGeometry2DLBM,
    LBM2DSolver,
    SimulationResult2DLBM,
    build_standard_hvac_lbm_geometry,
)


class TestLBM2DSolver(unittest.TestCase):
    """Rigorous unit test suite for 2D LBM D2Q9 + Smagorinsky LES solver."""

    def setUp(self):
        """Build standard geometry for tests."""
        self.nx = 80
        self.ny = 40
        self.dx = 0.04
        self.dy = 0.04
        self.geo = build_standard_hvac_lbm_geometry(
            nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, depth=0.1178, q_inlet=0.50
        )

    def test_geometry_creation_and_area_ratios(self):
        """Tests that 2D LBM duct geometry matches 3D ASHRAE area ratios."""
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
        # Branch 1 ratio H1/H0 = 0.15 / 0.60 = 0.250
        ratio_b1 = b1.width / inlet.width
        self.assertAlmostEqual(ratio_b1, 0.250, places=2)

        # Branch 2 ratio H2/H0 = 0.4167 / 0.60 = 0.694
        ratio_b2 = b2.width / inlet.width
        self.assertAlmostEqual(ratio_b2, 0.694, places=2)

    def test_lbm_initialization_and_equilibrium(self):
        """Tests resting equilibrium distribution and macroscopic recovery."""
        solver = LBM2DSolver(geometry=self.geo, tau_0=0.55, smagorinsky_cs=0.12)

        self.assertEqual(solver.f.shape, (9, self.ny, self.nx))
        self.assertEqual(solver.rho_lat.shape, (self.ny, self.nx))
        self.assertEqual(solver.u_lat.shape, (self.ny, self.nx))

        np.testing.assert_allclose(solver.rho_lat, 1.0, atol=1e-6)
        self.assertAlmostEqual(float(np.sum(solver.W)), 1.0, places=6)

    def test_smagorinsky_eddy_viscosity(self):
        """Tests that Smagorinsky model produces tau_eff >= tau_0 and positive eddy viscosity."""
        solver = LBM2DSolver(geometry=self.geo, tau_0=0.55, smagorinsky_cs=0.15, enable_smagorinsky=True)

        for _ in range(25):
            solver.step()

        self.assertTrue(np.all(solver.tau_eff >= solver.tau_0 - 1e-8))
        self.assertTrue(np.all(solver.nu_t_lat >= 0.0))

        max_nu_t = float(np.max(solver.nu_t_lat))
        self.assertGreater(max_nu_t, 0.0)

    def test_single_step_and_residual(self):
        """Tests that a single LBM step executes and returns a valid finite residual."""
        solver = LBM2DSolver(geometry=self.geo, tau_0=0.55)
        res = solver.step()
        self.assertTrue(np.isfinite(res))
        self.assertGreaterEqual(res, 0.0)

    def test_darcy_drag_forcing(self):
        """Tests that quasi-3D Darcy-Weisbach drag force is applied properly."""
        solver = LBM2DSolver(
            geometry=self.geo,
            tau_0=0.55,
            enable_darcy_drag=True,
            hydraulic_diameter=0.20,
            darcy_f=0.02,
        )

        for _ in range(20):
            solver.step()

        fx, fy = solver._apply_darcy_drag_forcing()
        u_fluid = solver.u_lat[solver.fluid_mask]
        fx_fluid = fx[solver.fluid_mask]
        opposing = np.sum(fx_fluid * u_fluid <= 1e-12)
        self.assertGreater(opposing, 0.90 * len(u_fluid))

    def test_simulation_run_and_mass_conservation(self):
        """Tests multi-step LBM simulation and verifies mass flow recovery."""
        solver = LBM2DSolver(geometry=self.geo, tau_0=0.55, smagorinsky_cs=0.12)
        # Run 400 steps so acoustic wave traverses the full domain and develops steady flow
        result = solver.run(num_steps=400, min_steps=100)

        self.assertIsInstance(result, SimulationResult2DLBM)
        self.assertGreater(result.inlet_flow_rate, 0.0)
        self.assertGreater(result.inlet_static_pressure, 0.0)
        self.assertGreater(result.inlet_total_pressure, 0.0)

        q1 = result.outlet_flow_rates["Branch_1_Elbow"]
        q2 = result.outlet_flow_rates["Branch_2_Straight"]
        self.assertGreater(q1, 0.0)
        self.assertGreater(q2, 0.0)

        q_sum = q1 + q2
        self.assertAlmostEqual(q_sum, result.inlet_flow_rate, delta=0.20 * result.inlet_flow_rate)

        dp1 = result.pressure_drops["Branch_1_Elbow"]
        dp2 = result.pressure_drops["Branch_2_Straight"]
        self.assertGreater(dp1, 0.0)
        self.assertGreater(dp2, 0.0)

    def test_execution_speed_and_fps(self):
        """Tests that the vectorized 2D LBM solver runs fast (> 100 FPS on test grid)."""
        solver = LBM2DSolver(geometry=self.geo, tau_0=0.55)

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

        print(f"\n[LBM 2D Benchmark] Grid {self.nx}x{self.ny}: {ms_per_step:.3f} ms/step ({fps:.1f} FPS)")
        self.assertGreater(fps, 100.0, f"Expected FPS > 100, got {fps:.1f} FPS")


if __name__ == "__main__":
    unittest.main()
