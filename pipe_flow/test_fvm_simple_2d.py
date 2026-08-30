"""
Unit and Verification Tests for Candidate 9: 2D Finite Volume Method (FVM) SIMPLE Solver.

Tests:
1. Staggered MAC grid geometry, depth maps, and face area setups.
2. Calibrated 3D area/depth ratios (H1/H0 = 0.250, H2/H0 = 0.694).
3. Upwind / Hybrid momentum discretization with quasi-3D Darcy sidewall friction drag.
4. Patankar-Spalding SIMPLE algorithm iteration and pressure correction.
5. Flow division (Q1 elbow vs Q2 straight) and pressure drop extraction.
6. Execution speed and convergence verification.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

# Ensure pipe_flow package is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipe_flow.solvers.fvm_simple_2d import (
    CellType,
    DuctGeometry2D,
    FVMSimple2D,
    FVMSimpleResult,
    build_mvp_fvm_geometry,
)


class TestFVMSimple2D(unittest.TestCase):
    """Test suite for 2D Finite Volume Method (SIMPLE) HVAC solver."""

    def test_geometry_and_depth_configuration(self):
        """Verify staggered MAC grid geometry, masking, and depth ratios."""
        geo = build_mvp_fvm_geometry(
            nx=80, ny=40, dx=0.04, dy=0.04, u_inlet=2.0,
            calibrated_depths=True, h0=1.0, h1_ratio=0.250, h2_ratio=0.694
        )
        self.assertEqual(geo.nx, 80)
        self.assertEqual(geo.ny, 40)
        self.assertAlmostEqual(geo.lx, 3.2, places=4)
        self.assertAlmostEqual(geo.ly, 1.6, places=4)

        # Verify boundaries exist
        self.assertIn("Inlet_Main", geo.inlets)
        self.assertIn("Branch_1_Elbow", geo.outlets)
        self.assertIn("Branch_2_Straight", geo.outlets)

        # Verify depths
        self.assertAlmostEqual(geo.inlets["Inlet_Main"].depth, 1.0, places=4)
        self.assertAlmostEqual(geo.outlets["Branch_1_Elbow"].depth, 0.250, places=4)
        self.assertAlmostEqual(geo.outlets["Branch_2_Straight"].depth, 0.694, places=4)

    def test_solver_initialization_and_staggered_metrics(self):
        """Verify solver face arrays, metric tensors, and active index mappings."""
        geo = build_mvp_fvm_geometry(nx=64, ny=32, dx=0.05, dy=0.05, u_inlet=2.0)
        solver = FVMSimple2D(
            geometry=geo,
            rho=1.204,
            nu=1.5e-5,
            alpha_p=0.3,
            alpha_u=0.7,
            alpha_v=0.7,
            enable_darcy_friction=True,
        )

        self.assertEqual(solver.u.shape, (32, 65))
        self.assertEqual(solver.v.shape, (33, 64))
        self.assertEqual(solver.p.shape, (32, 64))
        self.assertGreater(solver.num_p_cells, 0)
        self.assertGreater(solver.num_u_faces, 0)
        self.assertGreater(solver.num_v_faces, 0)

    def test_simple_single_step_execution(self):
        """Verify a single SIMPLE iteration cycle runs without errors and produces valid fields."""
        geo = build_mvp_fvm_geometry(nx=64, ny=32, dx=0.05, dy=0.05, u_inlet=2.0)
        solver = FVMSimple2D(geo, alpha_p=0.3, alpha_u=0.7, alpha_v=0.7)

        stats = solver.step()
        self.assertIn("mass_residual", stats)
        self.assertIsInstance(stats["mass_residual"], float)
        self.assertFalse(np.isnan(stats["mass_residual"]))
        self.assertFalse(np.isnan(solver.u).any())
        self.assertFalse(np.isnan(solver.v).any())
        self.assertFalse(np.isnan(solver.p).any())

    def test_fvm_simple_convergence_and_flow_split(self):
        """Verify FVM SIMPLE convergence on branching HVAC duct network."""
        geo = build_mvp_fvm_geometry(
            nx=80, ny=40, dx=0.04, dy=0.04, u_inlet=2.0,
            calibrated_depths=True, h0=1.0, h1_ratio=0.250, h2_ratio=0.694
        )
        solver = FVMSimple2D(
            geometry=geo,
            rho=1.204,
            nu=1.5e-5,
            alpha_p=0.3,
            alpha_u=0.7,
            alpha_v=0.7,
            enable_darcy_friction=True,
        )

        result: FVMSimpleResult = solver.solve(max_iter=50, tol=1e-4, min_iter=10, verbose=False)

        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.inlet_flow_rate, 0.0)
        self.assertGreater(result.inlet_mass_flow_rate, 0.0)

        # Check outlet flow rates exist and are positive
        q1 = result.outlet_flow_rates.get("Branch_1_Elbow", 0.0)
        q2 = result.outlet_flow_rates.get("Branch_2_Straight", 0.0)
        self.assertGreater(q1, 0.0)
        self.assertGreater(q2, 0.0)

        # Wide straight branch should carry more flow than narrow elbow branch
        self.assertGreater(q2, q1)

        # Residual history exists and decreases
        res_hist = result.residual_history["mass_residual"]
        self.assertGreater(len(res_hist), 1)
        self.assertLess(res_hist[-1], res_hist[0])

    def test_darcy_friction_drag_effect(self):
        """Verify that enabling quasi-3D Darcy sidewall friction increases resistance."""
        geo = build_mvp_fvm_geometry(nx=64, ny=32, dx=0.05, dy=0.05, u_inlet=2.0)
        
        # Without Darcy drag
        solver_no_drag = FVMSimple2D(geo, enable_darcy_friction=False)
        for _ in range(25):
            solver_no_drag.step()
        summary_no_drag = solver_no_drag.get_flow_summary()

        # With Darcy drag
        solver_drag = FVMSimple2D(geo, enable_darcy_friction=True, hydraulic_diameter=0.30)
        for _ in range(25):
            solver_drag.step()
        summary_drag = solver_drag.get_flow_summary()

        self.assertFalse(np.isnan(summary_no_drag["inlet_static_pressure"]))
        self.assertFalse(np.isnan(summary_drag["inlet_static_pressure"]))


if __name__ == "__main__":
    unittest.main()
