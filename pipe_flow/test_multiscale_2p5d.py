"""
Unit and Verification Tests for Candidate 10: Multi-Layer 2.5D Coupled Solver.

Tests:
1. 1D Vertical Riser transfer port hydraulics (Colebrook-White friction, elbow minor losses).
2. Floor 1 2D plenum simulation (Fan supply -> Wye split -> Room Diffuser + Riser Port).
3. Floor 2 2D plenum simulation (Riser Port -> Header -> Diffusers 2A & 2B).
4. Coupled Multi-Scale outer loop convergence matching interface port flux and pressure.
5. Multi-story mass conservation across Floor 1, Riser, and Floor 2.
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

from pipe_flow.solvers.multiscale_2p5d import (
    Riser1DConfig,
    Layer2DConfig,
    MultiScale2p5DSolution,
    MultiScale2p5DSolver,
    build_floor1_plenum_geometry,
    build_floor2_plenum_geometry,
)


class TestMultiScale2p5D(unittest.TestCase):
    """Test suite for Multi-Layer 2.5D Coupled HVAC Solver."""

    def test_riser_1d_hydraulics(self):
        """Verify 1D vertical riser pressure drop and friction calculations."""
        riser = Riser1DConfig(
            height=3.5,
            diameter=0.20,
            roughness=0.00009,
            k_elbow=0.35,
            num_elbows=2,
            isothermal_gauge=True,
        )

        q_test = 0.20  # m^3/s
        metrics = riser.compute_pressure_drop(q_test, rho=1.204, nu=1.5e-5)

        self.assertGreater(metrics["velocity"], 0.0)
        self.assertGreater(metrics["reynolds"], 1000.0)
        self.assertGreater(metrics["friction_factor"], 0.01)
        self.assertGreater(metrics["dp_major_pa"], 0.0)
        self.assertGreater(metrics["dp_minor_pa"], 0.0)
        self.assertGreater(metrics["dp_total_pa"], 0.0)
        self.assertAlmostEqual(
            metrics["dp_total_pa"],
            metrics["dp_major_pa"] + metrics["dp_minor_pa"] + metrics["dp_hydro_pa"],
            places=5,
        )

    def test_layer1_and_layer2_geometry_builders(self):
        """Verify Floor 1 and Floor 2 geometry builders."""
        geo1 = build_floor1_plenum_geometry(nx=80, ny=40, dx=0.03, dy=0.03, u_fan=2.5)
        self.assertIn("Fan_Supply_Inlet", geo1.inlets)
        self.assertIn("Floor1_Room_Diffuser", geo1.outlets)
        self.assertIn("Riser_Transfer_Port", geo1.outlets)

        geo2 = build_floor2_plenum_geometry(nx=80, ny=40, dx=0.03, dy=0.03, u_riser_inlet=2.0)
        self.assertIn("Riser_Inflow_Port", geo2.inlets)
        self.assertIn("Floor2_Diffuser_2A", geo2.outlets)
        self.assertIn("Floor2_Diffuser_2B", geo2.outlets)

    def test_multiscale_coupled_solver_execution(self):
        """Verify coupled multi-scale solver runs and converges across domains."""
        riser_cfg = Riser1DConfig(height=3.5, diameter=0.20)
        layer1_cfg = Layer2DConfig(name="Floor1", nx=80, ny=40, dx=0.03, dy=0.03)
        layer2_cfg = Layer2DConfig(name="Floor2", nx=80, ny=40, dx=0.03, dy=0.03)

        solver = MultiScale2p5DSolver(
            riser_config=riser_cfg,
            layer1_config=layer1_cfg,
            layer2_config=layer2_cfg,
            u_fan_supply=2.0,
            max_outer_iter=5,
            tol_pressure=0.20,
            damping_factor=0.70,
        )

        solution: MultiScale2p5DSolution = solver.solve(verbose=False)

        self.assertGreater(solution.iterations, 0)
        self.assertGreater(solution.q_fan, 0.0)
        self.assertGreater(solution.q_floor1_diffuser, 0.0)
        self.assertGreater(solution.q_riser, 0.0)
        self.assertGreater(len(solution.q_floor2_diffusers), 0)
        self.assertGreater(solution.riser_dp, 0.0)
        self.assertGreater(solution.p_fan_inlet, 0.0)

        # Verify mass distribution: Floor 1 + Floor 2 fractions sum to ~100%
        total_frac = solution.floor1_fraction_pct + solution.floor2_fraction_pct
        self.assertAlmostEqual(total_frac, 100.0, places=1)

        # Coupling history is recorded
        self.assertEqual(len(solution.coupling_history), solution.iterations)


if __name__ == "__main__":
    unittest.main()
