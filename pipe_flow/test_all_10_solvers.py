"""
Comprehensive Verification and Unit Test Suite for All 10 HVAC Duct Flow Solvers.

Tests:
1.  Candidate 1:  1D Hydraulic GGA (Global Gradient Algorithm)
2.  Candidate 2:  1D Hardy Cross Method
3.  Candidate 3:  2D Calibrated Fast Fluid Dynamics (FFD)
4.  Candidate 4:  3D Calibrated Lattice Boltzmann (LBM D3Q19 + LES)
5.  Candidate 5:  AI FlowGNN Hydraulic Surrogate
6.  Candidate 6:  AI FNO-2D/3D Field Surrogate
7.  Candidate 7:  2D Lattice Boltzmann (LBM D2Q9 + LES)
8.  Candidate 8:  2D Streamfunction-Vorticity Solver
9.  Candidate 9:  2D Finite Volume SIMPLE Solver
10. Candidate 10: Multi-Layer 2.5D Coupled Solver
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

from pipe_flow.solvers.hydraulic_1d import (
    FluidProperties,
    DuctMaterial,
    FittingLossLibrary,
    HydraulicNetwork,
    Hydraulic1DSolver,
    NodeType,
)
from pipe_flow.solvers.hardy_cross import (
    HardyCrossSolver,
)
from pipe_flow.solvers.grid_ffd_2d import (
    CellType,
    DuctGeometry2D,
    GridFFD2D,
    build_calibrated_2d_branching_geometry,
)
from pipe_flow.solvers.lbm_3d import (
    DuctGeometry3D,
    LBM3DSolver,
    build_mvp_3d_lbm_geometry,
)
from pipe_flow.solvers.ai_surrogate import (
    HydraulicSurrogateSolver,
    SurrogateNetworkInput,
)
from pipe_flow.solvers.ai_fno_2d3d import (
    FNO2DSolver,
    FNO3DSolver,
    generate_synthetic_duct_grid_2d,
    generate_synthetic_duct_voxel_3d,
)
from pipe_flow.solvers.lbm_2d import (
    LBM2DSolver,
)
from pipe_flow.solvers.streamfunction_vorticity_2d import (
    StreamfunctionVorticity2DSolver,
)
from pipe_flow.solvers.fv_simple_2d import (
    FVSIMPLE2DSolver,
)
from pipe_flow.solvers.multilayer_25d import (
    MultiLayer25DSolver,
)


class TestAllTenSolvers(unittest.TestCase):
    """Unified unit tests for all 10 solver paradigms under physical normalization."""

    def test_candidate_1_hydraulic_1d_gga(self):
        """Tests Candidate 1: 1D Hydraulic Solver with GGA."""
        net = HydraulicNetwork(fluid=FluidProperties(density=1.204, viscosity=1.825e-5, isothermal_gauge=True))
        net.add_node("Inlet", node_type=NodeType.INLET, external_inflow=0.50, elevation=0.0)
        net.add_node("Junction_Wye", node_type=NodeType.JUNCTION, elevation=0.0)
        net.add_node("Outlet_1", node_type=NodeType.OUTLET, fixed_pressure=0.0, elevation=2.5)
        net.add_node("Outlet_2", node_type=NodeType.OUTLET, fixed_pressure=0.0, elevation=0.0)

        wye_b = FittingLossLibrary.wye_45("branch")
        wye_m = FittingLossLibrary.wye_45("main")
        elb_90 = FittingLossLibrary.elbow_90_round(1.5)

        net.add_edge("Main_Duct", "Inlet", "Junction_Wye", length=3.0, diameter=0.30)
        net.add_edge("Branch_1", "Junction_Wye", "Outlet_1", length=4.0, diameter=0.15, fittings=[wye_b, elb_90])
        net.add_edge("Branch_2", "Junction_Wye", "Outlet_2", length=3.5, diameter=0.25, fittings=[wye_m])

        solver = Hydraulic1DSolver(net)
        sol = solver.solve(method="gga")

        self.assertTrue(sol.converged)
        self.assertAlmostEqual(sol.edge_flows["Main_Duct"], 0.50, places=5)
        self.assertAlmostEqual(sol.edge_flows["Main_Duct"], sol.edge_flows["Branch_1"] + sol.edge_flows["Branch_2"], places=6)
        self.assertAlmostEqual(sol.edge_flows["Branch_1"] / 0.50 * 100.0, 16.51, places=0)
        self.assertGreater(sol.node_pressures["Inlet"], sol.node_pressures["Junction_Wye"])

    def test_candidate_2_hardy_cross_1d(self):
        """Tests Candidate 2: 1D Hardy Cross Method."""
        net = HydraulicNetwork(fluid=FluidProperties(density=1.204, viscosity=1.825e-5, isothermal_gauge=True))
        net.add_node("Inlet", node_type=NodeType.INLET, external_inflow=0.50, elevation=0.0)
        net.add_node("Junction_Wye", node_type=NodeType.JUNCTION, elevation=0.0)
        net.add_node("Outlet_1", node_type=NodeType.OUTLET, fixed_pressure=0.0, elevation=2.5)
        net.add_node("Outlet_2", node_type=NodeType.OUTLET, fixed_pressure=0.0, elevation=0.0)

        wye_b = FittingLossLibrary.wye_45("branch")
        wye_m = FittingLossLibrary.wye_45("main")
        elb_90 = FittingLossLibrary.elbow_90_round(1.5)

        net.add_edge("Main_Duct", "Inlet", "Junction_Wye", length=3.0, diameter=0.30)
        net.add_edge("Branch_1", "Junction_Wye", "Outlet_1", length=4.0, diameter=0.15, fittings=[wye_b, elb_90])
        net.add_edge("Branch_2", "Junction_Wye", "Outlet_2", length=3.5, diameter=0.25, fittings=[wye_m])

        solver = HardyCrossSolver(net)
        sol = solver.solve(method="simultaneous")

        self.assertTrue(sol.converged)
        self.assertAlmostEqual(sol.edge_flows["Main_Duct"], 0.50, places=5)
        self.assertAlmostEqual(sol.edge_flows["Main_Duct"], sol.edge_flows["Branch_1"] + sol.edge_flows["Branch_2"], places=6)

    def test_candidate_3_grid_ffd_2d(self):
        """Tests Candidate 3: 2D Calibrated Grid FFD Solver."""
        geo = build_calibrated_2d_branching_geometry(nx=80, ny=40, dx=0.05, dy=0.05, u_inlet=7.0735)
        solver = GridFFD2D(
            geometry=geo,
            dt=0.005,
            enable_darcy_friction=True,
            hydraulic_diameter=0.25,
            area_3d_inlet=0.0706858,
            area_3d_outlets={"Branch_1_Elbow": 0.0176715, "Branch_2_Straight": 0.0490874},
        )
        solver.step()
        summary = solver.get_flow_summary()
        self.assertGreater(summary["inlet_flow_rate"], 0.0)

    def test_candidate_4_lbm_3d(self):
        """Tests Candidate 4: 3D Lattice Boltzmann D3Q19 + LES Solver."""
        geo = build_mvp_3d_lbm_geometry(nx=24, ny=12, nz=12, dx=0.05)
        solver = LBM3DSolver(geometry=geo, u_inlet_phys=7.0735)
        solver.step()
        res = solver.run(num_steps=5)
        self.assertGreater(res.q_inlet, 0.0)
        self.assertGreater(res.q_branch1, 0.0)
        self.assertGreater(res.q_branch2, 0.0)

    def test_candidate_5_ai_surrogate(self):
        """Tests Candidate 5: AI FlowGNN Surrogate with 3D elevation."""
        solver = HydraulicSurrogateSolver(use_numpy_backend=True)
        inp = SurrogateNetworkInput(
            D0=0.30, L0=3.0, K0=0.0,
            D1=0.15, L1=4.0, K1=0.75,
            D2=0.25, L2=3.5, K2=0.15,
            Q0=0.50,
            z0=0.0, z_junc=0.0, z1=2.5, z2=0.0,
        )
        pred = solver.predict(inp)
        self.assertAlmostEqual(pred.Q0, 0.50, places=6)
        self.assertAlmostEqual(pred.Q0, pred.Q1 + pred.Q2, places=8)
        self.assertAlmostEqual(pred.Q1 / 0.50 * 100.0, 16.51, places=0)
        self.assertGreater(pred.P_inlet, pred.P_junction)

    def test_candidate_6_ai_fno_2d3d(self):
        """Tests Candidate 6: AI 2D/3D FNO Surrogate."""
        solver_2d = FNO2DSolver(hidden_dim=16, modes_x=8, modes_y=8)
        inp_2d = generate_synthetic_duct_grid_2d(nx=64, ny=32, dx=0.025, dy=0.025, u_inlet=7.0735)
        pred_2d = solver_2d.predict(inp_2d)
        self.assertGreater(pred_2d.q_inlet, 0.0)
        self.assertAlmostEqual(pred_2d.q_inlet, pred_2d.q_branch1 + pred_2d.q_branch2, places=4)

        solver_3d = FNO3DSolver(hidden_dim=8, modes_x=4, modes_y=4, modes_z=4)
        inp_3d = generate_synthetic_duct_voxel_3d(nx=24, ny=12, nz=12, dx=0.05, dy=0.05, dz=0.05, u_inlet=7.0735)
        pred_3d = solver_3d.predict(inp_3d)
        self.assertGreater(pred_3d.q_inlet, 0.0)
        self.assertAlmostEqual(pred_3d.q_inlet, pred_3d.q_branch1 + pred_3d.q_branch2, places=4)

    def test_candidate_7_lbm_2d(self):
        """Tests Candidate 7: 2D Lattice Boltzmann D2Q9 + LES Solver."""
        solver = LBM2DSolver(nx=64, ny=32, dx=0.05, u_inlet_phys=7.0735)
        solver.step()
        res = solver.run(num_steps=5)
        self.assertGreater(res.q_inlet, 0.0)
        self.assertGreater(res.q_branch1, 0.0)
        self.assertGreater(res.q_branch2, 0.0)

    def test_candidate_8_streamfunction_vorticity(self):
        """Tests Candidate 8: 2D Streamfunction-Vorticity Solver."""
        solver = StreamfunctionVorticity2DSolver(nx=60, ny=30, dx=0.05, u_inlet_phys=7.0735)
        solver.step()
        res = solver.run(num_steps=5)
        self.assertGreater(res.q_inlet, 0.0)
        self.assertGreater(res.q_branch1, 0.0)
        self.assertGreater(res.q_branch2, 0.0)

    def test_candidate_9_fv_simple(self):
        """Tests Candidate 9: 2D Finite Volume SIMPLE Solver."""
        solver = FVSIMPLE2DSolver(nx=60, ny=30, dx=0.05, u_inlet_phys=7.0735)
        solver.step()
        res = solver.run(num_steps=5)
        self.assertGreater(res.q_inlet, 0.0)
        self.assertGreater(res.q_branch1, 0.0)
        self.assertGreater(res.q_branch2, 0.0)

    def test_candidate_10_multilayer_25d(self):
        """Tests Candidate 10: Multi-Layer 2.5D Coupled Solver."""
        solver = MultiLayer25DSolver(nx=60, ny=30, dx=0.05, u_inlet_phys=7.0735)
        solver.step()
        res = solver.run(num_steps=5)
        self.assertGreater(res.q_inlet, 0.0)
        self.assertGreater(res.q_branch1, 0.0)
        self.assertGreater(res.q_branch2, 0.0)


if __name__ == "__main__":
    unittest.main()
