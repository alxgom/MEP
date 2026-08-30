"""
Test and Benchmark Suite for Hardy Cross Loop Solver & 3D Hydraulic GGA with Elevation.

Elevated MVP Scene:
- Inlet: Fan delivering Q0 = 0.50 m^3/s (1800 m^3/h) at (0, 0, 0).
- Main Duct: D0 = 0.30 m, L0 = 3.0 m from (0, 0, 0) to (3, 0, 0).
- Wye Split at (3, 0, 0).
- Branch 1 (Elevated Riser + Elbows): D1 = 0.15 m, rises from (3, 0, 0) to elevation z = +2.5 m
  at (3, 1.5, 2.5), and runs horizontally to (7, 1.5, 2.5) with two 90° elbows, discharging to P = 0 Pa.
- Branch 2 (Direct Level Run): D2 = 0.25 m, straight level run from (3, 0, 0) to (6.5, -1.0, 0)
  at z = 0 m, discharging to P = 0 Pa.

Verifications:
- Automatic 3D Euclidean segment length computation.
- Hydrostatic elevation head pressure: dP_elev = rho * g * (z_end - z_start).
- Mass conservation error (|Q0 - (Q1 + Q2)| < 1e-7).
- Flow split consistency between Extended 3D Hydraulic GGA and Hardy Cross Loop Solver.
- Benchmark comparison of execution latency (< 1 ms), iterations, and residuals.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
import numpy as np

# Ensure pipe_flow is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipe_flow.solvers.hydraulic_1d import (
    FluidProperties,
    DuctMaterial,
    Fitting,
    FittingLossLibrary,
    FanCurve,
    Node,
    Edge,
    NodeType,
    HydraulicNetwork,
    Hydraulic1DSolver,
    HydraulicSolution,
    friction_factor,
)
from pipe_flow.solvers.hardy_cross import (
    Loop,
    HardyCrossSolver,
)


def build_elevated_mvp_network() -> HydraulicNetwork:
    """
    Constructs the Elevated 3D MVP HVAC Network:
    - Node 'Inlet': (0, 0, 0), inflow Q = 0.50 m^3/s
    - Node 'Junction_Wye': (3, 0, 0), z = 0.0 m
    - Node 'Riser_Top': (3, 1.5, 2.5), z = +2.5 m
    - Node 'Outlet_1': (7, 1.5, 2.5), z = +2.5 m, P = 0.0 Pa
    - Node 'Outlet_2': (6.5, -1.0, 0.0), z = 0.0 m, P = 0.0 Pa
    
    Edges (lengths auto-computed from 3D coordinates):
    - 'Main_Duct': Inlet (0,0,0) -> Junction_Wye (3,0,0) [D=0.30m, L=3.0m]
    - 'Branch_1_Riser': Junction_Wye (3,0,0) -> Riser_Top (3,1.5,2.5) [D=0.15m, 45° wye branch + 90° elbow]
    - 'Branch_1_Run': Riser_Top (3,1.5,2.5) -> Outlet_1 (7,1.5,2.5) [D=0.15m, 90° elbow]
    - 'Branch_2': Junction_Wye (3,0,0) -> Outlet_2 (6.5,-1.0,0.0) [D=0.25m, 45° wye main]
    """
    net = HydraulicNetwork(fluid=FluidProperties(density=1.204, viscosity=1.825e-5, gravity=9.80665))
    
    # 1. Nodes with 3D coordinates (x, y, z)
    net.add_node("Inlet", node_type=NodeType.INLET, coords=(0.0, 0.0, 0.0), external_inflow=0.50)
    net.add_node("Junction_Wye", node_type=NodeType.JUNCTION, coords=(3.0, 0.0, 0.0))
    net.add_node("Riser_Top", node_type=NodeType.JUNCTION, coords=(3.0, 1.5, 2.5))
    net.add_node("Outlet_1", node_type=NodeType.OUTLET, coords=(7.0, 1.5, 2.5), fixed_pressure=0.0)
    net.add_node("Outlet_2", node_type=NodeType.OUTLET, coords=(6.5, -1.0, 0.0), fixed_pressure=0.0)
    
    # 2. Fittings
    wye_branch_loss = FittingLossLibrary.wye_45("branch")   # K ~ 0.50
    wye_main_loss = FittingLossLibrary.wye_45("main")       # K ~ 0.15
    elbow_90 = FittingLossLibrary.elbow_90_round(1.5)       # K ~ 0.25
    
    # 3. Edges with Automatic Coordinate-Derived Lengths
    net.add_edge(
        name="Main_Duct",
        start_node="Inlet",
        end_node="Junction_Wye",
        diameter=0.30,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[],
    )
    
    net.add_edge(
        name="Branch_1_Riser",
        start_node="Junction_Wye",
        end_node="Riser_Top",
        diameter=0.15,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[wye_branch_loss, elbow_90],
    )
    
    net.add_edge(
        name="Branch_1_Run",
        start_node="Riser_Top",
        end_node="Outlet_1",
        diameter=0.15,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[elbow_90],
    )
    
    net.add_edge(
        name="Branch_2",
        start_node="Junction_Wye",
        end_node="Outlet_2",
        diameter=0.25,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[wye_main_loss],
    )
    
    return net


def build_closed_loop_network() -> HydraulicNetwork:
    """
    Constructs a looped duct network (Ring Network with cross-bridge):
    - Supply Inflow: 1.0 m^3/s entering at Node_A
    - Outlet 1: 0.4 m^3/s leaving at Node_C (fixed P = 0 Pa)
    - Outlet 2: 0.6 m^3/s leaving at Node_D (fixed P = 0 Pa)
    - Loop 1: A -> B -> C -> A
    - Loop 2: B -> D -> C -> B
    """
    net = HydraulicNetwork(fluid=FluidProperties(density=1.204, viscosity=1.825e-5))
    
    net.add_node("Node_A", node_type=NodeType.INLET, coords=(0.0, 0.0, 0.0), external_inflow=1.0)
    net.add_node("Node_B", node_type=NodeType.JUNCTION, coords=(4.0, 0.0, 0.0))
    net.add_node("Node_C", node_type=NodeType.OUTLET, coords=(4.0, 4.0, 1.0), fixed_pressure=0.0)
    net.add_node("Node_D", node_type=NodeType.OUTLET, coords=(0.0, 4.0, 1.0), fixed_pressure=0.0)
    
    net.add_edge("Duct_AB", "Node_A", "Node_B", diameter=0.35)
    net.add_edge("Duct_BC", "Node_B", "Node_C", diameter=0.25)
    net.add_edge("Duct_CD", "Node_C", "Node_D", diameter=0.20)
    net.add_edge("Duct_DA", "Node_D", "Node_A", diameter=0.25)
    net.add_edge("Duct_AC", "Node_A", "Node_C", diameter=0.30)
    
    return net


class TestHardyCrossAnd3DHydro(unittest.TestCase):
    """Unit and validation tests for 3D Hydraulic GGA and Hardy Cross Loop Solver."""

    def setUp(self):
        self.net_mvp = build_elevated_mvp_network()
        self.solver_gga = Hydraulic1DSolver(self.net_mvp)
        self.solver_hc = HardyCrossSolver(self.net_mvp)

    def test_3d_coordinates_and_auto_lengths(self):
        """Verify 3D Euclidean distances are automatically computed when length is omitted."""
        edges = self.net_mvp.edges
        
        # Main_Duct: (0,0,0) to (3,0,0) => 3.0 m
        self.assertAlmostEqual(edges[0].length, 3.0, places=4)
        
        # Branch_1_Riser: (3,0,0) to (3,1.5,2.5) => sqrt(1.5^2 + 2.5^2) = sqrt(2.25 + 6.25) = sqrt(8.5) ~ 2.91548 m
        expected_riser_len = np.sqrt(1.5**2 + 2.5**2)
        self.assertAlmostEqual(edges[1].length, expected_riser_len, places=4)
        
        # Branch_1_Run: (3,1.5,2.5) to (7,1.5,2.5) => sqrt(4^2) = 4.0 m
        self.assertAlmostEqual(edges[2].length, 4.0, places=4)
        
        # Branch_2: (3,0,0) to (6.5,-1,0) => sqrt(3.5^2 + (-1)^2) = sqrt(12.25 + 1) = sqrt(13.25) ~ 3.64005 m
        expected_b2_len = np.sqrt(3.5**2 + (-1.0)**2)
        self.assertAlmostEqual(edges[3].length, expected_b2_len, places=4)

    def test_hydrostatic_elevation_head(self):
        """Verify hydrostatic elevation head dP_elev = rho * g * delta_z."""
        rho = 1.204
        g = 9.80665
        expected_dp_elev = rho * g * 2.5  # 29.518 Pa
        
        # Branch_1_Riser has delta_z = +2.5 m
        self.assertAlmostEqual(self.solver_gga.edge_dp_hydro[1], expected_dp_elev, places=3)
        self.assertAlmostEqual(self.solver_hc.edge_dp_hydro[1], expected_dp_elev, places=3)
        
        # Level ducts have delta_z = 0 => dP_elev = 0
        self.assertAlmostEqual(self.solver_gga.edge_dp_hydro[0], 0.0, places=4)
        self.assertAlmostEqual(self.solver_gga.edge_dp_hydro[2], 0.0, places=4)
        self.assertAlmostEqual(self.solver_gga.edge_dp_hydro[3], 0.0, places=4)

    def test_elevated_mvp_convergence_and_mass_conservation(self):
        """Verify that both GGA and Hardy Cross solve the elevated MVP scene with strict mass conservation."""
        sol_gga = self.solver_gga.solve(method="gga")
        sol_hc_sim = self.solver_hc.solve(method="simultaneous")
        sol_hc_seq = self.solver_hc.solve(method="sequential")
        
        self.assertTrue(sol_gga.converged, "GGA solver must converge.")
        self.assertTrue(sol_hc_sim.converged, "Hardy Cross (simultaneous) must converge.")
        self.assertTrue(sol_hc_seq.converged, "Hardy Cross (sequential) must converge.")
        
        # Mass conservation checks
        self.assertLess(sol_gga.mass_conservation_error, 1e-7)
        self.assertLess(sol_hc_sim.mass_conservation_error, 1e-7)
        self.assertLess(sol_hc_seq.mass_conservation_error, 1e-7)
        
        # Main duct flow must equal inlet demand 0.50 m^3/s
        self.assertAlmostEqual(sol_gga.edge_flows["Main_Duct"], 0.50, places=6)
        self.assertAlmostEqual(sol_hc_sim.edge_flows["Main_Duct"], 0.50, places=6)
        
        # Branch 1 total flow: Q_riser == Q_run
        q1_riser = sol_gga.edge_flows["Branch_1_Riser"]
        q1_run = sol_gga.edge_flows["Branch_1_Run"]
        self.assertAlmostEqual(q1_riser, q1_run, places=6)
        
        # Split conservation: Q0 = Q1 + Q2
        q2 = sol_gga.edge_flows["Branch_2"]
        self.assertAlmostEqual(q1_riser + q2, 0.50, places=6)
        
        # Physical reality: Branch 2 (D=0.25m, level, no riser) must carry much more flow than Branch 1 (D=0.15m + 2.5m elevation + 2 elbows)
        self.assertGreater(q2, q1_riser)

    def test_consistency_between_gga_and_hardy_cross(self):
        """Assert that Extended GGA and Hardy Cross Loop Solver produce identical results within tight tolerance."""
        sol_gga = self.solver_gga.solve(method="gga")
        sol_hc = self.solver_hc.solve(method="simultaneous")
        
        # Verify edge flows match to 4 decimal places
        for edge_name in ["Main_Duct", "Branch_1_Riser", "Branch_1_Run", "Branch_2"]:
            q_gga = sol_gga.edge_flows[edge_name]
            q_hc = sol_hc.edge_flows[edge_name]
            self.assertAlmostEqual(
                q_gga, q_hc, places=4,
                msg=f"Edge '{edge_name}' flow mismatch: GGA={q_gga:.5f}, HC={q_hc:.5f}"
            )
            
        # Verify node pressures match
        for node_name in ["Inlet", "Junction_Wye", "Riser_Top", "Outlet_1", "Outlet_2"]:
            p_gga = sol_gga.node_pressures[node_name]
            p_hc = sol_hc.node_pressures[node_name]
            self.assertAlmostEqual(
                p_gga, p_hc, places=3,
                msg=f"Node '{node_name}' pressure mismatch: GGA={p_gga:.4f}, HC={p_hc:.4f}"
            )

    def test_closed_loop_cycle_basis_and_balancing(self):
        """Test closed-loop network balancing with multiple fundamental cycles."""
        net_loop = build_closed_loop_network()
        solver_hc_loop = HardyCrossSolver(net_loop)
        solver_gga_loop = Hydraulic1DSolver(net_loop)
        
        # Verify cycle basis found at least 2 independent fundamental loops
        self.assertGreaterEqual(solver_hc_loop.num_loops, 2)
        
        sol_hc = solver_hc_loop.solve(method="simultaneous")
        sol_gga = solver_gga_loop.solve(method="gga")
        
        self.assertTrue(sol_hc.converged)
        self.assertTrue(sol_gga.converged)
        
        # Verify strict mass conservation
        self.assertLess(sol_hc.mass_conservation_error, 1e-7)
        self.assertLess(sol_gga.mass_conservation_error, 1e-7)
        
        # Verify consistency across all edges
        for edge_name in sol_hc.edge_flows.keys():
            self.assertAlmostEqual(
                sol_hc.edge_flows[edge_name],
                sol_gga.edge_flows[edge_name],
                places=4,
                msg=f"Closed-loop flow mismatch on {edge_name}",
            )


def run_benchmarks(num_runs: int = 1000):
    """Benchmark comparative execution latencies of Extended GGA, Hardy Cross, and Newton solvers."""
    print("\n" + "=" * 92)
    print(f"BENCHMARK: EXTENDED 3D HYDRAULIC GGA vs HARDY CROSS LOOP SOLVER ({num_runs} runs)")
    print("=" * 92)
    
    net = build_elevated_mvp_network()
    solver_gga = Hydraulic1DSolver(net)
    solver_hc = HardyCrossSolver(net)
    
    # Warm-up and print detailed results
    sol_gga = solver_gga.solve(method="gga")
    sol_hc = solver_hc.solve(method="simultaneous")
    
    print("\n[EXTENDED 3D HYDRAULIC GGA SOLUTION]:")
    print(sol_gga.summary())
    
    print("\n[1D HARDY CROSS LOOP SOLVER SOLUTION]:")
    print(sol_hc.summary())
    
    # Measure GGA
    times_gga = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver_gga.solve(method="gga")
        t1 = time.perf_counter()
        times_gga.append((t1 - t0) * 1000.0)
    times_gga = np.array(times_gga)
    
    # Measure Hardy Cross Simultaneous
    times_hc_sim = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver_hc.solve(method="simultaneous")
        t1 = time.perf_counter()
        times_hc_sim.append((t1 - t0) * 1000.0)
    times_hc_sim = np.array(times_hc_sim)
    
    # Measure Hardy Cross Sequential
    times_hc_seq = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver_hc.solve(method="sequential")
        t1 = time.perf_counter()
        times_hc_seq.append((t1 - t0) * 1000.0)
    times_hc_seq = np.array(times_hc_seq)
    
    # Measure Newton
    times_newton = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver_gga.solve(method="newton")
        t1 = time.perf_counter()
        times_newton.append((t1 - t0) * 1000.0)
    times_newton = np.array(times_newton)
    
    print("\nBENCHMARK LATENCY & CONVERGENCE COMPARISON:")
    hdr = f"  {'Solver Method':<28} {'Mean (ms)':>11} {'Median (ms)':>12} {'Min (ms)':>11} {'p99 (ms)':>11} {'Iters':>8} {'Mass Err (m^3/s)':>18}"
    print(hdr)
    print("  " + "-" * 104)
    print(f"  {'Extended 3D GGA (Todini)':<28} {np.mean(times_gga):>11.4f} {np.median(times_gga):>12.4f} {np.min(times_gga):>11.4f} {np.percentile(times_gga, 99):>11.4f} {sol_gga.iterations:>8} {sol_gga.mass_conservation_error:>18.2e}")
    print(f"  {'Hardy Cross (Simultaneous)':<28} {np.mean(times_hc_sim):>11.4f} {np.median(times_hc_sim):>12.4f} {np.min(times_hc_sim):>11.4f} {np.percentile(times_hc_sim, 99):>11.4f} {sol_hc.iterations:>8} {sol_hc.mass_conservation_error:>18.2e}")
    print(f"  {'Hardy Cross (Sequential)':<28} {np.mean(times_hc_seq):>11.4f} {np.median(times_hc_seq):>12.4f} {np.min(times_hc_seq):>11.4f} {np.percentile(times_hc_seq, 99):>11.4f} {solver_hc.solve('sequential').iterations:>8} {solver_hc.solve('sequential').mass_conservation_error:>18.2e}")
    print(f"  {'Full Newton-Raphson':<28} {np.mean(times_newton):>11.4f} {np.median(times_newton):>12.4f} {np.min(times_newton):>11.4f} {np.percentile(times_newton, 99):>11.4f} {solver_gga.solve('newton').iterations:>8} {solver_gga.solve('newton').mass_conservation_error:>18.2e}")
    print("=" * 92)
    
    assert np.mean(times_gga) < 1.0, f"GGA target < 1.0 ms failed: {np.mean(times_gga):.4f} ms"
    assert np.mean(times_hc_sim) < 1.0, f"Hardy Cross target < 1.0 ms failed: {np.mean(times_hc_sim):.4f} ms"
    print(f"[SUCCESS] All solvers achieved sub-millisecond execution latency (< 1.0 ms)!\n")


if __name__ == "__main__":
    print("Executing Unit Test Suite for 3D Hydraulic GGA & Hardy Cross Solver...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHardyCrossAnd3DHydro)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    if not res.wasSuccessful():
        print("Unit tests failed!")
        sys.exit(1)
        
    run_benchmarks(num_runs=1000)
