"""
Test and Benchmark Suite for 1D Hydraulic Network Solver.

MVP Scene:
- Inlet: Fan delivering Q0 = 0.5 m^3/s (1800 m^3/h)
- Main duct: D0 = 0.30 m, L0 = 3.0 m
- Split: 45° wye / pant junction
- Branch 1: D1 = 0.15 m, L1 = 4.0 m with one 90° elbow, outlet to P=0
- Branch 2: D2 = 0.25 m, L2 = 3.5 m straight, outlet to P=0

Verifications:
- Execution time (< 1 ms benchmark)
- Flow split (Q1, Q2)
- Velocities (v0, v1, v2)
- Pressure drops (dP0, dP1, dP2)
- Mass conservation (|Q0 - (Q1 + Q2)| < 1e-7)
"""

from __future__ import annotations

import sys
import time
import unittest
import numpy as np

# Ensure pipe_flow is in sys.path
import os
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


def build_mvp_network() -> HydraulicNetwork:
    """
    Constructs the MVP HVAC Duct Network:
    - Node 'Inlet': Inflow Q = +0.5 m^3/s (1800 m^3/h), pressure unknown
    - Node 'Junction_Wye': Unknown pressure, mass conservation sum(Q) = 0
    - Node 'Outlet_1': Fixed pressure P = 0 Pa (Atmospheric)
    - Node 'Outlet_2': Fixed pressure P = 0 Pa (Atmospheric)
    
    Edges:
    - 'Main_Duct': Inlet -> Junction_Wye (D=0.30m, L=3.0m)
    - 'Branch_1': Junction_Wye -> Outlet_1 (D=0.15m, L=4.0m, 45° wye branch takeoff + 90° elbow)
    - 'Branch_2': Junction_Wye -> Outlet_2 (D=0.25m, L=3.5m, 45° wye main/branch straight)
    """
    net = HydraulicNetwork(fluid=FluidProperties(density=1.204, viscosity=1.825e-5))
    
    # Nodes
    net.add_node("Inlet", node_type=NodeType.INLET, external_inflow=0.5)
    net.add_node("Junction_Wye", node_type=NodeType.JUNCTION)
    net.add_node("Outlet_1", node_type=NodeType.OUTLET, fixed_pressure=0.0)
    net.add_node("Outlet_2", node_type=NodeType.OUTLET, fixed_pressure=0.0)
    
    # Fittings
    wye_branch_loss = FittingLossLibrary.wye_45("branch")  # K ~ 0.50
    wye_main_loss = FittingLossLibrary.wye_45("main")      # K ~ 0.15
    elbow_90_loss = FittingLossLibrary.elbow_90_round(1.5)  # K ~ 0.25
    
    # Edges
    net.add_edge(
        name="Main_Duct",
        start_node="Inlet",
        end_node="Junction_Wye",
        length=3.0,
        diameter=0.30,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[],
    )
    
    net.add_edge(
        name="Branch_1",
        start_node="Junction_Wye",
        end_node="Outlet_1",
        length=4.0,
        diameter=0.15,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[wye_branch_loss, elbow_90_loss],
    )
    
    net.add_edge(
        name="Branch_2",
        start_node="Junction_Wye",
        end_node="Outlet_2",
        length=3.5,
        diameter=0.25,
        roughness=DuctMaterial.GALVANIZED_STEEL,
        fittings=[wye_main_loss],
    )
    
    return net


class TestHydraulic1DSolver(unittest.TestCase):
    """Unit tests for the 1D Hydraulic Network Solver."""

    def setUp(self):
        self.net = build_mvp_network()
        self.solver = Hydraulic1DSolver(self.net)

    def test_friction_factor_regimes(self):
        """Test laminar, transition, and turbulent friction factor evaluations."""
        # Laminar Re = 1000: f = 64/1000 = 0.064
        f_lam = friction_factor(1000.0, 0.0001)
        self.assertAlmostEqual(f_lam, 0.064, places=4)
        
        # Turbulent Re = 100,000, eps/D = 0.0003
        f_turb = friction_factor(100000.0, 0.0003, method="haaland")
        self.assertTrue(0.015 < f_turb < 0.025)

    def test_mvp_scene_convergence_and_mass_conservation(self):
        """Test MVP scene solution convergence and strict mass conservation."""
        sol = self.solver.solve(method="gga")
        
        self.assertTrue(sol.converged, "Solver must converge successfully.")
        self.assertLess(sol.mass_conservation_error, 1e-7, "Mass conservation error must be < 1e-7 m^3/s.")
        
        # Verify edge flows
        Q0 = sol.edge_flows["Main_Duct"]
        Q1 = sol.edge_flows["Branch_1"]
        Q2 = sol.edge_flows["Branch_2"]
        
        self.assertAlmostEqual(Q0, 0.5, places=6, msg="Main duct flow must match inlet flow 0.5 m^3/s.")
        self.assertAlmostEqual(Q0, Q1 + Q2, places=6, msg="Q0 must equal Q1 + Q2 (mass conservation).")
        self.assertGreater(Q1, 0.0, "Branch 1 flow must be positive.")
        self.assertGreater(Q2, 0.0, "Branch 2 flow must be positive.")
        
        # Physical reality check: Branch 2 (D=0.25m) has much lower resistance than Branch 1 (D=0.15m + elbow)
        # Therefore Q2 must be significantly greater than Q1
        self.assertGreater(Q2, Q1, "Branch 2 (D=0.25m) should receive more flow than Branch 1 (D=0.15m).")

    def test_pressure_drops_and_nodal_pressures(self):
        """Verify pressure drops consistency across branches discharging to P=0."""
        sol = self.solver.solve(method="gga")
        
        P_inlet = sol.node_pressures["Inlet"]
        P_wye = sol.node_pressures["Junction_Wye"]
        P_out1 = sol.node_pressures["Outlet_1"]
        P_out2 = sol.node_pressures["Outlet_2"]
        
        self.assertEqual(P_out1, 0.0)
        self.assertEqual(P_out2, 0.0)
        self.assertGreater(P_wye, 0.0, "Junction pressure must be positive above atmospheric.")
        self.assertGreater(P_inlet, P_wye, "Inlet pressure must exceed junction pressure.")
        
        # Pressure drop across branch 1 and branch 2 must both equal P_wye - 0 = P_wye
        dp1 = sol.edge_pressure_drops["Branch_1"]
        dp2 = sol.edge_pressure_drops["Branch_2"]
        self.assertAlmostEqual(dp1, P_wye, places=4)
        self.assertAlmostEqual(dp2, P_wye, places=4)
        self.assertAlmostEqual(dp1, dp2, places=4, msg="Parallel branches to P=0 must have identical pressure drops.")

    def test_solver_methods_consistency(self):
        """Verify all solver methods (GGA, Newton, SciPy Root, SciPy LeastSquares) match."""
        sol_gga = self.solver.solve(method="gga")
        sol_newton = self.solver.solve(method="newton")
        sol_root = self.solver.solve(method="scipy_root")
        sol_lsq = self.solver.solve(method="scipy_least_squares")
        
        self.assertTrue(sol_gga.converged)
        self.assertTrue(sol_newton.converged)
        self.assertTrue(sol_root.converged)
        self.assertTrue(sol_lsq.converged)
        
        # Verify flows match within 1e-4
        for name in ["Main_Duct", "Branch_1", "Branch_2"]:
            self.assertAlmostEqual(sol_gga.edge_flows[name], sol_newton.edge_flows[name], places=4)
            self.assertAlmostEqual(sol_gga.edge_flows[name], sol_root.edge_flows[name], places=4)
            self.assertAlmostEqual(sol_gga.edge_flows[name], sol_lsq.edge_flows[name], places=4)
            
        # Verify pressures match
        for name in ["Inlet", "Junction_Wye"]:
            self.assertAlmostEqual(sol_gga.node_pressures[name], sol_newton.node_pressures[name], places=3)
            self.assertAlmostEqual(sol_gga.node_pressures[name], sol_root.node_pressures[name], places=3)
            self.assertAlmostEqual(sol_gga.node_pressures[name], sol_lsq.node_pressures[name], places=3)

    def test_fan_curve_network(self):
        """Test duct network driven by a fan curve dP = 150 - 50*Q^2."""
        net = HydraulicNetwork()
        net.add_node("Plenum_In", node_type=NodeType.PLENUM, fixed_pressure=0.0)
        net.add_node("Plenum_Out", node_type=NodeType.OUTLET, fixed_pressure=0.0)
        
        fan = FanCurve(a0=250.0, a1=0.0, a2=-100.0)  # dP_fan = 250 - 100*Q^2
        net.add_edge("Duct_With_Fan", "Plenum_In", "Plenum_Out", length=10.0, diameter=0.3, fan=fan)
        
        solver = Hydraulic1DSolver(net)
        sol = solver.solve(method="gga")
        
        self.assertTrue(sol.converged)
        Q = sol.edge_flows["Duct_With_Fan"]
        self.assertGreater(Q, 0.0)


def run_benchmarks(num_runs: int = 1000):
    """Benchmarks solver execution speed and prints performance statistics."""
    print("\n" + "=" * 80)
    print(f"RUNNING HIGH-PERFORMANCE 1D HYDRAULIC SOLVER BENCHMARK ({num_runs} runs)")
    print("=" * 80)
    
    net = build_mvp_network()
    solver = Hydraulic1DSolver(net)
    
    # Warm-up run
    sol = solver.solve(method="gga")
    sol.print_results()
    
    # Benchmark GGA (Global Gradient Algorithm)
    times_gga = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver.solve(method="gga")
        t1 = time.perf_counter()
        times_gga.append((t1 - t0) * 1000.0)  # ms
        
    times_gga = np.array(times_gga)
    
    # Benchmark Newton
    times_newton = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        solver.solve(method="newton")
        t1 = time.perf_counter()
        times_newton.append((t1 - t0) * 1000.0)  # ms
    times_newton = np.array(times_newton)
    
    # Benchmark SciPy Root (100 runs for scipy)
    times_scipy = []
    for _ in range(100):
        t0 = time.perf_counter()
        solver.solve(method="scipy_root")
        t1 = time.perf_counter()
        times_scipy.append((t1 - t0) * 1000.0)
    times_scipy = np.array(times_scipy)
    
    print("\nBENCHMARK EXECUTION LATENCY RESULTS:")
    print(f"  {'Method':<20} {'Mean (ms)':>12} {'Median (ms)':>12} {'Min (ms)':>12} {'p99 (ms)':>12} {'Speedup vs SciPy':>18}")
    print("  " + "-" * 88)
    print(f"  {'GGA (Analytical)':<20} {np.mean(times_gga):>12.4f} {np.median(times_gga):>12.4f} {np.min(times_gga):>12.4f} {np.percentile(times_gga, 99):>12.4f} {np.mean(times_scipy) / np.mean(times_gga):>17.1f}x")
    print(f"  {'Full Newton':<20} {np.mean(times_newton):>12.4f} {np.median(times_newton):>12.4f} {np.min(times_newton):>12.4f} {np.percentile(times_newton, 99):>12.4f} {np.mean(times_scipy) / np.mean(times_newton):>17.1f}x")
    print(f"  {'SciPy Root (hybr)':<20} {np.mean(times_scipy):>12.4f} {np.median(times_scipy):>12.4f} {np.min(times_scipy):>12.4f} {np.percentile(times_scipy, 99):>12.4f} {'1.0x':>18}")
    print("=" * 80)
    
    # Target assertion check
    mean_gga_ms = np.mean(times_gga)
    assert mean_gga_ms < 1.0, f"Target execution time < 1.0 ms failed: got {mean_gga_ms:.4f} ms"
    print(f"[OK] Execution time target met: {mean_gga_ms:.4f} ms (< 1.0 ms target)")


if __name__ == "__main__":
    # 1. Run Unit Tests
    print("Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHydraulic1DSolver)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("Unit tests failed!")
        sys.exit(1)
        
    # 2. Run Performance Benchmark
    run_benchmarks(num_runs=1000)
