"""
Test suite and MVP Scene benchmark for 2D Grid Fast Fluid Dynamics (FFD) HVAC Solver.

MVP Scene:
- Inlet duct (width = 0.6m) with prescribed inflow velocity u_in = 2.0 m/s.
- Splits into two parallel branches:
  * Branch 1: Narrow channel (width = 0.2m) with 90-degree elbow bend.
  * Branch 2: Wide straight channel (width = 0.4m).
- Simulates flow field to steady state.
- Benchmarks execution time per step and convergence time.
- Extracts branch flow division (Q1, Q2), pressure drops (Delta P1, Delta P2),
  and verifies mass conservation.
- Generates publication-grade flow field visualization to 'pipe_flow/ffd_flow_field.png'.
"""

import os
import sys
import time

# Ensure pipe_flow package is resolvable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from solvers.grid_ffd_2d import DuctGeometry2D, GridFFD2D, CellType, SimulationResult


def build_mvp_split_geometry(
    nx: int = 160,
    ny: int = 80,
    dx: float = 0.02,
    dy: float = 0.02,
    u_inlet: float = 2.0,
) -> DuctGeometry2D:
    """
    Construct the HVAC duct branching geometry:
    - Main trunk: [0.0, 1.0] x [0.4, 1.0] (H = 0.6m, rows 20..49, cols 0..49)
    - Branch 2 (wide straight): [1.0, 3.2] x [0.4, 0.8] (W = 0.4m, rows 20..39, cols 50..159)
    - Branch 1 (narrow + elbow):
      * Vertical riser: [0.8, 1.0] x [1.0, 1.4] (W = 0.2m, rows 50..69, cols 40..49)
      * Horizontal run: [1.0, 3.2] x [1.2, 1.4] (W = 0.2m, rows 60..69, cols 50..159)
    - Inlet: col 0, rows 20..49, u = 2.0 m/s
    - Outlet 1: col 159, rows 60..69, p = 0 Pa
    - Outlet 2: col 159, rows 20..39, p = 0 Pa
    """
    geo = DuctGeometry2D(nx=nx, ny=ny, dx=dx, dy=dy)

    # 1. Main trunk channel
    geo.add_box_cells(j0=0, i0=20, j1=49, i1=49, cell_type=CellType.FLUID)

    # 2. Branch 2 (Wide Straight Duct)
    geo.add_box_cells(j0=50, i0=20, j1=159, i1=39, cell_type=CellType.FLUID)

    # 3. Branch 1 (Vertical riser of elbow bend)
    geo.add_box_cells(j0=40, i0=50, j1=49, i1=69, cell_type=CellType.FLUID)

    # 4. Branch 1 (Horizontal run after elbow)
    geo.add_box_cells(j0=50, i0=60, j1=159, i1=69, cell_type=CellType.FLUID)

    # 5. Inlet Face at column 0 (rows 20..49, 30 cells = 0.6m)
    geo.add_inlet_cells(
        j0=0,
        i0=20,
        j1=0,
        i1=49,
        velocity=(u_inlet, 0.0),
        tracer=1.0,
        normal=(1.0, 0.0),
        name="Inlet_Main",
    )

    # 6. Outlet 1 Face (Branch 1 - Elbow, column 159, rows 60..69, 10 cells = 0.2m)
    geo.add_outlet_cells(
        j0=nx - 1,
        i0=60,
        j1=nx - 1,
        i1=69,
        pressure=0.0,
        normal=(1.0, 0.0),
        name="Branch_1_Elbow",
    )

    # 7. Outlet 2 Face (Branch 2 - Straight, column 159, rows 20..39, 20 cells = 0.4m)
    geo.add_outlet_cells(
        j0=nx - 1,
        i0=20,
        j1=nx - 1,
        i1=39,
        pressure=0.0,
        normal=(1.0, 0.0),
        name="Branch_2_Straight",
    )

    return geo


def test_grid_ffd_2d_mvp():
    """Main test and verification function for Candidate 2 FFD Solver."""
    print("=" * 80)
    print(" CANDIDATE 2: 2D GRID FAST FLUID DYNAMICS (FFD) SOLVER - MVP BENCHMARK")
    print("=" * 80)

    nx, ny = 160, 80
    dx, dy = 0.02, 0.02
    u_inlet = 2.0
    dt = 0.005

    print(f"Grid Configuration: {nx} x {ny} cells (dx = {dx*1000:.1f}mm, dy = {dy*1000:.1f}mm)")
    print(f"Domain Size: {nx*dx:.2f}m x {ny*dy:.2f}m | Inflow Velocity: {u_inlet:.2f} m/s")

    # 1. Build Geometry
    geo = build_mvp_split_geometry(nx=nx, ny=ny, dx=dx, dy=dy, u_inlet=u_inlet)
    print(f"Geometry Inlets: {list(geo.inlets.keys())}")
    print(f"Geometry Outlets: {list(geo.outlets.keys())}")

    # 2. Instantiate Solver
    solver = GridFFD2D(
        geometry=geo,
        dt=dt,
        nu=1.5e-5,
        rho=1.204,
        poisson_solver="lu",
    )
    print(f"Active Fluid Cells: {solver.num_fluid_cells} / {nx*ny} ({solver.num_fluid_cells/(nx*ny)*100:.1f}%)")

    # 3. Run Simulation to Steady State
    print("\nStarting Time-Stepping Simulation...")
    result: SimulationResult = solver.run_until_steady_state(
        max_steps=500,
        dt=dt,
        tol=1e-4,
        min_steps=60,
        verbose=True,
    )

    # 4. Print Comprehensive Benchmark Metrics
    print("\n" + "=" * 80)
    print(" BENCHMARK & FLOW DIVISION RESULTS")
    print("=" * 80)
    print(f"Convergence Status        : {'CONVERGED' if result.converged else 'MAX STEPS REACHED'}")
    print(f"Iterations Executed       : {result.iterations} steps")
    print(f"Simulated Physical Time   : {result.total_sim_time:.3f} s")
    print(f"Wall Clock Elapsed Time   : {result.elapsed_wall_time:.3f} s")
    print(f"Execution Speed           : {result.time_per_step_ms:.2f} ms/step ({result.steps_per_second:.1f} FPS)")
    print("-" * 80)
    print(f"Inlet Flow Rate (Q_in)    : {result.inlet_flow_rate:.5f} m^3/s ({result.inlet_mass_flow_rate:.5f} kg/s)")

    for name, q_val in result.outlet_flow_rates.items():
        frac = result.branch_fractions[name] * 100.0
        m_val = result.outlet_mass_flow_rates[name]
        dp_stat = result.pressure_drops[name]
        dp_tot = result.total_pressure_drops[name]
        print(f"  -> {name:20s}: Q = {q_val:.5f} m^3/s ({frac:5.1f}%) | mass = {m_val:.5f} kg/s | dP_stat = {dp_stat:6.2f} Pa | dP_tot = {dp_tot:6.2f} Pa")

    print("-" * 80)
    print(f"Mass Conservation Error   : {result.mass_conservation_error_pct:.4f} %")
    print(f"Inlet Static Pressure     : {result.inlet_static_pressure:.2f} Pa")
    print(f"Inlet Total Pressure      : {result.inlet_total_pressure:.2f} Pa")
    print("=" * 80)

    # 5. Physics Assertions
    assert result.mass_conservation_error_pct < 2.0, (
        f"Mass conservation error {result.mass_conservation_error_pct:.2f}% exceeds 2% limit!"
    )
    assert result.inlet_flow_rate > 0.0, "Inlet flow rate must be strictly positive"

    q1 = result.outlet_flow_rates["Branch_1_Elbow"]
    q2 = result.outlet_flow_rates["Branch_2_Straight"]
    assert q2 > q1, f"Wide straight branch flow (Q2={q2:.4f}) should exceed narrow elbow flow (Q1={q1:.4f})"

    for name, dp in result.pressure_drops.items():
        assert dp > 0.0, f"Pressure drop for {name} ({dp:.2f} Pa) must be positive"

    # 6. Save Flow Visualization
    save_img_path = os.path.join(current_dir, "ffd_flow_field.png")
    solver.plot_flow_field(
        save_path=save_img_path,
        show=False,
        title="2D Fast Fluid Dynamics (FFD) HVAC Solver - Branching Duct Simulation",
    )
    assert os.path.exists(save_img_path), f"Visualization file not found at {save_img_path}"
    print(f"\n[OK] Flow field visualization verified at: {save_img_path}")

    # 7. Compare Poisson Solvers (LU vs Red-Black SOR vs CG)
    print("\n" + "=" * 80)
    print(" POISSON SOLVER COMPARISON BENCHMARK (50 steps each)")
    print("=" * 80)

    for method in ["lu", "sor", "cg"]:
        geo_test = build_mvp_split_geometry(nx=nx, ny=ny, dx=dx, dy=dy, u_inlet=u_inlet)
        solver_test = GridFFD2D(
            geometry=geo_test,
            dt=dt,
            poisson_solver=method,
            sor_omega=1.65,
            poisson_max_iter=40,
        )
        t0 = time.perf_counter()
        for _ in range(50):
            solver_test.step()
        t1 = time.perf_counter()
        ms_per_step = (t1 - t0) / 50.0 * 1000.0
        fps = 50.0 / (t1 - t0)
        summary = solver_test.get_flow_summary()
        print(
            f"Solver: {method.upper():5s} | Speed: {ms_per_step:6.2f} ms/step ({fps:5.1f} FPS) | "
            f"Mass Err: {summary['mass_conservation_error_pct']:6.3f}% | "
            f"Q1/Q2 Split: {summary['branch_fractions']['Branch_1_Elbow']*100:.1f}% / "
            f"{summary['branch_fractions']['Branch_2_Straight']*100:.1f}%"
        )

    print("=" * 80)
    print(" ALL TESTS & BENCHMARKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_grid_ffd_2d_mvp()
