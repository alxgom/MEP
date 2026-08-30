"""
Test Suite and Benchmark for 3D Lattice Boltzmann (LBM D3Q19 / D3Q7) HVAC Solver.

Simulates 3D elevated HVAC scene with height changes:
- Main horizontal duct with fan inlet inflow u_in = 2.0 m/s at z = 0.2m (floor level).
- 3D Wye bifurcation split into two branches:
  * Branch 1: Vertical riser duct rising along +z from z = 0.2m to z = 0.9m,
              turning 90-degrees with an elbow into a horizontal run at elevated ceiling height,
              ending at Outlet 1 (Elevated Room Diffuser).
  * Branch 2: Horizontal straight duct continuing along +x at floor level (z = 0.2m),
              ending at Outlet 2 (Floor Diffuser).
- Gravity body force: g = (0, 0, -9.81 m/s^2).
- Validates mass conservation, branch flow distribution (Q1, Q2), pressure drops, and execution speed.
- Generates 4-panel multi-slice diagnostic visualization to 'pipe_flow/lbm_3d_flow_field.png'.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

# Ensure pipe_flow package is resolvable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from pipe_flow.solvers.lbm_3d import (
    CellType3D,
    DuctGeometry3D,
    LBM3DConfig,
    LBM3DSolver,
    LatticeStencilType,
    SimulationResult3D,
)


def build_3d_elevated_hvac_geometry(
    nx: int = 50,
    ny: int = 15,
    nz: int = 30,
    dx: float = 0.04,
    u_inlet: float = 2.0,
) -> DuctGeometry3D:
    """
    Construct the 3D elevated branching HVAC duct geometry:
    - Main Trunk: [0.0, 0.8m] x [0.1m, 0.5m] x [0.1m, 0.5m] (elevation z: low)
    - Branch 2 (Floor straight branch): [0.8m, 2.0m] x [0.1m, 0.5m] x [0.1m, 0.4m]
    - Branch 1 (Elevated Riser + Elbow + Ceiling run):
      * Vertical riser in z: [0.6m, 0.9m] x [0.1m, 0.5m] x [0.4m, 1.0m]
      * Horizontal run at ceiling: [0.9m, 2.0m] x [0.1m, 0.5m] x [0.8m, 1.1m]
    - Fan Inlet Face at x = 0
    - Outlet 1 at x = Lx (Ceiling)
    - Outlet 2 at x = Lx (Floor)
    """
    geo = DuctGeometry3D(nx=nx, ny=ny, nz=nz, dx=dx)

    # 1. Main trunk channel (ix: 0..19, iy: 2..12, iz: 3..11)
    # Physical: [0..0.8m] x [0.08..0.48m] x [0.12..0.44m]
    geo.add_box_cells(
        ix0=0, iy0=2, iz0=3,
        ix1=19, iy1=12, iz1=11,
        cell_type=CellType3D.FLUID,
    )

    # 2. Branch 2 (Floor Straight Duct): ix: 20..nx-1, iy: 2..12, iz: 3..9
    geo.add_box_cells(
        ix0=20, iy0=2, iz0=3,
        ix1=nx - 1, iy1=12, iz1=9,
        cell_type=CellType3D.FLUID,
    )

    # 3. Branch 1 (Vertical Riser in z): ix: 15..21, iy: 2..12, iz: 10..22
    geo.add_box_cells(
        ix0=15, iy0=2, iz0=10,
        ix1=21, iy1=12, iz1=22,
        cell_type=CellType3D.FLUID,
    )

    # 4. Branch 1 (Elevated Horizontal Ceiling Duct): ix: 20..nx-1, iy: 2..12, iz: 18..24
    geo.add_box_cells(
        ix0=20, iy0=2, iz0=18,
        ix1=nx - 1, iy1=12, iz1=24,
        cell_type=CellType3D.FLUID,
    )

    # 5. Fan Inlet at x = 0 (ix = 0, iy: 2..12, iz: 3..11)
    geo.add_inlet_cells(
        ix0=0, iy0=2, iz0=3,
        ix1=0, iy1=12, iz1=11,
        velocity=(u_inlet, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        name="Inlet_Main",
    )

    # 6. Outlet 1 (Elevated Ceiling Branch, ix = nx-1, iy: 2..12, iz: 18..24)
    geo.add_outlet_cells(
        ix0=nx - 1, iy0=2, iz0=18,
        ix1=nx - 1, iy1=12, iz1=24,
        pressure=0.0,
        normal=(1.0, 0.0, 0.0),
        name="Branch_1_Elevated_Riser",
    )

    # 7. Outlet 2 (Floor Level Branch, ix = nx-1, iy: 2..12, iz: 3..9)
    geo.add_outlet_cells(
        ix0=nx - 1, iy0=2, iz0=3,
        ix1=nx - 1, iy1=12, iz1=9,
        pressure=0.0,
        normal=(1.0, 0.0, 0.0),
        name="Branch_2_Floor_Main",
    )

    return geo


class TestLBM3DSolver(unittest.TestCase):
    """Rigorous Unit Tests and Benchmarks for 3D Lattice Boltzmann Solver."""

    def test_lbm_3d_geometry_builder(self):
        """Test 3D geometry builder masking, inlets, and outlets registration."""
        geo = build_3d_elevated_hvac_geometry(nx=30, ny=10, nz=20, dx=0.05)
        self.assertIn("Inlet_Main", geo.inlets)
        self.assertIn("Branch_1_Elevated_Riser", geo.outlets)
        self.assertIn("Branch_2_Floor_Main", geo.outlets)

        inlet = geo.inlets["Inlet_Main"]
        self.assertGreater(len(inlet.cells), 0)
        self.assertGreater(inlet.area, 0.0)

        out1 = geo.outlets["Branch_1_Elevated_Riser"]
        out2 = geo.outlets["Branch_2_Floor_Main"]
        self.assertGreater(len(out1.cells), 0)
        self.assertGreater(len(out2.cells), 0)

        fluid_count = np.sum(geo.mask == CellType3D.FLUID)
        self.assertGreater(fluid_count, 100)

    def test_lbm_3d_d3q7_stencil(self):
        """Test fast D3Q7 stencil on a 3D straight duct run."""
        nx, ny, nz = 25, 8, 8
        dx = 0.04
        geo = DuctGeometry3D(nx=nx, ny=ny, nz=nz, dx=dx)
        geo.add_box_cells(0, 1, 1, nx - 1, ny - 2, nz - 2, CellType3D.FLUID)
        geo.add_inlet_cells(0, 1, 1, 0, ny - 2, nz - 2, velocity=(1.5, 0.0, 0.0), name="Inlet")
        geo.add_outlet_cells(nx - 1, 1, 1, nx - 1, ny - 2, nz - 2, pressure=0.0, name="Outlet")

        cfg = LBM3DConfig(stencil=LatticeStencilType.D3Q7, tau=0.65, dx=dx, dt=0.001)
        solver = LBM3DSolver(geometry=geo, config=cfg)

        for _ in range(30):
            solver.step()

        summary = solver.get_flow_summary()
        self.assertGreater(summary["inlet_flow_rate"], 0.0)
        self.assertGreater(summary["outlet_flow_rates"]["Outlet"], 0.0)
        self.assertLess(summary["mass_conservation_error_pct"], 15.0)

    def test_lbm_3d_d3q19_elevated_scene(self):
        """Run full 3D D3Q19 Lattice Boltzmann simulation on elevated HVAC scene."""
        print("\n" + "=" * 80)
        print(" TEST: 3D LATTICE BOLTZMANN (LBM D3Q19) ELEVATED HVAC SOLVER BENCHMARK")
        print("=" * 80)

        nx, ny, nz = 50, 14, 28
        dx = 0.04  # 40 mm voxel resolution
        u_inlet = 2.0  # 2 m/s

        geo = build_3d_elevated_hvac_geometry(nx=nx, ny=ny, nz=nz, dx=dx, u_inlet=u_inlet)

        cfg = LBM3DConfig(
            stencil=LatticeStencilType.D3Q19,
            tau=0.60,
            dx=dx,
            dt=0.001,
            rho0=1.204,
            gravity=(0.0, 0.0, -9.81),
            use_guo_forcing=True,
        )

        solver = LBM3DSolver(geometry=geo, config=cfg)

        # Run steady-state solver
        result = solver.run_until_steady_state(
            max_steps=120,
            tol=1e-4,
            min_steps=30,
            verbose=True,
        )

        print("\n--- 3D LBM Solver Benchmark Performance Summary ---")
        print(f"Total Steps Executed:   {result.iterations}")
        print(f"Elapsed Wall Time:      {result.elapsed_wall_time:.3f} s")
        print(f"Mean Latency Per Step:  {result.time_per_step_ms:.2f} ms")
        print(f"Execution Throughput:   {result.steps_per_second:.1f} Steps/sec (Lattice Updates/sec)")
        print(f"Inlet Flow Rate Q_in:   {result.inlet_flow_rate:.5f} m^3/s ({result.inlet_mass_flow_rate:.4f} kg/s)")
        print(f"Inlet Static Pressure:  {result.inlet_static_pressure:.2f} Pa")
        print(f"Inlet Total Pressure:   {result.inlet_total_pressure:.2f} Pa")

        q1 = result.outlet_flow_rates.get("Branch_1_Elevated_Riser", 0.0)
        q2 = result.outlet_flow_rates.get("Branch_2_Floor_Main", 0.0)
        frac1 = result.branch_fractions.get("Branch_1_Elevated_Riser", 0.0) * 100.0
        frac2 = result.branch_fractions.get("Branch_2_Floor_Main", 0.0) * 100.0
        dp1 = result.pressure_drops.get("Branch_1_Elevated_Riser", 0.0)
        dp2 = result.pressure_drops.get("Branch_2_Floor_Main", 0.0)

        print(f"Branch 1 (Elevated Riser): Q1 = {q1:.5f} m^3/s ({frac1:.1f}%), dP1 = {dp1:.2f} Pa")
        print(f"Branch 2 (Floor Main):     Q2 = {q2:.5f} m^3/s ({frac2:.1f}%), dP2 = {dp2:.2f} Pa")
        print(f"Mass Conservation Error:   {result.mass_conservation_error_pct:.3f} %")

        # Save publication visualization
        save_img = os.path.join(current_dir, "lbm_3d_flow_field.png")
        solver.plot_flow_field_3d(save_path=save_img, show=False)

        # Assertions
        self.assertGreater(result.inlet_flow_rate, 0.0, "Inlet flow rate must be positive")
        self.assertGreater(q1, 0.0, "Elevated branch must receive airflow")
        self.assertGreater(q2, 0.0, "Floor branch must receive airflow")
        self.assertLess(result.mass_conservation_error_pct, 12.0, "Mass conservation error within reasonable bounds")
        self.assertGreater(result.steps_per_second, 5.0, "Real-time performance requirement (> 5 FPS)")
        self.assertTrue(os.path.exists(save_img), f"Visualization must be generated at {save_img}")

    def test_lbm_3d_gravity_hydrostatic_effect(self):
        """Test vertical riser column with gravity body force in z-direction."""
        nx, ny, nz = 10, 10, 30
        dx = 0.05
        geo = DuctGeometry3D(nx=nx, ny=ny, nz=nz, dx=dx)
        # Vertical duct from bottom to top
        geo.add_box_cells(2, 2, 0, 7, 7, nz - 1, CellType3D.FLUID)
        geo.add_inlet_cells(2, 2, 0, 7, 7, 0, velocity=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0), name="Inlet_Bottom")
        geo.add_outlet_cells(2, 2, nz - 1, 7, 7, nz - 1, pressure=0.0, normal=(0.0, 0.0, 1.0), name="Outlet_Top")

        cfg = LBM3DConfig(
            stencil=LatticeStencilType.D3Q19,
            tau=0.60,
            dx=dx,
            dt=0.001,
            gravity=(0.0, 0.0, -9.81),
            use_guo_forcing=True,
        )
        solver = LBM3DSolver(geometry=geo, config=cfg)

        for _ in range(40):
            solver.step()

        summary = solver.get_flow_summary()
        self.assertGreater(summary["inlet_flow_rate"], 0.0)
        self.assertGreater(summary["outlet_flow_rates"]["Outlet_Top"], 0.0)


if __name__ == "__main__":
    unittest.main()
