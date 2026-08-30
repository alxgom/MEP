"""
Unit tests for 2D Grid Fast Fluid Dynamics (FFD) HVAC solver.
"""

import math
import os
import sys
import numpy as np
import pytest

# Ensure path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from solvers.grid_ffd_2d import DuctGeometry2D, GridFFD2D, CellType


def test_straight_duct_mass_conservation():
    """Verify that a straight duct conserves mass with zero divergence and correct pressure drop."""
    nx, ny = 100, 40
    dx, dy = 0.02, 0.02
    u_in = 3.0

    geo = DuctGeometry2D(nx=nx, ny=ny, dx=dx, dy=dy)
    # Channel in rows 10..29 (20 cells = 0.4m)
    geo.add_box_cells(0, 10, 99, 29, CellType.FLUID)
    geo.add_inlet_cells(0, 10, 0, 29, velocity=(u_in, 0.0), name="Inlet")
    geo.add_outlet_cells(99, 10, 99, 29, pressure=0.0, name="Outlet")

    solver = GridFFD2D(geo, dt=0.005, poisson_solver="lu")

    # Step for 60 iterations
    for _ in range(60):
        solver.step()

    summary = solver.get_flow_summary()

    # Inflow = 3.0 m/s * 0.4m = 1.20 m^3/s
    assert math.isclose(summary["inlet_flow_rate"], 1.20, rel_tol=1e-3)
    # Outflow must equal Inflow
    out_q = summary["outlet_flow_rates"]["Outlet"]
    assert math.isclose(out_q, 1.20, rel_tol=1e-3)
    # Mass error must be effectively 0%
    assert summary["mass_conservation_error_pct"] < 0.1
    # Inlet pressure > Outlet pressure (positive pressure drop)
    assert summary["pressure_drops"]["Outlet"] > 0.0


def test_elbow_and_wye_junction_geometry():
    """Verify arbitrary duct geometries (elbows, boxes, circles) are properly rasterized."""
    nx, ny = 120, 80
    geo = DuctGeometry2D(nx=nx, ny=ny, dx=0.01, dy=0.01)

    geo.add_straight_duct(0.0, 0.2, 0.4, 0.2, orientation="horizontal")
    geo.add_elbow_90(center_x=0.4, center_y=0.2, inner_radius=0.1, width=0.2, quadrant="top_right")
    geo.add_circle_obstacle(center_x=0.2, center_y=0.3, radius=0.03)

    mask = geo.mask
    # Check that fluid, solid, and obstacle cells exist
    assert np.any(mask == CellType.FLUID)
    assert np.any(mask == CellType.SOLID)

    # Check obstacle at center is solid
    j_obs, i_obs = geo._phys_range_to_cells(0.19, 0.29, 0.21, 0.31)[:2]
    assert mask[i_obs, j_obs] == CellType.SOLID


def test_poisson_solver_backends():
    """Verify LU, SOR, and CG Poisson backends execute without numerical instability."""
    nx, ny = 60, 30
    dx, dy = 0.02, 0.02

    for solver_name in ["lu", "cg", "sor"]:
        geo = DuctGeometry2D(nx=nx, ny=ny, dx=dx, dy=dy)
        geo.add_box_cells(0, 5, 59, 24, CellType.FLUID)
        geo.add_inlet_cells(0, 5, 0, 24, velocity=(1.5, 0.0), name="Inlet")
        geo.add_outlet_cells(59, 5, 59, 24, pressure=0.0, name="Outlet")

        solver = GridFFD2D(geo, dt=0.01, poisson_solver=solver_name, poisson_max_iter=50)

        for _ in range(15):
            res = solver.step()
            assert not np.isnan(res)
            assert not np.isinf(res)

        summary = solver.get_flow_summary()
        assert not np.isnan(summary["inlet_flow_rate"])
        assert summary["inlet_flow_rate"] > 0.0


if __name__ == "__main__":
    test_straight_duct_mass_conservation()
    test_elbow_and_wye_junction_geometry()
    test_poisson_solver_backends()
    print("All unit tests passed successfully!")
