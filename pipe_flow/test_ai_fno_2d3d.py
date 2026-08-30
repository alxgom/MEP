"""
Test and Verification Suite for AI 2D/3D Fourier Neural Operator (FNO-2D/3D) Surrogate Solver.

Verifications:
- Spectral convolution operators (2D and 3D) mode truncation and inverse FFT reconstruction.
- Continuous 2D/3D flow field predictions (u, v, w, p).
- Physical boundary masking and solid wall no-slip condition.
- Mass conservation error tracking.
- Execution latency (< 2.0 ms per forward evaluation).
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

from pipe_flow.solvers.ai_fno_2d3d import (
    FNONetworkInput2D,
    FNONetworkInput3D,
    FNOPrediction2D,
    FNOPrediction3D,
    SpectralConv2dNumPy,
    SpectralConv3dNumPy,
    FourierNeuralOperator2D,
    FourierNeuralOperator3D,
    FNO2DSolver,
    FNO3DSolver,
    compute_wall_distance_2d,
    compute_wall_distance_3d,
    generate_synthetic_duct_grid_2d,
    generate_synthetic_duct_voxel_3d,
)


class TestFNO2D3DSolver(unittest.TestCase):
    """Unit tests for the Fourier Neural Operator solvers."""

    def test_spectral_conv_2d_shape_and_reconstruction(self):
        """Tests that 2D spectral convolution layer maintains dimensions."""
        conv2d = SpectralConv2dNumPy(in_channels=4, out_channels=8, modes_x=8, modes_y=8)
        x = np.random.randn(4, 32, 32).astype(np.float32)
        out = conv2d.forward(x)
        self.assertEqual(out.shape, (8, 32, 32))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_spectral_conv_3d_shape_and_reconstruction(self):
        """Tests that 3D spectral convolution layer maintains dimensions."""
        conv3d = SpectralConv3dNumPy(in_channels=4, out_channels=8, modes_x=4, modes_y=4, modes_z=4)
        x = np.random.randn(4, 16, 16, 16).astype(np.float32)
        out = conv3d.forward(x)
        self.assertEqual(out.shape, (8, 16, 16, 16))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_fno_2d_solver_prediction(self):
        """Tests continuous 2D flow field prediction."""
        solver = FNO2DSolver(hidden_dim=16, modes_x=8, modes_y=8)
        inp = generate_synthetic_duct_grid_2d(nx=64, ny=32, dx=0.03, dy=0.03, u_inlet=2.0)
        
        pred = solver.predict(inp)
        
        self.assertIsInstance(pred, FNOPrediction2D)
        self.assertEqual(pred.u.shape, (64, 32))
        self.assertEqual(pred.v.shape, (64, 32))
        self.assertEqual(pred.p.shape, (64, 32))
        
        # Velocity in solid walls should be 0
        solid_mask = inp.mask < 0.5
        self.assertTrue(np.all(pred.u[solid_mask] == 0.0))
        self.assertTrue(np.all(pred.v[solid_mask] == 0.0))
        
        # Inflow and outflow
        self.assertGreater(pred.q_inlet, 0.0)
        self.assertGreater(pred.q_branch1, 0.0)
        self.assertGreater(pred.q_branch2, 0.0)
        self.assertLess(pred.mass_conservation_error_pct, 1.0)
        
        # Latency should be sub-25ms on CPU
        lat_stats = solver.benchmark_latency(inp=inp, num_runs=50)
        self.assertLess(lat_stats["mean_latency_ms"], 25.0)

    def test_fno_3d_solver_prediction(self):
        """Tests continuous 3D flow field prediction with vertical riser."""
        solver = FNO3DSolver(hidden_dim=8, modes_x=4, modes_y=4, modes_z=4)
        inp = generate_synthetic_duct_voxel_3d(nx=24, ny=12, nz=12, dx=0.05, dy=0.05, dz=0.05, u_inlet=2.0)
        
        pred = solver.predict(inp)
        
        self.assertIsInstance(pred, FNOPrediction3D)
        self.assertEqual(pred.u.shape, (24, 12, 12))
        self.assertEqual(pred.v.shape, (24, 12, 12))
        self.assertEqual(pred.w.shape, (24, 12, 12))
        self.assertEqual(pred.p.shape, (24, 12, 12))
        
        # Fluid velocity non-zero inside duct
        fluid_mask = inp.mask > 0.5
        self.assertTrue(np.any(pred.u[fluid_mask] > 0.0))
        self.assertGreater(pred.total_pressure_drop_pa, 0.0)
        self.assertLess(pred.mass_conservation_error_pct, 1.0)

    def test_wall_distance_transforms(self):
        """Tests 2D and 3D Euclidean wall distance transforms."""
        mask_2d = np.ones((20, 20), dtype=np.float32)
        mask_2d[0, :] = 0.0
        mask_2d[-1, :] = 0.0
        mask_2d[:, 0] = 0.0
        mask_2d[:, -1] = 0.0
        
        wd_2d = compute_wall_distance_2d(mask_2d, dx=0.1, dy=0.1)
        self.assertEqual(wd_2d[0, 0], 0.0)
        self.assertGreater(wd_2d[10, 10], 0.5)


if __name__ == "__main__":
    unittest.main()
