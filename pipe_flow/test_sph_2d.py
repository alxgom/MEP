"""
Benchmark and Verification Test Suite for Candidate 11: 2D SPH Particle Solver.

Evaluates:
1. SPH latency scaling vs. particle count N (500, 1000, 2500, 5000).
2. Frames per second (FPS) and execution speed in Python vs. projected C++/Rust/GPU speeds.
3. Flow split across branches and continuous particle recirculation.
4. Generates high-resolution visualization saving to 'pipe_flow/visualizations/field_candidate_11_sph.png'.
"""

import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipe_flow.solvers.sph_2d import SPHSolver2D, SPHConfig2D


def run_sph_benchmarks():
    print("=" * 80)
    print(" BENCHMARKING CANDIDATE 11: 2D SMOOTHED PARTICLE HYDRODYNAMICS (SPH)")
    print("=" * 80)

    # 1. Main Simulation Run (N ~ 1,500 - 2,500 particles)
    cfg = SPHConfig2D(
        kernel_radius=0.045,
        particle_spacing=0.028,
        sound_speed=35.0,
        u_inlet=7.0735,
        enable_darcy_drag=True,
    )
    solver = SPHSolver2D(config=cfg)
    print(f"Initialized SPH solver with {solver.num_particles} initial particles (dt = {solver.dt*1000.0:.3f} ms)...")

    # Run for 150 steps
    res = solver.run(num_steps=150, verbose=True)

    print("\n--- SPH Simulation Benchmark Summary ---")
    print(f"Active Particles:       {res.num_particles:,}")
    print(f"Mean Latency per Step:  {res.time_per_step_ms:.2f} ms ({res.fps:.1f} FPS in pure Python/NumPy)")
    print(f"Total Spawned / Exited: Spawned: {solver.total_spawned:,} | Exited B1: {solver.total_exited_b1:,} | Exited B2: {solver.total_exited_b2:,}")
    print(f"Branch Flow Split:      Branch 1 (Elbow): {res.branch1_fraction_pct:.1f}% | Branch 2 (Straight): {res.branch2_fraction_pct:.1f}%")
    print(f"Mean Particle Speed:    {res.mean_speed:.2f} m/s (Max: {res.max_speed:.2f} m/s)")

    # 2. Scaling Benchmark across Particle Counts
    print("\n--- Particle Count Scaling Test (Python vs. C++/Rust vs. GPU Compute) ---")
    spacings = [0.045, 0.035, 0.025, 0.018]
    scaling_results = []

    for sp in spacings:
        c = SPHConfig2D(particle_spacing=sp, kernel_radius=sp * 1.5)
        s = SPHSolver2D(config=c)
        r = s.run(num_steps=30, verbose=False)
        
        # Theoretical projection factors:
        # C++/Rust/SIMD is typically 15x-30x faster than pure Python loops
        # GPU (CUDA / WebGPU / compute shaders) is typically 100x-300x faster
        ms_python = r.time_per_step_ms
        ms_cpp = ms_python / 22.0
        ms_gpu = ms_python / 180.0
        
        fps_python = 1000.0 / max(1e-4, ms_python)
        fps_cpp = 1000.0 / max(1e-4, ms_cpp)
        fps_gpu = 1000.0 / max(1e-4, ms_gpu)
        
        scaling_results.append({
            "particles": r.num_particles,
            "ms_python": ms_python,
            "fps_python": fps_python,
            "ms_cpp": ms_cpp,
            "fps_cpp": fps_cpp,
            "ms_gpu": ms_gpu,
            "fps_gpu": fps_gpu,
        })
        print(f"  * N = {r.num_particles:5d} | Python: {ms_python:6.2f} ms ({fps_python:5.1f} FPS) | C++/Rust: {ms_cpp:5.2f} ms ({fps_cpp:6.1f} FPS) | WebGPU/CUDA: {ms_gpu:5.3f} ms ({fps_gpu:7.1f} FPS)")

    # 3. Save Visualization Plot
    vis_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    plot_path = os.path.join(vis_dir, "field_candidate_11_sph.png")
    solver.render_plot(save_path=plot_path)

    return res, scaling_results


if __name__ == "__main__":
    run_sph_benchmarks()
