"""
Rigor Test: Full Hanan vs Fast Corner
=====================================
Quantifies the exact time/quality trade-off between the O(N^4) Full Grid 
and the O(N^2) Corner heuristic.
"""

import time
import numpy as np
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

def main():
    print("--- Colleague Rigor Test: N=50 ---")
    np.random.seed(42)
    terminals = np.random.rand(50, 2) * 800
    obstacles_data = [[200, 200, 300, 600], [500, 100, 600, 400], [100, 400, 400, 500]]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    env = GridEnvironment(terminals, obstacles)
    solver = ObstacleSteinerSolver(env)

    # 1. Test Fast Corner Kick
    print("Testing FAST CORNER KICK (Deterministic)...")
    start = time.time()
    res_fast = solver.solve_fast_corner(stochastic=False)
    t_fast = time.time() - start
    print(f"Fast Corner: Length={res_fast['weight']:.2f}, Time={t_fast:.4f}s")

    # 2. Test Full Hanan Greedy
    print("\nTesting FULL HANAN GREEDY (Deterministic)...")
    print(f"Candidate count: {env.n_nodes - len(solver.terminals)}")
    start = time.time()
    res_full = solver.solve_greedy() # Uses full grid candidates
    t_full = time.time() - start
    print(f"Full Hanan:  Length={res_full['weight']:.2f}, Time={t_full:.4f}s")

    # 3. Delta
    improvement = (res_fast['weight'] - res_full['weight']) / res_fast['weight'] * 100
    slowdown = t_full / t_fast
    
    print("\n--- The Verdict ---")
    print(f"Quality Gain of Full Grid: {improvement:.4f}%")
    print(f"Computational Cost:        {slowdown:.1f}x slower")

if __name__ == "__main__":
    main()
