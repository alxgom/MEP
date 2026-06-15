import os
import sys
import numpy as np
import pytest

# Add parent directory of this test to path so we can import environment and solver
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

from environment import Obstacle, GridEnvironment, EscapeGraphEnvironment
from solver import ObstacleSteinerSolver

def test_stochastic_escape_behavior():
    """Verify GridEnvironment and EscapeGraphEnvironment build cleanly and solve_fast_corner runs without errors."""
    seeds = [42, 101, 2023]
    n_terminals = 30
    
    hanan_weights = []
    eg_weights = []
    
    # Standard obstacle configuration
    obstacles_data = [
        [200, 200, 300, 600],
        [500, 100, 600, 400],
        [100, 400, 400, 500]
    ]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Generate terminals that are not inside any obstacle
        terminals = []
        while len(terminals) < n_terminals:
            tx = float(np.random.rand() * 800.0)
            ty = float(np.random.rand() * 800.0)
            if not any(o.contains(tx, ty) for o in obstacles):
                terminals.append([tx, ty])
        terminals = np.array(terminals)
        
        # 1. Verify environment construction is clean and doesn't crash
        env_hanan = GridEnvironment(terminals, obstacles)
        env_eg = EscapeGraphEnvironment(terminals, obstacles)
        
        assert env_hanan.n_nodes > 0, "Hanan Grid should have nodes."
        assert env_eg.n_nodes > 0, "Escape Graph should have nodes."
        assert env_eg.n_nodes < env_hanan.n_nodes, "Escape Graph should reduce node count dramatically."
        
        # 2. Run solve_fast_corner(stochastic=True)
        # Hanan Solver
        solver_hanan = ObstacleSteinerSolver(env_hanan)
        np.random.seed(seed)
        res_hanan = solver_hanan.solve_fast_corner(max_steiner=25, stochastic=True)
        assert res_hanan["weight"] > 0, "Hanan solver should return a valid positive path weight."
        hanan_weights.append(res_hanan["weight"])
        
        # Escape Graph Solver
        solver_eg = ObstacleSteinerSolver(env_eg)
        np.random.seed(seed)
        res_eg = solver_eg.solve_fast_corner(max_steiner=25, stochastic=True)
        assert res_eg["weight"] > 0, "Escape Graph solver should return a valid positive path weight."
        eg_weights.append(res_eg["weight"])
        
        # Print for visibility
        print(f"Seed {seed} | Hanan Weight: {res_hanan['weight']:.2f} | EG Weight: {res_eg['weight']:.2f}")

    avg_hanan = np.mean(hanan_weights)
    avg_eg = np.mean(eg_weights)
    
    # Calculate weight gap: positive gap means EG is longer, negative means EG is shorter
    gap = (avg_eg - avg_hanan) / avg_hanan
    
    print(f"\n--- Aggregated Results (N={n_terminals}) ---")
    print(f"Average Hanan Weight: {avg_hanan:.4f}")
    print(f"Average Escape Graph Weight: {avg_eg:.4f}")
    print(f"Stochastic Weight Gap: {gap * 100.0:.4f}%")
    
    # 3. Assert average weight gap is highly competitive (within 1% tolerance, or Escape Graph is better)
    assert gap <= 0.01, f"Escape Graph weight is not within 1% of Hanan! Gap: {gap*100.0:.4f}%"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
