import numpy as np
from environment import Obstacle, GridEnvironment, EscapeGraphEnvironment
from solver import ObstacleSteinerSolver

def test_escape_graph_properties():
    # Set up a seed for reproducibility
    np.random.seed(42)
    
    # 30 terminals
    terminals = np.random.rand(30, 2) * 800
    obstacles_data = [[200, 200, 300, 600], [500, 100, 600, 400], [100, 400, 400, 500]]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    
    # Check that terminals aren't in obstacles
    cleaned_terminals = []
    for t in terminals:
        if not any(o.contains(t[0], t[1]) for o in obstacles):
            cleaned_terminals.append(t)
    terminals = np.array(cleaned_terminals)
    
    # 1. Construct environments
    env_hanan = GridEnvironment(terminals, obstacles)
    env_eg = EscapeGraphEnvironment(terminals, obstacles)
    
    # Assert nodes count is strictly less for EscapeGraph
    print(f"Hanan Grid Nodes: {env_hanan.n_nodes}")
    print(f"Escape Graph Nodes: {env_eg.n_nodes}")
    assert env_eg.n_nodes < env_hanan.n_nodes, "Escape Graph should have strictly fewer nodes than Hanan Grid!"
    
    # 2. Solve using same solver
    solver_hanan = ObstacleSteinerSolver(env_hanan)
    res_hanan = solver_hanan.solve_fast_corner(max_steiner=25, stochastic=False)
    
    solver_eg = ObstacleSteinerSolver(env_eg)
    res_eg = solver_eg.solve_fast_corner(max_steiner=25, stochastic=False)
    
    print(f"Hanan Steiner Weight: {res_hanan['weight']:.2f}")
    print(f"Escape Graph Steiner Weight: {res_eg['weight']:.2f}")
    
    # Assert Steiner weights are identical or very close (floating point tolerance)
    assert abs(res_hanan["weight"] - res_eg["weight"]) < 1e-3, f"Weights should be identical! Hanan={res_hanan['weight']:.4f}, EG={res_eg['weight']:.4f}"
    
    print("All assertions passed cleanly!")

if __name__ == "__main__":
    test_escape_graph_properties()
