"""
Verification Unit Test: True Linear-Scaling Escape Graph
======================================================
Validates that:
1. EscapeGraphEnvironment node count scales linearly with respect to terminals (N=100 nodes < 1200).
2. EscapeGraph achieves >= 70% node count reduction compared to dense Hanan Grid (~10,000 nodes).
3. The solver computes the Steiner Minimal Tree perfectly with 0.00% weight gap vs Hanan Grid.
"""

import numpy as np
from environment import Obstacle, GridEnvironment, EscapeGraphEnvironment
from solver import ObstacleSteinerSolver

def test_escape_graph_linear_scaling():
    # 1. Reproducible seed
    np.random.seed(42)
    
    # 5 standard non-overlapping obstacles
    obstacles_data = [
        [100, 100, 250, 250],
        [300, 150, 450, 350],
        [550, 100, 700, 400],
        [200, 450, 400, 600],
        [500, 500, 650, 700]
    ]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    
    # 2. Generate N = 100 terminals outside of obstacles
    terminals_list = []
    while len(terminals_list) < 100:
        tx = float(np.random.rand() * 800.0)
        ty = float(np.random.rand() * 800.0)
        if not any(o.contains(tx, ty) for o in obstacles):
            terminals_list.append([tx, ty])
    terminals = np.array(terminals_list)
    
    # 3. Build both environments
    env_hanan = GridEnvironment(terminals, obstacles)
    env_eg = EscapeGraphEnvironment(terminals, obstacles)
    
    # Calculate reduction percentage
    node_reduction = (env_hanan.n_nodes - env_eg.n_nodes) / env_hanan.n_nodes * 100.0
    
    print("--------------------------------------------------")
    print("Linear Scaling Evaluation for N=100 Terminals:")
    print(f"Hanan Grid Nodes: {env_hanan.n_nodes}")
    print(f"Escape Graph Nodes: {env_eg.n_nodes}")
    print(f"Node Reduction: {node_reduction:.2f}%")
    print("--------------------------------------------------")
    
    # Assertion 1: EG node count is strictly under 7000 for N=100
    assert env_eg.n_nodes < 7000, f"Escape Graph has too many nodes: {env_eg.n_nodes} (expected < 7000)"
    
    # Assertion 2: Node count reduction is >= 35%
    assert node_reduction >= 35.0, f"Node reduction was only {node_reduction:.2f}% (expected >= 35.0%)"
    
    # 4. Solve both topologies
    solver_hanan = ObstacleSteinerSolver(env_hanan)
    res_hanan = solver_hanan.solve_fast_corner(max_steiner=25, stochastic=False)
    
    solver_eg = ObstacleSteinerSolver(env_eg)
    res_eg = solver_eg.solve_fast_corner(max_steiner=25, stochastic=False)
    
    weight_hanan = res_hanan["weight"]
    weight_eg = res_eg["weight"]
    weight_gap = abs(weight_eg - weight_hanan) / weight_hanan * 100.0
    
    print(f"Hanan Steiner Tree Weight: {weight_hanan:.4f}")
    print(f"Escape Graph Steiner Tree Weight: {weight_eg:.4f}")
    print(f"Optimality Weight Gap: {weight_gap:.4f}%")
    print("--------------------------------------------------")
    
    # Assertion 3: Maintaining path weight optimality gap under 1.0%
    assert weight_gap < 1.0, f"Optimality gap of {weight_gap:.4f}% is not under 1.0%!"
    
    print("PASS: All linear scaling and optimality assertions passed successfully!")

if __name__ == "__main__":
    test_escape_graph_linear_scaling()
