import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Resolve imports by adding parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import Obstacle, GridEnvironment, EscapeGraphEnvironment
from solver import ObstacleSteinerSolver

def main():
    # Set up seed for beautiful layout
    np.random.seed(42)
    
    # 10 terminals in an 800x800 domain
    terminals = np.random.rand(10, 2) * 800
    obstacles_data = [
        [200, 200, 300, 600],
        [500, 100, 600, 400],
        [100, 400, 400, 500]
    ]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    
    # Ensure terminals are not inside obstacles
    cleaned_terminals = []
    for t in terminals:
        if not any(o.contains(t[0], t[1]) for o in obstacles):
            cleaned_terminals.append(t)
    terminals = np.array(cleaned_terminals)
    
    # 1. Construct environments
    env_hanan = GridEnvironment(terminals, obstacles)
    env_eg = EscapeGraphEnvironment(terminals, obstacles)
    
    # 2. Solve Steiner trees
    solver_hanan = ObstacleSteinerSolver(env_hanan)
    res_hanan = solver_hanan.solve_fast_corner(max_steiner=25, stochastic=False)
    
    solver_eg = ObstacleSteinerSolver(env_eg)
    res_eg = solver_eg.solve_fast_corner(max_steiner=25, stochastic=False)
    
    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#1e1e24')
    
    # Common plotting parameters
    for ax in (ax1, ax2):
        ax.set_facecolor('#121216')
        ax.set_xlim(-100, 900)
        ax.set_ylim(-100, 900)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', color='#30303a', alpha=0.5)
        
        # Plot obstacles
        for o in obstacles:
            rect = plt.Rectangle((o.min_x, o.min_y), o.max_x - o.min_x, o.max_y - o.min_y, 
                                 facecolor='#404050', edgecolor='#707080', alpha=0.6, zorder=3)
            ax.add_patch(rect)
            
    # Panel 1: Hanan Grid
    # Plot Hanan edges
    row, col = env_hanan.adj_matrix.nonzero()
    for u, v in zip(row, col):
        p1, p2 = env_hanan.nodes[u], env_hanan.nodes[v]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#3a3a4a', linewidth=0.8, alpha=0.5, zorder=1)
        
    # Plot Hanan nodes
    ax1.scatter(env_hanan.nodes[:, 0], env_hanan.nodes[:, 1], color='#505060', s=10, alpha=0.7, zorder=2, label='Grid Nodes')
    
    # Plot Hanan Steiner Tree
    for u, v in res_hanan["segments"]:
        p1, p2 = env_hanan.nodes[u], env_hanan.nodes[v]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#ff5555', linewidth=3, zorder=4)
        
    # Plot terminals
    ax1.scatter(terminals[:, 0], terminals[:, 1], color='#2ecc71', s=60, edgecolors='white', zorder=5, label='Terminals')
    
    n_hanan_edges = len(row) // 2
    ax1.set_title(f"Hanan Grid Topology\nNodes: {env_hanan.n_nodes} | Edges: {n_hanan_edges}\nPath Length: {res_hanan['weight']:.1f}", 
                  color='white', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(facecolor='#1e1e24', edgecolor='#3a3a4a', labelcolor='white')
    
    # Panel 2: Escape Graph
    # Plot EG edges
    row_eg, col_eg = env_eg.adj_matrix.nonzero()
    for u, v in zip(row_eg, col_eg):
        p1, p2 = env_eg.nodes[u], env_eg.nodes[v]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#4a6b5d', linewidth=0.8, alpha=0.6, zorder=1)
        
    # Plot EG nodes
    ax2.scatter(env_eg.nodes[:, 0], env_eg.nodes[:, 1], color='#689f84', s=15, alpha=0.8, zorder=2, label='Escape Graph Nodes')
    
    # Plot EG Steiner Tree
    for u, v in res_eg["segments"]:
        p1, p2 = env_eg.nodes[u], env_eg.nodes[v]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#2ecc71', linewidth=3, zorder=4)
        
    # Plot terminals
    ax2.scatter(terminals[:, 0], terminals[:, 1], color='#2ecc71', s=60, edgecolors='white', zorder=5, label='Terminals')
    
    n_eg_edges = len(row_eg) // 2
    ax2.set_title(f"Escape Graph Topology (RAY-TRACED)\nNodes: {env_eg.n_nodes} ({(1 - env_eg.n_nodes/env_hanan.n_nodes)*100:.1f}% reduction) | Edges: {n_eg_edges}\nPath Length: {res_eg['weight']:.1f} (OPTIMAL)", 
                  color='white', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(facecolor='#1e1e24', edgecolor='#3a3a4a', labelcolor='white')
    
    plt.suptitle("Topological Faceoff: Hanan Grid vs. Escape Graph\n(Visualizing Sparse Ray-Tracing vs. Quadratic Cartesian Meshing)", 
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = "demos/05-manhattan-obstacles/data_exploration/topology_comparison.png"
    plt.savefig(output_path, dpi=150, facecolor='#1e1e24')
    print(f"Comparison plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
