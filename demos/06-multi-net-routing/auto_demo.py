"""
Headless Multi-Net Tournament Runner
====================================
Generates a random 3-net configuration and saves a comparison plot.
Used for visual verification of collision-avoidance logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import MultiNetEnvironment
from solver import MultiNetSolver

def run_auto_benchmark():
    print("Initializing Automated Headless Benchmark...")
    
    # 1. Generate 3 Random Nets (10 terminals each)
    np.random.seed(42) # Reproducible randomness
    nets = {
        "Net_1": np.random.randint(100, 700, (10, 2)),
        "Net_2": np.random.randint(100, 700, (10, 2)),
        "Net_3": np.random.randint(100, 700, (10, 2))
    }
    
    # 2. Build Environment and Solver
    env = MultiNetEnvironment(nets)
    solver = MultiNetSolver(env)
    
    # 3. Solve using the 4 Contenders
    print("\n--- Solver Metrics ---")
    print("Solving Negotiated...")
    res_neg, len_neg, iss_neg = solver.solve_negotiated()
    print(f"  [NEGOTIATED]  Total Length: {len_neg:.2f} | Collisions/Fails: {iss_neg}")
    
    print("Solving Surgical Hybrid (Ideal)...")
    res_hyb, len_hyb, iss_hyb = solver.solve_hybrid_ripup()
    print(f"  [SURGICAL]    Total Length: {len_hyb:.2f} | Collisions/Fails: {iss_hyb}")

    print("Solving Negotiated Hybrid (Soft)...")
    res_nhb, len_nhb, iss_nhb = solver.solve_negotiated_hybrid()
    print(f"  [NEG-HYBRID]  Total Length: {len_nhb:.2f} | Collisions/Fails: {iss_nhb}")
    
    print("Solving Global Permutation...")
    res_per, len_per, iss_per = solver.solve_best_permutation()
    print(f"  [GLOBAL PERM] Total Length: {len_per:.2f} | Collisions/Fails: {iss_per}")
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    plt.subplots_adjust(top=0.85)
    
    titles = [
        f"1. Negotiated (Hard)\nLen: {len_neg:.1f} | Issues: {iss_neg}",
        f"2. Surgical (Ideal)\nLen: {len_hyb:.1f} | Issues: {iss_hyb}",
        f"3. Neg-Hybrid (Soft)\nLen: {len_nhb:.1f} | Issues: {iss_nhb}",
        f"4. Global Permutation\nLen: {len_per:.1f} | Issues: {iss_per}"
    ]
    
    results = [res_neg, res_hyb, res_nhb, res_per]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"] # Green, Blue, Orange, Purple
    
    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx], fontweight='bold')
        ax.set_xlim(-100, 900)
        ax.set_ylim(-100, 900)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        sol = results[idx]
        if sol:
            for net_idx, (net_name, data) in enumerate(sol.items()):
                color = colors[net_idx % len(colors)]
                # Draw Terminals
                t = nets[net_name]
                ax.scatter(t[:,0], t[:,1], c=color, s=30, zorder=5, edgecolors='white')
                
                # Draw Segments
                for u, v in data["segments"]:
                    p1, p2 = env.nodes[u], env.nodes[v]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.8, zorder=2)
                    
        ax.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle("Automated Multi-Net Routing Tournament (v4.3)", fontsize=16, fontweight='bold')
    out_path = "tournament_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"Result saved to: {out_path}")

if __name__ == "__main__":
    run_auto_benchmark()
