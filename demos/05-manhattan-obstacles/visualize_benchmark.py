import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

def plot_to_ax(ax, env, solution, title, color_path="#2ecc71"):
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs.min_x, obs.min_y), 
                                  obs.max_x - obs.min_x, 
                                  obs.max_y - obs.min_y, 
                                  color='#e74c3c', alpha=0.3))
    lines = []
    for u, v in solution["segments"]:
        lines.append([env.nodes[u], env.nodes[v]])
    lc = LineCollection(lines, colors=color_path, linewidths=2, alpha=0.8)
    ax.add_collection(lc)
    if "steiner_indices" in solution and len(solution["steiner_indices"]) > 0:
        s_pts = env.nodes[solution["steiner_indices"]]
        ax.scatter(s_pts[:, 0], s_pts[:, 1], c="#e67e22", s=30, marker="s", label="Steiner", zorder=5)
    ax.scatter(env.terminals[:, 0], env.terminals[:, 1], c="#3498db", s=50, edgecolors="white", label="Terminals", zorder=6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.axis("off")

def main():
    print("Generating Enhanced Benchmark Visualizations (80 Trials)...")
    np.random.seed(42)
    obstacles_data = [[200, 200, 300, 600], [500, 100, 600, 400], [100, 400, 400, 500]]
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    terminals = []
    while len(terminals) < 40:
        p = np.random.rand(2) * 800
        if not any(o.contains(p[0], p[1]) for o in obstacles):
            terminals.append(p)
    terminals = np.array(terminals)
    env = GridEnvironment(terminals, obstacles)
    solver = ObstacleSteinerSolver(env)
    print("Computing Baseline MST...")
    sol_mst = solver.solve_mst()
    mst_w = sol_mst['weight']
    print("Computing Deterministic Greedy...")
    sol_greedy = solver.solve_fast_corner(stochastic=False)
    greedy_w = sol_greedy['weight']
    print("Running 80 Stochastic Trials...")
    best_stoch = None
    best_w = float('inf')
    weights = []
    for i in range(80):
        np.random.seed(i)
        res = solver.solve_fast_corner(stochastic=True, temperature=0.1)
        weights.append(res["weight"])
        if res["weight"] < best_w:
            best_w = res["weight"]; best_stoch = res
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#f8f9fa")
    plot_to_ax(axes[0, 0], env, sol_mst, f"1. Baseline MST\nLength: {mst_w:.2f}")
    plot_to_ax(axes[0, 1], env, sol_greedy, f"2. Deterministic Greedy\nLength: {greedy_w:.2f}")
    plot_to_ax(axes[1, 0], env, best_stoch, f"3. Best Stochastic (v1.6)\nLength: {best_w:.2f}")
    ax_hist = axes[1, 1]
    ax_hist.hist(weights, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(greedy_w, color='red', linestyle='dashed', linewidth=2, label=f'Greedy ({greedy_w:.1f})')
    ax_hist.axvline(best_w, color='green', linestyle='dashed', linewidth=2, label=f'Best Stoch ({best_w:.1f})')
    ax_hist.set_title(f"Stochastic Spread (80 trials)\nBaseline MST: {mst_w:.1f}", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel("Total Tree Length"); ax_hist.set_ylabel("Frequency"); ax_hist.legend(); ax_hist.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_comparison_80.png", dpi=150)
    print(f"\nFinal Stats:\nMST: {mst_w:.2f}\nGreedy: {greedy_w:.2f}\nBest: {best_w:.2f}")
    print("\nComparison visualization saved to benchmark_comparison_80.png")

if __name__ == "__main__":
    main()
