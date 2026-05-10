"""
Tournament Visualizer
=====================
Generates side-by-side comparisons of different solver results.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os

def plot_comparison(case_name: str, results_file: str = "tournament_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
    
    # Filter results for the specific case
    case_results = [r for r in data if r["case"] == case_name]
    if not case_results:
        print(f"No results found for case: {case_name}")
        return

    n_algos = len(case_results)
    cols = 3
    rows = (n_algos + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    fig.patch.set_facecolor("#f0f0f5")
    axes = axes.flatten()
    
    for i, res in enumerate(case_results):
        ax = axes[i]
        points = np.array(res["points"])
        edges = res["edges"]
        
        # Plot edges
        lines = []
        for u, v in edges:
            lines.append([points[u], points[v]])
        lc = LineCollection(lines, colors="green", linewidths=2, alpha=0.7)
        ax.add_collection(lc)
        
        # Plot points
        n_terminals = len(points) - res["steiner_count"]
        ax.scatter(points[:n_terminals, 0], points[:n_terminals, 1], 
                   c="dodgerblue", s=80, label="Terminals", zorder=5)
        if res["steiner_count"] > 0:
            ax.scatter(points[n_terminals:, 0], points[n_terminals:, 1], 
                       c="orange", s=60, label="Steiner", zorder=5)
            
        ax.set_title(f"{res['algorithm']}\nLength: {res['length']:.4f} (Gap: {res['gap_pct']:.2f}%)", 
                     fontsize=12, fontweight='bold')
        ax.set_aspect("equal")
        ax.axis("off")
        
    # Hide unused axes
    for i in range(n_algos, len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    output_path = f"comparison_{case_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    # Example: plot comparison for the first case in results
    if os.path.exists("tournament_results.json"):
        with open("tournament_results.json", "r") as f:
            data = json.load(f)
            if data:
                plot_comparison(data[0]["case"])
    else:
        print("Run framework.py first to generate results.")
