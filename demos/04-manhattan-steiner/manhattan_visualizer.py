"""
Manhattan Steiner Visualizer
============================
Visualizes Rectilinear Steiner Trees with orthogonal routing.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Tuple, Dict, Any


def plot_manhattan_tree(
    terminals: np.ndarray,
    steiner_points: np.ndarray,
    edges: List[Tuple[int, int]],
    title: str = "Rectilinear Steiner Tree",
    save_path: str = None
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#f8f9fa")
    
    all_points = np.vstack([terminals, steiner_points]) if len(steiner_points) > 0 else terminals
    
    # 1. Draw Hanan Grid (faint background)
    unique_x = np.unique(terminals[:, 0])
    unique_y = np.unique(terminals[:, 1])
    
    for x in unique_x:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    for y in unique_y:
        ax.axhline(y, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
        
    # 2. Draw Rectilinear Edges
    lines = []
    for u, v in edges:
        p1, p2 = all_points[u], all_points[v]
        
        # If the edge is already orthogonal, just draw it
        if np.isclose(p1[0], p2[0]) or np.isclose(p1[1], p2[1]):
            lines.append([p1, p2])
        else:
            # If diagonal, draw an L-shape (X then Y)
            corner = np.array([p2[0], p1[1]])
            lines.append([p1, corner])
            lines.append([corner, p2])
            
    lc = LineCollection(lines, colors="#2ecc71", linewidths=3, alpha=0.8, zorder=1)
    ax.add_collection(lc)
    
    # 3. Draw Terminals
    ax.scatter(terminals[:, 0], terminals[:, 1], c="#3498db", s=100, 
               label="Terminals", edgecolors="white", zorder=3)
    
    # 4. Draw Steiner Points
    if len(steiner_points) > 0:
        ax.scatter(steiner_points[:, 0], steiner_points[:, 1], c="#e67e22", s=80, 
                   label="Steiner Junctions", edgecolors="white", marker="s", zorder=2)
        
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend()
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Test visualization
    t = np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8]])
    s = np.array([[0.2, 0.8]]) # Corner
    e = [(0, 3), (1, 2)] # This is just a dummy edge list
    plot_manhattan_tree(t, s, e, title="Manhattan Test Plot")
