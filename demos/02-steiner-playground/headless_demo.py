"""
Steiner Tree Demo — Fully Headless (no display required)
=========================================================
Generates an animated GIF of the Steiner tree optimization using matplotlib.

Run: python headless_demo.py
Output: steiner_optimization.gif
"""

import os
import math
import random
from io import BytesIO

import imageio.v2 as iio
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from steiner_solver import SteinerTreeSolver


# ── Problem Setup ────────────────────────────────────────────────────────────

TERMINALS = [
    (0.2, 0.3), (0.8, 0.3), (0.5, 0.7),
    (0.35, 0.15), (0.65, 0.15),
]
N_STEINER = 3
SEED = 42

solver = SteinerTreeSolver(TERMINALS, n_steiner=N_STEINER, seed=SEED)
solver._compute_mst()

# ── Visualization Setup ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#12121c")
ax.set_facecolor("#12121c")
margin = 0.15
ax.set_xlim(-margin, 1 + margin)
ax.set_ylim(-margin, 1 + margin)
ax.set_aspect("equal")
ax.axis("off")

ax.text(
    0.5, 1.08,
    "Steiner Tree Optimization — Gradient Descent + SA",
    fontsize=16, fontweight="bold", color="#5eb8ff",
    ha="center", va="center", fontfamily="monospace",
    transform=ax.transAxes,
)


def _to_arr(pt):
    return np.array(pt)


def render_frame(solver, step, highlight_edge=None, status="",
                 show_forces=True, show_angles=True):
    ax.clear()
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#12121c")
    ax.set_facecolor("#12121c")

    ax.text(
        0.5, 1.08,
        "Steiner Tree Optimization — Gradient Descent + SA",
        fontsize=16, fontweight="bold", color="#5eb8ff",
        ha="center", va="center", fontfamily="monospace",
        transform=ax.transAxes,
    )

    points = solver.points
    n = len(points)
    n_t = solver.n_terminals

    # Candidate edges dimly
    for i in range(n):
        for j in range(i + 1, n):
            a = _to_arr(points[i])
            b = _to_arr(points[j])
            lc = LineCollection([np.stack([a, b])], colors="#3a3a50", linewidths=1, alpha=0.4)
            ax.add_collection(lc)

    # MST edges
    for u, v in solver.mst_edges:
        a = _to_arr(points[u])
        b = _to_arr(points[v])
        lc = LineCollection([np.stack([a, b])], colors="#3cff70", linewidths=2.5, alpha=0.9)
        ax.add_collection(lc)

    # Highlight current edge
    if highlight_edge is not None:
        u, v = highlight_edge
        a = _to_arr(points[u])
        b = _to_arr(points[v])
        lc = LineCollection([np.stack([a, b])], colors="#ffc832", linewidths=3.5, alpha=1.0)
        ax.add_collection(lc)

    # Force arrows on Steiner points
    if show_forces and step > 0:
        steiner_start = n_t
        for i in range(steiner_start, n):
            fx, fy = solver._compute_gradient(i)
            mag = math.hypot(fx, fy)
            if mag < 1e-8:
                continue
            px, py = points[i]
            scale = 0.04
            ax.annotate(
                "", xy=(px + fx * scale, py + fy * scale),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="#ff6464", lw=1.5, alpha=0.8),
            )

    # 120° angle arcs at Steiner points
    if show_angles:
        steiner_start = n_t
        for i in range(steiner_start, n):
            nb_indices = solver.adj.get(i, [])
            if len(nb_indices) < 2:
                continue
            px, py = points[i][0], points[i][1]
            
            # Sort neighbors by polar angle
            sorted_nb = sorted(nb_indices, key=lambda idx: math.atan2(points[idx][1]-py, points[idx][0]-px))
            
            for j in range(len(sorted_nb)):
                nj = sorted_nb[j]
                nk = sorted_nb[(j + 1) % len(sorted_nb)]
                ax_ = points[nj][0] - px
                ay_ = points[nj][1] - py
                bx_ = points[nk][0] - px
                by_ = points[nk][1] - py
                
                ang_a = math.atan2(ay_, ax_)
                ang_b = math.atan2(by_, bx_)
                diff = ang_b - ang_a
                while diff < 0: diff += 2 * math.pi
                while diff >= 2 * math.pi: diff -= 2 * math.pi
                
                actual = math.degrees(diff)
                deviation = abs(actual - 120.0)
                color = "#3cff70" if deviation < 5 else ("#ffa500" if deviation < 15 else "#ff4040")
                
                arc_radius = 0.04
                theta = np.linspace(ang_a, ang_a + diff, 50)
                arc_x = px + arc_radius * np.cos(theta)
                arc_y = py + arc_radius * np.sin(theta)
                ax.plot(arc_x, arc_y, color=color, linewidth=1.5, alpha=0.8)
                
                mid_angle = ang_a + diff / 2
                label_r = arc_radius + 0.025
                lx = px + label_r * math.cos(mid_angle)
                ly = py + label_r * math.sin(mid_angle)
                ax.text(lx, ly, f"{actual:.1f}°", fontsize=7, color=color,
                        ha="center", va="center", fontfamily="monospace")

    # Vertices
    for i in range(n):
        px, py = points[i]
        is_terminal = i < n_t
        color = "#46b4ff" if is_terminal else "#ffb450"
        ec = "#ffffff"
        radius = 0.04 if is_terminal else 0.03
        circle = plt.Circle((px, py), radius, color=color, ec=ec, linewidth=2, zorder=5)
        ax.add_patch(circle)
        label_char = f"T{i}" if is_terminal else f"S{i - n_t}"
        ax.text(px, py, label_char, fontsize=9, fontweight="bold",
                ha="center", va="center", color="#12121c", fontfamily="monospace", zorder=6)

    # Stats
    panel_x = 0.62
    panel_y = 0.92
    stats_left = [
        f"Terminals : {n_t}",
        f"Steiner   : {n - n_t}",
        f"Edges     : {len(solver.mst_edges)}",
    ]
    for j, s in enumerate(stats_left):
        ax.text(panel_x, panel_y - j * 0.04, s, fontsize=10, color="#c0c0d0",
                fontfamily="monospace", va="top", ha="left", transform=ax.transAxes)

    stats_right = [
        f"MST Weight: {solver.mst_weight:.4f}",
        f"Iteration : {solver.iteration}",
    ]
    if solver.history:
        last = solver.history[-1]
        stats_right.append(f"Max Force : {last['max_force']:.6f}")
        stats_right.append(f"120° Dev  : {last['max_120_deviation']:.2f}°")
    for j, s in enumerate(stats_right):
        ax.text(panel_x + 0.15, panel_y - j * 0.04, s, fontsize=10, color="#c0c0d0",
                fontfamily="monospace", va="top", ha="left", transform=ax.transAxes)

    # Status
    status_color = "#3cff70" if "DONE" in status or "CONVERGED" in status else "#ffc832"
    ax.text(0.5, -0.06, status, fontsize=11, fontweight="bold", color=status_color,
            ha="center", va="center", fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#3cff70", edgecolor="#3cff70", label="MST edge"),
        mpatches.Patch(facecolor="#ff6464", edgecolor="#ff6464", label="Force vector"),
        mpatches.Patch(facecolor="#46b4ff", edgecolor="#ffffff", label="Terminal"),
        mpatches.Patch(facecolor="#ffb450", edgecolor="#ffffff", label="Steiner point"),
    ]
    legend = ax.legend(handles=legend_items, loc="lower right", facecolor="#1a1a2e",
                       edgecolor="#404060", fontsize=8, labelcolor="#c0c0d0", framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = iio.imread(buf)
    buf.close()
    return img


# ── Run Optimization + Generate Frames ──────────────────────────────────────

frames = []

# Initial frame
img = render_frame(solver, 0, status="Initializing...")
frames.append(img)
for _ in range(10):
    frames.append(img.copy())

# Run optimization with fewer iterations for speed
MAX_ITER = 50
for iteration in range(MAX_ITER):
    prev_weight = solver.mst_weight
    solver.step_gradient(learning_rate=0.1, sa_temperature=0.2 * (0.98 ** iteration))

    record = solver.history[-1]
    status = (f"iter {solver.iteration} | weight={solver.mst_weight:.4f} | "
              f"force={record['max_force']:.6f} | "
              f"120° dev={record['max_120_deviation']:.2f}°")

    img = render_frame(solver, iteration + 1, status=status)
    frames.append(img)

    # Hold on significant events
    extra_hold = 3 if record["merges"] > 0 else 0
    for _ in range(2 + extra_hold):
        frames.append(img.copy())

    # Check convergence
    if record["max_force"] < 0.01 and record["merges"] == 0:
        break

# Final hold
img = render_frame(solver, solver.iteration,
                   status=f"✓ CONVERGED — Weight={solver.mst_weight:.4f}")
frames.append(img)
for _ in range(20):
    frames.append(img.copy())

# ── Generate GIF ─────────────────────────────────────────────────────────────

out_path = "steiner_optimization.gif"
iio.mimsave(out_path, frames, duration=0.08, loop=0)

print(f"GIF saved:          {os.path.abspath(out_path)}")
print(f"Frames:             {len(frames)}")
print(f"Iterations:         {solver.iteration}")
print(f"Terminals:          {solver.n_terminals}")
print(f"Steiner points:     {len(solver.points) - solver.n_terminals}")
print(f"MST weight:         {solver.mst_weight:.4f}")
print(f"Final MST edges:    {solver.mst_edges}")
print(f"Max 120° deviation: {solver.history[-1]['max_120_deviation']:.3f}°")

plt.close(fig)
