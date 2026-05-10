"""
MST Demo — Fully Headless (no display required)
================================================
Generates an animated GIF of Kruskal's algorithm using only numpy + matplotlib.
Run: python headless_demo.py
Output: mst_kruskal.gif
"""

import math
import os
from io import BytesIO

import imageio.v2 as iio
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must be before pyplot import — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ── DSU ────────────────────────────────────────────────────────────────────


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ── Problem Setup ──────────────────────────────────────────────────────────
# 6 vertices in an interesting hexagonal-ish layout
vertices = np.array(
    [[1.5, 0.0], [4.5, 0.0], [3.0, 3.0], [2.0, 1.2], [4.0, 1.2], [3.0, 2.0]],
    dtype=float,
)
n = len(vertices)

edges = []
for i in range(n):
    for j in range(i + 1, n):
        w = np.linalg.norm(vertices[i] - vertices[j])
        edges.append((i, j, w))
edges.sort(key=lambda e: e[2])

dsu = DSU(n)
mst = []
rejected = []
total_weight = 0.0

# ── Visualization Setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#12121c")
ax.set_facecolor("#12121c")
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.8, 3.8)
ax.set_aspect("equal")
ax.axis("off")

ax.text(
    2.5,
    3.6,
    "Kruskal's MST Algorithm",
    fontsize=20,
    fontweight="bold",
    color="#5eb8ff",
    ha="center",
    va="center",
    fontfamily="monospace",
)

# ── Render Function ────────────────────────────────────────────────────────


def render_frame(step, highlight_edge=None, status=""):
    ax.clear()
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#12121c")
    ax.set_facecolor("#12121c")

    ax.text(
        2.5,
        3.6,
        "Kruskal's MST Algorithm",
        fontsize=20,
        fontweight="bold",
        color="#5eb8ff",
        ha="center",
        va="center",
        fontfamily="monospace",
    )

    # Candidate edges (gray, dim)
    for i, (u, v, w) in enumerate(edges):
        if i < step:
            continue
        lc = LineCollection(
            [vertices[[u, v]]], colors="#3a3a50", linewidths=1, alpha=0.7
        )
        ax.add_collection(lc)

    # MST edges (green, thick)
    for u, v, w in mst:
        lc = LineCollection(
            [vertices[[u, v]]], colors="#3cff70", linewidths=3.5, alpha=0.9
        )
        ax.add_collection(lc)

    # Rejected edges (red)
    for u, v in rejected:
        lc = LineCollection(
            [vertices[[u, v]]], colors="#ff4040", linewidths=2, alpha=0.6
        )
        ax.add_collection(lc)
        mid = (vertices[u] + vertices[v]) / 2
        ax.plot(mid[0], mid[1], "x", color="#ff4040", markersize=12, markeredgewidth=2)

    # Current edge (yellow, highlighted)
    if highlight_edge is not None:
        u, v, w = highlight_edge
        lc = LineCollection(
            [vertices[[u, v]]], colors="#ffc832", linewidths=3.5, alpha=1.0
        )
        ax.add_collection(lc)
        mid = (vertices[u] + vertices[v]) / 2
        ax.text(
            mid[0],
            mid[1] + 0.18,
            f"{w:.1f}",
            fontsize=11,
            color="#ffc832",
            fontweight="bold",
            ha="center",
            va="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#12121c",
                edgecolor="#ffc832",
                alpha=0.9,
            ),
        )

    # Vertices
    for i in range(n):
        color = "#c8e8ff"
        edge_color = "#ffffff"
        if highlight_edge and (i == highlight_edge[0] or i == highlight_edge[1]):
            color = "#ffe080"
            edge_color = "#ffc832"
        circle = plt.Circle(
            vertices[i], 0.28, color=color, ec=edge_color, linewidth=2.5, zorder=5
        )
        ax.add_patch(circle)
        ax.text(
            vertices[i][0],
            vertices[i][1],
            str(i),
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            color="#12121c",
            fontfamily="monospace",
            zorder=6,
        )

    # Stats panel
    panel_x = 4.55
    panel_y_top = 3.2
    stats_left = [f"Vertices : {n}", f"Edges    : {len(edges)}", f"Step     : {step}/{len(edges)}"]
    for j, s in enumerate(stats_left):
        ax.text(panel_x, panel_y_top - j * 0.25, s, fontsize=11, color="#c0c0d0",
                fontfamily="monospace", va="top", ha="left")

    stats_right = [f"In MST  : {len(mst)}", f"Reject  : {len(rejected)}", f"Weight  : {total_weight:.1f}"]
    for j, s in enumerate(stats_right):
        ax.text(panel_x, panel_y_top - 1.0 - j * 0.25, s, fontsize=11, color="#c0c0d0",
                fontfamily="monospace", va="top", ha="left")

    # Status
    status_color = "#3cff70" if "DONE" in status else "#ffc832"
    ax.text(2.5, -0.55, status, fontsize=13, fontweight="bold", color=status_color,
            ha="center", va="center", fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#3cff70", edgecolor="#3cff70", label="MST edge"),
        mpatches.Patch(facecolor="#ff4040", edgecolor="#ff4040", label="Rejected (cycle)"),
        mpatches.Patch(facecolor="#ffc832", edgecolor="#ffc832", label="Considering"),
        mpatches.Patch(facecolor="#c8e8ff", edgecolor="#ffffff", label="Vertex"),
    ]
    legend = ax.legend(handles=legend_items, loc="lower left", facecolor="#1a1a2e",
                       edgecolor="#404060", fontsize=8, labelcolor="#c0c0d0", framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    # Render to buffer
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = iio.imread(buf)
    buf.close()
    return img


# ── Run Algorithm + Generate Frames ────────────────────────────────────────
frames = []

# Initial frame
img = render_frame(0, status="Initializing...")
frames.append(img)
for _ in range(15):
    frames.append(img.copy())

for idx, (u, v, w) in enumerate(edges):
    highlight = (u, v, w)

    if dsu.union(u, v):
        mst.append((u, v, w))
        total_weight += w
        status = f"✓ Edge {u}—{v} accepted (w={w:.1f})"
    else:
        rejected.append((u, v))
        status = f"✗ Edge {u}—{v} rejected (cycle!)"

    img = render_frame(idx + 1, highlight_edge=highlight, status=status)
    frames.append(img)
    for _ in range(8):  # hold each step
        frames.append(img.copy())

# Final hold
img = render_frame(len(edges), status="✓ DONE — MST complete!")
frames.append(img)
for _ in range(30):
    frames.append(img.copy())

# ── Generate GIF ───────────────────────────────────────────────────────────
out_path = "mst_kruskal.gif"
iio.mimsave(out_path, frames, duration=0.08, loop=0)

print(f"GIF saved:     {os.path.abspath(out_path)}")
print(f"Frames:        {len(frames)}")
print(f"MST edges:     {len(mst)} / {n - 1} expected")
print(f"MST weight:    {total_weight:.2f}")
print(f"MST edge set:  {[(u, v) for u, v, w in mst]}")

plt.close(fig)