"""
MST Auto-Demo (Enhanced) — generates a GIF so you can watch Kruskal's algorithm run.
Run: python auto_demo.py
Output: mst_demo.gif in the current directory
"""

import pygame
import sys
import math
import os

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# --- DSU ---
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


# --- Setup ---
WIDTH, HEIGHT = 1000, 650
CANVAS_LEFT = 320
CANVAS_TOP = 20
CANVAS_W = WIDTH - CANVAS_LEFT - 20
CANVAS_H = HEIGHT - 60

os.makedirs("frames", exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MST Auto-Demo")
font_sm = pygame.font.SysFont("consolas", 14)
font_md = pygame.font.SysFont("consolas", 18)
font_lg = pygame.font.SysFont("consolas", 26, bold=True)
clock = pygame.time.Clock()

# Nice vertex layout (6 points in an interesting pattern)
vertices = [
    (150, 480),
    (650, 480),
    (400, 100),
    (250, 250),
    (550, 250),
    (400, 380),
]
n = len(vertices)


def dist(i, j):
    return math.hypot(
        vertices[i][0] - vertices[j][0], vertices[i][1] - vertices[j][1]
    )


# Build and sort all edges
edges = []
for i in range(n):
    for j in range(i + 1, n):
        edges.append((i, j, dist(i, j)))
edges.sort(key=lambda e: e[2])

dsu = DSU(n)
mst_edges = []
rejected_edges = []
total_weight = 0.0
step = 0
frame = 0
done = False

FPS = 30
HOLD_FRAMES = 60  # hold final frame for 2 seconds at 30fps

running = True
while running:
    clock.tick(FPS)
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # --- Logic ---
    if not done and step < len(edges):
        u, v, w = edges[step]
        if dsu.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            status = "ACCEPTED"
        else:
            rejected_edges.append((u, v))
            status = "rejected (cycle)"
        step += 1
    else:
        done = True
        status = "DONE"

    # --- Draw ---
    screen.fill((18, 18, 28))

    # Canvas background
    pygame.draw.rect(screen, (22, 22, 36),
                     (CANVAS_LEFT - 2, CANVAS_TOP - 2, CANVAS_W + 4, CANVAS_H + 4))

    # Grid
    for x in range(0, CANVAS_W, 40):
        pygame.draw.line(screen, (35, 35, 55),
                         (CANVAS_LEFT + x, CANVAS_TOP),
                         (CANVAS_LEFT + x, CANVAS_TOP + CANVAS_H))
    for y in range(0, CANVAS_H, 40):
        pygame.draw.line(screen, (35, 35, 55),
                         (CANVAS_LEFT, CANVAS_TOP + y),
                         (CANVAS_LEFT + CANVAS_W, CANVAS_TOP + y))

    # All candidate edges (dim)
    for i, (u, v, w) in enumerate(edges):
        if i < step:
            continue
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        pygame.draw.line(screen, (50, 50, 70),
                         (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                         (CANVAS_LEFT + x2, CANVAS_TOP + y2), 1)

    # Rejected edges (red, translucent)
    for (u, v) in rejected_edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        pygame.draw.line(screen, (180, 60, 60),
                         (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                         (CANVAS_LEFT + x2, CANVAS_TOP + y2), 2)

    # MST edges (green, thick)
    for (u, v, w) in mst_edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        pygame.draw.line(screen, (60, 200, 100),
                         (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                         (CANVAS_LEFT + x2, CANVAS_TOP + y2), 3)

    # Current edge being considered (yellow highlight)
    if not done and step < len(edges):
        u, v, w = edges[step]
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        pygame.draw.line(screen, (255, 200, 50),
                         (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                         (CANVAS_LEFT + x2, CANVAS_TOP + y2), 3)
        # Pulsing glow on endpoints
        r = RADIUS = 16
        pulse = 3 + int(2 * math.sin(pygame.time.get_ticks() * 0.01))
        pygame.draw.circle(screen, (255, 200, 50),
                           (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                           r + pulse, 2)
        pygame.draw.circle(screen, (255, 200, 50),
                           (CANVAS_LEFT + x2, CANVAS_TOP + y2),
                           r + pulse, 2)

    # Vertices
    for i, (x, y) in enumerate(vertices):
        px, py = CANVAS_LEFT + x, CANVAS_TOP + y
        pygame.draw.circle(screen, (70, 180, 255), (px, py), RADIUS)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), RADIUS, 2)
        label = font_sm.render(str(i), True, (0, 0, 0))
        screen.blit(label, (px - label.get_width() // 2,
                            py - label.get_height() // 2))

    # Edge weight labels on candidate edges
    if not done and step < len(edges):
        u, v, w = edges[step]
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        wlabel = font_sm.render(f"w={w:.0f}", True, (255, 200, 50))
        screen.blit(wlabel, (CANVAS_LEFT + mx - wlabel.get_width() // 2,
                             CANVAS_TOP + my - 10))

    # Divider line
    pygame.draw.line(screen, (50, 50, 70), (CANVAS_LEFT - 2, 0),
                     (CANVAS_LEFT - 2, HEIGHT), 2)

    # --- Panel ---
    px = 10
    py = 10

    screen.blit(font_lg.render("KRUSKAL'S", True, (80, 180, 255)), (px, py))
    py += 32

    # Legend
    legend = [
        ((60, 200, 100), "MST edge"),
        ((180, 60, 60), "Rejected"),
        ((255, 200, 50), "Considering"),
        ((70, 180, 255), "Vertex"),
    ]
    for col, txt in legend:
        pygame.draw.rect(screen, col, (px, py, 12, 12))
        screen.blit(font_sm.render(txt, True, (200, 200, 210)), (px + 18, py))
        py += 20

    py += 8
    pygame.draw.line(screen, (50, 50, 70), (px, py), (CANVAS_LEFT - 12, py), 1)
    py += 12

    # Stats
    stats = [
        f"Vertices  : {n}",
        f"Edges     : {len(edges)}",
        f"Step      : {step}/{len(edges)}",
        f"In MST    : {len(mst_edges)}",
        f"Rejected  : {len(rejected_edges)}",
        f"MST weight: {total_weight:.1f}",
    ]
    for s in stats:
        screen.blit(font_md.render(s, True, (220, 220, 230)), (px, py))
        py += 24

    py += 8
    pygame.draw.line(screen, (50, 50, 70), (px, py), (CANVAS_LEFT - 12, py), 1)
    py += 12

    # Status
    status_surface = font_md.render(f"Status: {status}", True,
                                     (60, 200, 100) if done else (255, 200, 50))
    screen.blit(status_surface, (px, py))
    py += 24

    # DSU info
    roots = {}
    for i in range(n):
        roots.setdefault(dsu.find(i), []).append(i)
    py += 4
    screen.blit(font_sm.render("Components:", True, (120, 120, 140)), (px, py))
    py += 18
    for r, members in roots.items():
        s = f"  root {r} <- {members}"
        screen.blit(font_sm.render(s, True, (140, 180, 255)), (px, py))
        py += 18

    # Controls hint at bottom
    controls = "Controls: [N]ext step  [A]uto  [R]eset  [C]lear"
    screen.blit(font_sm.render(controls, True, (80, 80, 100)),
                (10, HEIGHT - 22))

    pygame.display.flip()

    # Save frame
    pygame.image.save(screen, "frames/frame_{:04d}.png".format(frame))
    frame += 1

    # Hold final frames
    if done:
        for _ in range(HOLD_FRAMES):
            clock.tick(FPS)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            pygame.display.flip()
            pygame.image.save(screen, "frames/frame_{:04d}.png".format(frame))
            frame += 1
        break

    # Slow down steps so we can see each one
    if not done:
        pygame.time.wait(400)

pygame.quit()

# Generate GIF
if HAS_IMAGEIO:
    frames = sorted([f for f in os.listdir("frames") if f.endswith(".png")])
    images = [imageio.imread("frames/{}".format(f)) for f in frames]
    # Duplicate last frame a few times so the result lingers
    images.extend([images[-1]] * 10)
    imageio.mimsave("mst_demo.gif", images, duration=0.12, loop=0)
    print("GIF saved to mst_demo.gif ({} frames)".format(len(frames)))
else:
    print("{} PNG frames saved to frames/".format(frame))
    print("Install imageio to auto-generate a GIF: pip install imageio")

print("MST total weight: {:.1f}".format(total_weight))
print("MST edges: {}".format([(u, v) for u, v, w in mst_edges]))