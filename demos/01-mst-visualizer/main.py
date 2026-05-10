"""
MST Visualizer — Kruskal's Algorithm (Pygame)
===============================================
Interactive: click to add vertices, drag to reposition.
Auto-plays through the algorithm, then exits cleanly.

Run: python main.py
Controls: [N]ext step  [A]uto  [R]eset  [C]lear  [F]ast mode  [Escape] quit
          Click dropdown to select preset  [G]enerate random
"""

import json
import os
import random
import pygame
import sys
import math
from typing import List, Tuple, Optional

# ─── Config ─────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1000, 650
CANVAS_LEFT = 320
CANVAS_TOP = 20
CANVAS_W = WIDTH - CANVAS_LEFT - 20
CANVAS_H = HEIGHT - 60

WHITE        = (255, 255, 255)
BLACK        = (20, 20, 30)
DARK_BG      = (18, 18, 28)
CANVAS_BG    = (22, 22, 36)
PANEL_BG     = (28, 28, 42)
GRID_COLOR   = (30, 30, 50)
GRAY         = (100, 100, 120)
GRAY_DARK    = (60, 60, 80)
GRAY_LIGHT   = (50, 50, 70)
RED          = (220, 60, 60)
GREEN        = (60, 200, 100)
BLUE         = (80, 140, 255)
YELLOW       = (255, 200, 50)
LIGHT_BLUE   = (120, 180, 255)
CYAN         = (80, 220, 220)
MAGENTA      = (200, 100, 255)
VERTEX_COLOR = (70, 180, 255)
TEXT_COLOR   = (220, 220, 230)
MUTED        = (90, 90, 110)
BTN_BG       = (35, 35, 55)
BTN_HOVER    = (50, 50, 75)
BTN_BORDER   = (60, 60, 90)

RADIUS = 16
STEP_DELAY = 600     # ms between auto-steps (normal speed)
FAST_DELAY = 20      # ms between auto-steps (fast/benchmark mode)
HOLD_FINALE = 3000   # ms to hold final screen

FPS = 60


# ─── Preset Database ────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_PATH = os.path.join(SCRIPT_DIR, "presets.json")

def load_presets() -> list:
    try:
        with open(PRESETS_PATH, "r") as f:
            data = json.load(f)
        return data.get("presets", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load presets: {e}")
        return []

PRESETS = load_presets()


def scale_vertices(normalized: List[List[float]],
                   margin: float = 0.12) -> List[Tuple[float, float]]:
    """Scale normalized [0,1] coordinates to canvas pixel space."""
    effective_w = CANVAS_W * (1 - 2 * margin)
    effective_h = CANVAS_H * (1 - 2 * margin)
    offset_x = CANVAS_W * margin
    offset_y = CANVAS_H * margin
    return [(offset_x + v[0] * effective_w, offset_y + v[1] * effective_h)
            for v in normalized]


def generate_random(n: int, seed: int) -> List[Tuple[float, float]]:
    """Generate n random vertices using a fixed seed for reproducibility."""
    rng = random.Random(seed)
    margin = 0.10
    return [
        (round(rng.uniform(margin, 1.0 - margin), 3),
         round(rng.uniform(margin, 1.0 - margin), 3))
        for _ in range(n)
    ]


# ─── Dropdown Widget ────────────────────────────────────────────────────────

class PresetDropdown:
    """A simple clickable dropdown for preset selection."""

    def __init__(self, x: int, y: int, width: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = 26
        self.open = False
        self.selected_index = 0
        self.scroll_offset = 0
        self.max_visible = 10
        self.font_sm = pygame.font.SysFont("consolas", 13)

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_item_rect(self, idx: int) -> pygame.Rect:
        return pygame.Rect(
            self.x,
            self.y + self.height + (idx - self.scroll_offset) * 22,
            self.width,
            22
        )

    def draw(self, surface, presets: list):
        # Main box
        color = CYAN if self.open else BTN_BORDER
        pygame.draw.rect(surface, BTN_BG, self.get_rect())
        pygame.draw.rect(surface, color, self.get_rect(), 1)

        # Selected label
        label = self.font_sm.render(
            presets[self.selected_index].get("name", "Unknown")
            if presets else "No presets",
            True, TEXT_COLOR
        )
        surface.blit(label, (self.x + 6, self.y + 5))

        # Arrow
        arrow = "▾" if not self.open else "▴"
        arr_surf = self.font_sm.render(arrow, True, GRAY)
        surface.blit(arr_surf, (self.x + self.width - 18, self.y + 4))

        # Dropdown list
        if self.open and presets:
            visible = presets[self.scroll_offset : self.scroll_offset + self.max_visible]
            for i, p in enumerate(visible):
                idx = self.scroll_offset + i
                rect = self.get_item_rect(idx)
                if idx == self.selected_index:
                    pygame.draw.rect(surface, BTN_HOVER, rect)
                name_surf = self.font_sm.render(p.get("name", ""), True, TEXT_COLOR)
                surface.blit(name_surf, (rect.x + 4, rect.y + 3))

            # Scroll indicators
            if self.scroll_offset > 0:
                pygame.draw.polygon(surface, GRAY, [
                    (self.x + 6, self.y + self.height - 2),
                    (self.x + 12, self.y + self.height + 6),
                    (self.x, self.y + self.height + 6),
                ])
            if self.scroll_offset + self.max_visible < len(presets):
                pygame.draw.polygon(surface, GRAY, [
                    (self.x + 6, self.y + self.height + (len(visible)) * 22 + 2),
                    (self.x + 12, self.y + self.height + (len(visible)) * 22 - 6),
                    (self.x, self.y + self.height + (len(visible)) * 22 - 6),
                ])

    def handle_event(self, event) -> Optional[int]:
        """Returns selected preset index if selection changed, None otherwise."""
        if not PRESETS:
            return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check click on dropdown box
                if self.get_rect().collidepoint(event.pos):
                    self.open = not self.open
                    return None

                if self.open:
                    # Check click on an item
                    for idx in range(self.scroll_offset,
                                     min(len(PRESETS), self.scroll_offset + self.max_visible)):
                        if self.get_item_rect(idx).collidepoint(event.pos):
                            self.selected_index = idx
                            self.open = False
                            return idx

                # Click outside — close
                self.open = False

        elif event.type == pygame.MOUSEWHEEL and self.open:
            # Scroll in dropdown
            if event.y < 0 and self.scroll_offset > 0:
                self.scroll_offset -= 1
            elif event.y > 0 and self.scroll_offset + self.max_visible < len(PRESETS):
                self.scroll_offset += 1

        return None


# ─── DSU (Disjoint Set Union) ───────────────────────────────────────────────

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ─── Visualizer ─────────────────────────────────────────────────────────────

class MSTVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MST Visualizer — Kruskal's Algorithm")
        self.clock = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_md = pygame.font.SysFont("consolas", 18)
        self.font_lg = pygame.font.SysFont("consolas", 26, bold=True)

        self.vertices: List[Tuple[float, float]] = []
        self.n = 0
        self.edges: List[Tuple[int, int, float]] = []
        self.dsu: Optional[DSU] = None
        self.mst_edges: List[Tuple[int, int]] = []
        self.rejected_edges: List[Tuple[int, int]] = []
        self.current_idx = 0
        self.total_weight = 0.0
        self.status_msg = "Initializing..."
        self.complete = False
        self.step_time = 0
        self.waiting = True
        self.frames_saved = 0
        self.dragging = -1

        # Preset and dropdown
        self.preset_index = -1
        self.preset_label = ""
        self.random_seed = 42
        self.dropdown = PresetDropdown(panel_x := 10, 65, 280)

        # Speed mode
        self.fast_mode = False

        # Startup delay
        self.startup_timer = 1500
        self.startup_done = False

        self._load_preset(0)

    # ── Preset Management ────────────────────────────────────────────────

    def _load_preset(self, idx: int):
        if not PRESETS:
            self.status_msg = "No presets.json found — use click/drag."
            return
        idx = idx % len(PRESETS)
        self.dropdown.selected_index = idx
        p = PRESETS[idx]
        raw = p.get("vertices", [])
        seed = p.get("seed")
        # Generate on-the-fly if preset defines a seed but no vertices
        if not raw and seed is not None:
            n = p.get("count", 20)  # default count for generated presets
            raw = generate_random(n, seed)
        if not raw:
            return
        self.preset_index = idx
        self.preset_label = p.get("name", f"Preset {idx}")
        self.vertices = scale_vertices(raw)
        self.n = len(self.vertices)
        self._reset_algorithm()
        self._draw()

    def _load_random(self, n: int = None, seed: int = None):
        if n is None:
            n = random.randint(5, 12)
        if seed is None:
            seed = self.random_seed
        else:
            self.random_seed = seed
        raw = generate_random(n, seed)
        self.preset_index = -1
        self.preset_label = f"Random n={n} seed={seed}"
        self.dropdown.selected_index = -1
        self.vertices = scale_vertices(raw)
        self.n = len(self.vertices)
        self._reset_algorithm()
        self._draw()

    def _next_preset(self):
        if PRESETS:
            self._load_preset(self.dropdown.selected_index + 1)

    def _prev_preset(self):
        if PRESETS:
            self._load_preset(self.dropdown.selected_index - 1)

    # ── Distance / Graph helpers ──────────────────────────────────────────

    def _dist(self, i: int, j: int) -> float:
        x1, y1 = self.vertices[i]
        x2, y2 = self.vertices[j]
        return math.hypot(x2 - x1, y2 - y1)

    def _build_edges(self):
        self.edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.edges.append((i, j, self._dist(i, j)))
        self.edges.sort(key=lambda e: e[2])

    def _init_dsu(self):
        self.dsu = DSU(self.n)
        self.mst_edges = []
        self.rejected_edges = []
        self.current_idx = 0
        self.total_weight = 0.0
        self.complete = False
        self.waiting = True
        delay = FAST_DELAY if self.fast_mode else self.startup_timer
        self.step_time = pygame.time.get_ticks() + delay
        self.startup_done = False

    def _reset_algorithm(self):
        self.n = len(self.vertices)
        self._build_edges()
        self._init_dsu()
        if self.n >= 2:
            self.status_msg = (f"Ready — {len(self.edges)} edges. "
                               f"[N]ext / [A]uto / [F]ast")
        else:
            self.status_msg = "Need at least 2 vertices."

    # ── Clear ─────────────────────────────────────────────────────────────

    def clear_all(self):
        self.vertices = []
        self.n = 0
        self.edges = []
        self.preset_index = -1
        self.preset_label = ""
        self._init_dsu()
        self.status_msg = "Cleared. Click canvas to add vertices."
        self._draw()

    # ── Hit test ──────────────────────────────────────────────────────────

    def _hit(self, mx: int, my: int) -> int:
        for i, (vx, vy) in enumerate(self.vertices):
            dx = vx - mx
            dy = vy - my
            if dx * dx + dy * dy < (RADIUS + 6) ** 2:
                return i
        return -1

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_grid(self):
        for x in range(0, CANVAS_W, 40):
            pygame.draw.line(self.screen, GRID_COLOR,
                             (CANVAS_LEFT + x, CANVAS_TOP),
                             (CANVAS_LEFT + x, CANVAS_TOP + CANVAS_H))
        for y in range(0, CANVAS_H, 40):
            pygame.draw.line(self.screen, GRID_COLOR,
                             (CANVAS_LEFT, CANVAS_TOP + y),
                             (CANVAS_LEFT + CANVAS_W, CANVAS_TOP + y))

    def _draw_vertices(self):
        highlight = set()
        if self.current_idx < len(self.edges):
            e = self.edges[self.current_idx]
            highlight = {e[0], e[1]}

        for i, (x, y) in enumerate(self.vertices):
            px, py = CANVAS_LEFT + x, CANVAS_TOP + y
            color = YELLOW if i in highlight else VERTEX_COLOR
            pygame.draw.circle(self.screen, color, (int(px), int(py)), RADIUS)
            pygame.draw.circle(self.screen, WHITE, (int(px), int(py)), RADIUS, 2)
            label = self.font_sm.render(str(i), True, BLACK)
            self.screen.blit(label, (int(px - label.get_width() // 2),
                                      int(py - label.get_height() // 2)))

    def _draw_edges(self):
        for i, (u, v, w) in enumerate(self.edges):
            if i < self.current_idx:
                continue
            x1, y1 = self.vertices[u]
            x2, y2 = self.vertices[v]
            pygame.draw.line(self.screen, GRAY_DARK,
                             (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                             (CANVAS_LEFT + x2, CANVAS_TOP + y2), 1)

        for (u, v) in self.rejected_edges:
            x1, y1 = self.vertices[u]
            x2, y2 = self.vertices[v]
            pygame.draw.line(self.screen, RED,
                             (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                             (CANVAS_LEFT + x2, CANVAS_TOP + y2), 2)
            mid = ((x1 + x2) // 2, (y1 + y2) // 2)
            pygame.draw.line(self.screen, RED,
                             (CANVAS_LEFT + mid[0] - 5, CANVAS_TOP + mid[1] - 5),
                             (CANVAS_LEFT + mid[0] + 5, CANVAS_TOP + mid[1] + 5), 2)
            pygame.draw.line(self.screen, RED,
                             (CANVAS_LEFT + mid[0] + 5, CANVAS_TOP + mid[1] - 5),
                             (CANVAS_LEFT + mid[0] - 5, CANVAS_TOP + mid[1] + 5), 2)

        for (u, v) in self.mst_edges:
            x1, y1 = self.vertices[u]
            x2, y2 = self.vertices[v]
            pygame.draw.line(self.screen, GREEN,
                             (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                             (CANVAS_LEFT + x2, CANVAS_TOP + y2), 3)

        if self.current_idx < len(self.edges):
            u, v, w = self.edges[self.current_idx]
            x1, y1 = self.vertices[u]
            x2, y2 = self.vertices[v]
            pygame.draw.line(self.screen, YELLOW,
                             (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                             (CANVAS_LEFT + x2, CANVAS_TOP + y2), 3)
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            pw = self.font_sm.render(f"w={w:.1f}", True, YELLOW)
            self.screen.blit(pw, (CANVAS_LEFT + mx - pw.get_width() // 2,
                                   CANVAS_TOP + my - 12))
            pulse = 2 + int(2 * math.sin(pygame.time.get_ticks() * 0.005))
            pygame.draw.circle(self.screen, YELLOW,
                               (CANVAS_LEFT + x1, CANVAS_TOP + y1),
                               RADIUS + pulse, 2)
            pygame.draw.circle(self.screen, YELLOW,
                               (CANVAS_LEFT + x2, CANVAS_TOP + y2),
                               RADIUS + pulse, 2)

    def _draw_panel(self):
        global panel_x  # referenced in dropdown init
        panel_x = 10
        y = 10

        self.screen.blit(self.font_lg.render("KRUSKAL'S MST", True, CYAN),
                         (panel_x, y))
        y += 32

        self.screen.blit(self.font_md.render(f"[ {self.preset_label} ]", True, MAGENTA),
                         (panel_x, y))
        y += 24

        # Legend
        for col, txt in [(GREEN, "MST"), (RED, "Rejected"),
                          (YELLOW, "Current"), (VERTEX_COLOR, "Vertex")]:
            pygame.draw.rect(self.screen, col, (panel_x, y, 12, 12))
            self.screen.blit(self.font_sm.render(txt, True, TEXT_COLOR),
                             (panel_x + 18, y + 2))
            y += 20

        y += 8
        pygame.draw.line(self.screen, GRAY_DARK, (panel_x, y),
                         (CANVAS_LEFT - 12, y), 1)
        y += 12

        # Stats
        step_label = "FAST" if self.fast_mode else f"{self.current_idx}/{len(self.edges)}"
        for label, val in [
            ("Vertices", str(self.n)),
            ("Edges", str(len(self.edges))),
            ("Progress", step_label),
            ("In MST", str(len(self.mst_edges))),
            ("Rejected", str(len(self.rejected_edges))),
            ("MST weight", f"{self.total_weight:.1f}"),
        ]:
            self.screen.blit(self.font_md.render(f"{label:>10}: {val}",
                           True, WHITE), (panel_x, y))
            y += 24

        y += 8
        pygame.draw.line(self.screen, GRAY_DARK, (panel_x, y),
                         (CANVAS_LEFT - 12, y), 1)
        y += 12

        # Status
        words = self.status_msg.split()
        line = ""
        max_w = CANVAS_LEFT - 24
        for w in words:
            test = line + (" " if line else "") + w
            if self.font_md.size(test)[0] > max_w:
                self.screen.blit(self.font_md.render(line, True, YELLOW),
                                 (panel_x, y))
                y += 20
                line = w
            else:
                line = test
        if line:
            self.screen.blit(self.font_md.render(line, True, YELLOW),
                             (panel_x, y))
            y += 20

        y += 10
        controls = [
            "[N]ext   [A]uto   [R]eset   [C]lear   [F]ast",
            "[,] prev  [.] next  [G] random",
            "Click dropdown above to pick preset",
        ]
        for c in controls:
            self.screen.blit(self.font_sm.render(c, True, MUTED),
                             (panel_x, y))
            y += 18

        # Dropdown
        y += 8
        dropdown_y = y + 10
        self.dropdown.x = panel_x
        self.dropdown.y = dropdown_y
        self.dropdown.draw(self.screen, PRESETS)

        # DSU roots
        y = dropdown_y + self.dropdown.height + (self.dropdown.max_visible * 22 if self.dropdown.open else 0) + 20
        if y > CANVAS_H - 20:
            y = CANVAS_H - 20
        if self.dsu:
            pygame.draw.line(self.screen, GRAY_DARK, (panel_x, y - 6),
                             (CANVAS_LEFT - 12, y - 6), 1)
            self.screen.blit(self.font_sm.render("DSU Components:", True, MUTED),
                             (panel_x, y))
            y += 18
            roots = {}
            for i in range(self.n):
                r = self.dsu.find(i)
                roots.setdefault(r, []).append(i)
            for r, members in roots.items():
                s = f"  root {r} <- {members}"
                self.screen.blit(self.font_sm.render(s, True, LIGHT_BLUE),
                                 (panel_x, y))
                y += 18

    def _draw(self):
        self.screen.fill(DARK_BG)
        self._draw_grid()
        self._draw_edges()
        self._draw_vertices()
        pygame.draw.line(self.screen, GRAY, (CANVAS_LEFT - 5, 0),
                         (CANVAS_LEFT - 5, HEIGHT), 2)
        self._draw_panel()
        pygame.display.flip()

    # ── Step Logic ───────────────────────────────────────────────────────

    def do_step(self):
        if self.dsu is None or self.current_idx >= len(self.edges):
            return False
        u, v, w = self.edges[self.current_idx]
        if self.dsu.union(u, v):
            self.mst_edges.append((u, v))
            self.total_weight += w
            self.status_msg = f"Edge ({u}—{v}) accepted  w={w:.1f}  MST={self.total_weight:.1f}"
        else:
            self.rejected_edges.append((u, v))
            self.status_msg = f"Edge ({u}—{v}) rejected  w={w:.1f}  (cycle)"
        self.current_idx += 1
        if self.current_idx >= len(self.edges):
            expected = self.n - 1 if self.n > 0 else 0
            if len(self.mst_edges) == expected:
                self.status_msg = f"✓ DONE — MST complete! Weight={self.total_weight:.1f}"
            else:
                self.status_msg = "Graph disconnected; MST incomplete."
            self.complete = True
        return True

    # ── Main Loop ────────────────────────────────────────────────────────

    def run(self):
        self._draw()

        while True:
            self.clock.tick(FPS)
            now = pygame.time.get_ticks()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._cleanup()
                    return

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    cx, cy = mx - CANVAS_LEFT, my - CANVAS_TOP

                    # Check if dropdown was clicked
                    dresult = self.dropdown.handle_event(ev)
                    if dresult is not None:
                        self._load_preset(dresult)
                        continue

                    if 0 <= cx <= CANVAS_W and 0 <= cy <= CANVAS_H:
                        idx = self._hit(cx, cy)
                        if idx >= 0:
                            self.dragging = idx
                        else:
                            self.vertices.append((cx, cy))
                            self.n = len(self.vertices)
                            self.preset_index = -1
                            self.preset_label = "(custom)"
                            self._reset_algorithm()
                            self._draw()

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = -1

                elif ev.type == pygame.MOUSEMOTION:
                    if self.dragging >= 0:
                        mx, my = ev.pos
                        cx = max(0, min(CANVAS_W, mx - CANVAS_LEFT))
                        cy = max(0, min(CANVAS_H, my - CANVAS_TOP))
                        self.vertices[self.dragging] = (cx, cy)
                        self.preset_index = -1
                        self.preset_label = "(custom)"
                        self._reset_algorithm()
                        self._draw()

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_n and not self.complete and self.n >= 2:
                        self.waiting = False
                        self.do_step()
                        self._draw()
                        if not self.complete:
                            self.waiting = True
                            delay = FAST_DELAY if self.fast_mode else STEP_DELAY
                            self.step_time = now + delay

                    elif ev.key in (pygame.K_a, pygame.K_SPACE):
                        if self.complete:
                            pass
                        elif self.n < 2:
                            self.status_msg = "Need at least 2 vertices."
                        else:
                            self.waiting = False

                    elif ev.key == pygame.K_r:
                        if self.n >= 2:
                            self._reset_algorithm()
                            self._draw()

                    elif ev.key == pygame.K_c:
                        self.clear_all()

                    elif ev.key == pygame.K_f:
                        self.fast_mode = not self.fast_mode
                        mode_name = "FAST" if self.fast_mode else "NORMAL"
                        self.status_msg = f"Speed mode: {mode_name}"
                        self._draw()
                        if not self.complete and self.n >= 2:
                            self.waiting = False

                    elif ev.key == pygame.K_PERIOD:
                        self._next_preset()
                    elif ev.key == pygame.K_COMMA:
                        self._prev_preset()

                    elif ev.key == pygame.K_g:
                        self.random_seed += 1
                        n = random.choice([4, 5, 6, 7, 8, 10])
                        self._load_random(n=n, seed=self.random_seed)
                        self.status_msg = f"Random {n} vertices (seed={self.random_seed}). [N] or [A] to start."

                    elif ev.key == pygame.K_ESCAPE:
                        self._cleanup()
                        return

            # Auto-advance
            if not self.waiting and not self.complete and self.n >= 2:
                if now >= self.step_time:
                    self.do_step()
                    self._draw()
                    if not self.complete:
                        delay = FAST_DELAY if self.fast_mode else STEP_DELAY
                        self.step_time = now + delay
                    else:
                        pygame.time.wait(HOLD_FINALE if not self.fast_mode else 500)
                        self._cleanup()
                        return

    def _cleanup(self):
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    v = MSTVisualizer()
    v.run()