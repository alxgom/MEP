"""
Steiner Point Playground — Interactive Pygame Demo
===================================================
Visualize gradient-descent optimization of Steiner points in real time.

Controls:
  [N]ext step   [A]uto play   [R]eset   [C]lear   [F]ast mode
  [G]enerate random   [+] add Steiner   [-] remove Steiner
  [,] prev preset   [.] next preset   [Escape] quit
  [[ / ]] decrease/increase Force Strength (Learning Rate)
  Click+drag vertices to reposition
  Scroll to zoom (canvas)

The solver runs the hybrid variational algorithm:
  1. MST topology snap (Kruskal's)
  2. Gradient descent on Steiner points (sum of unit vectors)
  3. Simulated Annealing thermal perturbation
  4. Annihilation (merge nearby Steiner points)
  5. 120° angle verification
"""

import json
import os
import random
import sys
import math
import pygame
from typing import List, Tuple, Optional, Dict

from steiner_solver import SteinerTreeSolver, DSU

# ─── Config ─────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1100, 700
CANVAS_LEFT = 340
CANVAS_TOP = 20
CANVAS_W = WIDTH - CANVAS_LEFT - 20
CANVAS_H = HEIGHT - 60

WHITE        = (255, 255, 255)
BLACK        = (20, 20, 30)
DARK_BG      = (18, 18, 28)
CANVAS_BG    = (22, 22, 36)
PANEL_BG     = (28, 28, 42)
PANEL_BG_ALT = (33, 33, 52)
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
ORANGE       = (255, 160, 50)
TERMINAL_COLOR  = (70, 180, 255)
STEINER_COLOR   = (255, 180, 80)
MST_COLOR       = GREEN
REJECTED_COLOR  = RED
CURRENT_COLOR   = YELLOW
TEXT_COLOR   = (220, 220, 230)
MUTED        = (90, 90, 110)
BTN_BG       = (35, 35, 55)
BTN_HOVER    = (50, 50, 75)
BTN_BORDER   = (60, 60, 90)
FORCE_COLOR  = (255, 100, 100)
ANGLE_COLOR  = (200, 200, 100)

RADIUS_TERMINAL = 16
RADIUS_STEINER = 13
RADIUS_HOVER = 20
STEP_DELAY = 600
FAST_DELAY = 20
FPS = 60


# ─── Presets ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_PATH = os.path.join(SCRIPT_DIR, "steiner_presets.json")


def load_presets() -> list:
    try:
        with open(PRESETS_PATH, "r") as f:
            data = json.load(f)
        return data.get("presets", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load presets: {e}")
        return []


PRESETS = load_presets()


def generate_equilateral_triangle(cx: float, cy: float, side: float) -> List[Tuple[float, float]]:
    """Generate 3 terminals forming an equilateral triangle."""
    h = side * math.sqrt(3) / 2
    return [
        (cx - side / 2, cy + h / 3),
        (cx + side / 2, cy + h / 3),
        (cx, cy - 2 * h / 3),
    ]


def generate_square(cx: float, cy: float, side: float) -> List[Tuple[float, float]]:
    """Generate 4 terminals forming a square."""
    s2 = side / 2
    return [
        (cx - s2, cy - s2),
        (cx + s2, cy - s2),
        (cx + s2, cy + s2),
        (cx - s2, cy + s2),
    ]


def generate_hexagon(cx: float, cy: float, r: float) -> List[Tuple[float, float]]:
    """Generate 6 terminals forming a regular hexagon."""
    return [
        (cx + r * math.cos(math.radians(60 * i)),
         cy + r * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def generate_random(n: int, seed: int) -> List[Tuple[float, float]]:
    """Generate n random normalized points."""
    rng = random.Random(seed)
    margin = 0.1
    return [
        (round(rng.uniform(margin, 1.0 - margin), 3),
         round(rng.uniform(margin, 1.0 - margin), 3))
        for _ in range(n)
    ]


# ─── Dropdown Widget ────────────────────────────────────────────────────────

class PresetDropdown:
    """Dropdown for preset selection."""

    def __init__(self, x: int, y: int, width: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = 26
        self.open = False
        self.selected_index = 0
        self.scroll_offset = 0
        self.max_visible = 10
        try:
            self.font_sm = pygame.font.SysFont("consolas", 13)
        except Exception:
            self.font_sm = pygame.font.SysFont("monospace", 13)

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_item_rect(self, idx: int) -> pygame.Rect:
        return pygame.Rect(
            self.x,
            self.y + self.height + (idx - self.scroll_offset) * 22,
            self.width,
            22,
        )

    def draw(self, surface, presets: list):
        color = CYAN if self.open else BTN_BORDER
        pygame.draw.rect(surface, BTN_BG, self.get_rect())
        pygame.draw.rect(surface, color, self.get_rect(), 1)

        label = self.font_sm.render(
            presets[self.selected_index].get("name", "Unknown")
            if presets
            else "No presets",
            True,
            TEXT_COLOR,
        )
        surface.blit(label, (self.x + 6, self.y + 5))

        arrow = "▾" if not self.open else "▴"
        arr_surf = self.font_sm.render(arrow, True, GRAY)
        surface.blit(arr_surf, (self.x + self.width - 18, self.y + 4))

        if self.open and presets:
            visible = presets[self.scroll_offset : self.scroll_offset + self.max_visible]
            for i, p in enumerate(visible):
                idx = self.scroll_offset + i
                rect = self.get_item_rect(idx)
                if idx == self.selected_index:
                    pygame.draw.rect(surface, BTN_HOVER, rect)
                name_surf = self.font_sm.render(p.get("name", ""), True, TEXT_COLOR)
                surface.blit(name_surf, (rect.x + 4, rect.y + 3))

    def handle_event(self, event) -> Optional[int]:
        if not PRESETS:
            return None
        
        mouse_pos = pygame.mouse.get_pos()
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.get_rect().collidepoint(mouse_pos):
                self.open = not self.open
                return None
            if self.open:
                for idx in range(
                    self.scroll_offset,
                    min(len(PRESETS), self.scroll_offset + self.max_visible),
                ):
                    if self.get_item_rect(idx).collidepoint(mouse_pos):
                        self.selected_index = idx
                        self.open = False
                        return idx
            self.open = False
        elif event.type == pygame.MOUSEWHEEL and self.open:
            if event.y > 0: # Scroll up
                if self.scroll_offset > 0:
                    self.scroll_offset -= 1
            elif event.y < 0: # Scroll down
                if self.scroll_offset + self.max_visible < len(PRESETS):
                    self.scroll_offset += 1
        return None


# ─── Visualizer ─────────────────────────────────────────────────────────────

class SteinerVisualizer:
    def __init__(self, terminals: List[Tuple[float, float]] = None, n_steiner: int = 2):
        # Pygame init must come FIRST because children (like PresetDropdown) use fonts
        pygame.init()
        
        self.orig_terminals = terminals or [(0.2, 0.3), (0.8, 0.3), (0.5, 0.7)]
        self.user_terminals = list(self.orig_terminals)
        self.manual_steiner = []
        self.n_steiner = n_steiner
        self.learning_rate = 0.08
        
        # Baselines for dashboard
        self.terminal_mst_weight = 0.0
        self.optimal_target_weight = 0.0

        # Solver instance (recreated on reset)
        self.solver: SteinerTreeSolver | None = None

        # Animation state
        self.auto_running = False
        self.fast_mode = False
        self.waiting = True
        self.step_time = 0
        self.startup_timer = 1500
        self.startup_done = False
        self.complete = False
        self.status_msg = "Initializing..."

        # Interaction
        self.dragging = -1  # -1 = none, >= n_terminals means Steiner point
        self.hover = -1
        self.panning = False
        self.pan_last = (0, 0)
        self.zoom = 1.0
        self.pan_offset = [0.0, 0.0]  # in world coords [0,1]

        # Preset
        self.preset_index = -1
        self.preset_label = "(custom)"
        self.dropdown = PresetDropdown(10, 65, 300)

        # Pygame setup
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Steiner Point Playground — Gradient Descent + SA")
        self.clock = pygame.time.Clock()
        def _get_font(size: int, bold: bool = False):
            """Get a monospace font with fallback."""
            for name in ("consolas", "monospace", "courier", "dejavu sans mono"):
                try:
                    return pygame.font.SysFont(name, size, bold=bold)
                except Exception:
                    continue
            return pygame.font.Font(None, size)

        self.font_sm = _get_font(13)
        self.font_md = _get_font(16)
        self.font_lg = _get_font(24, bold=True)

        # Initial solve
        self._reset_solver()

    # ── Coordinate transforms ──────────────────────────────────────────────

    def _world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world [0,1] coords (with pan/zoom) to pixel coords on canvas."""
        effective_w = CANVAS_W * 0.85
        effective_h = CANVAS_H * 0.85
        cx = CANVAS_LEFT + CANVAS_W / 2
        cy = CANVAS_TOP + CANVAS_H / 2
        sx = (wx - 0.5 - self.pan_offset[0]) * self.zoom * effective_w + cx
        sy = (wy - 0.5 - self.pan_offset[1]) * self.zoom * effective_h + cy
        return int(sx), int(sy)

    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coords on canvas to world [0,1] coords."""
        effective_w = CANVAS_W * 0.85
        effective_h = CANVAS_H * 0.85
        cx = CANVAS_LEFT + CANVAS_W / 2
        cy = CANVAS_TOP + CANVAS_H / 2
        wx = (px - cx) / (self.zoom * effective_w) + 0.5 + self.pan_offset[0]
        wy = (py - cy) / (self.zoom * effective_h) + 0.5 + self.pan_offset[1]
        return wx, wy

    def _get_radius(self) -> int:
        return max(8, int(RADIUS_TERMINAL / self.zoom * 0.85))

    # ── Solver management ──────────────────────────────────────────────────

    def _get_solver_steiner_count(self) -> int:
        return max(0, self.n_steiner)

    def _compute_terminal_mst(self):
        """Compute the MST of terminals only for dashboard comparison."""
        if len(self.user_terminals) < 2:
            self.terminal_mst_weight = 0.0
            return
        temp_solver = SteinerTreeSolver(self.user_terminals, n_steiner=0)
        temp_solver._compute_mst()
        self.terminal_mst_weight = temp_solver.mst_weight

    def _reset_solver(self, randomize_steiner=False):
        n_s = self._get_solver_steiner_count()
        
        # Calculate dashboard baseline
        self._compute_terminal_mst()
        
        # Ensure terminals are mutable lists
        initial_points = [list(p) for p in self.user_terminals]
        
        if randomize_steiner:
            # Full reset including steiner positions
            self.manual_steiner = []
            # Cloud Seeding: start with more points than requested to ensure survival
            requested = n_s
            actual_to_seed = requested * 2 if requested > 0 else 0
            self.solver = SteinerTreeSolver(self.user_terminals, n_steiner=actual_to_seed, seed=42)
        else:
            # Topology rebuild: keep current Steiner positions if they exist
            current_steiner = []
            if self.solver:
                current_steiner = [list(p) for p in self.solver.points[self.solver.n_terminals:]]
            
            # Combine manual placements and survivors, ensuring all are mutable lists
            all_steiner = current_steiner + [list(p) for p in self.manual_steiner]
            self.manual_steiner = [] # Clear once absorbed
            
            # Reconstruct solver without randomizing
            self.solver = SteinerTreeSolver(self.user_terminals, n_steiner=0)
            self.solver.points = initial_points + all_steiner
            self.solver.n_steiner = len(all_steiner)
            self.solver.n_total = len(self.solver.points)

        self.solver._compute_mst()
        self.complete = False
        self.waiting = True
        self.startup_done = False
        self.step_time = pygame.time.get_ticks() + self.startup_timer
        self.status_msg = (
            f"Ready — {len(self.solver.mst_edges)} edges. "
            "[N]ext / [A]uto / [F]ast"
        )

    # ── Step logic ──────────────────────────────────────────────────────────

    def do_step(self) -> bool:
        if self.solver is None:
            return False
        
        # Determine current temperature with cooling
        if not self.solver.history:
            temp = 0.02
        else:
            # Aggressive cooling for visual feedback
            last_temp = self.solver.history[-1]["sa_temperature"]
            # Fast cooling: drop by 2% each step
            temp = last_temp * 0.98
            # Quench phase: once noise is small enough, set to 0 to settle
            if temp < 0.001:
                temp = 0.0
            
        record = self.solver.step_gradient(
            learning_rate=self.learning_rate,
            sa_temperature=temp,
        )
        self.status_msg = (
            f"iter {record['iteration']} | "
            f"weight: {record['mst_weight']:.4f} | "
            f"temp: {temp:.4f} | "
            f"max_force: {record['max_force']:.4f} | "
            f"merges: {record['merges']} | "
            f"120° deviation: {record['max_120_deviation']:.2f}°"
        )
        # Convergence requires low force AND zero noise
        if record["max_force"] < 0.001 and record["merges"] == 0 and temp == 0.0:
            self.complete = True
            self.status_msg += " ✓ CONVERGED"
        return True

    # ── Drawing helpers ─────────────────────────────────────────────────────

    def _draw_grid(self):
        """Draw a subtle grid on the canvas area."""
        for x in range(0, CANVAS_W, 40):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (CANVAS_LEFT + x, CANVAS_TOP),
                (CANVAS_LEFT + x, CANVAS_TOP + CANVAS_H),
            )
        for y in range(0, CANVAS_H, 40):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (CANVAS_LEFT, CANVAS_TOP + y),
                (CANVAS_LEFT + CANVAS_W, CANVAS_TOP + y),
            )

    def _draw_edges(self):
        """Draw MST edges, candidate edges, and rejected edges."""
        if self.solver is None:
            return
        # Draw all candidate edges dimly
        n = len(self.solver.points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self._world_to_pixel(*self.solver.points[i])
                x2, y2 = self._world_to_pixel(*self.solver.points[j])
                pygame.draw.line(self.screen, GRAY_DARK, (x1, y1), (x2, y2), 1)

        # Draw MST edges
        for u, v in self.solver.mst_edges:
            x1, y1 = self._world_to_pixel(*self.solver.points[u])
            x2, y2 = self._world_to_pixel(*self.solver.points[v])
            pygame.draw.line(self.screen, MST_COLOR, (x1, y1), (x2, y2), 3)

    def _draw_forces(self):
        """Draw force arrows on Steiner points."""
        if self.solver is None:
            return
        steiner_start = self.solver.n_terminals
        for i in range(steiner_start, len(self.solver.points)):
            fx, fy = self.solver._compute_gradient(i)
            force_mag = math.hypot(fx, fy)
            if force_mag < 1e-8:
                continue
            px, py = self._world_to_pixel(*self.solver.points[i])
            scale = 40.0
            end_x = px + fx * scale
            end_y = py + fy * scale
            # Arrow line
            pygame.draw.line(
                self.screen, FORCE_COLOR, (int(px), int(py)), (int(end_x), int(end_y)), 2
            )
            # Arrow head
            angle = math.atan2(fy, fx)
            head_len = 8
            pygame.draw.line(
                self.screen,
                FORCE_COLOR,
                (int(end_x), int(end_y)),
                (
                    int(end_x - head_len * math.cos(angle - 0.4)),
                    int(end_y - head_len * math.sin(angle - 0.4)),
                ),
                2,
            )
            pygame.draw.line(
                self.screen,
                FORCE_COLOR,
                (int(end_x), int(end_y)),
                (
                    int(end_x - head_len * math.cos(angle + 0.4)),
                    int(end_y - head_len * math.sin(angle + 0.4)),
                ),
                2,
            )

    def _draw_vertices(self):
        """Draw terminal and Steiner vertices with labels."""
        if self.solver is None:
            return
        r = self._get_radius()
        for i, pt in enumerate(self.solver.points):
            px, py = self._world_to_pixel(pt[0], pt[1])
            is_terminal = i < self.solver.n_terminals
            
            if is_terminal:
                color = TERMINAL_COLOR
            else:
                # Color code Steiner points by degree
                deg = len(self.solver.adj.get(i, []))
                if deg == 3:
                    color = ORANGE # Optimal junction candidate
                else:
                    color = RED # Non-optimal (trapped d=4 or redundant d=2)

            if i == self.hover:
                pygame.draw.circle(
                    self.screen, YELLOW, (px, py), r + RADIUS_HOVER, 2
                )
            # Glow
            glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            glow_alpha = 30 if is_terminal else 50
            pygame.draw.circle(
                glow_surf, (*color[:3], glow_alpha), (r * 2, r * 2), r * 2
            )
            self.screen.blit(glow_surf, (px - r * 2, py - r * 2))
            # Main circle
            pygame.draw.circle(self.screen, color, (px, py), r)
            pygame.draw.circle(self.screen, WHITE, (px, py), r, 2)
            # Label
            label_char = f"T{i}" if is_terminal else f"S{i - self.solver.n_terminals}"
            label = self.font_sm.render(label_char, True, BLACK)
            self.screen.blit(
                label, (px - label.get_width() // 2, py - label.get_height() // 2)
            )

    def _draw_120_angles(self):
        """Visualize angles at Steiner points and flag deviations."""
        if self.solver is None or self.solver.n_steiner == 0:
            return
        steiner_start = self.solver.n_terminals
        for i in range(steiner_start, len(self.solver.points)):
            nb_indices = self.solver.adj.get(i, [])
            if len(nb_indices) < 2:
                continue
            
            px, py = self.solver.points[i]
            
            # SORT neighbors by polar angle to ensure sectors sum to 360 and arcs are adjacent
            def polar_angle(nb_idx):
                nx, ny = self.solver.points[nb_idx]
                return math.atan2(ny - py, nx - px)
            
            sorted_nb = sorted(nb_indices, key=polar_angle)
            
            px_pix, py_pixel = self._world_to_pixel(px, py)
            arc_radius = 35 / self.zoom
            rect_arc = pygame.Rect(
                px_pix - arc_radius, py_pixel - arc_radius, arc_radius * 2, arc_radius * 2
            )
            
            # Draw sectors
            n = len(sorted_nb)
            for j in range(n):
                nj = sorted_nb[j]
                nk = sorted_nb[(j + 1) % n]
                
                # Polar angles in screen space (Y-down)
                ang_a = polar_angle(nj)
                ang_b = polar_angle(nk)
                
                # The "step" from a to b in the sorted list (CW direction in screen pixels)
                diff = ang_b - ang_a
                while diff < 0: diff += 2 * math.pi
                
                actual_angle_deg = math.degrees(diff)
                
                # Color based on deviation from 120
                deviation = abs(actual_angle_deg - 120.0)
                if n == 3 and deviation < 5:
                    arc_color = GREEN
                elif n == 3 and deviation < 15:
                    arc_color = ORANGE
                else:
                    arc_color = RED
                
                # Pygame draw.arc uses standard math (CCW positive, Y-up).
                # To draw a CW sector (ang_a to ang_b) in Pygame, we negate the angles.
                # Sector: ang_a (start) to ang_b (stop).
                # Negated: -ang_b (start) to -ang_a (stop).
                pygame.draw.arc(
                    self.screen,
                    arc_color,
                    rect_arc,
                    -ang_b,
                    -ang_a,
                    3,
                )
                
                # Label at the middle of the sector
                mid_angle = ang_a + diff / 2
                label_r = arc_radius + 18
                lx = px_pix + label_r * math.cos(mid_angle)
                ly = py_pixel + label_r * math.sin(mid_angle)
                angle_text = self.font_sm.render(f"{actual_angle_deg:.1f}°", True, arc_color)
                self.screen.blit(
                    angle_text,
                    (int(lx - angle_text.get_width() // 2), int(ly - angle_text.get_height() // 2)),
                )

    def _draw_dashboard(self, x, y, width):
        """Draw the optimization dashboard with bar charts."""
        if not self.solver or self.terminal_mst_weight == 0:
            return y
        
        # Section Title
        title = self.font_md.render("OPTIMIZATION DASHBOARD", True, CYAN)
        self.screen.blit(title, (x, y))
        y += 25
        
        # Baseline MST (No Steiner)
        base = self.terminal_mst_weight
        curr = self.solver.mst_weight
        target = self.optimal_target_weight
        
        # Scaling for bars: Baseline is always 100% width
        bar_h = 16
        bar_w_max = width - 80
        
        def draw_bar(val, label, color, row_y, baseline):
            # Label
            lbl = self.font_sm.render(f"{label}", True, TEXT_COLOR)
            self.screen.blit(lbl, (x, row_y))
            
            # Bar background
            pygame.draw.rect(self.screen, GRAY_DARK, (x + 85, row_y, bar_w_max, bar_h))
            
            # Bar foreground
            w = (val / baseline) * bar_w_max if baseline > 0 else 0
            w = min(bar_w_max, w)
            pygame.draw.rect(self.screen, color, (x + 85, row_y, w, bar_h))
            
            # Value
            val_lbl = self.font_sm.render(f"{val:.3f}", True, WHITE)
            self.screen.blit(val_lbl, (x + 85 + bar_w_max + 5, row_y))
            return row_y + 22

        # 1. MST Baseline
        y = draw_bar(base, "MST Baseline", TERMINAL_COLOR, y, base)
        
        # 2. Current Result
        savings = (1 - curr / base) * 100 if base > 0 else 0
        y = draw_bar(curr, "Current Tree", GREEN if savings > 0.01 else YELLOW, y, base)
        
        # Savings text
        save_lbl = self.font_md.render(f"Length Saved: {savings:.2f}%", True, GREEN if savings > 0 else WHITE)
        self.screen.blit(save_lbl, (x + 85, y))
        y += 28
        
        # 3. Theoretical Optimal (if preset has it)
        if target > 0:
            y = draw_bar(target, "Ideal Target", LIGHT_BLUE, y, base)
            potential = (1 - target / base) * 100
            pot_lbl = self.font_sm.render(f"Max Potential: {potential:.1f}%", True, MUTED)
            self.screen.blit(pot_lbl, (x + 85, y))
            y += 20

        return y

    def _draw_panel(self):
        """Draw the side panel with controls and stats."""
        panel_x = 10
        y = 10

        # Title
        self.screen.blit(
            self.font_lg.render("STEINER TREE", True, CYAN), (panel_x, y)
        )
        y += 30
        self.screen.blit(
            self.font_md.render("Point Playground", True, YELLOW),
            (panel_x + 10, y),
        )
        y += 30

        # Preset label
        self.screen.blit(
            self.font_md.render(f"[ {self.preset_label} ]", True, MAGENTA),
            (panel_x, y),
        )
        y += 24

        # Dashboard Section
        y = self._draw_dashboard(panel_x, y, CANVAS_LEFT - 20)
        y += 10

        # Stats from solver
        if self.solver:
            stats = [
                ("Terminals", str(self.solver.n_terminals)),
                ("Steiner pts", str(self.solver.n_steiner)),
                ("Total pts", str(len(self.solver.points))),
                ("Iteration", str(self.solver.iteration)),
                ("Force Str", f"{self.learning_rate:.3f}"),
            ]
            if self.solver.history:
                last = self.solver.history[-1]
                stats.extend([
                    ("Max force", f"{last['max_force']:.6f}"),
                    ("120° dev", f"{last['max_120_deviation']:.2f}°"),
                ])
        else:
            stats = []

        for label, val in stats:
            self.screen.blit(
                self.font_md.render(f"{label:>12}: {val}", True, WHITE), (panel_x, y)
            )
            y += 22

        y += 8
        pygame.draw.line(
            self.screen, GRAY_DARK, (panel_x, y), (CANVAS_LEFT - 12, y), 1
        )
        y += 12

        # Status
        words = self.status_msg.split()
        line = ""
        max_w = CANVAS_LEFT - 24
        for w in words:
            test = line + (" " if line else "") + w
            if self.font_md.size(test)[0] > max_w:
                self.screen.blit(
                    self.font_md.render(line, True, YELLOW), (panel_x, y)
                )
                y += 20
                line = w
            else:
                line = test
        if line:
            self.screen.blit(
                self.font_md.render(line, True, YELLOW), (panel_x, y)
            )
            y += 20

        y += 10
        controls = [
            "[N]ext   [A]uto   [R]eset   [C]lear",
            "[F]ast   [G]enerate  [+] Stein  [-] Stein",
            "[[ / ]] Adjust Force Strength",
            "Left: Terminal | Right: Steiner",
            "[,] prev  [.] next  [Scroll] zoom",
            "Click+drag to move vertices",
        ]
        for c in controls:
            self.screen.blit(
                self.font_sm.render(c, True, MUTED), (panel_x, y)
            )
            y += 18

        # Dropdown
        y += 10
        self.dropdown.x = panel_x
        self.dropdown.y = y
        self.dropdown.draw(self.screen, PRESETS)

    def _draw(self):
        self.screen.fill(DARK_BG)
        self._draw_grid()
        self._draw_edges()
        self._draw_120_angles()
        self._draw_forces()
        self._draw_vertices()
        # Canvas border
        pygame.draw.line(
            self.screen, GRAY, (CANVAS_LEFT - 5, 0), (CANVAS_LEFT - 5, HEIGHT), 2
        )
        # Zoom indicator
        zoom_text = self.font_sm.render(f"Zoom: {self.zoom:.2f}x", True, MUTED)
        self.screen.blit(zoom_text, (CANVAS_LEFT + 5, CANVAS_TOP + 5))
        # Mode indicator
        mode = "FAST" if self.fast_mode else "NORMAL"
        mode_text = self.font_sm.render(f"Speed: {mode}", True, MUTED)
        self.screen.blit(mode_text, (CANVAS_LEFT + 5, CANVAS_TOP + 25))
        self._draw_panel()
        pygame.display.flip()

    # ── Hit testing ─────────────────────────────────────────────────────────

    def _hit_test(self, mx: int, my: int) -> int:
        """Find the vertex index at pixel (mx, my), or -1."""
        if self.solver is None:
            return -1
        r = self._get_radius() + 8
        for i, pt in enumerate(self.solver.points):
            px, py = self._world_to_pixel(pt[0], pt[1])
            if (mx - px) ** 2 + (my - py) ** 2 < r ** 2:
                return i
        return -1

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        self._draw()

        while True:
            self.clock.tick(FPS)
            now = pygame.time.get_ticks()
            mouse_pos = pygame.mouse.get_pos()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._cleanup()
                    return

                # Dropdown handles both click and scroll
                dresult = self.dropdown.handle_event(ev)
                if dresult is not None:
                    self._load_preset(dresult)
                    continue

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = ev.pos
                    # Check canvas click
                    if (
                        CANVAS_LEFT <= mx <= CANVAS_LEFT + CANVAS_W
                        and CANVAS_TOP <= my <= CANVAS_TOP + CANVAS_H
                    ):
                        self.hover = -1
                        idx = self._hit_test(mx, my)
                        if idx >= 0:
                            if ev.button == 1: # Left click drag
                                self.dragging = idx
                        else:
                            wx, wy = self._pixel_to_world(mx, my)
                            if ev.button == 1: # Left click add terminal
                                self.user_terminals.append((round(wx, 3), round(wy, 3)))
                                self.preset_index = -1
                                self.preset_label = "(custom)"
                                self.optimal_target_weight = 0.0 # Clear optimal for custom
                                self._reset_solver()
                            elif ev.button == 3: # Right click add steiner
                                self.manual_steiner.append([round(wx, 3), round(wy, 3)])
                                self._reset_solver()
                            self._draw()

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = -1

                elif ev.type == pygame.MOUSEMOTION:
                    mx, my = ev.pos
                    # Update hover
                    if CANVAS_LEFT <= mx <= CANVAS_LEFT + CANVAS_W and CANVAS_TOP <= my <= CANVAS_TOP + CANVAS_H:
                        self.hover = self._hit_test(mx, my)

                    if self.dragging >= 0:
                        wx, wy = self._pixel_to_world(mx, my)
                        # Clamp
                        wx = max(0.02, min(0.98, wx))
                        wy = max(0.02, min(0.98, wy))
                        self.solver.points[self.dragging] = [wx, wy]
                        if self.dragging < self.solver.n_terminals:
                            self.user_terminals[self.dragging] = (wx, wy)
                            self.preset_index = -1
                            self.preset_label = "(custom)"
                            self.optimal_target_weight = 0.0
                        self._reset_solver()
                        self._draw()

                elif ev.type == pygame.MOUSEWHEEL:
                    # Dropdown handles its own scroll, but if closed, zoom canvas
                    if not self.dropdown.open:
                        old_zoom = self.zoom
                        if ev.y > 0:
                            self.zoom = min(5.0, self.zoom * 1.15)
                        else:
                            self.zoom = max(0.2, self.zoom / 1.15)
                        
                        # Pan so zoom centers on mouse
                        if CANVAS_LEFT <= mouse_pos[0] <= CANVAS_LEFT + CANVAS_W and CANVAS_TOP <= mouse_pos[1] <= CANVAS_TOP + CANVAS_H:
                            wx, wy = self._pixel_to_world(mouse_pos[0], mouse_pos[1])
                            new_px, new_py = self._world_to_pixel(wx, wy)
                            dx_world = (new_px - mouse_pos[0]) / (self.zoom * CANVAS_W * 0.85)
                            dy_world = (new_py - mouse_pos[1]) / (self.zoom * CANVAS_H * 0.85)
                            self.pan_offset[0] -= dx_world
                            self.pan_offset[1] -= dy_world
                        self._draw()

                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE,):
                        self._cleanup()
                        return

                    elif ev.key == pygame.K_n and not self.complete and self.solver.n_terminals >= 2:
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
                        elif self.solver is None or self.solver.n_terminals < 2:
                            self.status_msg = "Need at least 2 terminals."
                        else:
                            self.waiting = False

                    elif ev.key == pygame.K_r:
                        # Full reset (randomize steiner)
                        self._reset_solver(randomize_steiner=True)
                        self._draw()

                    elif ev.key == pygame.K_c:
                        self.user_terminals = []
                        self.manual_steiner = []
                        self.optimal_target_weight = 0.0
                        self._reset_solver(randomize_steiner=True)
                        self._draw()

                    elif ev.key == pygame.K_f:
                        self.fast_mode = not self.fast_mode
                        mode_name = "FAST" if self.fast_mode else "NORMAL"
                        self.status_msg = f"Speed mode: {mode_name}"
                        self._draw()
                        if not self.complete and self.solver and self.solver.n_terminals >= 2:
                            self.waiting = False

                    elif ev.key == pygame.K_g:
                        # Generate random terminals
                        n = random.choice([3, 4, 5, 6, 8])
                        seed = random.randint(1, 9999)
                        self.user_terminals = generate_random(n, seed)
                        self.n_steiner = max(0, n - 2)
                        self.preset_index = -1
                        self.preset_label = f"Random n={n} seed={seed}"
                        self.optimal_target_weight = 0.0
                        self._reset_solver(randomize_steiner=True)
                        self._draw()
                        self.status_msg = f"Random {n} terminals (seed={seed}). [N] or [A] to optimize."

                    elif ev.key == pygame.K_PERIOD:
                        self._next_preset()
                    elif ev.key == pygame.K_COMMA:
                        self._prev_preset()

                    elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                        self.n_steiner = min(20, self.n_steiner + 1)
                        self._reset_solver()
                        self._draw()
                        self.status_msg = f"Steiner points: {self.n_steiner}"

                    elif ev.key == pygame.K_MINUS:
                        self.n_steiner = max(0, self.n_steiner - 1)
                        self._reset_solver()
                        self._draw()
                        self.status_msg = f"Steiner points: {self.n_steiner}"

                    elif ev.key == pygame.K_LEFTBRACKET:
                        self.learning_rate = max(0.001, self.learning_rate - 0.01)
                        self.status_msg = f"Force Strength: {self.learning_rate:.3f}"
                        self._draw()

                    elif ev.key == pygame.K_RIGHTBRACKET:
                        self.learning_rate = min(0.5, self.learning_rate + 0.01)
                        self.status_msg = f"Force Strength: {self.learning_rate:.3f}"
                        self._draw()

                    elif ev.key == pygame.K_l:
                        # Load from dropdown index
                        if PRESETS:
                            self._load_preset(self.dropdown.selected_index)

                    elif ev.key == pygame.K_h:
                        self._show_help()

                    elif ev.key == pygame.K_p:
                        # Print solver result to console
                        if self.solver:
                            print(f"\n--- Steiner Tree Result ---")
                            print(f"Terminals: {self.solver.terminals}")
                            print(f"Steiner points: {self.solver.points[self.solver.n_terminals:]}")
                            print(f"MST edges: {self.solver.mst_edges}")
                            print(f"MST weight: {self.solver.mst_weight:.4f}")
                            print(f"Iterations: {self.solver.iteration}")

            # Auto-advance
            if not self.waiting and not self.complete and self.solver and self.solver.n_terminals >= 2:
                if now >= self.step_time:
                    self.do_step()
                    self._draw()
                    if not self.complete:
                        delay = FAST_DELAY if self.fast_mode else STEP_DELAY
                        self.step_time = now + delay
                    else:
                        if not self.fast_mode:
                            pygame.time.wait(2000)

    def _load_preset(self, idx: int):
        if not PRESETS:
            self.status_msg = "No presets found."
            return
        idx = idx % len(PRESETS)
        self.dropdown.selected_index = idx
        p = PRESETS[idx]
        raw = p.get("terminals", [])
        self.preset_index = idx
        self.preset_label = p.get("name", f"Preset {idx}")
        self.user_terminals = [(v[0], v[1]) for v in raw]
        self.n_steiner = p.get("n_steiner", max(0, len(self.user_terminals) - 2))
        self.optimal_target_weight = p.get("optimal_weight", 0.0)
        self._reset_solver(randomize_steiner=True)
        self._draw()

    def _next_preset(self):
        if PRESETS:
            self._load_preset(self.dropdown.selected_index + 1)

    def _prev_preset(self):
        if PRESETS:
            self._load_preset(self.dropdown.selected_index - 1)

    def _show_help(self):
        """Overlay a help screen."""
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        help_lines = [
            "HELP — Steinberg Point Playground",
            "",
            "MOUSE:",
            "  Left Click canvas: add Terminal",
            "  Right Click canvas: add Steiner",
            "  Click+drag vertex: move it",
            "  Scroll: zoom in/out (or scroll dropdown)",
            "",
            "KEYBOARD:",
            "  N: next algorithm step",
            "  A/SPACE: auto-run",
            "  R: reset algorithm (randomize Steiner)",
            "  C: clear all nodes",
            "  F: toggle fast mode",
            "  +/-: add/remove Steiner point",
            "  [ / ]: Adjust Force Strength (Learning Rate)",
            "  G: generate random terminals",
            "  ,/: prev/next preset",
            "  L: load selected preset",
            "  H: show/hide this help",
            "  P: print result to console",
            "  ESC: quit",
            "",
            "Click anywhere to close",
        ]

        y = 40
        for i, line in enumerate(help_lines):
            color = CYAN if i == 0 else (YELLOW if line.startswith("  ") else WHITE)
            surf = self.font_md.render(line, True, color)
            self.screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))
            y += 24

        pygame.display.flip()

        # Wait for click to close
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting = False
                    break

    def _cleanup(self):
        pygame.quit()
        sys.exit(0)


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default: equilateral triangle with 2 Steiner points
    terminals = [(0.2, 0.3), (0.8, 0.3), (0.5, 0.7)]

    import argparse
    parser = argparse.ArgumentParser(description="Steiner Point Playground")
    parser.add_argument("--terminals", type=int, default=0, help="Number of random terminals (overrides defaults)")
    parser.add_argument("--steiner", type=int, default=2, help="Number of Steiner points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for terminal generation")
    args = parser.parse_args()

    if args.terminals > 0:
        terminals = generate_random(args.terminals, args.seed)

    v = SteinerVisualizer(terminals=terminals, n_steiner=args.steiner)
    v.run()
