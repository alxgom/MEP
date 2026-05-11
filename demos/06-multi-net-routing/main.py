"""
Advanced 3-Panel Multi-Net Comparison Dashboard (v4.0)
======================================================
Panels:
1. Negotiated Congestion (Pathfinder Soft-Locking)
2. Rip-up and Re-route (Ordering Failure Recovery)
3. Bottleneck-Aware (Density-Based Priority)
"""

import pygame
import numpy as np
from typing import Dict, List, Any
from environment import MultiNetEnvironment
from solver import MultiNetSolver

# Window Config
PANEL_W, PANEL_H = 450, 550
WIDTH, HEIGHT = PANEL_W * 4, PANEL_H + 50

# Colors
COLOR_BG = (20, 20, 30)
COLOR_NETS = {
    "Net_1": (46, 204, 113),  # Green
    "Net_2": (52, 152, 219),  # Blue
    "Net_3": (155, 89, 182),  # Purple
    "Net_4": (241, 196, 15)   # Yellow
}
COLOR_TEXT = (220, 220, 220)
COLOR_FAILED = (231, 76, 60)
COLOR_BORDER = (40, 40, 60)

class HeuristicTournamentVisualizer:
    def __init__(self):
        pygame.init()
        # Rigid window size to prevent scrolling issues
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Multi-Net Heuristic Tournament v4.8")
        self.font = pygame.font.SysFont("Arial", 14)
        self.font_bold = pygame.font.SysFont("Arial", 16, bold=True)
        self.clock = pygame.time.Clock()
        
        self.nets: Dict[str, List[List[float]]] = {"Net_1":[], "Net_2":[], "Net_3":[], "Net_4":[]}
        self.current_net = "Net_1"
        self.results: Dict[str, Any] = {} # {HeuristicName: (solution, total_len, fails)}

    def _solve(self):
        active_nets = {k: np.array(v) for k, v in self.nets.items() if len(v) >= 2}
        if not active_nets: return

        print("\n--- Dispatching Tournament v4.8 ---")
        env = MultiNetEnvironment(active_nets)
        solver = MultiNetSolver(env)
        
        # 1. Negotiated Congestion
        print("Running Negotiated Congestion (Aggressive)...")
        self.results["Negotiated"] = solver.solve_negotiated(max_iters=30, congestion_base=500.0)
        
        # 2. Surgical Hybrid (Ideal Draft)
        print("Running Surgical Hybrid (Ideal Draft)...")
        self.results["Hybrid"] = solver.solve_hybrid_ripup()
        
        # 3. Negotiated Hybrid (Soft Draft)
        print("Running Negotiated Hybrid (Soft Draft)...")
        self.results["NegHybrid"] = solver.solve_negotiated_hybrid()
        
        # 4. Global Permutation (Standard)
        print("Running Global Permutation Search...")
        self.results["GlobalPerm"] = solver.solve_best_permutation()

    def draw_panel(self, x_offset, title, h_key):
        rect = pygame.Rect(x_offset, 50, PANEL_W, PANEL_H)
        pygame.draw.rect(self.screen, (25, 25, 40), rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, rect, 1)
        
        t_surf = self.font_bold.render(title, True, (255, 255, 255))
        self.screen.blit(t_surf, (x_offset + 10, 15))
        
        if h_key in self.results:
            sol, total_l, issues = self.results[h_key]
            stat_text = f"Len: {total_l:.1f} | Issues: {issues}"
            c = COLOR_FAILED if issues > 0 else (100, 255, 100)
            self.screen.blit(self.font.render(stat_text, True, c), (x_offset + PANEL_W - 160, 18))
            
            # Rendering using a cached environment setup
            active_nets = {k: np.array(v) for k, v in self.nets.items() if len(v) >= 2}
            env = MultiNetEnvironment(active_nets)
            
            for net_name, res in sol.items():
                color = COLOR_NETS[net_name]
                for u, v in res["segments"]:
                    p1 = (env.nodes[u][0] + x_offset, env.nodes[u][1] + 50)
                    p2 = (env.nodes[v][0] + x_offset, env.nodes[v][1] + 50)
                    pygame.draw.line(self.screen, color, p1, p2, 3)
        
        for net_name, pts in self.nets.items():
            color = COLOR_NETS[net_name]
            for p in pts:
                pos = (int(p[0] + x_offset), int(p[1] + 50))
                pygame.draw.circle(self.screen, color, pos, 6)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 6, 1)

    def run(self):
        running = True
        while running:
            self.screen.fill(COLOR_BG)
            mx, my = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Mirror click to panel coordinate system
                    local_x = mx % PANEL_W
                    local_y = my - 50
                    if 0 <= local_y <= PANEL_H:
                        self.nets[self.current_net].append([float(local_x), float(local_y)])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: self.current_net = "Net_1"
                    elif event.key == pygame.K_2: self.current_net = "Net_2"
                    elif event.key == pygame.K_3: self.current_net = "Net_3"
                    elif event.key == pygame.K_4: self.current_net = "Net_4"
                    elif event.key == pygame.K_s: self._solve()
                    elif event.key == pygame.K_c: self.nets={k:[] for k in self.nets}; self.results={}

            self.draw_panel(0, "1. Negotiated (Aggressive)", "Negotiated")
            self.draw_panel(PANEL_W, "2. Surgical Hybrid (Ideal)", "Hybrid")
            self.draw_panel(PANEL_W * 2, "3. Negotiated Hybrid (Soft)", "NegHybrid")
            self.draw_panel(PANEL_W * 3, "4. Global Permutation", "GlobalPerm")

            help_t = f"Brush: {self.current_net} | Keys 1-4: Switch | S: Solve | C: Clear"
            self.screen.blit(self.font.render(help_t, True, COLOR_TEXT), (10, HEIGHT - 30))

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    v = HeuristicTournamentVisualizer()
    v.run()
