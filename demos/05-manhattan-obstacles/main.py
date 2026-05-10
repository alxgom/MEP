"""
Main Interactive Visualizer: Manhattan Obstacles
================================================
Controls:
- LEFT CLICK: Add Terminal
- RIGHT CLICK + DRAG: Draw Obstacle
- 1: Manhattan MST (Baseline)
- 2: Greedy Steiner (Iterated 1-Steiner)
- 3: Grid Pruning (Dense Grid)
- 4: Fast Corner Kick (targeted selection)
- 5: Monte Carlo Population (Evolutionary)
- C: Clear map
- R: Generate Random terminals
"""

import pygame
import numpy as np
from typing import List
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

# Colors
COLOR_BG = (20, 20, 35)
COLOR_GRID = (40, 40, 60)
COLOR_OBSTACLE = (100, 40, 40)
COLOR_TERMINAL = (50, 150, 255)
COLOR_STEINER = (255, 150, 50)
COLOR_PATH = (46, 204, 113)
COLOR_TEXT = (220, 220, 220)

class Visualizer:
    def __init__(self, width=1000, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MEP Optimizer: Manhattan Steiner with Obstacles")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        self.terminals = []
        self.obstacles: List[Obstacle] = []
        self.mode = 2 
        self.mode_names = {
            1: "Manhattan MST", 
            2: "Greedy Steiner", 
            3: "Grid Pruning",
            4: "Fast Corner Kick",
            5: "Monte Carlo Population"
        }
        
        self.current_obstacle_start = None
        self.solution = None
        self.baseline_solution = None
        self.env = None

    def _solve(self):
        if len(self.terminals) < 2:
            self.solution = None; self.baseline_solution = None
            return

        pts = np.array(self.terminals)
        self.env = GridEnvironment(pts, self.obstacles)
        
        solver = ObstacleSteinerSolver(self.env)
        self.baseline_solution = solver.solve_mst()
        
        if self.mode == 1: self.solution = self.baseline_solution
        elif self.mode == 2: self.solution = solver.solve_greedy()
        elif self.mode == 3: self.solution = solver.solve_prune()
        elif self.mode == 4: self.solution = solver.solve_fast_corner()
        elif self.mode == 5: self.solution = solver.solve_monte_carlo()

    def run(self):
        running = True
        while running:
            self.screen.fill(COLOR_BG)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if event.button == 1: self.terminals.append(list(pos)); self._solve()
                    elif event.button == 3: self.current_obstacle_start = pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3 and self.current_obstacle_start:
                        pos = pygame.mouse.get_pos()
                        obs = Obstacle(self.current_obstacle_start[0], self.current_obstacle_start[1], pos[0], pos[1])
                        if abs(obs.max_x - obs.min_x) > 5 and abs(obs.max_y - obs.min_y) > 5:
                            self.obstacles.append(obs); self._solve()
                        self.current_obstacle_start = None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.terminals = []; self.obstacles = []; self.solution = None; self.env = None; self.baseline_solution = None
                    elif event.key == pygame.K_r:
                        w, h = self.screen.get_size()
                        self.terminals = [[np.random.randint(50, w-50), np.random.randint(50, h-50)] for _ in range(8)]
                        self._solve()
                    elif event.key == pygame.K_1: self.mode = 1; self._solve()
                    elif event.key == pygame.K_2: self.mode = 2; self._solve()
                    elif event.key == pygame.K_3: self.mode = 3; self._solve()
                    elif event.key == pygame.K_4: self.mode = 4; self._solve()
                    elif event.key == pygame.K_5: self.mode = 5; self._solve()

            if self.current_obstacle_start:
                curr_pos = pygame.mouse.get_pos()
                x, y = min(self.current_obstacle_start[0], curr_pos[0]), min(self.current_obstacle_start[1], curr_pos[1])
                w, h = abs(self.current_obstacle_start[0] - curr_pos[0]), abs(self.current_obstacle_start[1] - curr_pos[1])
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, w, h), 2)

            for obs in self.obstacles:
                pygame.draw.rect(self.screen, COLOR_OBSTACLE, (obs.min_x, obs.min_y, obs.max_x - obs.min_x, obs.max_y - obs.min_y))

            if self.env:
                for node in self.env.nodes: pygame.draw.circle(self.screen, COLOR_GRID, (int(node[0]), int(node[1])), 1)

            if self.solution and self.env:
                for u_idx, v_idx in self.solution["segments"]:
                    pygame.draw.line(self.screen, COLOR_PATH, self.env.nodes[u_idx], self.env.nodes[v_idx], 4)
                for s_idx in self.solution["steiner_indices"]:
                    p = self.env.nodes[s_idx]
                    pygame.draw.rect(self.screen, COLOR_STEINER, (p[0]-5, p[1]-5, 10, 10))

            for p in self.terminals:
                pygame.draw.circle(self.screen, COLOR_TERMINAL, (int(p[0]), int(p[1])), 8)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(p[0]), int(p[1])), 8, 2)

            baseline_w = self.baseline_solution['weight'] if self.baseline_solution else 0
            current_w = self.solution['weight'] if self.solution else 0
            saving = ((baseline_w - current_w) / baseline_w * 100) if baseline_w > 0 else 0
            
            info = [
                f"MODE: {self.mode_names[self.mode]} (Keys 1-5)",
                "LEFT CLICK: Add Terminal | RIGHT DRAG: Obstacle",
                "C: Clear | R: Random",
                f"Terminals: {len(self.terminals)}",
                f"Baseline MST: {baseline_w:.2f}",
                f"Optimized:    {current_w:.2f}",
                f"Saving:       {saving:.1f}%"
            ]
            for i, text in enumerate(info):
                self.screen.blit(self.font.render(text, True, COLOR_TEXT), (20, 20 + i * 25))

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    v = Visualizer()
    v.run()
