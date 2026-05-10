"""
Steiner Tree Solver — Pure Logic (no pygame dependency)
======================================================
Hybrid variational approach: MST topology + gradient descent + Simulated Annealing.

Algorithm
---------
1. Initialize Steiner points randomly inside the convex hull of terminals.
2. Repeat until convergence:
   a. Build MST over all points (terminals + Steiner).
   b. Compute gradient on each Steiner point (sum of unit vectors toward MST neighbors).
   c. Move Steiner points along gradient (learning_rate * force).
   d. Optionally apply thermal perturbation (Simulated Annealing).
   e. Merge Steiner points that are too close (annihilation).
3. Verify 120° angle condition at each Steiner point.
"""

import math
import random
from typing import List, Tuple, Set


class DSU:
    """Disjoint Set Union with path compression and union-by-rank."""

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


class SteinerTreeSolver:
    """
    Hybrid Steiner Tree solver using gradient descent + SA + MST topology.
    """

    def __init__(
        self,
        terminals: List[Tuple[float, float]],
        n_steiner: int = 0,
        seed=None,
    ):
        self.rng = random.Random(seed)
        self.terminals = list(terminals)
        self.n_terminals = len(terminals)
        self.n_steiner = n_steiner
        self.n_total = self.n_terminals + n_steiner

        xs = [p[0] for p in terminals]
        ys = [p[1] for p in terminals]
        self._min_x = min(xs) if xs else 0.0
        self._max_x = max(xs) if xs else 1.0
        self._min_y = min(ys) if ys else 0.0
        self._max_y = max(ys) if ys else 1.0
        self._bb_diag = math.hypot(self._max_x - self._min_x, self._max_y - self._min_y)

        self.points: List[List[float]] = [list(p) for p in terminals]
        for _ in range(n_steiner):
            self.points.append(self._random_point_in_hull())

        self.mst_edges: List[Tuple[int, int]] = []
        self.mst_weight: float = 0.0
        self.adj: dict = {}
        
        # Physics state: track velocities to dampen oscillations (momentum decay)
        self.velocities: List[List[float]] = [[0.0, 0.0] for _ in range(self.n_total)]

        self.history: List[dict] = []
        self.iteration = 0

    def _random_point_in_hull(self) -> List[float]:
        margin = 0.05
        x = self.rng.uniform(self._min_x + margin, self._max_x - margin)
        y = self.rng.uniform(self._min_y + margin, self._max_y - margin)
        return [x, y]

    def _dist(self, i: int, j: int) -> float:
        x1, y1 = self.points[i]
        x2, y2 = self.points[j]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _dist_pts(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _compute_mst(self):
        n = len(self.points)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                w = self._dist(i, j)
                edges.append((w, i, j))
        edges.sort()

        dsu = DSU(n)
        mst = []
        total = 0.0
        for w, u, v in edges:
            if dsu.union(u, v):
                mst.append((u, v))
                total += w
                if len(mst) == n - 1:
                    break

        self.mst_edges = mst
        self.mst_weight = total
        self.adj = {i: [] for i in range(n)}
        for u, v in mst:
            self.adj[u].append(v)
            self.adj[v].append(u)
        return mst

    def _compute_gradient(self, idx: int):
        px, py = self.points[idx]
        fx = fy = 0.0
        for nb in self.adj.get(idx, []):
            nx, ny = self.points[nb]
            dx = nx - px
            dy = ny - py
            d = math.hypot(dx, dy)
            if d > 1e-10:
                fx += dx / d
                fy += dy / d
        return fx, fy

    def _angles_at_steiner(self, idx: int):
        nb_indices = self.adj.get(idx, [])
        if len(nb_indices) < 2:
            return []
        px, py = self.points[idx]
        
        # Sort neighbors by polar angle around (px, py)
        def polar_angle(nb_idx):
            nx, ny = self.points[nb_idx]
            return math.atan2(ny - py, nx - px)
        
        sorted_nb = sorted(nb_indices, key=polar_angle)
        
        angles = []
        n = len(sorted_nb)
        for i in range(n):
            a = sorted_nb[i]
            b = sorted_nb[(i + 1) % n]
            
            ax, ay = self.points[a][0] - px, self.points[a][1] - py
            bx, by = self.points[b][0] - px, self.points[b][1] - py
            
            # Use atan2 difference for better precision and to ensure it's the interior angle
            ang_a = math.atan2(ay, ax)
            ang_b = math.atan2(by, bx)
            
            diff = ang_b - ang_a
            while diff < 0: diff += 2 * math.pi
            while diff >= 2 * math.pi: diff -= 2 * math.pi
            
            angles.append(math.degrees(diff))
            
        return sorted(angles)

    def _max_angle_deviation(self, idx: int):
        angles = self._angles_at_steiner(idx)
        if not angles:
            return 0.0
        return max(abs(a - 120.0) for a in angles)

    def _merge_close_points(self, threshold=None):
        if threshold is None:
            threshold = 0.01 * self._bb_diag
        merges = 0
        n = len(self.points)
        steiner_indices = list(range(self.n_terminals, n))
        to_remove = set()
        
        # 1. Merge Steiner points with other Steiner points
        for si in range(len(steiner_indices)):
            i = steiner_indices[si]
            if i in to_remove:
                continue
            for sj in range(si + 1, len(steiner_indices)):
                j = steiner_indices[sj]
                if j in to_remove:
                    continue
                if self._dist(i, j) < threshold:
                    self.points[i][0] = (self.points[i][0] + self.points[j][0]) / 2
                    self.points[i][1] = (self.points[i][1] + self.points[j][1]) / 2
                    # Preserving momentum via velocity averaging
                    self.velocities[i][0] = (self.velocities[i][0] + self.velocities[j][0]) / 2
                    self.velocities[i][1] = (self.velocities[i][1] + self.velocities[j][1]) / 2
                    to_remove.add(j)
                    merges += 1
        
        # 2. Merge Steiner points with Terminal points
        for i in steiner_indices:
            if i in to_remove:
                continue
            for t_idx in range(self.n_terminals):
                if self._dist(i, t_idx) < threshold:
                    to_remove.add(i)
                    merges += 1
                    break

        for idx in sorted(to_remove, reverse=True):
            self.points.pop(idx)
            if idx < len(self.velocities): self.velocities.pop(idx)
            self.n_steiner -= 1
            self.n_total -= 1
        return merges

    def _prune_redundant_steiner(self, min_iteration=30, force_threshold=0.01):
        """
        Remove Steiner points that have degree 1 or 2 in the MST.
        Only happens after min_iteration to give points time to migrate.
        Only happens if the point is at equilibrium (low force).
        """
        if self.iteration < min_iteration:
            return 0

        pruned = 0
        steiner_start = self.n_terminals
        to_remove = []
        
        for i in range(len(self.points) - 1, steiner_start - 1, -1):
            deg = len(self.adj.get(i, []))
            if deg <= 2:
                # Check physical equilibrium: if force is high, it's still moving to a junction
                fx, fy = self._compute_gradient(i)
                if math.hypot(fx, fy) < force_threshold:
                    to_remove.append(i)
                    pruned += 1
        
        for idx in to_remove:
            self.points.pop(idx)
            if idx < len(self.velocities): self.velocities.pop(idx)
            self.n_steiner -= 1
            self.n_total -= 1
        
        if pruned > 0:
            self._compute_mst()
        return pruned

    def step_gradient(self, learning_rate=0.05, sa_temperature=0.0):
        self._compute_mst()
        old_weight = self.mst_weight

        steiner_start = self.n_terminals
        forces = []
        for i in range(steiner_start, len(self.points)):
            fx, fy = self._compute_gradient(i)
            forces.append((fx, fy))

        max_force = 0.0
        # Adaptive friction: higher friction (momentum decay) when cooling down to stop waves
        friction = 0.7 if sa_temperature < 0.001 else 0.4
        
        for k, i in enumerate(range(steiner_start, len(self.points))):
            fx, fy = forces[k]
            
            if i >= len(self.velocities):
                self.velocities.extend([[0.0, 0.0]] * (i - len(self.velocities) + 1))
            
            vx, vy = self.velocities[i]
            
            # Update velocity: momentum + force
            vx = vx * friction + learning_rate * fx
            vy = vy * friction + learning_rate * fy
            
            # HARD CLAMP: Prevent wild jumps
            limit = 0.02 * self._bb_diag
            vm = math.hypot(vx, vy)
            if vm > limit:
                vx *= limit / vm
                vy *= limit / vm
            
            self.points[i][0] += vx
            self.points[i][1] += vy
            self.velocities[i] = [vx, vy]
            
            max_force = max(max_force, math.hypot(fx, fy))

        # Thermal perturbation (Simulated Annealing)
        if sa_temperature > 1e-6:
            for i in range(steiner_start, len(self.points)):
                dx = self.rng.gauss(0, sa_temperature * 0.5)
                dy = self.rng.gauss(0, sa_temperature * 0.5)
                self.points[i][0] += dx
                self.points[i][1] += dy
            margin = 0.02
            for i in range(steiner_start, len(self.points)):
                self.points[i][0] = max(self._min_x + margin, min(self._max_x - margin, self.points[i][0]))
                self.points[i][1] = max(self._min_y + margin, min(self._max_y - margin, self.points[i][1]))

        # ACTIVE FISSION: Break degree-4 traps
        fissions = 0
        if len(self.points) < self.n_terminals + self.n_steiner + 5:
            for i in range(steiner_start, len(self.points)):
                deg = len(self.adj.get(i, []))
                if deg >= 4:
                    fx, fy = self._compute_gradient(i)
                    if math.hypot(fx, fy) < 0.1:
                        ang = self.rng.uniform(0, 2 * math.pi)
                        eps = 0.005
                        new_pt = [self.points[i][0] + eps * math.cos(ang),
                                  self.points[i][1] + eps * math.sin(ang)]
                        self.points.append(new_pt)
                        self.velocities.append([0.0, 0.0])
                        self.n_steiner += 1
                        self.n_total += 1
                        fissions += 1
        
        if fissions > 0:
            self._compute_mst()

        # Merging (Annihilation)
        merges = self._merge_close_points(threshold=0.005 * self._bb_diag)
        if merges > 0:
            self._compute_mst()
        
        # Redundancy Pruning (Delayed + Equilibrium Check)
        pruned = self._prune_redundant_steiner()
        
        # Final MST check
        self._compute_mst()
        new_weight = self.mst_weight

        max_deviation = 0.0
        for i in range(steiner_start, len(self.points)):
            dev = self._max_angle_deviation(i)
            max_deviation = max(max_deviation, dev)

        self.iteration += 1
        record = {
            "iteration": self.iteration,
            "n_points": len(self.points),
            "n_steiner": self.n_steiner,
            "mst_weight": self.mst_weight,
            "weight_delta": new_weight - old_weight,
            "max_force": max_force,
            "merges": merges,
            "pruned": pruned,
            "max_120_deviation": max_deviation,
            "points": [tuple(p) for p in self.points],
            "mst_edges": list(self.mst_edges),
            "sa_temperature": sa_temperature,
            "terminal_count": self.n_terminals,
        }
        self.history.append(record)
        return record

    def run(self, max_iterations=200, learning_rate=0.05,
            sa_initial_temp=0.02, sa_cooling=0.98, sa_min_temp=1e-6,
            convergence_threshold=1e-5):
        temperature = sa_initial_temp
        prev_weight = float("inf")

        for _ in range(max_iterations):
            current_temp = temperature if temperature > 0.001 else 0.0
            record = self.step_gradient(
                learning_rate=learning_rate,
                sa_temperature=current_temp,
            )
            if abs(prev_weight - record["mst_weight"]) < convergence_threshold and current_temp == 0:
                record["converged"] = True
                break
            prev_weight = record["mst_weight"]
            temperature = max(sa_min_temp, temperature * sa_cooling)
            record["sa_temperature"] = temperature

        self._compute_mst()
        result_edges = []
        for u, v in self.mst_edges:
            result_edges.append((u, v, self._dist(u, v)))

        return {
            "mst_weight": self.mst_weight,
            "steiner_points": [tuple(p) for p in self.points[self.n_terminals:]],
            "all_points": [tuple(p) for p in self.points],
            "edges": result_edges,
            "mst_edges": list(self.mst_edges),
            "iterations": self.iteration,
            "history": self.history,
            "terminals": list(self.terminals),
            "120_deviations": [
                self._max_angle_deviation(i) for i in range(self.n_terminals, len(self.points))
            ],
        }
