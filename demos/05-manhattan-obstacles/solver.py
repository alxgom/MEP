"""
Obstacle-Aware Manhattan Steiner Solver
=======================================
Finds the shortest rectilinear network connecting terminals while avoiding obstacles.
Optimized for pre-computed environment data.

Solver Hierarchy
----------------
 Baseline
   solve_mst()              Geodesic MST on terminals only (KMB steps 1-3, no re-MST).

 Classical / Guaranteed
   solve_kou()              Full Kou-Markowsky-Berman '81 (steps 1-5).
                            Approximation ratio: <= 2(1 - 1/|S|).

 Greedy Heuristics
   solve_greedy()           Iterated 1-Steiner: exhaustive greedy candidate search.
   solve_fast_corner()      KMB-style: only tests L-bend corner candidates.
   solve_fast_corner(stochastic=True)   Same but picks randomly from Top-3 gains.
   solve_prune()            Start from all-nodes MST, prune non-terminal leaves.

 Stochastic / Escape
   solve_stochastic_kou()   KMB with perturbed terminal metric closure; best of N trials.
   solve_anisotropic_kou()  KMB with directionally-weighted A* heuristic (NetworkX).
                            Optionally stochastic when n_trials > 1.

 Population-Based
   solve_monte_carlo()      Evolutionary: tournament-select + random mutation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx


class ObstacleSteinerSolver:
    def __init__(self, env):
        self.env = env
        self.terminals = env.terminal_node_indices
        self.active_nodes = list(self.terminals)
        self.mst_edges = []
        self.mst_weight = 0.0
        # Build NetworkX graph once from the environment's CSR adjacency matrix.
        # Used by A*-based solvers (solve_anisotropic_kou).
        self.nx_graph = nx.from_scipy_sparse_array(
            env.adj_matrix, edge_attribute="weight", create_using=nx.Graph()
        )

    @classmethod
    def from_precomputed(
        cls,
        terminal_indices,
        dist_matrix,
        predecessors,
        n_nodes,
        node_map,
        nodes,
        obstacles,
        adj_matrix=None,
    ):
        """
        Worker-friendly factory to avoid pickling the full environment.
        Pass adj_matrix (CSR) to enable A*-based solvers in batch contexts.
        """
        instance = cls.__new__(cls)

        class MockEnv:
            def __init__(self, ti, dm, pred, nn, nm, nd, obs, am):
                self.terminal_node_indices = ti
                self.dist_matrix = dm
                self.predecessors = pred
                self.n_nodes = nn
                self.node_map = nm
                self.nodes = nd
                self.obstacles = obs
                self.adj_matrix = am

            def get_path(self, s, e):
                path = []
                curr = e
                while curr != s:
                    if curr == -9999 or curr < 0:
                        return []
                    path.append(curr)
                    curr = self.predecessors[s, curr]
                path.append(s)
                return path[::-1]

        instance.env = MockEnv(
            terminal_indices, dist_matrix, predecessors,
            n_nodes, node_map, nodes, obstacles, adj_matrix,
        )
        instance.terminals = terminal_indices
        instance.active_nodes = list(terminal_indices)
        instance.mst_edges = []
        instance.mst_weight = 0.0
        if adj_matrix is not None:
            instance.nx_graph = nx.from_scipy_sparse_array(
                adj_matrix, edge_attribute="weight", create_using=nx.Graph()
            )
        else:
            instance.nx_graph = None
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_geodesic_mst(self, node_indices: List[int]) -> float:
        """MST on the submatrix of precomputed geodesic distances."""
        k = len(node_indices)
        if k < 2:
            return 0.0
        sub_dist = self.env.dist_matrix[np.ix_(node_indices, node_indices)]
        if np.any(np.isinf(sub_dist)):
            return 1e9
        mst_sparse = minimum_spanning_tree(csr_matrix(sub_dist))
        u_sub, v_sub = mst_sparse.nonzero()
        self.mst_edges = [
            (node_indices[u], node_indices[v]) for u, v in zip(u_sub, v_sub)
        ]
        self.mst_weight = float(np.sum(mst_sparse.data))
        return self.mst_weight

    def _get_mst_corner_candidates(self) -> np.ndarray:
        """Return grid nodes at L-bend corners of current MST edges."""
        candidates = []
        for u, v in self.mst_edges:
            p1, p2 = self.env.nodes[u], self.env.nodes[v]
            if not np.isclose(p1[0], p2[0]) and not np.isclose(p1[1], p2[1]):
                c1 = (float(p1[0]), float(p2[1]))
                c2 = (float(p2[0]), float(p1[1]))
                for c in [c1, c2]:
                    if c in self.env.node_map:
                        idx = self.env.node_map[c]
                        if idx not in self.active_nodes:
                            candidates.append(idx)
        return np.unique(np.array(candidates)) if candidates else np.array([])

    def _prune_leaves(self, terminal_set: set) -> None:
        """Iteratively remove non-terminal degree-1 nodes from active_nodes."""
        while True:
            leaves = [
                i for i in self.active_nodes
                if i not in terminal_set
                and sum(1 for e in self.mst_edges if i in e) <= 1
            ]
            if not leaves:
                break
            self.active_nodes = [i for i in self.active_nodes if i not in leaves]
            self._compute_geodesic_mst(self.active_nodes)

    def _prune_collinear_passthrough(self, terminal_set: set) -> None:
        """
        Remove degree-2 non-terminal nodes that are straight pass-throughs.

        A degree-2 Steiner node where both MST neighbors lie on the same
        horizontal or vertical line is not a real junction — it is just an
        intermediate waypoint on a straight pipe segment. Removing it does
        not change routing or total length; it only cleans up the Steiner
        node count so reported numbers reflect real T/X-junctions only.

        This is the second pruning stage after _prune_leaves().
        """
        nodes = self.env.nodes
        while True:
            to_remove = []
            # Build adjacency from current mst_edges for fast lookup
            adj: dict = {i: [] for i in self.active_nodes}
            for u, v in self.mst_edges:
                adj[u].append(v)
                adj[v].append(u)
            for i in self.active_nodes:
                if i in terminal_set:
                    continue
                nbrs = adj[i]
                if len(nbrs) != 2:
                    continue
                j, k = nbrs
                pi, pj, pk = nodes[i], nodes[j], nodes[k]
                horiz = np.isclose(pi[1], pj[1]) and np.isclose(pi[1], pk[1])
                vert  = np.isclose(pi[0], pj[0]) and np.isclose(pi[0], pk[0])
                if horiz or vert:
                    to_remove.append(i)
            if not to_remove:
                break
            self.active_nodes = [i for i in self.active_nodes if i not in to_remove]
            self._compute_geodesic_mst(self.active_nodes)

    def _full_prune(self, terminal_set: set) -> None:
        """Full two-stage pruning: leaves first, then collinear pass-throughs."""
        self._prune_leaves(terminal_set)
        self._prune_collinear_passthrough(terminal_set)

    def _expand_segments(self) -> List[Tuple[int, int]]:
        """
        Expand MST edges to actual grid segments, deduplicated.

        Deduplication matters when two MST edges expand to paths that share
        intermediate grid segments (shared trunks). Without deduplication,
        mst_weight would overcount those segments in the metric closure.
        """
        seen: set = set()
        segments: List[Tuple[int, int]] = []
        for u, v in self.mst_edges:
            path = self.env.get_path(u, v)
            for i in range(len(path) - 1):
                key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                if key not in seen:
                    seen.add(key)
                    segments.append((path[i], path[i + 1]))
        return segments

    def _true_routed_length(self, segments: List[Tuple[int, int]]) -> float:
        """
        True physical pipe length: sum of actual coordinate distances
        over deduplicated segments. This is the ground-truth MEP metric —
        independent of how the metric closure was constructed.
        """
        nodes = self.env.nodes
        return sum(
            abs(nodes[u][0] - nodes[v][0]) + abs(nodes[u][1] - nodes[v][1])
            for u, v in segments
        )

    def _result(self, terminal_set: set) -> Dict[str, Any]:
        segments = self._expand_segments()
        from collections import defaultdict
        degrees = defaultdict(int)
        for u, v in segments:
            degrees[u] += 1
            degrees[v] += 1
        
        branching_steiners = [
            i for i in self.active_nodes
            if i not in terminal_set
            and degrees[i] >= 3
        ]
        return {
            "weight": self._true_routed_length(segments),  # true physical length
            "steiner_indices": branching_steiners,
            "segments": segments,
        }

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def solve_mst(self) -> Dict[str, Any]:
        """
        Geodesic MST Baseline (Shortest Path Heuristic).

        Builds the MST on terminals only, using precomputed geodesic distances.
        No Steiner points are added. Equivalent to KMB steps 1-2 + path expansion
        (without the re-MST or pruning steps). Always obstacle-aware.

        Complexity: O(|T|^2) distance lookups + O(|T| log |T|) Kruskal.
        """
        self.active_nodes = list(self.terminals)
        self._compute_geodesic_mst(self.active_nodes)
        segments = self._expand_segments()
        return {
            "weight": self._true_routed_length(segments),
            "steiner_indices": [],
            "segments": segments,
        }

    def solve_kou(self) -> Dict[str, Any]:
        """
        Kou-Markowsky-Berman (KMB) Algorithm, 1981.

        Guaranteed approximation ratio: <= 2(1 - 1/|S|) of the optimal Steiner tree.

        Steps:
          1+2. Build metric closure on terminals (geodesic distances); compute MST.
          3.   Expand each MST edge to its actual shortest grid path;
               harvest all intermediate nodes as Steiner candidates.
          4.   Re-MST on terminals ∪ candidate set (removes cycles from shared paths).
          5.   Prune: iteratively remove non-terminal leaves.

        Contrast with solve_mst(): that method skips the re-MST and pruning,
        leaving redundant segments when expanded paths share intermediate nodes.

        Complexity: O(|T|^2) + O(|T| log |T|) + O(|T| * path_len) + re-MST.
        """
        terminal_set = set(self.terminals)

        # Steps 1+2
        self.active_nodes = list(self.terminals)
        self._compute_geodesic_mst(self.active_nodes)
        terminal_mst_edges = list(self.mst_edges)

        # Step 3: harvest intermediate nodes
        candidates = set()
        for u, v in terminal_mst_edges:
            for node in self.env.get_path(u, v):
                if node not in terminal_set:
                    candidates.add(node)

        # Step 4: re-MST
        self.active_nodes = list(self.terminals) + list(candidates)
        self._compute_geodesic_mst(self.active_nodes)

        # Step 5: prune
        self._full_prune(terminal_set)

        return self._result(terminal_set)

    def solve_greedy(self, max_steiner: int = 15) -> Dict[str, Any]:
        """
        Iterated 1-Steiner (Exhaustive Greedy).

        At each round, evaluates every non-terminal grid node as a potential
        Steiner point and adds the one that gives the maximum weight reduction.
        Repeats until no improving candidate exists or max_steiner is reached.

        Quality: typically the best deterministic heuristic but O(V * max_steiner)
        MST evaluations makes it slow on large grids.
        """
        terminal_set = set(self.terminals)
        self.active_nodes = list(self.terminals)
        best_w = self._compute_geodesic_mst(self.active_nodes)
        candidates = [i for i in range(self.env.n_nodes) if i not in terminal_set]
        for _ in range(max_steiner):
            best_gain, best_cand = 0.0, -1
            for cand in candidates:
                if cand in self.active_nodes:
                    continue
                w = self._compute_geodesic_mst(self.active_nodes + [cand])
                gain = best_w - w
                if gain > best_gain + 1e-6:
                    best_gain, best_cand = gain, cand
            if best_cand != -1:
                self.active_nodes.append(best_cand)
                best_w -= best_gain
            else:
                break
        self._compute_geodesic_mst(self.active_nodes)
        return self._result(terminal_set)

    def solve_fast_corner(
        self, max_steiner: int = 25, stochastic: bool = False, temperature: float = 0.5
    ) -> Dict[str, Any]:
        """
        Fast Corner Kick (Targeted L-Bend Heuristic).

        Instead of evaluating all grid nodes, only considers L-bend corner nodes
        of current MST edges — the two axis-aligned corners that connect any
        diagonal MST edge pair. This restricts the search to geometrically motivated
        candidates, making each round O(|MST_edges|) instead of O(V).

        When stochastic=True, picks randomly from the Top-3 gain candidates
        instead of always taking the best. This gives stochastic escape at
        almost no extra cost.
        """
        terminal_set = set(self.terminals)
        self.active_nodes = list(self.terminals)
        self._compute_geodesic_mst(self.active_nodes)
        for _ in range(max_steiner):
            candidates = list(self._get_mst_corner_candidates())
            if not candidates:
                break
            current_w = self.mst_weight
            gains = np.array([
                max(0, current_w - self._compute_geodesic_mst(self.active_nodes + [c]))
                for c in candidates
            ])
            best_cand = -1
            if stochastic:
                valid = np.where(gains > 1e-6)[0]
                if len(valid) == 0:
                    break
                top_k = min(3, len(valid))
                top_idx = np.argsort(gains[valid])[-top_k:]
                best_cand = candidates[valid[np.random.choice(top_idx)]]
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] > 1e-6:
                    best_cand = candidates[best_idx]
            if best_cand != -1:
                self.active_nodes.append(best_cand)
                self._compute_geodesic_mst(self.active_nodes)
            else:
                break
        return self._result(terminal_set)

    def solve_prune(self) -> Dict[str, Any]:
        """
        Dense Grid Pruning (Top-Down).

        Starts with ALL grid nodes included in the MST, then iteratively removes
        non-terminal nodes that are degree <= 2 (i.e., passing-through or leaf
        nodes that contribute no branching value). Converges to a locally minimal
        Steiner tree. Expensive to initialize (O(V^2) for the first MST) but finds
        Steiner points that greedy bottom-up approaches miss.
        """
        terminal_set = set(self.terminals)
        self.active_nodes = list(range(self.env.n_nodes))
        self._compute_geodesic_mst(self.active_nodes)
        while True:
            to_remove = [
                i for i in self.active_nodes
                if i not in terminal_set
                and len([e for e in self.mst_edges if i in e]) <= 2
            ]
            if not to_remove:
                break
            self.active_nodes = [i for i in self.active_nodes if i not in to_remove]
            self._compute_geodesic_mst(self.active_nodes)
        return self._result(terminal_set)

    def solve_stochastic_kou(
        self, n_trials: int = 20, perturbation: float = 0.10
    ) -> Dict[str, Any]:
        """
        Stochastic KMB — Metric Closure Perturbation.

        Runs KMB n_trials times, each with a slightly different terminal metric
        closure (edge weights multiplied by symmetric random noise in
        [1-perturbation, 1+perturbation]). This changes which MST topology is
        chosen in Step 2, which in turn changes which intermediate nodes are
        harvested in Step 3, diversifying the Steiner candidate pool.

        All candidates are evaluated with TRUE (unperturbed) distances.
        Returns the best result across all trials.

        Why this escapes local minima
        ------------------------------
        Standard KMB is trapped by Dijkstra's deterministic tie-breaking:
        on a uniform Hanan grid, many paths have identical cost but different shapes.
        Perturbation breaks these ties differently each trial, surfacing intermediate
        nodes from equal-cost routes that the deterministic solver never visits.

        Complexity: n_trials * O(KMB).
        """
        terminal_set = set(self.terminals)
        t_idx = list(self.terminals)
        base_sub = self.env.dist_matrix[np.ix_(t_idx, t_idx)].copy()

        best_weight = float("inf")
        best_active, best_edges = None, None

        for _ in range(n_trials):
            # Symmetric noise matrix
            noise = np.random.uniform(1 - perturbation, 1 + perturbation, base_sub.shape)
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 1.0)
            perturbed = base_sub * noise

            # Step 2: MST on perturbed metric closure
            mst = minimum_spanning_tree(csr_matrix(perturbed))
            u_sub, v_sub = mst.nonzero()
            trial_edges = [(t_idx[u], t_idx[v]) for u, v in zip(u_sub, v_sub)]

            # Step 3: Expand with TRUE predecessors
            candidates = set()
            for u, v in trial_edges:
                for node in self.env.get_path(u, v):
                    if node not in terminal_set:
                        candidates.add(node)

            # Step 4: Re-MST with TRUE distances
            self.active_nodes = t_idx + list(candidates)
            self._compute_geodesic_mst(self.active_nodes)

            # Step 5: Prune
            self._full_prune(terminal_set)

            # Evaluate best trial using true, deduplicated physical routed length
            trial_segs = self._expand_segments()
            trial_len = self._true_routed_length(trial_segs)
            if trial_len < best_weight:
                best_weight = trial_len
                best_active = list(self.active_nodes)
                best_edges = list(self.mst_edges)

        self.active_nodes = best_active
        self.mst_edges = best_edges
        self.mst_weight = best_weight
        return self._result(terminal_set)

    def solve_anisotropic_kou(
        self,
        w_x: float = 1.0,
        w_y: float = 1.0,
        n_trials: int = 1,
        sigma: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Anisotropic KMB — Directionally-Weighted A* (NetworkX).

        Replaces KMB Step 3's deterministic predecessor backtrace with NetworkX
        A* using a directionally-weighted Manhattan heuristic:

            h(u, v) = w_x * |Δx| + w_y * |Δy|

        This biases A* toward different axis preferences, tracing physically
        different route shapes between the same terminal pair even when geodesic
        distances are identical — surfacing different intermediate Steiner nodes.

        Parameters
        ----------
        w_x, w_y : float
            Directional weights. w_y > w_x penalises vertical movement (models
            real MEP costs: risers need floor penetrations, slope constraints).
            Admissibility: guaranteed when w_x, w_y <= min edge cost (~1.0).
        n_trials : int
            Number of independent A* passes. If > 1, randomizes (w_x, w_y)
            around (w_x ± sigma, w_y ± sigma) each trial for stochastic escape.
        sigma : float
            Standard deviation of per-trial weight noise. 0 = deterministic.

        Steps 1+2 always use TRUE geodesic distances (APSP) for the terminal
        metric closure. Only Step 3 (path shape discovery) uses A*.
        Steps 4+5 (re-MST + pruning) use TRUE distances.

        Complexity: n_trials * (O(|T|^2) + O(|T| * A*_cost) + O(re-MST)).
        """
        if self.nx_graph is None:
            raise RuntimeError(
                "solve_anisotropic_kou requires nx_graph. "
                "Pass adj_matrix to from_precomputed()."
            )

        terminal_set = set(self.terminals)
        t_idx = list(self.terminals)
        nodes = self.env.nodes

        best_weight = float("inf")
        best_active, best_edges = None, None

        for _ in range(n_trials):
            # Sample directional weights (captured via default args for closure safety)
            trial_wx = max(0.01, w_x + (np.random.uniform(-sigma, sigma) if sigma > 0 else 0.0))
            trial_wy = max(0.01, w_y + (np.random.uniform(-sigma, sigma) if sigma > 0 else 0.0))

            def heuristic(u, v, wx=trial_wx, wy=trial_wy):
                return wx * abs(nodes[u][0] - nodes[v][0]) + wy * abs(nodes[u][1] - nodes[v][1])

            # Steps 1+2: MST on TRUE terminal metric closure
            sub_dist = self.env.dist_matrix[np.ix_(t_idx, t_idx)]
            mst = minimum_spanning_tree(csr_matrix(sub_dist))
            u_sub, v_sub = mst.nonzero()
            trial_edges = [(t_idx[u], t_idx[v]) for u, v in zip(u_sub, v_sub)]

            # Step 3: Expand with anisotropic A* (different path shapes)
            candidates = set()
            for u, v in trial_edges:
                try:
                    path = nx.astar_path(
                        self.nx_graph, u, v, heuristic=heuristic, weight="weight"
                    )
                    for node in path:
                        if node not in terminal_set:
                            candidates.add(node)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Fallback: use precomputed Dijkstra path
                    for node in self.env.get_path(u, v):
                        if node not in terminal_set:
                            candidates.add(node)

            # Step 4: Re-MST with TRUE distances
            self.active_nodes = t_idx + list(candidates)
            self._compute_geodesic_mst(self.active_nodes)

            # Step 5: Prune
            self._full_prune(terminal_set)

            # Evaluate best trial using true, deduplicated physical routed length
            trial_segs = self._expand_segments()
            trial_len = self._true_routed_length(trial_segs)
            if trial_len < best_weight:
                best_weight = trial_len
                best_active = list(self.active_nodes)
                best_edges = list(self.mst_edges)

        self.active_nodes = best_active
        self.mst_edges = best_edges
        self.mst_weight = best_weight
        return self._result(terminal_set)

    def solve_monte_carlo(
        self, population_size: int = 5, generations: int = 15
    ) -> Dict[str, Any]:
        """
        Monte Carlo Population (Evolutionary / Genetic-Style).

        Maintains a population of random Steiner node sets. Each generation:
          - Evaluate all individuals by MST weight.
          - Keep top-2 as elite parents.
          - Fill remaining slots by mutating a random parent
            (replace one random Steiner node with a random grid node).
        Terminates after `generations` rounds. Applies leaf pruning to the winner.

        The least constrained solver — no geometric bias, pure stochastic search.
        Useful as an upper-bound check on what random exploration can find.

        Complexity: O(population_size * generations * |T+S|^2).
        """
        terminal_set = set(self.terminals)
        best_nodes = list(self.terminals)
        best_w = self._compute_geodesic_mst(best_nodes)
        all_indices = list(range(self.env.n_nodes))
        population = [
            list(self.terminals) + np.random.choice(
                all_indices, min(len(self.terminals), 15), replace=False
            ).tolist()
            for _ in range(population_size)
        ]
        for _ in range(generations):
            weights = [self._compute_geodesic_mst(nodes) for nodes in population]
            for w, nodes in zip(weights, population):
                if w < best_w:
                    best_w, best_nodes = w, list(nodes)
            sorted_pop = [p for _, p in sorted(zip(weights, population))]
            new_pop = sorted_pop[:2]
            while len(new_pop) < population_size:
                child = list(new_pop[np.random.randint(0, len(new_pop))])
                s_idx = [i for i in range(len(child)) if child[i] not in terminal_set]
                if s_idx:
                    child[np.random.choice(s_idx)] = np.random.choice(all_indices)
                new_pop.append(child)
            population = new_pop
        self.active_nodes = best_nodes
        self._compute_geodesic_mst(self.active_nodes)
        self._full_prune(terminal_set)
        return self._result(terminal_set)
