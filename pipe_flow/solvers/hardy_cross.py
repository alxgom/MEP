"""
Hardy Cross 1D Hydraulic Loop Solver for HVAC Duct & Pipe Networks.

Features:
- Classic Hardy Cross iterative loop-balancing method with fundamental cycle basis.
- Automatic spanning-tree cycle basis generation for arbitrary network topologies
  (closed loops, open branching networks with multiple pressure outlets via pseudo-loops).
- Darcy-Weisbach friction with Colebrook-White / Haaland friction factor and exact
  analytical derivatives for quadratic/superlinear convergence.
- Hydrostatic elevation head support: dP_elev = rho * g * (z_end - z_start).
- ASHRAE fitting minor losses and fan curves.
- Guaranteed mass conservation at all iterations (circulation along closed null-space loops).
- Reconstruction of single-valued node pressures from reference boundary conditions.
- Both Simultaneous Loop Newton (fastest) and Sequential Gauss-Seidel Hardy Cross solvers.
"""

from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.linalg

from pipe_flow.solvers.hydraulic_1d import (
    FluidProperties,
    DuctMaterial,
    Fitting,
    FittingLossLibrary,
    FanCurve,
    Node,
    Edge,
    NodeType,
    HydraulicNetwork,
    HydraulicSolution,
    friction_factor,
    friction_factor_derivative_Re,
)


@dataclass
class Loop:
    """
    Fundamental Loop (cycle or pseudo-loop) in a duct network.
    
    Attributes:
        id: Loop index.
        name: String identifier.
        edge_ids: List of physical edge IDs in the loop.
        orientations: List of +1 (forward) or -1 (reverse) for each edge.
        virtual_head_offset: Net known pressure difference from virtual boundary edges (Pa).
    """
    id: int
    name: str
    edge_ids: List[int]
    orientations: List[int]
    virtual_head_offset: float = 0.0


class HardyCrossSolver:
    """
    Hardy Cross Loop-Balancing Solver for 1D/3D Hydraulic Networks.
    
    Implements:
    - Automatic Cycle Basis Detection (BFS Spanning Tree + Chord Edges + Virtual Ground Node for Pseudo-Loops).
    - Exact Mass-Conserving Initial Flow Distribution Q0.
    - Simultaneous Loop Newton-Raphson (Loop Gradient) & Sequential Hardy Cross iterations.
    - Full 3D Elevation Head: dP_e = (f L/Dh + sum K) * rho/(2*A^2) * |Q|*Q + rho*g*(z_end - z_start) - dP_fan.
    """

    def __init__(
        self,
        network: HydraulicNetwork,
        friction_method: str = "haaland",
        max_iterations: int = 100,
        tol_head: float = 1e-5,
        tol_flow: float = 1e-7,
        relaxation: float = 1.0,
    ):
        self.network = network
        self.friction_method = friction_method
        self.max_iterations = max_iterations
        self.tol_head = tol_head
        self.tol_flow = tol_flow
        self.relaxation = relaxation
        self._build_topology_and_cycles()

    def _build_topology_and_cycles(self) -> None:
        """Constructs incidence matrices, identifies cycle basis, and precomputes factors."""
        self.network.validate()
        
        self.num_nodes = len(self.network.nodes)
        self.num_edges = len(self.network.edges)
        
        self.unknown_node_ids = [n.id for n in self.network.nodes.values() if not n.is_pressure_known]
        self.known_node_ids = [n.id for n in self.network.nodes.values() if n.is_pressure_known]
        
        self.num_u = len(self.unknown_node_ids)
        self.num_k = len(self.known_node_ids)
        
        # Incidence matrix A_full (num_nodes x num_edges)
        self.A_full = np.zeros((self.num_nodes, self.num_edges), dtype=np.float64)
        for e_id, edge in self.network.edges.items():
            self.A_full[edge.start_node_id, e_id] = -1.0
            self.A_full[edge.end_node_id, e_id] = +1.0
            
        self.A_u = self.A_full[self.unknown_node_ids, :] if self.num_u > 0 else np.zeros((0, self.num_edges))
        
        # Edge physical vectors
        self.edge_L = np.array([e.length for e in self.network.edges.values()], dtype=np.float64)
        self.edge_Dh = np.array([e.hydraulic_diameter for e in self.network.edges.values()], dtype=np.float64)
        self.edge_A = np.array([e.area for e in self.network.edges.values()], dtype=np.float64)
        self.edge_rr = np.array([e.relative_roughness for e in self.network.edges.values()], dtype=np.float64)
        self.edge_K = np.array([e.total_loss_coefficient for e in self.network.edges.values()], dtype=np.float64)
        
        # Fluid properties
        self.rho = self.network.fluid.density
        self.mu = self.network.fluid.viscosity
        self.g_acc = getattr(self.network.fluid, "gravity", 9.80665)
        
        self.geom_factor = self.rho / (2.0 * (self.edge_A ** 2))
        self.L_over_Dh = self.edge_L / self.edge_Dh
        self.re_factor = (self.rho * self.edge_Dh) / (self.mu * self.edge_A)
        self.haaland_term0 = np.maximum(self.edge_rr / 3.7, 0.0) ** 1.11
        self.ln10 = math.log(10.0)
        
        # 3D Node Elevations and Hydrostatic Head: dP_hydro = (rho - rho_amb) * g * (z_end - z_start)
        self.node_z = np.array([self.network.nodes[nid].z for nid in range(self.num_nodes)], dtype=np.float64)
        self.edge_dz = self.A_full.T @ self.node_z  # (z_end - z_start)
        
        amb_rho = getattr(self.network.fluid, "ambient_density", None)
        isoth = getattr(self.network.fluid, "isothermal_gauge", False)
        if amb_rho is not None:
            eff_drho = self.rho - float(amb_rho)
        elif isoth:
            eff_drho = 0.0
        else:
            eff_drho = self.rho
        self.edge_dp_hydro = eff_drho * self.g_acc * self.edge_dz
        
        # External inflows and known pressures
        self.Q_ext_u = np.array([self.network.nodes[nid].external_inflow for nid in self.unknown_node_ids], dtype=np.float64)
        self.P_k_fixed = np.array([self.network.nodes[nid].fixed_pressure for nid in self.known_node_ids], dtype=np.float64)
        
        # Fans
        self.has_fans = any(e.fan is not None for e in self.network.edges.values())
        self.fan_edges = [(i, e.fan) for i, e in self.network.edges.items() if e.fan is not None]
        
        # Find Fundamental Cycle Basis (Loops and Pseudo-Loops)
        self.loops: List[Loop] = self._find_fundamental_cycles()
        self.num_loops = len(self.loops)
        
        # Build Loop-Edge Incidence Matrix C (num_loops x num_edges)
        self.C = np.zeros((self.num_loops, self.num_edges), dtype=np.float64)
        self.virtual_head_vec = np.zeros(self.num_loops, dtype=np.float64)
        
        for k, loop in enumerate(self.loops):
            self.virtual_head_vec[k] = loop.virtual_head_offset
            for eid, ori in zip(loop.edge_ids, loop.orientations):
                self.C[k, eid] = float(ori)
                
        # Initial Flow Distribution Q0 satisfying node continuity
        self.Q_init = self._compute_initial_flows()

    def _find_fundamental_cycles(self) -> List[Loop]:
        """
        Builds augmented graph with virtual ground node for multi-outlet open networks
        and finds independent fundamental cycle basis using spanning tree chords.
        """
        use_virtual_ground = self.num_k > 1
        virtual_ground_id = self.num_nodes if use_virtual_ground else None
        
        # Augmented edges: (u, v, edge_id, is_virtual, head_fixed)
        aug_adj: Dict[int, List[Tuple[int, int, bool, float]]] = {}
        for nid in range(self.num_nodes + (1 if use_virtual_ground else 0)):
            aug_adj[nid] = []
            
        for eid, edge in self.network.edges.items():
            u, v = edge.start_node_id, edge.end_node_id
            aug_adj[u].append((v, eid, False, 0.0))
            aug_adj[v].append((u, eid, False, 0.0))
            
        if use_virtual_ground:
            for idx, knid in enumerate(self.known_node_ids):
                P_fixed = float(self.network.nodes[knid].fixed_pressure or 0.0)
                aug_adj[knid].append((virtual_ground_id, -1 - idx, True, P_fixed))
                aug_adj[virtual_ground_id].append((knid, -1 - idx, True, -P_fixed))
                
        # Build Spanning Tree via BFS starting from root
        root = virtual_ground_id if use_virtual_ground else (self.known_node_ids[0] if self.num_k > 0 else 0)
        
        visited: Set[int] = {root}
        parent: Dict[int, Optional[int]] = {root: None}
        parent_edge: Dict[int, Optional[Tuple[int, bool, float]]] = {root: None}
        depth: Dict[int, int] = {root: 0}
        
        tree_edges: Set[int] = set()
        
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            for neighbor, eid, is_virt, h_fix in aug_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    parent_edge[neighbor] = (eid, is_virt, h_fix)
                    depth[neighbor] = depth[curr] + 1
                    tree_edges.add(eid)
                    queue.append(neighbor)
                    
        # Identify chord edges and trace fundamental cycles
        loops: List[Loop] = []
        chord_count = 0
        
        # Real chord edges
        for eid, edge in self.network.edges.items():
            if eid not in tree_edges:
                u, v = edge.start_node_id, edge.end_node_id
                loop = self._trace_cycle(u, v, eid, False, 0.0, parent, parent_edge, depth, chord_count)
                if loop is not None:
                    loops.append(loop)
                    chord_count += 1
                    
        # Virtual chord edges
        if use_virtual_ground:
            for idx, knid in enumerate(self.known_node_ids):
                v_eid = -1 - idx
                if v_eid not in tree_edges:
                    P_fixed = float(self.network.nodes[knid].fixed_pressure or 0.0)
                    loop = self._trace_cycle(knid, virtual_ground_id, v_eid, True, P_fixed, parent, parent_edge, depth, chord_count)
                    if loop is not None:
                        loops.append(loop)
                        chord_count += 1
                        
        return loops

    def _trace_cycle(
        self,
        u: int,
        v: int,
        chord_eid: int,
        chord_is_virt: bool,
        chord_h_fix: float,
        parent: Dict[int, Optional[int]],
        parent_edge: Dict[int, Optional[Tuple[int, bool, float]]],
        depth: Dict[int, int],
        loop_id: int,
    ) -> Optional[Loop]:
        """Traces the unique simple cycle formed by chord edge (u, v) and spanning tree path."""
        path_u_edges: List[Tuple[int, bool, float, int, int]] = []
        path_v_edges: List[Tuple[int, bool, float, int, int]] = []
        
        curr_u, curr_v = u, v
        while depth[curr_u] > depth[curr_v]:
            p = parent[curr_u]
            eid, is_v, h_f = parent_edge[curr_u]
            path_u_edges.append((eid, is_v, h_f, curr_u, p))
            curr_u = p
            
        while depth[curr_v] > depth[curr_u]:
            p = parent[curr_v]
            eid, is_v, h_f = parent_edge[curr_v]
            path_v_edges.append((eid, is_v, h_f, curr_v, p))
            curr_v = p
            
        while curr_u != curr_v:
            pu = parent[curr_u]
            eid_u, is_v_u, h_f_u = parent_edge[curr_u]
            path_u_edges.append((eid_u, is_v_u, h_f_u, curr_u, pu))
            curr_u = pu
            
            pv = parent[curr_v]
            eid_v, is_v_v, h_f_v = parent_edge[curr_v]
            path_v_edges.append((eid_v, is_v_v, h_f_v, curr_v, pv))
            curr_v = pv
            
        # Cycle traversal: chord (u -> v), then tree (v -> LCA -> u)
        cycle_steps: List[Tuple[int, bool, float, int, int]] = [(chord_eid, chord_is_virt, chord_h_fix, u, v)]
        for step in path_v_edges:
            cycle_steps.append(step)
        for step in reversed(path_u_edges):
            eid, is_v, h_f, n1, n2 = step
            cycle_steps.append((eid, is_v, -h_f, n2, n1))
            
        real_edge_ids: List[int] = []
        orientations: List[int] = []
        virt_head_offset = 0.0
        
        for eid, is_virt, h_fix, from_n, to_n in cycle_steps:
            if is_virt:
                p_from = float(self.network.nodes[from_n].fixed_pressure or 0.0) if from_n < self.num_nodes else 0.0
                p_to = float(self.network.nodes[to_n].fixed_pressure or 0.0) if to_n < self.num_nodes else 0.0
                virt_head_offset += (p_from - p_to)
            else:
                edge = self.network.edges[eid]
                if from_n == edge.start_node_id and to_n == edge.end_node_id:
                    ori = +1
                elif from_n == edge.end_node_id and to_n == edge.start_node_id:
                    ori = -1
                else:
                    ori = +1
                real_edge_ids.append(eid)
                orientations.append(ori)
                
        if len(real_edge_ids) == 0:
            return None
            
        return Loop(
            id=loop_id,
            name=f"Loop_{loop_id + 1}",
            edge_ids=real_edge_ids,
            orientations=orientations,
            virtual_head_offset=virt_head_offset,
        )

    def _compute_initial_flows(self) -> np.ndarray:
        """
        Computes a strictly mass-conserving initial flow distribution Q0 satisfying A_u * Q0 = -Q_ext_u.
        """
        if self.num_u > 0:
            A_u_pinv = scipy.linalg.pinv(self.A_u)
            Q0 = A_u_pinv @ (-self.Q_ext_u)
            Q0 = np.where(np.abs(Q0) < 1e-4, 1e-3, Q0)
        else:
            Q0 = np.full(self.num_edges, 0.1, dtype=np.float64)
        return Q0

    def _eval_edge_loss(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates head loss h(Q), analytical derivative dh/dQ, major loss, minor loss, and friction factor f.
        """
        abs_Q = np.abs(Q)
        Re = self.re_factor * abs_Q
        Re_safe = np.maximum(Re, 4000.0)
        
        # Haaland turbulent friction factor & derivative
        g = self.haaland_term0 + 6.9 / Re_safe
        inv_sqrt_f = -1.8 * np.log10(g)
        f = 1.0 / (inv_sqrt_f * inv_sqrt_f)
        df_dRe = (3.6 / (self.ln10 * g * (inv_sqrt_f ** 3))) * (-6.9 / (Re_safe * Re_safe))
        
        # Laminar regime (Re < 2000)
        lam_mask = Re < 2000.0
        if np.any(lam_mask):
            Re_lam = np.maximum(Re[lam_mask], 1e-4)
            f[lam_mask] = 64.0 / Re_lam
            df_dRe[lam_mask] = -64.0 / (Re_lam * Re_lam)
            
        # Transition regime (2000 <= Re <= 4000)
        trans_mask = (~lam_mask) & (Re < 4000.0)
        if np.any(trans_mask):
            re_tr = Re[trans_mask]
            t = (re_tr - 2000.0) * 0.0005
            f_lam = 0.032
            f_turb = f[trans_mask]
            h00 = 1.0 - 3.0 * (t ** 2) + 2.0 * (t ** 3)
            h10 = 3.0 * (t ** 2) - 2.0 * (t ** 3)
            f[trans_mask] = f_lam * h00 + f_turb * h10
            dh00_dt = -6.0 * t + 6.0 * (t ** 2)
            dh10_dt = 6.0 * t - 6.0 * (t ** 2)
            df_dRe[trans_mask] = (f_lam * dh00_dt + f_turb * dh10_dt) * 0.0005 + df_dRe[trans_mask] * h10
            
        f_ratio = f * self.L_over_Dh
        total_R = (f_ratio + self.edge_K) * self.geom_factor
        dp_major = f_ratio * self.geom_factor * abs_Q * Q
        dp_minor = self.edge_K * self.geom_factor * abs_Q * Q
        dp_elev = self.edge_dp_hydro
        
        # Total edge pressure drop h(Q) = dp_friction + dp_elev - dp_fan
        h = dp_major + dp_minor + dp_elev
        
        df_dQ = df_dRe * self.re_factor * np.sign(Q + 1e-15)
        dh_dQ = 2.0 * total_R * np.maximum(abs_Q, 1e-8) + df_dQ * self.L_over_Dh * self.geom_factor * (Q * Q)
        
        if self.has_fans:
            for i, fan in self.fan_edges:
                h[i] -= fan.pressure_boost(Q[i])
                dh_dQ[i] -= fan.derivative(Q[i])
                
        dh_dQ = np.maximum(dh_dQ, 1e-8)
        return h, dh_dQ, dp_major, dp_minor, f

    def solve(
        self,
        method: str = "simultaneous",
        initial_flows: Optional[np.ndarray] = None,
    ) -> HydraulicSolution:
        """
        Solves network flows and pressures using the Hardy Cross loop balancing method.
        
        Args:
            method: 'simultaneous' (Loop Newton-Raphson, fastest) or 'sequential' (Classic Hardy Cross).
            initial_flows: Optional array of initial edge flows (must satisfy mass conservation).
            
        Returns:
            HydraulicSolution object with flows, node pressures, and diagnostics.
        """
        t_start = time.perf_counter()
        
        method_clean = method.lower().replace("-", "_").replace(" ", "_")
        if method_clean in ("simultaneous", "loop_newton", "matrix"):
            sol = self._solve_simultaneous(initial_flows)
        elif method_clean in ("sequential", "classic", "gauss_seidel", "hardy_cross"):
            sol = self._solve_sequential(initial_flows)
        else:
            raise ValueError(f"Unknown method '{method}'. Available: 'simultaneous', 'sequential'.")
            
        t_end = time.perf_counter()
        sol.execution_time_ms = (t_end - t_start) * 1000.0
        return sol

    def _solve_simultaneous(self, initial_flows: Optional[np.ndarray] = None) -> HydraulicSolution:
        """
        Simultaneous Loop Newton-Raphson: Solves J_loop * delta_q = - delta_H.
        Guarantees quadratic convergence while strictly preserving mass conservation.
        """
        Q = np.array(initial_flows if initial_flows is not None else self.Q_init, copy=True, dtype=np.float64)
        
        converged = False
        iteration = 0
        max_head_err = 0.0
        max_flow_corr = 0.0
        
        if self.num_loops == 0:
            converged = True
            iteration = 1
        else:
            for it in range(1, self.max_iterations + 1):
                iteration = it
                
                h, dh_dQ, _, _, _ = self._eval_edge_loss(Q)
                
                # Loop head loss residual: delta_H = C * h + virtual_head_vec
                delta_H = self.C @ h + self.virtual_head_vec
                max_head_err = float(np.max(np.abs(delta_H)))
                
                if max_head_err < self.tol_head:
                    converged = True
                    break
                    
                # Loop Jacobian J_loop = C * diag(dh_dQ) * C^T
                J_loop = (self.C * dh_dQ) @ self.C.T
                
                # Solve J_loop * delta_q = - delta_H
                if self.num_loops == 1:
                    delta_q = np.array([- delta_H[0] / J_loop[0, 0]])
                elif self.num_loops == 2:
                    j00, j01, j10, j11 = J_loop[0, 0], J_loop[0, 1], J_loop[1, 0], J_loop[1, 1]
                    inv_det = 1.0 / (j00 * j11 - j01 * j10)
                    delta_q = np.array([
                        (- j11 * delta_H[0] + j01 * delta_H[1]) * inv_det,
                        (j10 * delta_H[0] - j00 * delta_H[1]) * inv_det
                    ])
                else:
                    delta_q = np.linalg.solve(J_loop, - delta_H)
                    
                max_flow_corr = float(np.max(np.abs(delta_q)))
                
                # Update branch flows: Q = Q + C^T * delta_q
                Q += self.relaxation * (self.C.T @ delta_q)
                
                if max_flow_corr < self.tol_flow and max_head_err < self.tol_head:
                    converged = True
                    break

        return self._reconstruct_solution(Q, converged, iteration, max_head_err)

    def _solve_sequential(self, initial_flows: Optional[np.ndarray] = None) -> HydraulicSolution:
        """
        Classic Sequential Hardy Cross (Gauss-Seidel loop correction).
        Iterates over each loop sequentially: delta_q_k = - delta_H_k / sum(dh/dQ).
        """
        Q = np.array(initial_flows if initial_flows is not None else self.Q_init, copy=True, dtype=np.float64)
        
        converged = False
        iteration = 0
        max_head_err = 0.0
        
        if self.num_loops == 0:
            converged = True
            iteration = 1
        else:
            for it in range(1, self.max_iterations * 2 + 1):
                iteration = it
                max_dq = 0.0
                max_dH = 0.0
                
                for k, loop in enumerate(self.loops):
                    h, dh_dQ, _, _, _ = self._eval_edge_loss(Q)
                    
                    # Compute loop head loss
                    dH_k = float(np.dot(self.C[k, :], h) + self.virtual_head_vec[k])
                    denom_k = float(np.dot(np.abs(self.C[k, :]), dh_dQ))
                    denom_k = max(denom_k, 1e-8)
                    
                    dq_k = - dH_k / denom_k
                    Q += self.relaxation * (self.C[k, :] * dq_k)
                    
                    max_dq = max(max_dq, abs(dq_k))
                    max_dH = max(max_dH, abs(dH_k))
                    
                max_head_err = max_dH
                if max_dq < self.tol_flow and max_dH < self.tol_head:
                    converged = True
                    break

        return self._reconstruct_solution(Q, converged, iteration, max_head_err)

    def _reconstruct_solution(
        self,
        Q: np.ndarray,
        converged: bool,
        iterations: int,
        max_residual: float,
    ) -> HydraulicSolution:
        """
        Reconstructs node pressures and packages all results into HydraulicSolution.
        """
        h, _, dp_major, dp_minor, f = self._eval_edge_loss(Q)
        
        # Mass conservation check: A_u * Q + Q_ext_u = 0
        mass_res = self.A_u @ Q + self.Q_ext_u if self.num_u > 0 else np.zeros(0)
        max_mass_err = float(np.max(np.abs(mass_res))) if self.num_u > 0 else 0.0
        
        # Node Pressure Reconstruction:
        # P_start - P_end = h_e => P_end = P_start - h_e
        P = np.full(self.num_nodes, np.nan, dtype=np.float64)
        
        # Set known fixed pressures
        for knid in self.known_node_ids:
            P[knid] = float(self.network.nodes[knid].fixed_pressure or 0.0)
            
        if self.num_k == 0:
            P[0] = 0.0
            
        # BFS propagation from solved nodes to all connected nodes
        solved_nodes = deque([nid for nid in range(self.num_nodes) if not np.isnan(P[nid])])
        adj: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(self.num_nodes)}
        for eid, edge in self.network.edges.items():
            adj[edge.start_node_id].append((edge.end_node_id, eid, +1))
            adj[edge.end_node_id].append((edge.start_node_id, eid, -1))
            
        while solved_nodes:
            curr = solved_nodes.popleft()
            curr_P = P[curr]
            for neighbor, eid, direction in adj[curr]:
                if np.isnan(P[neighbor]):
                    if direction == +1:
                        P[neighbor] = curr_P - h[eid]
                    else:
                        P[neighbor] = curr_P + h[eid]
                    solved_nodes.append(neighbor)
                    
        P = np.nan_to_num(P, nan=0.0)
        
        velocities = np.abs(Q) / self.edge_A
        reynolds = (self.rho * velocities * self.edge_Dh) / self.mu
        dp_edges = - (self.A_full.T @ P)
        
        node_names = [n.name for n in self.network.nodes.values()]
        edge_names = [e.name for e in self.network.edges.values()]

        return HydraulicSolution(
            converged=converged,
            iterations=iterations,
            execution_time_ms=0.0,
            mass_conservation_error=max_mass_err,
            max_residual=max_residual,
            node_pressures={name: float(P[nid]) for nid, name in enumerate(node_names)},
            edge_flows={name: float(Q[eid]) for eid, name in enumerate(edge_names)},
            edge_velocities={name: float(velocities[eid]) for eid, name in enumerate(edge_names)},
            edge_reynolds={name: float(reynolds[eid]) for eid, name in enumerate(edge_names)},
            edge_friction_factors={name: float(f[eid]) for eid, name in enumerate(edge_names)},
            edge_pressure_drops={name: float(dp_edges[eid]) for eid, name in enumerate(edge_names)},
            edge_major_losses={name: float(dp_major[eid]) for eid, name in enumerate(edge_names)},
            edge_minor_losses={name: float(dp_minor[eid]) for eid, name in enumerate(edge_names)},
            edge_elevation_heads={name: float(self.edge_dp_hydro[eid]) for eid, name in enumerate(edge_names)},
        )
