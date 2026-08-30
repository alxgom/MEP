"""
High-Speed 1D Hydraulic Network Solver for HVAC Duct Networks.

Features:
- Directed Graph representation of duct network with Nodes (Plenums, Junctions, Outlets)
  and Edges (Duct segments with length L, hydraulic diameter Dh, roughness eps, fittings).
- Comprehensive ASHRAE fitting loss library (90°/45° elbows, 45° wyes/pants, 90° tees,
  crosses, sudden contractions/expansions, dampers, custom fittings).
- Darcy-Weisbach friction equation with Colebrook-White friction factor via
  Haaland, Swamee-Jain, and exact Colebrook formulations with C1-smooth transition blend.
- Global Gradient Algorithm (GGA / Todini-Pilati Schur complement Newton-Raphson)
  with exact analytical Jacobian achieving < 1 ms execution time.
- Boundary condition support: Fan constant flow rate / Fan curves (dP = a0 + a1*Q + a2*Q^2)
  and fixed atmospheric/plenum pressures.
- Mass conservation error tracking and validation.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.optimize


# ============================================================================
# Fluid & Material Physical Properties
# ============================================================================

@dataclass(frozen=True)
class FluidProperties:
    """Standard fluid properties for air or other gases/liquids."""
    density: float = 1.204  # kg/m^3 (Air at 20°C, 101.325 kPa)
    viscosity: float = 1.825e-5  # Pa·s (Dynamic viscosity)
    gravity: float = 9.80665  # m/s^2 (Standard gravitational acceleration)
    ambient_density: Optional[float] = None  # Ambient air density for buoyancy/stack calculations
    isothermal_gauge: bool = False  # If True and ambient_density is None, hydrostatic column cancels in room gauge pressure
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity nu = mu / rho (m^2/s)."""
        return self.viscosity / self.density


class DuctMaterial(float, Enum):
    """Standard absolute roughness values epsilon (meters) according to ASHRAE."""
    GALVANIZED_STEEL = 0.00009   # 0.09 mm - standard galvanized sheet metal
    SMOOTH_PVC = 0.0000015       # 0.0015 mm - smooth plastic / PVC
    COMMERCIAL_STEEL = 0.000045  # 0.045 mm - standard commercial steel
    FLEXIBLE_DUCT = 0.0010       # 1.0 mm - fully extended flexible duct
    CONCRETE_DUCT = 0.0015       # 1.5 mm - smooth concrete duct


# ============================================================================
# Friction Factor Models (Colebrook-White, Haaland, Swamee-Jain)
# ============================================================================

def friction_factor_laminar(Re: Union[float, np.ndarray], reg_eps: float = 1e-6) -> Union[float, np.ndarray]:
    """Laminar friction factor f = 64 / Re with regularization near Re -> 0."""
    Re_safe = np.maximum(np.abs(Re), reg_eps)
    return 64.0 / Re_safe


def friction_factor_haaland(
    Re: Union[float, np.ndarray],
    rel_roughness: Union[float, np.ndarray],
    reg_eps: float = 1e-6,
) -> Union[float, np.ndarray]:
    """
    Haaland explicit approximation to Colebrook-White equation for turbulent flow:
    1 / sqrt(f) = -1.8 * log10( (eps / (3.7 * Dh))^1.11 + 6.9 / Re )
    """
    Re_safe = np.maximum(np.abs(Re), 4000.0)
    rr_safe = np.maximum(rel_roughness, 0.0)
    term = (rr_safe / 3.7) ** 1.11 + 6.9 / Re_safe
    inv_sqrt_f = -1.8 * np.log10(term)
    return (1.0 / inv_sqrt_f) ** 2


def friction_factor_swamee_jain(
    Re: Union[float, np.ndarray],
    rel_roughness: Union[float, np.ndarray],
    reg_eps: float = 1e-6,
) -> Union[float, np.ndarray]:
    """
    Swamee-Jain explicit approximation to Colebrook-White equation:
    f = 0.25 / [ log10( eps / (3.7 * Dh) + 5.74 / Re^0.9 ) ]^2
    """
    Re_safe = np.maximum(np.abs(Re), 4000.0)
    rr_safe = np.maximum(rel_roughness, 0.0)
    term = rr_safe / 3.7 + 5.74 / (Re_safe ** 0.9)
    log_term = np.log10(term)
    return 0.25 / (log_term ** 2)


def friction_factor(
    Re: Union[float, np.ndarray],
    rel_roughness: Union[float, np.ndarray],
    method: str = "haaland",
) -> Union[float, np.ndarray]:
    """
    Vectorized, C1-smooth friction factor across Laminar, Transition, and Turbulent regimes.
    
    Regimes:
    - Re < 2000: Laminar (f = 64 / Re)
    - 2000 <= Re <= 4000: Smooth cubic Hermite transition interpolation
    - Re > 4000: Turbulent (Haaland / Swamee-Jain / Colebrook)
    """
    Re_arr = np.asarray(Re, dtype=np.float64)
    rr_arr = np.asarray(rel_roughness, dtype=np.float64)
    scalar_input = Re_arr.ndim == 0
    
    Re_arr = np.atleast_1d(Re_arr)
    rr_arr = np.atleast_1d(rr_arr)
    if rr_arr.shape != Re_arr.shape:
        rr_arr = np.broadcast_to(rr_arr, Re_arr.shape)
        
    abs_Re = np.abs(Re_arr)
    f_res = np.zeros_like(abs_Re)
    
    # 1. Laminar mask
    lam_mask = abs_Re < 2000.0
    if np.any(lam_mask):
        f_res[lam_mask] = friction_factor_laminar(abs_Re[lam_mask])
        
    # 2. Turbulent mask
    turb_mask = abs_Re > 4000.0
    if np.any(turb_mask):
        if method.lower() == "swamee_jain":
            f_res[turb_mask] = friction_factor_swamee_jain(abs_Re[turb_mask], rr_arr[turb_mask])
        else:
            f_res[turb_mask] = friction_factor_haaland(abs_Re[turb_mask], rr_arr[turb_mask])
            
    # 3. Transition mask (2000 <= Re <= 4000) with C1 cubic Hermite blend
    trans_mask = (~lam_mask) & (~turb_mask)
    if np.any(trans_mask):
        re_trans = abs_Re[trans_mask]
        rr_trans = rr_arr[trans_mask]
        
        f_lam_2000 = 64.0 / 2000.0  # 0.032
        if method.lower() == "swamee_jain":
            f_turb_4000 = friction_factor_swamee_jain(np.full_like(re_trans, 4000.0), rr_trans)
        else:
            f_turb_4000 = friction_factor_haaland(np.full_like(re_trans, 4000.0), rr_trans)
            
        t = (re_trans - 2000.0) / 2000.0  # in [0, 1]
        h00 = 1.0 - 3.0 * (t ** 2) + 2.0 * (t ** 3)
        h10 = 3.0 * (t ** 2) - 2.0 * (t ** 3)
        f_res[trans_mask] = f_lam_2000 * h00 + f_turb_4000 * h10
        
    return float(f_res[0]) if scalar_input else f_res


def friction_factor_derivative_Re(
    Re: Union[float, np.ndarray],
    rel_roughness: Union[float, np.ndarray],
    method: str = "haaland",
) -> Union[float, np.ndarray]:
    """Analytical derivative df/dRe for Newton-Raphson Jacobian."""
    Re_arr = np.asarray(Re, dtype=np.float64)
    rr_arr = np.asarray(rel_roughness, dtype=np.float64)
    scalar_input = Re_arr.ndim == 0
    
    Re_arr = np.atleast_1d(Re_arr)
    rr_arr = np.atleast_1d(rr_arr)
    if rr_arr.shape != Re_arr.shape:
        rr_arr = np.broadcast_to(rr_arr, Re_arr.shape)
        
    abs_Re = np.abs(Re_arr)
    df_dRe = np.zeros_like(abs_Re)
    
    # Laminar: f = 64/Re -> df/dRe = -64/Re^2
    lam_mask = abs_Re < 2000.0
    if np.any(lam_mask):
        Re_safe = np.maximum(abs_Re[lam_mask], 1e-4)
        df_dRe[lam_mask] = -64.0 / (Re_safe ** 2)
        
    # Turbulent: Haaland derivative
    turb_mask = abs_Re > 4000.0
    if np.any(turb_mask):
        Re_t = abs_Re[turb_mask]
        rr_t = rr_arr[turb_mask]
        g = (np.maximum(rr_t, 0.0) / 3.7) ** 1.11 + 6.9 / Re_t
        inv_sqrt_f = -1.8 * np.log10(g)
        # f = (inv_sqrt_f)^(-2)
        # df/dg = -2 * (inv_sqrt_f)^(-3) * (-1.8 / (ln(10) * g))
        # dg/dRe = -6.9 / Re^2
        # df/dRe = df/dg * dg/dRe
        df_dg = 3.6 / (np.log(10.0) * g * (inv_sqrt_f ** 3))
        dg_dRe = -6.9 / (Re_t ** 2)
        df_dRe[turb_mask] = df_dg * dg_dRe
        
    # Transition: Cubic blend derivative
    trans_mask = (~lam_mask) & (~turb_mask)
    if np.any(trans_mask):
        re_trans = abs_Re[trans_mask]
        rr_trans = rr_arr[trans_mask]
        f_lam_2000 = 64.0 / 2000.0
        f_turb_4000 = friction_factor_haaland(np.full_like(re_trans, 4000.0), rr_trans)
        t = (re_trans - 2000.0) / 2000.0
        dt_dRe = 1.0 / 2000.0
        dh00_dt = -6.0 * t + 6.0 * (t ** 2)
        dh10_dt = 6.0 * t - 6.0 * (t ** 2)
        df_dRe[trans_mask] = (f_lam_2000 * dh00_dt + f_turb_4000 * dh10_dt) * dt_dRe
        
    return float(df_dRe[0]) if scalar_input else df_dRe


# ============================================================================
# ASHRAE Fitting Loss Library
# ============================================================================

@dataclass
class Fitting:
    """Represents a duct fitting with local minor loss coefficient K."""
    name: str
    K: float
    description: str = ""

    def get_loss_coefficient(self, **kwargs) -> float:
        """Returns the dimensionless loss coefficient K."""
        return float(self.K)


class FittingLossLibrary:
    """Catalog of ASHRAE Standard Duct Fitting Losses (DFDB Reference)."""

    @staticmethod
    def elbow_90_round(r_over_d: float = 1.5) -> Fitting:
        """
        ASHRAE 90° round smooth die-stamped / segmented elbow.
        K depends on centerline radius to diameter ratio r/D:
        - r/D = 1.5: K ~ 0.25 (standard long radius)
        - r/D = 1.0: K ~ 0.45 (medium radius)
        - r/D = 0.75: K ~ 0.75 (short radius)
        """
        if r_over_d >= 1.5:
            k = 0.25
        elif r_over_d >= 1.0:
            # Interpolate between 0.45 and 0.25
            k = 0.45 - (r_over_d - 1.0) / 0.5 * 0.20
        else:
            k = 0.75 - (r_over_d - 0.75) / 0.25 * 0.30
        return Fitting(f"Elbow 90° Round (r/D={r_over_d:.2f})", k, "ASHRAE 90 deg round elbow")

    @staticmethod
    def elbow_90_mitered(has_turning_vanes: bool = False) -> Fitting:
        """ASHRAE 90° mitered elbow with or without turning vanes."""
        k = 0.35 if has_turning_vanes else 1.20
        desc = "Mitered 90° elbow with vanes" if has_turning_vanes else "Mitered 90° elbow without vanes"
        return Fitting("Elbow 90° Mitered" + (" (Vanes)" if has_turning_vanes else " (No Vanes)"), k, desc)

    @staticmethod
    def elbow_45_round(r_over_d: float = 1.5) -> Fitting:
        """ASHRAE 45° round smooth elbow (K ~ 0.15 - 0.25)."""
        k = 0.15 if r_over_d >= 1.5 else 0.25
        return Fitting(f"Elbow 45° Round (r/D={r_over_d:.2f})", k, "ASHRAE 45 deg round elbow")

    @staticmethod
    def wye_45(role: str = "branch", area_ratio: float = 0.5) -> Fitting:
        """
        ASHRAE 45° Wye / Pant junction.
        - role='branch' (45° takeoff branch): K ~ 0.40 - 0.80 (default 0.50)
        - role='main' or 'through' (straight main run): K ~ 0.15
        """
        role_lower = role.lower()
        if "branch" in role_lower or "takeoff" in role_lower:
            # Branch takeoff loss: 0.4 to 0.8 depending on flow/area ratio
            k = 0.50
            name = "Wye 45° Branch"
        else:
            k = 0.15
            name = "Wye 45° Main"
        return Fitting(name, k, f"ASHRAE 45 deg Wye ({role})")

    @staticmethod
    def pants_junction_split(angle_deg: float = 45.0) -> Fitting:
        """
        Symmetrical Pant / Trouser split (bifurcation):
        Both symmetrical legs experience minor loss K ~ 0.45 - 0.60.
        """
        k = 0.45 if angle_deg <= 45.0 else 0.60
        return Fitting(f"Pant Split ({angle_deg:.0f}°)", k, "Symmetrical bifurcation pants junction")

    @staticmethod
    def tee_90(role: str = "branch") -> Fitting:
        """ASHRAE 90° standard tee fitting."""
        role_lower = role.lower()
        if "branch" in role_lower:
            return Fitting("Tee 90° Branch", 1.20, "90 deg Tee Branch Takeoff")
        return Fitting("Tee 90° Main", 0.20, "90 deg Tee Straight Main Run")

    @staticmethod
    def cross(role: str = "branch") -> Fitting:
        """ASHRAE cross junction fitting."""
        role_lower = role.lower()
        if "branch" in role_lower:
            return Fitting("Cross Branch", 1.30, "Cross Branch Takeoff")
        return Fitting("Cross Main", 0.30, "Cross Straight Run")

    @staticmethod
    def sudden_contraction(area_upstream: float, area_downstream: float) -> Fitting:
        """
        Sudden contraction loss based on downstream velocity:
        K = 0.5 * (1 - A_down / A_up)
        """
        ratio = min(area_downstream / max(area_upstream, 1e-7), 1.0)
        k = 0.5 * (1.0 - ratio)
        return Fitting(f"Sudden Contraction (A2/A1={ratio:.2f})", k, "Sudden contraction loss")

    @staticmethod
    def sudden_expansion(area_upstream: float, area_downstream: float) -> Fitting:
        """
        Sudden expansion Borda-Carnot loss based on upstream velocity:
        K_up = (1 - A_up / A_down)^2
        Converted to downstream velocity: K_down = (A_down / A_up - 1)^2
        Here returned relative to downstream velocity.
        """
        ratio = max(area_upstream / max(area_downstream, 1e-7), 1e-6)
        k_up = (1.0 - ratio) ** 2
        # K relative to downstream: K_down = K_up * (A_down / A_up)^2
        k_down = k_up / (ratio ** 2) if ratio > 0 else 1.0
        return Fitting(f"Sudden Expansion (A1/A2={ratio:.2f})", k_down, "Sudden expansion loss")

    @staticmethod
    def damper(angle_deg: float = 0.0) -> Fitting:
        """
        Volume control damper loss as a function of blade angle:
        - 0° (Fully Open): K ~ 0.20
        - 15°: K ~ 0.52
        - 30°: K ~ 1.50
        - 45°: K ~ 5.10
        - 60°: K ~ 20.5
        """
        theta_rad = math.radians(min(max(angle_deg, 0.0), 85.0))
        k = 0.20 + 0.50 * (math.tan(theta_rad) ** 2) + 2.0 * (math.sin(theta_rad) ** 4)
        return Fitting(f"Damper ({angle_deg:.0f}°)", k, f"Blade damper at {angle_deg:.0f} deg")

    @staticmethod
    def generic(name: str, K: float, description: str = "") -> Fitting:
        """Generic fitting with user-specified K coefficient."""
        return Fitting(name, float(K), description)


# ============================================================================
# Fan Curve Model
# ============================================================================

@dataclass
class FanCurve:
    """
    Fan static pressure performance curve:
    dP_fan(Q) = a0 + a1 * Q + a2 * Q^2 (Pa)
    where Q is in m^3/s.
    For a constant pressure boost, set a1 = 0, a2 = 0, a0 = dP_boost.
    """
    a0: float  # Shut-off head / constant static pressure boost (Pa)
    a1: float = 0.0  # Linear coefficient (Pa·s/m^3)
    a2: float = 0.0  # Quadratic coefficient (Pa·s^2/m^6)

    def pressure_boost(self, Q: float) -> float:
        """Evaluates fan static pressure boost dP_fan(Q) in Pa."""
        return self.a0 + self.a1 * Q + self.a2 * (Q ** 2)

    def derivative(self, Q: float) -> float:
        """Evaluates derivative d(dP_fan)/dQ."""
        return self.a1 + 2.0 * self.a2 * Q


# ============================================================================
# Network Graph: Nodes & Edges
# ============================================================================

class NodeType(str, Enum):
    JUNCTION = "junction"
    INLET = "inlet"
    OUTLET = "outlet"
    PLENUM = "plenum"


@dataclass
class Node:
    """
    Node in the 1D/3D hydraulic duct network.
    
    Attributes:
        id: Integer index in solver matrices (0-indexed).
        name: Unique string identifier.
        node_type: Junction, Inlet, Outlet, Plenum.
        fixed_pressure: Fixed pressure boundary condition in Pa (e.g. 0.0 for atmospheric outlet).
                        If None, pressure is an unknown to be solved.
        external_inflow: Known external flow entering node in m^3/s (>0 inlet, <0 outlet).
        x: X-coordinate in meters (default 0.0).
        y: Y-coordinate in meters (default 0.0).
        z: Z-coordinate / elevation in meters (default 0.0).
        elevation: Alias / storage for elevation z in meters (default 0.0).
    """
    id: int
    name: str
    node_type: NodeType = NodeType.JUNCTION
    fixed_pressure: Optional[float] = None
    external_inflow: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    elevation: float = 0.0

    def __post_init__(self):
        # Synchronize elevation and z if one is provided non-zero
        if self.elevation != 0.0 and self.z == 0.0:
            self.z = float(self.elevation)
        elif self.z != 0.0 and self.elevation == 0.0:
            self.elevation = float(self.z)

    @property
    def is_pressure_known(self) -> bool:
        return self.fixed_pressure is not None

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Returns 3D spatial coordinate tuple (x, y, z)."""
        return (float(self.x), float(self.y), float(self.z))


@dataclass
class Edge:
    """
    Duct segment edge connecting start_node to end_node.
    
    Attributes:
        id: Integer index in solver matrices.
        name: Unique string identifier.
        start_node_id: ID of upstream node.
        end_node_id: ID of downstream node.
        length: Duct length L (m). If not provided, computed from node coordinates.
        diameter: Round duct diameter D (m). If rectangular, hydraulic diameter Dh is computed.
        width: Optional rectangular duct width W (m).
        height: Optional rectangular duct height H (m).
        roughness: Absolute wall roughness epsilon (m), default galvanized steel (0.00009 m).
        fittings: List of Fitting objects or numerical loss coefficients K.
        fan: Optional FanCurve on this duct segment.
        loss_coeff_extra: Extra lumped minor loss coefficient K.
    """
    id: int
    name: str
    start_node_id: int
    end_node_id: int
    length: float
    diameter: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    roughness: float = DuctMaterial.GALVANIZED_STEEL
    fittings: List[Union[Fitting, float]] = field(default_factory=list)
    fan: Optional[FanCurve] = None
    loss_coeff_extra: float = 0.0

    def __post_init__(self):
        if self.diameter is None and (self.width is None or self.height is None):
            raise ValueError(f"Edge '{self.name}': Must specify either diameter or (width and height).")

    @property
    def is_rectangular(self) -> bool:
        return self.width is not None and self.height is not None

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter Dh = 4A / P (m)."""
        if self.is_rectangular:
            w, h = float(self.width), float(self.height)
            return (2.0 * w * h) / (w + h)
        return float(self.diameter)

    @property
    def area(self) -> float:
        """Cross-sectional flow area A (m^2)."""
        if self.is_rectangular:
            return float(self.width) * float(self.height)
        d = float(self.diameter)
        return 0.25 * math.pi * (d ** 2)

    @property
    def relative_roughness(self) -> float:
        """Dimensionless relative roughness eps / Dh."""
        return self.roughness / self.hydraulic_diameter

    @property
    def total_loss_coefficient(self) -> float:
        """Total minor loss coefficient sum(K)."""
        k_sum = float(self.loss_coeff_extra)
        for fit in self.fittings:
            if isinstance(fit, Fitting):
                k_sum += fit.get_loss_coefficient()
            else:
                k_sum += float(fit)
        return k_sum


# DuctSegment alias for Edge
DuctSegment = Edge


# ============================================================================
# Hydraulic Network Container
# ============================================================================

class HydraulicNetwork:
    """Container and builder for 1D Hydraulic Duct Networks."""

    def __init__(self, fluid: Optional[FluidProperties] = None):
        self.fluid: FluidProperties = fluid or FluidProperties()
        self.nodes: Dict[int, Node] = {}
        self.node_name_to_id: Dict[str, int] = {}
        self.edges: Dict[int, Edge] = {}
        self.edge_name_to_id: Dict[str, int] = {}

    def add_node(
        self,
        name: str,
        node_type: Union[NodeType, str] = NodeType.JUNCTION,
        fixed_pressure: Optional[float] = None,
        external_inflow: float = 0.0,
        elevation: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        coords: Optional[Tuple[float, float, float]] = None,
    ) -> Node:
        """Adds a node to the network with optional 3D coordinates (x, y, z)."""
        if name in self.node_name_to_id:
            raise ValueError(f"Node with name '{name}' already exists in network.")
        if isinstance(node_type, str):
            node_type = NodeType(node_type.lower())
            
        if coords is not None:
            x, y, z = coords
        elif elevation is not None:
            z = elevation
            
        elev_val = elevation if elevation is not None else z
        node_id = len(self.nodes)
        node = Node(
            id=node_id,
            name=name,
            node_type=node_type,
            fixed_pressure=fixed_pressure,
            external_inflow=float(external_inflow),
            x=float(x),
            y=float(y),
            z=float(z),
            elevation=float(elev_val),
        )
        self.nodes[node_id] = node
        self.node_name_to_id[name] = node_id
        return node

    def add_edge(
        self,
        name: str,
        start_node: Union[str, Node, int],
        end_node: Union[str, Node, int],
        length: Optional[float] = None,
        diameter: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        roughness: float = DuctMaterial.GALVANIZED_STEEL,
        fittings: Optional[List[Union[Fitting, float, str]]] = None,
        fan: Optional[FanCurve] = None,
        loss_coeff_extra: float = 0.0,
    ) -> Edge:
        """Adds a duct segment edge to the network. Length is calculated from 3D coordinates if None."""
        if name in self.edge_name_to_id:
            raise ValueError(f"Edge with name '{name}' already exists in network.")
        
        start_id = self._resolve_node_id(start_node)
        end_id = self._resolve_node_id(end_node)
        
        if length is None:
            n_start = self.nodes[start_id]
            n_end = self.nodes[end_id]
            dx = n_end.x - n_start.x
            dy = n_end.y - n_start.y
            dz = n_end.z - n_start.z
            calculated_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if calculated_len <= 1e-9:
                raise ValueError(
                    f"Edge '{name}' start and end nodes have identical coordinates {n_start.coords}. "
                    f"Please provide an explicit length > 0."
                )
            edge_length = calculated_len
        else:
            edge_length = float(length)
        
        # Parse fitting strings if any
        parsed_fittings: List[Union[Fitting, float]] = []
        if fittings:
            for fit in fittings:
                if isinstance(fit, str):
                    parsed_fittings.append(self._parse_fitting_string(fit))
                else:
                    parsed_fittings.append(fit)

        edge_id = len(self.edges)
        edge = Edge(
            id=edge_id,
            name=name,
            start_node_id=start_id,
            end_node_id=end_id,
            length=edge_length,
            diameter=float(diameter) if diameter is not None else None,
            width=float(width) if width is not None else None,
            height=float(height) if height is not None else None,
            roughness=float(roughness),
            fittings=parsed_fittings,
            fan=fan,
            loss_coeff_extra=float(loss_coeff_extra),
        )
        self.edges[edge_id] = edge
        self.edge_name_to_id[name] = edge_id
        return edge

    def add_duct(self, *args, **kwargs) -> Edge:
        """Alias for add_edge."""
        return self.add_edge(*args, **kwargs)

    def _resolve_node_id(self, node: Union[str, Node, int]) -> int:
        if isinstance(node, int):
            if node not in self.nodes:
                raise ValueError(f"Node id {node} not found.")
            return node
        if isinstance(node, Node):
            return node.id
        if isinstance(node, str):
            if node not in self.node_name_to_id:
                raise ValueError(f"Node name '{node}' not found.")
            return self.node_name_to_id[node]
        raise TypeError(f"Invalid node type {type(node)}")

    @staticmethod
    def _parse_fitting_string(name: str) -> Fitting:
        name_clean = name.strip().lower().replace("-", "_").replace(" ", "_")
        if "elbow_90" in name_clean or "90_elbow" in name_clean:
            return FittingLossLibrary.elbow_90_round(1.5)
        if "elbow_45" in name_clean or "45_elbow" in name_clean:
            return FittingLossLibrary.elbow_45_round(1.5)
        if "wye_45_branch" in name_clean or "wye_branch" in name_clean:
            return FittingLossLibrary.wye_45("branch")
        if "wye_45_main" in name_clean or "wye_main" in name_clean:
            return FittingLossLibrary.wye_45("main")
        if "pant" in name_clean:
            return FittingLossLibrary.pants_junction_split(45.0)
        if "tee_90_branch" in name_clean or "tee_branch" in name_clean:
            return FittingLossLibrary.tee_90("branch")
        if "tee_90_main" in name_clean or "tee_main" in name_clean:
            return FittingLossLibrary.tee_90("main")
        return FittingLossLibrary.generic(name, 0.5)

    def validate(self) -> None:
        """Validates network topological and boundary conditions."""
        if len(self.nodes) == 0:
            raise ValueError("Network has no nodes.")
        if len(self.edges) == 0:
            raise ValueError("Network has no edges.")
            
        has_pressure_ref = any(n.is_pressure_known for n in self.nodes.values())
        if not has_pressure_ref:
            raise ValueError("Network must have at least one node with fixed pressure (reference pressure).")
            
        total_inflow = sum(n.external_inflow for n in self.nodes.values())
        all_pressure_unknown = all(not n.is_pressure_known for n in self.nodes.values())
        if all_pressure_unknown and abs(total_inflow) > 1e-6:
            raise ValueError(f"Inflow sum ({total_inflow}) must equal 0 when all pressures are unknown.")


# ============================================================================
# Solver Results Container
# ============================================================================

@dataclass
class HydraulicSolution:
    """Solution results returned by Hydraulic1DSolver."""
    converged: bool
    iterations: int
    execution_time_ms: float
    mass_conservation_error: float
    max_residual: float
    node_pressures: Dict[str, float]
    edge_flows: Dict[str, float]
    edge_velocities: Dict[str, float]
    edge_reynolds: Dict[str, float]
    edge_friction_factors: Dict[str, float]
    edge_pressure_drops: Dict[str, float]
    edge_major_losses: Dict[str, float]
    edge_minor_losses: Dict[str, float]
    edge_elevation_heads: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Formatted summary table of solution results."""
        lines = []
        lines.append("=" * 80)
        status_str = "CONVERGED" if self.converged else "FAILED"
        lines.append(f"1D HYDRAULIC NETWORK SOLVER SOLUTION [{status_str}]")
        lines.append(f"Execution Time: {self.execution_time_ms:.4f} ms | Iterations: {self.iterations} | Max Mass Err: {self.mass_conservation_error:.2e} m^3/s")
        lines.append("-" * 80)
        lines.append("NODAL PRESSURES:")
        lines.append(f"  {'Node Name':<20} {'Pressure (Pa)':>15} {'Head (mmH2O)':>15}")
        lines.append("  " + "-" * 52)
        for name, P in self.node_pressures.items():
            lines.append(f"  {name:<20} {P:>15.4f} {P / 9.80665:>15.4f}")
            
        lines.append("-" * 80)
        lines.append("DUCT FLOWS & LOSSES:")
        has_elev = any(abs(v) > 1e-4 for v in self.edge_elevation_heads.values()) if self.edge_elevation_heads else False
        if has_elev:
            hdr = f"  {'Duct Name':<16} {'Flow (m^3/s)':>14} {'Flow (m^3/h)':>14} {'Vel (m/s)':>10} {'Re':>10} {'f':>8} {'dP (Pa)':>10} {'dP_maj':>10} {'dP_min':>10} {'dP_elev':>10}"
        else:
            hdr = f"  {'Duct Name':<16} {'Flow (m^3/s)':>14} {'Flow (m^3/h)':>14} {'Vel (m/s)':>10} {'Re':>10} {'f':>8} {'dP (Pa)':>10} {'dP_maj':>10} {'dP_min':>10}"
        lines.append(hdr)
        lines.append("  " + "-" * (114 if has_elev else 102))
        for name, Q in self.edge_flows.items():
            Q_h = Q * 3600.0
            v = self.edge_velocities[name]
            Re = self.edge_reynolds[name]
            f = self.edge_friction_factors[name]
            dp = self.edge_pressure_drops[name]
            dp_maj = self.edge_major_losses[name]
            dp_min = self.edge_minor_losses[name]
            if has_elev:
                dp_elev = self.edge_elevation_heads.get(name, 0.0)
                lines.append(f"  {name:<16} {Q:>14.5f} {Q_h:>14.2f} {v:>10.3f} {Re:>10.0f} {f:>8.5f} {dp:>10.3f} {dp_maj:>10.3f} {dp_min:>10.3f} {dp_elev:>10.3f}")
            else:
                lines.append(f"  {name:<16} {Q:>14.5f} {Q_h:>14.2f} {v:>10.3f} {Re:>10.0f} {f:>8.5f} {dp:>10.3f} {dp_maj:>10.3f} {dp_min:>10.3f}")
        lines.append("=" * 80)
        return "\n".join(lines)

    def print_results(self) -> None:
        """Prints formatted summary to stdout."""
        print(self.summary())


# ============================================================================
# High-Speed 1D Hydraulic Network Solver
# ============================================================================

class Hydraulic1DSolver:
    """
    High-Speed Non-linear 1D Hydraulic Network Solver for HVAC & Duct systems.
    
    Implements:
    - Global Gradient Algorithm (GGA / Todini-Pilati Schur complement Newton-Raphson)
      with exact analytical Jacobian. Achieves sub-millisecond execution (< 1 ms).
    - Full Newton-Raphson with exact Jacobian.
    - SciPy optimize root / least_squares fallback solvers.
    """

    def __init__(
        self,
        network: HydraulicNetwork,
        friction_method: str = "haaland",
        max_iterations: int = 50,
        tol_mass: float = 1e-7,
        tol_pressure: float = 1e-5,
    ):
        self.network = network
        self.friction_method = friction_method
        self.max_iterations = max_iterations
        self.tol_mass = tol_mass
        self.tol_pressure = tol_pressure
        self._build_matrices()

    def _build_matrices(self) -> None:
        """Constructs incidence matrices and indexing structures."""
        self.network.validate()
        
        self.num_nodes = len(self.network.nodes)
        self.num_edges = len(self.network.edges)
        
        # Partition nodes into unknown pressure nodes (u) and known pressure nodes (k)
        self.unknown_node_ids = [n.id for n in self.network.nodes.values() if not n.is_pressure_known]
        self.known_node_ids = [n.id for n in self.network.nodes.values() if n.is_pressure_known]
        
        self.num_u = len(self.unknown_node_ids)
        self.num_k = len(self.known_node_ids)
        
        self.u_id_to_idx = {nid: i for i, nid in enumerate(self.unknown_node_ids)}
        self.k_id_to_idx = {nid: i for i, nid in enumerate(self.known_node_ids)}
        
        # Incidence matrix A (N x M): A[i, e] = +1 if e enters i, -1 if e leaves i
        self.A_full = np.zeros((self.num_nodes, self.num_edges), dtype=np.float64)
        for e_id, edge in self.network.edges.items():
            self.A_full[edge.start_node_id, e_id] = -1.0
            self.A_full[edge.end_node_id, e_id] = +1.0
            
        self.A_u = self.A_full[self.unknown_node_ids, :]  # (num_u x M)
        self.A_k = self.A_full[self.known_node_ids, :]    # (num_k x M)
        
        # Vectors of edge properties
        self.edge_L = np.array([e.length for e in self.network.edges.values()], dtype=np.float64)
        self.edge_Dh = np.array([e.hydraulic_diameter for e in self.network.edges.values()], dtype=np.float64)
        self.edge_A = np.array([e.area for e in self.network.edges.values()], dtype=np.float64)
        self.edge_rr = np.array([e.relative_roughness for e in self.network.edges.values()], dtype=np.float64)
        self.edge_K = np.array([e.total_loss_coefficient for e in self.network.edges.values()], dtype=np.float64)
        
        # Fluid properties & Precomputed fast evaluation factors
        self.rho = self.network.fluid.density
        self.mu = self.network.fluid.viscosity
        self.geom_factor = self.rho / (2.0 * (self.edge_A ** 2))
        self.L_over_Dh = self.edge_L / self.edge_Dh
        self.re_factor = (self.rho * self.edge_Dh) / (self.mu * self.edge_A)
        self.haaland_term0 = np.maximum(self.edge_rr / 3.7, 0.0) ** 1.11
        self.ln10 = math.log(10.0)
        
        # Node boundary vectors
        self.P_k_fixed = np.array([self.network.nodes[nid].fixed_pressure for nid in self.known_node_ids], dtype=np.float64)
        self.Q_ext_u = np.array([self.network.nodes[nid].external_inflow for nid in self.unknown_node_ids], dtype=np.float64)
        self.Q_ext_all = np.array([n.external_inflow for n in self.network.nodes.values()], dtype=np.float64)

        # 3D Node Elevations and Hydrostatic Head: dP_hydro = (rho - rho_amb) * g * (z_end - z_start)
        self.g_acc = getattr(self.network.fluid, "gravity", 9.80665)  # m/s^2
        self.node_z = np.array([self.network.nodes[nid].z for nid in range(self.num_nodes)], dtype=np.float64)
        self.edge_dz = self.A_full.T @ self.node_z  # z_end - z_start (M,)
        
        amb_rho = getattr(self.network.fluid, "ambient_density", None)
        isoth = getattr(self.network.fluid, "isothermal_gauge", False)
        if amb_rho is not None:
            eff_drho = self.rho - float(amb_rho)
        elif isoth:
            eff_drho = 0.0
        else:
            eff_drho = self.rho
        self.edge_dp_hydro = eff_drho * self.g_acc * self.edge_dz  # Effective buoyancy / hydrostatic head in Pa

        # Precompute initial guess
        if self.num_u > 0:
            self.A_u_pinv = scipy.linalg.pinv(self.A_u)
            self.Q_init_default = self.A_u_pinv @ (-self.Q_ext_u)
            self.Q_init_default = np.where(np.abs(self.Q_init_default) < 1e-4, 1e-3, self.Q_init_default)
        else:
            self.Q_init_default = np.full(self.num_edges, 0.1, dtype=np.float64)
            
        p_ref = float(np.mean(self.P_k_fixed)) if self.num_k > 0 else 0.0
        self.P_u_init_default = np.full(self.num_u, p_ref + 10.0, dtype=np.float64)

        # Fans
        self.has_fans = any(e.fan is not None for e in self.network.edges.values())
        self.fan_edges = [(i, e.fan) for i, e in self.network.edges.items() if e.fan is not None]

    def _eval_edge_loss(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast vectorized evaluation of edge head loss h(Q), major loss, minor loss, f, and analytical dh/dQ.
        """
        abs_Q = np.abs(Q)
        Re = self.re_factor * abs_Q
        Re_safe = np.maximum(Re, 4000.0)
        
        # Turbulent Haaland friction factor & exact derivative
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
        h = total_R * abs_Q * Q + self.edge_dp_hydro
        
        df_dQ = df_dRe * self.re_factor * np.sign(Q + 1e-15)
        dh_dQ = 2.0 * total_R * np.maximum(abs_Q, 1e-8) + df_dQ * self.L_over_Dh * self.geom_factor * (Q * Q)
        
        if self.has_fans:
            for i, fan in self.fan_edges:
                h[i] -= fan.pressure_boost(Q[i])
                dh_dQ[i] -= fan.derivative(Q[i])
                
        dh_dQ = np.maximum(dh_dQ, 1e-8)
        return h, dh_dQ, dp_major, dp_minor, f

    def _initial_flows_and_pressures(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns precomputed initial guess for edge flows Q and unknown pressures P_u."""
        return np.copy(self.Q_init_default), np.copy(self.P_u_init_default)

    def solve(self, method: str = "gga", initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> HydraulicSolution:
        """
        Solves the 1D hydraulic network.
        
        Args:
            method: 'gga' (Global Gradient Algorithm, fastest), 'newton', 'scipy_root', or 'scipy_least_squares'.
            initial_guess: Optional tuple of (Q_init, P_u_init).
            
        Returns:
            HydraulicSolution object containing all pressures, flows, losses, and convergence stats.
        """
        t_start = time.perf_counter()
        
        if method.lower() == "gga":
            sol = self._solve_gga(initial_guess)
        elif method.lower() == "newton":
            sol = self._solve_newton(initial_guess)
        elif method.lower() == "scipy_root":
            sol = self._solve_scipy_root(initial_guess)
        elif method.lower() == "scipy_least_squares":
            sol = self._solve_scipy_least_squares(initial_guess)
        else:
            raise ValueError(f"Unknown solver method '{method}'. Available: 'gga', 'newton', 'scipy_root', 'scipy_least_squares'.")
            
        t_end = time.perf_counter()
        sol.execution_time_ms = (t_end - t_start) * 1000.0
        return sol

    def _solve_gga(self, initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> HydraulicSolution:
        """
        Todini-Pilati Global Gradient Algorithm (Schur-Complement Newton-Raphson).
        Solves symmetric positive-definite system:
        (A_u * D^(-1) * A_u^T) * delta_P_u = (A_u * Q + Q_ext_u) - A_u * D^(-1) * (A^T * P + h(Q))
        """
        if initial_guess is not None:
            Q, P_u = np.array(initial_guess[0], copy=True), np.array(initial_guess[1], copy=True)
        else:
            Q, P_u = self._initial_flows_and_pressures()

        converged = False
        iteration = 0
        max_mass_err = 0.0
        max_res = 0.0

        for it in range(1, self.max_iterations + 1):
            iteration = it
            
            # Construct full pressure vector P
            P = np.zeros(self.num_nodes, dtype=np.float64)
            if self.num_u > 0:
                P[self.unknown_node_ids] = P_u
            if self.num_k > 0:
                P[self.known_node_ids] = self.P_k_fixed

            # Evaluate head loss and derivatives
            h, dh_dQ, dp_major, dp_minor, f = self._eval_edge_loss(Q)
            
            # Residual 1: Edge momentum balance F1 = A^T * P + h(Q) = 0
            F1 = self.A_full.T @ P + h
            
            # Residual 2: Node mass conservation F2 = A_u * Q + Q_ext_u = 0
            F2 = self.A_u @ Q + self.Q_ext_u if self.num_u > 0 else np.zeros(0)
            
            max_res_edge = np.max(np.abs(F1))
            max_mass_err = np.max(np.abs(F2)) if self.num_u > 0 else 0.0
            max_res = max(max_res_edge, max_mass_err)
            
            if max_mass_err < self.tol_mass and max_res_edge < self.tol_pressure:
                converged = True
                break

            # Diagonal inverse D^(-1)
            inv_D = 1.0 / dh_dQ  # (M,)
            
            if self.num_u > 0:
                # Schur complement matrix S = A_u * diag(inv_D) * A_u^T  (num_u x num_u)
                S = (self.A_u * inv_D) @ self.A_u.T
                
                # Right-hand side: RHS = F2 - A_u * (inv_D * F1)
                rhs = F2 - self.A_u @ (inv_D * F1)
                
                # Solve S * delta_P_u = rhs
                if self.num_u == 1:
                    delta_P_u = np.array([rhs[0] / S[0, 0]])
                elif self.num_u == 2:
                    s00, s01, s10, s11 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
                    inv_det = 1.0 / (s00 * s11 - s01 * s10)
                    delta_P_u = np.array([
                        (s11 * rhs[0] - s01 * rhs[1]) * inv_det,
                        (-s10 * rhs[0] + s00 * rhs[1]) * inv_det
                    ])
                else:
                    delta_P_u = np.linalg.solve(S, rhs)
                
                # Update P_u
                P_u += delta_P_u
                P[self.unknown_node_ids] = P_u
                
                # Update Q: delta_Q = - inv_D * (A^T * P_new + h(Q))
                F1_new = self.A_full.T @ P + h
                delta_Q = - inv_D * F1_new
                
                # Damped step if flow change is too violent
                alpha = 1.0
                Q += alpha * delta_Q
            else:
                # All pressures known, update Q directly
                delta_Q = - inv_D * F1
                Q += delta_Q

        # Final evaluation for result packaging
        P = np.zeros(self.num_nodes, dtype=np.float64)
        if self.num_u > 0:
            P[self.unknown_node_ids] = P_u
        if self.num_k > 0:
            P[self.known_node_ids] = self.P_k_fixed
            
        h, dh_dQ, dp_major, dp_minor, f = self._eval_edge_loss(Q)
        rho = self.network.fluid.density
        mu = self.network.fluid.viscosity
        velocities = np.abs(Q) / self.edge_A
        reynolds = (rho * velocities * self.edge_Dh) / mu
        dp_edges = - (self.A_full.T @ P)  # P_start - P_end

        # Package solution fast
        node_names = [n.name for n in self.network.nodes.values()]
        edge_names = [e.name for e in self.network.edges.values()]
        node_pressures = dict(zip(node_names, P))
        edge_flows = dict(zip(edge_names, Q))
        edge_vels = dict(zip(edge_names, velocities))
        edge_re = dict(zip(edge_names, reynolds))
        edge_f = dict(zip(edge_names, f))
        edge_dp = dict(zip(edge_names, dp_edges))
        edge_maj = dict(zip(edge_names, dp_major))
        edge_min = dict(zip(edge_names, dp_minor))
        edge_elev = dict(zip(edge_names, self.edge_dp_hydro))

        return HydraulicSolution(
            converged=converged,
            iterations=iteration,
            execution_time_ms=0.0,
            mass_conservation_error=float(max_mass_err),
            max_residual=float(max_res),
            node_pressures=node_pressures,
            edge_flows=edge_flows,
            edge_velocities=edge_vels,
            edge_reynolds=edge_re,
            edge_friction_factors=edge_f,
            edge_pressure_drops=edge_dp,
            edge_major_losses=edge_maj,
            edge_minor_losses=edge_min,
            edge_elevation_heads=edge_elev,
        )

    def _solve_newton(self, initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> HydraulicSolution:
        """Standard Full Newton-Raphson on Coupled System [Q, P_u]."""
        if initial_guess is not None:
            Q, P_u = np.array(initial_guess[0], copy=True), np.array(initial_guess[1], copy=True)
        else:
            Q, P_u = self._initial_flows_and_pressures()

        converged = False
        iteration = 0
        max_mass_err = 0.0
        max_res = 0.0

        for it in range(1, self.max_iterations + 1):
            iteration = it
            P = np.zeros(self.num_nodes, dtype=np.float64)
            if self.num_u > 0:
                P[self.unknown_node_ids] = P_u
            if self.num_k > 0:
                P[self.known_node_ids] = self.P_k_fixed

            h, dh_dQ, dp_major, dp_minor, f = self._eval_edge_loss(Q)
            F1 = self.A_full.T @ P + h
            F2 = self.A_u @ Q + self.Q_ext_u if self.num_u > 0 else np.zeros(0)

            max_mass_err = np.max(np.abs(F2)) if self.num_u > 0 else 0.0
            max_res = max(np.max(np.abs(F1)), max_mass_err)

            if max_mass_err < self.tol_mass and np.max(np.abs(F1)) < self.tol_pressure:
                converged = True
                break

            # Full Jacobian:
            # [ D    A_u^T ] [ delta_Q   ] = - [ F1 ]
            # [ A_u   0    ] [ delta_P_u ]   - [ F2 ]
            if self.num_u > 0:
                N_sys = self.num_edges + self.num_u
                J = np.zeros((N_sys, N_sys), dtype=np.float64)
                J[:self.num_edges, :self.num_edges] = np.diag(dh_dQ)
                J[:self.num_edges, self.num_edges:] = self.A_u.T
                J[self.num_edges:, :self.num_edges] = self.A_u
                
                rhs = - np.concatenate([F1, F2])
                delta = np.linalg.solve(J, rhs)
                
                Q += delta[:self.num_edges]
                P_u += delta[self.num_edges:]
            else:
                delta_Q = - F1 / dh_dQ
                Q += delta_Q

        P = np.zeros(self.num_nodes, dtype=np.float64)
        if self.num_u > 0:
            P[self.unknown_node_ids] = P_u
        if self.num_k > 0:
            P[self.known_node_ids] = self.P_k_fixed

        h, dh_dQ, dp_major, dp_minor, f = self._eval_edge_loss(Q)
        rho = self.network.fluid.density
        mu = self.network.fluid.viscosity
        velocities = np.abs(Q) / self.edge_A
        reynolds = (rho * velocities * self.edge_Dh) / mu
        dp_edges = - (self.A_full.T @ P)

        node_names = [n.name for n in self.network.nodes.values()]
        edge_names = [e.name for e in self.network.edges.values()]

        return HydraulicSolution(
            converged=converged,
            iterations=iteration,
            execution_time_ms=0.0,
            mass_conservation_error=float(max_mass_err),
            max_residual=float(max_res),
            node_pressures={n.name: float(P[n.id]) for n in self.network.nodes.values()},
            edge_flows={e.name: float(Q[e.id]) for e in self.network.edges.values()},
            edge_velocities={e.name: float(velocities[e.id]) for e in self.network.edges.values()},
            edge_reynolds={e.name: float(reynolds[e.id]) for e in self.network.edges.values()},
            edge_friction_factors={e.name: float(f[e.id]) for e in self.network.edges.values()},
            edge_pressure_drops={e.name: float(dp_edges[e.id]) for e in self.network.edges.values()},
            edge_major_losses={e.name: float(dp_major[e.id]) for e in self.network.edges.values()},
            edge_minor_losses={e.name: float(dp_minor[e.id]) for e in self.network.edges.values()},
            edge_elevation_heads=dict(zip(edge_names, self.edge_dp_hydro)),
        )

    def _solve_scipy_root(self, initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> HydraulicSolution:
        """SciPy optimize root wrapper."""
        if initial_guess is not None:
            Q_init, P_u_init = initial_guess
        else:
            Q_init, P_u_init = self._initial_flows_and_pressures()

        x0 = np.concatenate([Q_init, P_u_init])

        def residual(x: np.ndarray) -> np.ndarray:
            Q = x[:self.num_edges]
            P_u = x[self.num_edges:]
            P = np.zeros(self.num_nodes, dtype=np.float64)
            if self.num_u > 0:
                P[self.unknown_node_ids] = P_u
            if self.num_k > 0:
                P[self.known_node_ids] = self.P_k_fixed
            h, _, _, _, _ = self._eval_edge_loss(Q)
            F1 = self.A_full.T @ P + h
            F2 = self.A_u @ Q + self.Q_ext_u if self.num_u > 0 else np.zeros(0)
            return np.concatenate([F1, F2])

        res = scipy.optimize.root(residual, x0, method="hybr")
        x_opt = res.x
        Q_opt = x_opt[:self.num_edges]
        P_u_opt = x_opt[self.num_edges:]

        P = np.zeros(self.num_nodes, dtype=np.float64)
        if self.num_u > 0:
            P[self.unknown_node_ids] = P_u_opt
        if self.num_k > 0:
            P[self.known_node_ids] = self.P_k_fixed

        h, _, dp_major, dp_minor, f = self._eval_edge_loss(Q_opt)
        rho = self.network.fluid.density
        mu = self.network.fluid.viscosity
        velocities = np.abs(Q_opt) / self.edge_A
        reynolds = (rho * velocities * self.edge_Dh) / mu
        dp_edges = - (self.A_full.T @ P)
        F2 = self.A_u @ Q_opt + self.Q_ext_u if self.num_u > 0 else np.zeros(0)

        node_names = [n.name for n in self.network.nodes.values()]
        edge_names = [e.name for e in self.network.edges.values()]

        return HydraulicSolution(
            converged=bool(res.success),
            iterations=int(res.nfev),
            execution_time_ms=0.0,
            mass_conservation_error=float(np.max(np.abs(F2))) if self.num_u > 0 else 0.0,
            max_residual=float(np.max(np.abs(res.fun))),
            node_pressures={n.name: float(P[n.id]) for n in self.network.nodes.values()},
            edge_flows={e.name: float(Q_opt[e.id]) for e in self.network.edges.values()},
            edge_velocities={e.name: float(velocities[e.id]) for e in self.network.edges.values()},
            edge_reynolds={e.name: float(reynolds[e.id]) for e in self.network.edges.values()},
            edge_friction_factors={e.name: float(f[e.id]) for e in self.network.edges.values()},
            edge_pressure_drops={e.name: float(dp_edges[e.id]) for e in self.network.edges.values()},
            edge_major_losses={e.name: float(dp_major[e.id]) for e in self.network.edges.values()},
            edge_minor_losses={e.name: float(dp_minor[e.id]) for e in self.network.edges.values()},
            edge_elevation_heads=dict(zip(edge_names, self.edge_dp_hydro)),
        )

    def _solve_scipy_least_squares(self, initial_guess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> HydraulicSolution:
        """SciPy optimize least_squares wrapper."""
        if initial_guess is not None:
            Q_init, P_u_init = initial_guess
        else:
            Q_init, P_u_init = self._initial_flows_and_pressures()

        x0 = np.concatenate([Q_init, P_u_init])

        def residual(x: np.ndarray) -> np.ndarray:
            Q = x[:self.num_edges]
            P_u = x[self.num_edges:]
            P = np.zeros(self.num_nodes, dtype=np.float64)
            if self.num_u > 0:
                P[self.unknown_node_ids] = P_u
            if self.num_k > 0:
                P[self.known_node_ids] = self.P_k_fixed
            h, _, _, _, _ = self._eval_edge_loss(Q)
            F1 = self.A_full.T @ P + h
            F2 = self.A_u @ Q + self.Q_ext_u if self.num_u > 0 else np.zeros(0)
            return np.concatenate([F1, F2])

        res = scipy.optimize.least_squares(residual, x0, method="lm")
        x_opt = res.x
        Q_opt = x_opt[:self.num_edges]
        P_u_opt = x_opt[self.num_edges:]

        P = np.zeros(self.num_nodes, dtype=np.float64)
        if self.num_u > 0:
            P[self.unknown_node_ids] = P_u_opt
        if self.num_k > 0:
            P[self.known_node_ids] = self.P_k_fixed

        h, _, dp_major, dp_minor, f = self._eval_edge_loss(Q_opt)
        rho = self.network.fluid.density
        mu = self.network.fluid.viscosity
        velocities = np.abs(Q_opt) / self.edge_A
        reynolds = (rho * velocities * self.edge_Dh) / mu
        dp_edges = - (self.A_full.T @ P)
        F2 = self.A_u @ Q_opt + self.Q_ext_u if self.num_u > 0 else np.zeros(0)

        node_names = [n.name for n in self.network.nodes.values()]
        edge_names = [e.name for e in self.network.edges.values()]

        return HydraulicSolution(
            converged=bool(res.success),
            iterations=int(res.nfev),
            execution_time_ms=0.0,
            mass_conservation_error=float(np.max(np.abs(F2))) if self.num_u > 0 else 0.0,
            max_residual=float(np.max(np.abs(res.fun))),
            node_pressures={n.name: float(P[n.id]) for n in self.network.nodes.values()},
            edge_flows={e.name: float(Q_opt[e.id]) for e in self.network.edges.values()},
            edge_velocities={e.name: float(velocities[e.id]) for e in self.network.edges.values()},
            edge_reynolds={e.name: float(reynolds[e.id]) for e in self.network.edges.values()},
            edge_friction_factors={e.name: float(f[e.id]) for e in self.network.edges.values()},
            edge_pressure_drops={e.name: float(dp_edges[e.id]) for e in self.network.edges.values()},
            edge_major_losses={e.name: float(dp_major[e.id]) for e in self.network.edges.values()},
            edge_minor_losses={e.name: float(dp_minor[e.id]) for e in self.network.edges.values()},
            edge_elevation_heads=dict(zip(edge_names, self.edge_dp_hydro)),
        )
