"""
Pipe Flow - High Performance HVAC & Fluid Network Solvers.
"""

from pipe_flow.solvers.hydraulic_1d import (
    FluidProperties,
    DuctMaterial,
    Fitting,
    FittingLossLibrary,
    FanCurve,
    Node,
    Edge,
    DuctSegment,
    HydraulicNetwork,
    Hydraulic1DSolver,
    HydraulicSolution,
)

from pipe_flow.solvers.grid_ffd_2d import (
    CellType,
    BoundaryFace,
    DuctGeometry2D,
    GridFFD2D,
    SimulationResult,
)

from pipe_flow.solvers.ai_surrogate import (
    SurrogateNetworkInput,
    SurrogatePrediction,
    DatasetSample,
    HydraulicSurrogateSolver,
    NumPySurrogateForward,
    extract_physics_features,
    generate_branching_dataset,
)

__version__ = "0.1.0"
__all__ = [
    # 1D Hydraulic Solver
    "FluidProperties",
    "DuctMaterial",
    "Fitting",
    "FittingLossLibrary",
    "FanCurve",
    "Node",
    "Edge",
    "DuctSegment",
    "HydraulicNetwork",
    "Hydraulic1DSolver",
    "HydraulicSolution",
    # 2D Grid FFD Solver
    "CellType",
    "BoundaryFace",
    "DuctGeometry2D",
    "GridFFD2D",
    "SimulationResult",
    # AI / GNN Surrogate Solver
    "SurrogateNetworkInput",
    "SurrogatePrediction",
    "DatasetSample",
    "HydraulicSurrogateSolver",
    "NumPySurrogateForward",
    "extract_physics_features",
    "generate_branching_dataset",
]
