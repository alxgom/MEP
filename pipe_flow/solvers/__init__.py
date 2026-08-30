"""
Solvers subpackage for pipe_flow.
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

from pipe_flow.solvers.hardy_cross import (
    Loop,
    HardyCrossSolver,
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

from pipe_flow.solvers.lbm_3d import (
    VoxelType,
    SimulationResult3D,
    DuctGeometry3D,
    LBM3DSolver,
)

from pipe_flow.solvers.ai_fno_2d3d import (
    FNONetworkInput2D,
    FNONetworkInput3D,
    FNOPrediction2D,
    FNOPrediction3D,
    SpectralConv2dNumPy,
    SpectralConv3dNumPy,
    FourierNeuralOperator2D,
    FourierNeuralOperator3D,
    FNO2DSolver,
    FNO3DSolver,
    compute_wall_distance_2d,
    compute_wall_distance_3d,
    generate_synthetic_duct_grid_2d,
    generate_synthetic_duct_voxel_3d,
)

from pipe_flow.solvers.lbm_2d import (
    CellType2D,
    SimulationResult2D,
    LBM2DSolver,
)

from pipe_flow.solvers.streamfunction_vorticity_2d import (
    StreamfunctionVorticityResult2D,
    StreamfunctionVorticity2DSolver,
)

from pipe_flow.solvers.fv_simple_2d import (
    FVSIMPLEResult2D,
    FVSIMPLE2DSolver,
)

from pipe_flow.solvers.multilayer_25d import (
    MultiLayer25DResult,
    MultiLayer25DSolver,
)

from pipe_flow.solvers.fvm_simple_2d import (
    FVMSimple2D,
    FVMSimpleResult,
    build_mvp_fvm_geometry,
)

from pipe_flow.solvers.multiscale_2p5d import (
    Riser1DConfig,
    Layer2DConfig,
    MultiScale2p5DSolution,
    MultiScale2p5DSolver,
    build_floor1_plenum_geometry,
    build_floor2_plenum_geometry,
)

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
    # Hardy Cross Loop Solver
    "Loop",
    "HardyCrossSolver",
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
    # 3D Lattice Boltzmann Solver
    "LatticeStencilType",
    "CellType3D",
    "BoundaryFace3D",
    "LBM3DConfig",
    "SimulationResult3D",
    "DuctGeometry3D",
    "LBM3DSolver",
    # AI 2D/3D Fourier Neural Operator (FNO)
    "FNONetworkInput2D",
    "FNONetworkInput3D",
    "FNOPrediction2D",
    "FNOPrediction3D",
    "SpectralConv2dNumPy",
    "SpectralConv3dNumPy",
    "FourierNeuralOperator2D",
    "FourierNeuralOperator3D",
    "FNO2DSolver",
    "FNO3DSolver",
    "compute_wall_distance_2d",
    "compute_wall_distance_3d",
    "generate_synthetic_duct_grid_2d",
    "generate_synthetic_duct_voxel_3d",
    # Candidate 7: 2D LBM D2Q9 + LES
    "CellType2D",
    "SimulationResult2D",
    "LBM2DSolver",
    # Candidate 8: 2D Streamfunction-Vorticity Solver
    "StreamfunctionVorticityResult2D",
    "StreamfunctionVorticity2DSolver",
    # Candidate 9: 2D FV SIMPLE Solver
    "FVSIMPLEResult2D",
    "FVSIMPLE2DSolver",
    "FVMSimple2D",
    "FVMSimpleResult",
    "build_mvp_fvm_geometry",
    # Candidate 10: Multi-Layer 2.5D Coupled Solver
    "MultiLayer25DResult",
    "MultiLayer25DSolver",
    "Riser1DConfig",
    "Layer2DConfig",
    "MultiScale2p5DSolution",
    "MultiScale2p5DSolver",
    "build_floor1_plenum_geometry",
    "build_floor2_plenum_geometry",
]
