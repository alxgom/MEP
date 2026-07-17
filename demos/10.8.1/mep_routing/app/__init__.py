"""Application-level runtime coordinators for the interactive router."""

from .dwelling_session import ActiveDwellingSession
from .dwelling_application import (
    CommonGraphConfig,
    InstallationGraphProfile,
    TerminalRuntimeConfig,
    create_active_dwelling,
    load_real_dwelling,
)
from .graph_build_session import GraphBuildSession
from .demo_config import DemoConfig, build_demo_config
from .workbench_application import DemoWorkbenchApplication
from .workbench_state import (
    DisplayState, DwellingState, InstallationState, MachinePose,
    SalWorkbenchState, WorkbenchState,
)
from .installation_ui import (
    InstallationMachineUi,
    InstallationUiSession,
    MachineSelectionTransition,
)
from .installation_context import installation_context_obstacles
from .installation_execution import InstallationCoordinator, InstallationExecution
from .installation_runtime import (
    CliInstallationRuntime, InstallationCapabilities, InstallationRegistry,
    InstallationRuntime, InstallationSolveContext, SalInstallationRuntime,
)

from .interaction_runtime import (
    InteractionCallbacks,
    InteractionState,
    RoutingRun,
    RoutingTransition,
    apply_auto_placement,
    apply_dwelling_change,
    apply_rotation,
    apply_routing_transition,
    apply_slider_change,
    apply_terminal_area,
    apply_terminal_point,
    reset_terminal_preferences,
)

__all__ = [
    "ActiveDwellingSession",
    "CommonGraphConfig",
    "InstallationGraphProfile",
    "TerminalRuntimeConfig",
    "create_active_dwelling",
    "DisplayState", "DwellingState", "InstallationState", "MachinePose",
    "SalWorkbenchState", "WorkbenchState",
    "load_real_dwelling",
    "GraphBuildSession",
    "DemoConfig", "build_demo_config",
    "DemoWorkbenchApplication",
    "InstallationMachineUi",
    "InstallationUiSession",
    "installation_context_obstacles",
    "InstallationCoordinator", "InstallationExecution",
    "InstallationCapabilities", "InstallationRegistry", "InstallationRuntime",
    "InstallationSolveContext", "SalInstallationRuntime", "CliInstallationRuntime",
    "MachineSelectionTransition",
    "InteractionCallbacks",
    "InteractionState",
    "RoutingRun",
    "RoutingTransition",
    "apply_auto_placement",
    "apply_dwelling_change",
    "apply_rotation",
    "apply_routing_transition",
    "apply_slider_change",
    "apply_terminal_area",
    "apply_terminal_point",
    "reset_terminal_preferences",
]
