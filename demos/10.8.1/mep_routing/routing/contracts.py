"""Small, installation-neutral contracts shared by routing solvers.

These value objects describe what a solver receives and returns.  They avoid
installation policy, UI state, and algorithm configuration so those concerns
can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeVar


Node = TypeVar("Node")
Target = TypeVar("Target")
Graph = TypeVar("Graph")
Route = TypeVar("Route")


@dataclass(frozen=True)
class RoutingRequest(Generic[Node, Target]):
    """One named connection between source nodes and eligible targets."""

    key: str
    source_nodes: tuple[Node, ...]
    target_candidates: tuple[Target, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("routing request key must not be empty")
        if not self.source_nodes:
            raise ValueError(f"routing request {self.key!r} requires a source node")
        if not self.target_candidates:
            raise ValueError(f"routing request {self.key!r} requires a target candidate")


@dataclass(frozen=True)
class RoutingProblem(Generic[Node, Target, Graph]):
    """Graph and connection requests supplied to one solver invocation."""

    graph: Graph
    requests: tuple[RoutingRequest[Node, Target], ...]
    edge_weights: Mapping[tuple[Node, Node], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = tuple(request.key for request in self.requests)
        if len(keys) != len(set(keys)):
            raise ValueError("routing request keys must be unique")


@dataclass(frozen=True)
class SolvedRoute(Generic[Route]):
    """Solver-owned route payload associated with its request key."""

    request_key: str
    route: Route


@dataclass(frozen=True)
class SolverFailure:
    """Machine-readable solver failure with optional request attribution."""

    code: str
    message: str
    request_key: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("solver failure code must not be empty")


@dataclass(frozen=True)
class SolverResult(Generic[Route]):
    """Normalized solver outcome, independent of UI presentation."""

    routes: tuple[SolvedRoute[Route], ...] = ()
    failure: SolverFailure | None = None
    elapsed_ms: float = 0.0
    route_node_count: int = 0

    def __post_init__(self) -> None:
        if self.routes and self.failure is not None:
            raise ValueError("solver result cannot contain both routes and a failure")
        if self.elapsed_ms < 0:
            raise ValueError("solver elapsed time must not be negative")
        if self.route_node_count < 0:
            raise ValueError("solver route-node count must not be negative")

    @property
    def success(self) -> bool:
        return self.failure is None
