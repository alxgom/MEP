from types import SimpleNamespace

from mep_routing.routing.workspace import RoutingWorkspace


class TerminalRuntimeProbe:
    def __init__(self):
        self.graphs = []

    def set_graph(self, nodes, adjacency, spatial_index):
        self.graphs.append((nodes, adjacency, spatial_index))


def _runtime(label):
    env = SimpleNamespace(nodes=[label], adj={label: []})
    return SimpleNamespace(
        nodes=[label], adjacency={label: []}, edge_list=[(0, 1)], edge_coords=[label],
        spatial_index=f"kd-{label}", env=env,
    )


def test_workspace_commit_and_dynamic_filter_keep_base_graph_distinct():
    committed = _runtime("base")
    dynamic_env = SimpleNamespace(nodes=["dynamic"], adj={"dynamic": []})
    lifecycle = SimpleNamespace(
        build_selected=lambda *_args, **_kwargs: SimpleNamespace(runtime=committed),
        apply_dynamic_obstacle=lambda *_args, **_kwargs: SimpleNamespace(env=dynamic_env),
    )
    terminal = TerminalRuntimeProbe()
    workspace = RoutingWorkspace(lifecycle=lifecycle)

    workspace.build_selected(0, {}, 0, terminal_runtime=terminal)
    workspace.apply_machine_obstacle(None, {}, 0, clearance_mm=10, terminal_runtime=terminal)

    assert workspace.active_graph is committed
    assert workspace.env is dynamic_env
    assert terminal.graphs == [
        (committed.env.nodes, committed.env.adj, "kd-base"),
        (dynamic_env.nodes, dynamic_env.adj, "kd-base"),
    ]


def test_workspace_dwelling_replacement_discards_graph_cache_and_overlay():
    workspace = RoutingWorkspace(lifecycle=object(), active_graph=_runtime("old"), env=object())
    workspace.overlay.values[(1, 2)] = 4
    old_cache = workspace.static_clearance_cache

    lifecycle = object()
    workspace.replace_dwelling(lifecycle)

    assert workspace.lifecycle is lifecycle
    assert workspace.active_graph is None and workspace.env is None
    assert workspace.overlay.values == {}
    assert workspace.static_clearance_cache is not old_cache
