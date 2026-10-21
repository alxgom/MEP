"""Compare moving-pose grid builds with the pre-cache implementation."""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import types
from time import perf_counter

import numpy as np


def benchmark(reference_ref="914d637", cases=2, repeats=3):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    demo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(demo))
    with redirect_stdout(io.StringIO()):
        import main as app
    from mep_routing.graphs import GraphMode, lifecycle, hybrid
    from mep_routing.graphs.cells import cached_cell_template, cached_machine_cell_template

    reference = types.ModuleType("mep_routing.graphs._cell_cache_reference")
    reference.__package__ = "mep_routing.graphs"
    sys.modules[reference.__name__] = reference
    source = subprocess.check_output(
        ["git", "show", f"{reference_ref}:demos/10.8.1/mep_routing/graphs/variants.py"], cwd=demo, text=True)
    exec(compile(source, f"{reference_ref}:variants.py", "exec"), reference.__dict__)
    current = (lifecycle.build_cell_variant, lifecycle.build_hannan_cell_variant)
    modes = (GraphMode.HANNAN, GraphMode.CELL_SKELETON, GraphMode.HANNAN_CELL)
    rows = []

    def clear_caches():
        cached_cell_template.cache_clear()
        cached_machine_cell_template.cache_clear()
        hybrid._cached_join.cache_clear()

    sources = [(0, i) for i in range(min(cases, len(app.REAL_DWELLING_SCENARIOS)))] + [(1, 0)]
    try:
        for source_index, case in sources:
            random.seed(42)
            np.random.seed(42)
            app.dwelling_source_idx, app.real_scenario_idx = source_index, case
            app.graph_type_idx, app.router_backend_idx = 1, 0
            with redirect_stdout(io.StringIO()):
                app.dwelling_runtime.generate()
            graph = app.workbench_state.dwelling.workspace.lifecycle
            pose = app.workbench_state.installation("sal").pose
            x, y, angle = pose.as_tuple()
            for repeat in range(repeats):
                paths = ("reference", "local_cache") if repeat % 2 == 0 else ("local_cache", "reference")
                for path in paths:
                    lifecycle.build_cell_variant, lifecycle.build_hannan_cell_variant = (
                        current if path == "local_cache" else
                        (reference.build_cell_variant, reference.build_hannan_cell_variant))
                    for mode in modes:
                        clear_caches()
                        # Prime one initial pose, then time different positions/rotations.
                        graph.build_selected(mode, app.routing_controller.machine_pins(x, y, angle), angle)
                        for dx, rotation in ((100, 0), (200, 0), (0, 90)):
                            pins = app.routing_controller.machine_pins(x + dx, y, angle + rotation)
                            before = cached_cell_template.cache_info()
                            started = perf_counter()
                            result = graph.build_selected(mode, pins, angle + rotation)
                            elapsed = (perf_counter() - started) * 1000
                            after = cached_cell_template.cache_info()
                            info = result.variant.diagnostics
                            rows.append({"case": app.current_scenario_label,
                                         "source": "real" if source_index == 0 else "synthetic",
                                         "repeat": repeat, "path": path, "grid": mode.name,
                                         "pose": [x + dx, y, angle + rotation], "build_ms": round(elapsed, 3),
                                         "partition_ms": round(result.variant.axes_ms, 3),
                                         "template_hits": after.hits - before.hits,
                                         "template_misses": after.misses - before.misses,
                                         "base_cells": info.get("base_cells"), "rebuilt_cells": info.get("rebuilt_cells"),
                                         "cells": len(info.get("cells", ())), "nodes": len(result.runtime.nodes)})
    finally:
        lifecycle.build_cell_variant, lifecycle.build_hannan_cell_variant = current
        clear_caches()
        sys.modules.pop(reference.__name__, None)
    summary = {}
    for mode in modes:
        summary[mode.name] = {}
        for path in ("reference", "local_cache"):
            selected = [r for r in rows if r["grid"] == mode.name and r["path"] == path]
            times = sorted(r["build_ms"] for r in selected)
            summary[mode.name][path] = {
                "samples": len(selected), "median_ms": round(statistics.median(times), 3),
                "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))], "max_ms": times[-1],
                "median_partition_ms": round(statistics.median(r["partition_ms"] for r in selected), 3),
                "template_hits": sum(r["template_hits"] for r in selected),
                "template_misses": sum(r["template_misses"] for r in selected),
                "median_nodes": statistics.median(r["nodes"] for r in selected),
            }
    return {"reference_ref": reference_ref,
            "note": "Moving-pose grid lifecycle builds, runtime included, routing/rendering/dwelling preparation excluded. Prime an initial pose then retain caches across three different poses. Alternating implementation order; caches reset between grids/paths/repeats. Local partition geometry intentionally differs from global machine-hole decomposition; Hanan is an unchanged control. For Hanan/merged, partition_ms includes Hanan axis preparation. Machine-dependent timings.",
            "summary": summary, "samples": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", default="914d637")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("cases and repeats must be positive")
    result = benchmark(args.reference_ref, args.cases, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
