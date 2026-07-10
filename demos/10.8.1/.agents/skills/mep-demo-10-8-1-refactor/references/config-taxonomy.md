# Config Taxonomy Reference

Use semantic keys:

```text
<SCOPE>.<AREA>.<SUBAREA>.<PARAMETER>
```

Name values by why they exist, not by the function that consumes them.

## Current Decisions

- Use English keys.
- Use integer millimetres internally wherever practical.
- UI may display metres.
- Do not use `REGULATION` without an explicit legal/code/project-code source.
- Current Demo 10.8.1 distances are mostly `FEASIBILITY`, `HEURISTIC`, `SOLVER`, or `PERFORMANCE`.
- Default values can remain checked in for now. If explicitly requested, parameter values may be sourced from local testing config at `C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Config\global_config.json`; that file is mutable and should not be treated as immutable truth.

## Classification

- `FEASIBILITY`: breaking it makes geometry invalid or impossible.
- `SOLVER`: changing it changes graph/search/cost algorithm behavior.
- `HEURISTIC`: professional preference or warning/scoring behavior, not validity.
- `MAINTENANCE`: serviceability/access after installation.
- `DEBUG`: inspection-only execution help.
- `OBSERVABILITY`: logs, traces, config snapshots.
- `PATHS`: directories, templates, families, logs.
- `REGULATION`: only confirmed legal/code/project-code constraints.

## Examples

Prefer:

```text
SALUBRIDAD.FEASIBILITY.ROUTING.MIN_DISTANCE_TO_WALL_MM
SALUBRIDAD.SOLVER.GRAPH.GRID_EPSILON_MM
SALUBRIDAD.HEURISTIC.DUCTS.MIN_PIECE_FACTOR_DEFAULT
DOMAIN.GEOMETRY.WALL_THICKNESS_MM
```

Avoid:

```text
SALUBRIDAD.SOLVER.HEURISTIC.BOCAS_MIN_DISTANCE_TO_WALL
SALUBRIDAD.REGULATION.*  # unless sourced
```

## Machine Metadata

Machine-family metadata lives at:

```text
C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Familias
```

Use this as a future source for machine catalogs by installation type. Sal currently has one machine, but Cli is expected to have multiple machines, so avoid APIs that assume a single global machine per domain.
