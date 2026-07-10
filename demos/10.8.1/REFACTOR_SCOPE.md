# Demo 10.8.1 Refactor Scope

Demo 10.8.1 starts as a behavior-preserving copy of Demo 10.8. The refactor goal is to turn the current single-file Pygame prototype into a modular routing workbench that can support Sal now and later add Cli, Coc, and San with their own parameters, graphs, strategies, UI affordances, machine families, terminal families, and validation rules.

This document is the working contract for the first refactor pass. It should be updated when a boundary proves wrong in code.

## Current Baseline

`main.py` currently mixes these concerns in one module:

- import/path bootstrapping for local demos and optional dwelling export data
- global configuration constants
- global mutable app state
- geometry normalization and Shapely helpers
- grid construction for regular, Hannan, and epsilon graphs
- machine geometry, connector pins, and access nodes
- terminal selection and preferred terminal tools
- static clearance fields, route-route interaction fields, and validation warnings
- A* and line-graph search backends
- sequential, negotiated, and min-cost-flow strategy orchestration
- dwelling source loading and synthetic generation
- placement scoring and auto-placement
- Pygame event handling, drawing, plots, logs, and UI widgets

The main risk is not only file length. The stronger issue is that domain rules, solver choices, UI state, and debug visualization share global variables, so adding Cli/Coc/San will multiply implicit coupling unless configuration and domain capabilities are made explicit.

## Refactor Principles

- Preserve behavior in small commits. Each extraction should keep `python main.py` runnable.
- Move from global variables to explicit context objects where state crosses module boundaries.
- Split by reason for existence, not by the current consuming function.
- Keep pure computation independent from Pygame and filesystem access.
- Keep adapters thin: Pygame, dwelling DB loading, and future CLI/export integrations should call domain services rather than contain domain logic.
- Use configuration keys that describe domain meaning first. A solver may consume a regulation value, but it should not own the value semantically.
- Normalize internal geometry and routing units to integer millimetres as much as possible. UI can expose metres where useful, but internal millimetre integers reduce floating-point tie-break ambiguity and may improve memory/performance later.

## Target Package Shape

Initial target under `demos/10.8.1/`:

```text
main.py
vent_router/
  __init__.py
  app.py
  state.py
  config/
    __init__.py
    schema.py
    defaults_sal.py
    legacy_keys.py
  domain/
    __init__.py
    units.py
    rooms.py
    machines.py
    terminals.py
    shafts.py
    clearances.py
    validation.py
  geometry/
    __init__.py
    transforms.py
    shapely_utils.py
    distances.py
    frames.py
  graphs/
    __init__.py
    env.py
    regular.py
    hannan.py
    epsilon.py
    weights.py
  routing/
    __init__.py
    astar.py
    line_graph.py
    min_cost_flow.py
    strategies.py
    scoring.py
  placement/
    __init__.py
    candidates.py
    scoring.py
    rotation.py
  data_sources/
    __init__.py
    synthetic.py
    dwelling_export.py
  ui/
    __init__.py
    pygame_app.py
    layout.py
    events.py
    drawing.py
    widgets.py
    plots.py
  observability/
    __init__.py
    logs.py
    snapshots.py
```

`main.py` should become only an entrypoint:

```python
from vent_router.app import run

if __name__ == "__main__":
    run()
```

## Boundary Semantics

`config`: typed defaults, key metadata, compatibility aliases, and config loading. It must not import Pygame or Shapely-heavy runtime state.

`domain`: names and rules from the installation domain: machine families, terminal families, room categories, shaft semantics, diameter rules, feasibility constraints, heuristic defaults, and route validation. Confirmed regulation values should only be introduced when a legal/code/project-code source is explicit. Machine families should be modeled as catalogs, not singletons, because Cli is expected to have multiple available machines even though Sal currently has one.

`geometry`: generic coordinate transforms and geometry operations. It may use Shapely/NumPy but should not know about Sal/Cli/Coc/San.

`graphs`: graph construction and edge fields. It can depend on `geometry` and read domain/config values through explicit arguments.

`routing`: path search and route orchestration. It consumes graph structures, start/target specs, and cost policies. It should not read Pygame state.

`placement`: machine candidate generation, rotation scoring, and placement ranking. It should consume domain machine models, graph/geometry services, and config.

`data_sources`: adapters for synthetic and real dwelling inputs. It should normalize into shared domain models.

`ui`: Pygame-specific state, drawing, interaction, plot widgets, and event bindings. It calls app services and does not own routing/domain rules.

`observability`: route snapshots, logs, debug dumps, and reproducibility metadata.

## Configuration Key Model

Use keys shaped as:

```text
<SCOPE>.<AREA>.<SUBAREA>.<PARAMETER>
```

Names should state why the parameter exists, not which function currently consumes it.

Good:

```text
SALUBRIDAD.FEASIBILITY.BOCAS.MIN_DISTANCE_TO_WALL_MM
SALUBRIDAD.SOLVER.GRAPH.GRID_EPSILON
DOMAIN.GEOMETRY.WALL_THICKNESS
EXPORT.IMAGES.ENABLED
```

Avoid:

```text
SALUBRIDAD.SOLVER.HEURISTIC.BOCAS_MIN_DISTANCE_TO_WALL
```

unless the value is truly only a solver heuristic. Do not use `REGULATION` for Demo 10.8.1 values unless we have an explicit legal/code/project-code source.

## Proposed Top-Level Scopes

- `SYSTEM`: process-level defaults and global runtime behavior.
- `PROJECT`: project/building-level metadata and selected scenario.
- `IO`: filesystem paths, templates, family paths, local datasets, logs.
- `EXECUTION`: phases, workers, max attempts, installation order, retry behavior.
- `DEBUG`: debug targets, debug images, graph dumps, continue-on-failure behavior.
- `EXPORT`: output JSON, images, route info, building export.
- `OBSERVABILITY`: log levels, trace metadata, config snapshots.
- `DOMAIN`: shared building geometry, spaces, categories, mapping, units.
- `PATINEJOS`: shaft-specific shared rules.
- `SALUBRIDAD`: Sal-specific rules, solver defaults, machines, terminals, UI choices.
- `SALUBRIDAD_DOBLE_FLUJO`: double-flow Sal-specific rules.
- `CLIMATIZACION`: Cli-specific rules, machines, terminals, solver defaults.
- `SANEAMIENTO`: San-specific rules, machines, terminals, solver defaults.
- `COCINA`: Coc-specific rules, machines, terminals, solver defaults.
- `FONTANERIA`: plumbing-specific rules, machines, terminals, solver defaults.
- `COLLISIONS`: cross-domain collision policies where they are not owned by one installation.
- `INTEGRATIONS`: Azure/cloud/external service configuration.

## Area Vocabulary

- `DOMAIN`: building, spaces, families, room categories.
- `SOLVER`: graph epsilon, graph kind, MST/Steiner mode, search backend, routing strategy.
- `FEASIBILITY`: hard geometric validity and allowed placement limits.
- `REGULATION`: legal/code/project-code constraints. Demo 10.8.1 should treat this area as reserved until a source is explicit.
- `HEURISTIC`: preference weights and professional judgement that are not validity rules.
- `MAINTENANCE`: access clearances and serviceability distances.
- `EXECUTION`: workers, phases, max solutions, backtracking, retry limits.
- `DEBUG`: graph dumps, images, targeted debug selectors, failure continuation.
- `EXPORT`: exported artifacts and output options.
- `OBSERVABILITY`: logger mode, log level, trace IDs, config-used snapshots.
- `PATHS`: directories, templates, families, logs.
- `INPUT`: input source selection and source-specific parameters.
- `OUTPUT`: output destination and naming policies.
- `MAPPING`: schema/category/name mapping.
- `GEOMETRY`: geometric constants and transforms.
- `PLACEMENT`: placement candidate generation and ranking.
- `COLLISIONS`: collision solving, jump margins, obstacle buffers.
- `PERFORMANCE`: pruning thresholds, caches, max attempts, performance ceilings.
- `COMPATIBILITY`: legacy flags, old schema names, old rerun behavior.
- `INTEGRATION`: cloud/external service integration settings.

## Classification Rules

- If breaking it violates legal/code/domain rules, use `REGULATION`.
- If breaking it makes geometry invalid or impossible, use `FEASIBILITY`.
- If changing it changes search strategy or algorithm behavior, use `SOLVER`.
- If changing it expresses preference but not validity, use `HEURISTIC`.
- If it exists for service/access after installation, use `MAINTENANCE`.
- If it only helps inspect execution, use `DEBUG` or `OBSERVABILITY`.
- If it is a file/directory/external artifact, use `IO.PATHS` or `INTEGRATIONS`.

## First Demo 10.8.1 Config Mapping

Current constant | Proposed key | Reason
--- | --- | ---
`GRID_SPACING` | `SALUBRIDAD.SOLVER.GRAPH.REGULAR_GRID_SPACING_MM` | Solver graph resolution for this domain.
`HANNAN_SCAFFOLD_SPACING` | `SALUBRIDAD.SOLVER.GRAPH.HANNAN_SCAFFOLD_SPACING_MM` | Solver graph connectivity scaffold.
`CORE_EPSILON_GRID_MM` | `SALUBRIDAD.SOLVER.GRAPH.GRID_EPSILON_MM` | Solver graph axis offset.
`WALL_THICKNESS` | `DOMAIN.GEOMETRY.WALL_THICKNESS_MM` | Shared building geometry.
`ROUTING_WALL_CLEARANCE_MM` | `SALUBRIDAD.FEASIBILITY.ROUTING.MIN_DISTANCE_TO_WALL_MM` | Current wall clearance behaves as a hard geometric feasibility rule.
`TERMINAL_REGULATION_CLEARANCE_MM` | `SALUBRIDAD.FEASIBILITY.TERMINALS.MIN_DISTANCE_TO_WALL_MM` | Terminal feasibility rule, not solver-owned.
`BUFFER_ROOM_TERMINALES_AIRE_MM` | `SALUBRIDAD.FEASIBILITY.TERMINALS.ROOM_BOUNDARY_BUFFER_MM` | Hard terminal candidate feasibility.
`PATINEJO_CLEARANCE_MM` | `PATINEJOS.FEASIBILITY.CLEARANCE.NON_SHAFT_DUCT_MM` | Shaft clearance for non-shaft ducts.
`SHAFT_ENTRY_SEARCH_MM` | `PATINEJOS.SOLVER.ENTRY.SEARCH_RADIUS_MM` | Search envelope for shaft entry candidate discovery.
`SHAFT_ENTRY_MAX_CANDIDATES` | `PATINEJOS.PERFORMANCE.ENTRY.MAX_CANDIDATES` | Candidate cap.
`MACHINE_BODY_W` | `SALUBRIDAD.DOMAIN.MACHINE.OZEO_FLAT.BODY_WIDTH_MM` | Machine family geometry.
`MACHINE_BODY_H` | `SALUBRIDAD.DOMAIN.MACHINE.OZEO_FLAT.BODY_HEIGHT_MM` | Machine family geometry.
`MACHINE_OVERALL_W` | `SALUBRIDAD.DOMAIN.MACHINE.OZEO_FLAT.OVERALL_WIDTH_MM` | Machine family geometry.
`MACHINE_SMALL_DUCT_D` | `SALUBRIDAD.DOMAIN.DUCTS.SMALL_DIAMETER_MM` | Sal duct family definition.
`MACHINE_LARGE_DUCT_D` | `SALUBRIDAD.DOMAIN.DUCTS.LARGE_DIAMETER_MM` | Sal duct family definition.
`SMALL_PIN_STUB_LENGTH` | `SALUBRIDAD.FEASIBILITY.CONNECTORS.SMALL_STUB_LENGTH_MM` | Connector reach geometry used for valid access.
`LARGE_PIN_STUB_LENGTH` | `SALUBRIDAD.FEASIBILITY.CONNECTORS.LARGE_STUB_LENGTH_MM` | Connector reach geometry used for valid access.
`DUCT_BUFFER_RATIO` | `SALUBRIDAD.FEASIBILITY.CLEARANCE.DUCT_BUFFER_RATIO` | Geometry validity inflation factor.
`MACHINE_CLEARANCE_SOFT_MARGIN_MM` | `SALUBRIDAD.HEURISTIC.PLACEMENT.MACHINE_CLEARANCE_SOFT_MARGIN_MM` | Soft penalty band, not a hard invalidation.
`C_BEND_DEFAULT` | `SALUBRIDAD.SOLVER.COST.BEND_PENALTY_DEFAULT_MM` | Search cost behavior.
`C_BEND_MIN` | `SALUBRIDAD.UI.CONTROLS.BEND_PENALTY_MIN_MM` | UI slider range.
`C_BEND_MAX` | `SALUBRIDAD.UI.CONTROLS.BEND_PENALTY_MAX_MM` | UI slider range.
`CROSSING_MULTIPLIER_DEFAULT` | `SALUBRIDAD.SOLVER.COST.CROSSING_PENALTY_MULTIPLIER_DEFAULT` | Search cost behavior.
`OVERLAP_BLOCK_WEIGHT` | `COLLISIONS.SOLVER.OVERLAP_BLOCK_WEIGHT` | Cross-domain collision solver policy.
`MIN_PIECE_FACTOR_DEFAULT` | `SALUBRIDAD.HEURISTIC.DUCTS.MIN_PIECE_FACTOR_DEFAULT` | Current warning/scoring threshold matching routing-core config; not confirmed as regulation.
`SHORT_PIECE_SCORE_PENALTY` | `SALUBRIDAD.SOLVER.COST.SHORT_PIECE_SCORE_PENALTY_MM` | Score consequence, not the regulation itself.
`REAL_DWELLING_DB` | `IO.PATHS.DWELLING_EXPORT_DB` | External local data source.
`REAL_DWELLING_SCENARIOS` | `PROJECT.INPUT.REAL_DWELLING_SCENARIOS` | Scenario selection for demo/project.
`PREFERRED_SHAFT_INSTALLATION` | `SALUBRIDAD.INPUT.PREFERRED_SHAFT_INSTALLATION` | Domain-specific input filter.
`FPS` | `SYSTEM.UI.FPS` | Runtime/UI process behavior.

## Config Metadata Direction

Flat keys are enough for the first pass, but the schema should allow metadata later:

```python
ConfigParameter(
    key="SALUBRIDAD.FEASIBILITY.BOCAS.MIN_DISTANCE_TO_WALL_MM",
    reason="feasibility",
    default=100,
    unit="mm",
    also_affects=("feasibility", "solver"),
    legacy_names=("BOCAS_MIN_DISTANCE_TO_WALL",),
)
```

Do not force every value into metadata before extraction. Start with typed defaults and a legacy alias map, then add metadata where migration decisions need traceability.

## External Reference Sources

- `C:\Users\ALEXIS GOMEL\Documents\Dwelling-export`: real-dwelling geometry export source.
- `C:\Users\ALEXIS GOMEL\Documents\letsmep-routing-core`: routing-core implementation to mimic or selectively port when aligning Sal behavior.
- `C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Familias`: machine-family metadata by installation type. Use this later for machine catalogs; do not assume a domain has only one machine.
- `C:\Users\ALEXIS GOMEL\Desktop\LETSMEP\AzureFile\Plantillas\Enrutado\Config\global_config.json`: mutable local-testing config. Use as a value source only when explicitly requested, not as an implicit replacement for the checked-in default catalog.

## Migration Strategy

1. Commit the untouched baseline copy.
2. Add this scope document and keep it current.
3. Introduce `vent_router.config` with typed defaults and legacy constant compatibility.
4. Extract pure value objects: units, env graph, machine spec, route spec, config object.
5. Extract geometry helpers that have no Pygame dependency.
6. Extract graph builders behind a common `build_graph(kind, context, config)` interface.
7. Extract search backends behind a common `route_one(...)` interface.
8. Extract route scoring and validation into domain/routing services.
9. Extract placement candidate generation and scoring.
10. Move Pygame drawing and event handling into `ui`, keeping `app.py` as orchestration.
11. Split Sal-specific defaults from shared domain defaults.
12. Add future Cli/Coc/San by adding config defaults and domain capabilities, not by branching UI/solver code everywhere.

## Compatibility Policy

During the refactor, preserve:

- the existing keyboard/mouse controls
- the default Sal behavior
- English semantic config keys for Demo 10.8.1
- current graph mode names
- current strategy names
- current score formula unless a commit explicitly says otherwise
- runtime ability to launch from `demos/10.8.1/main.py`

Expected temporary compromises:

- Some globals may remain until the state object is introduced.
- Some functions may move before their signatures are ideal.
- The first config layer can expose both old constant names and new semantic keys to reduce migration risk.

## Decisions

- Internal units: use integer millimetres as much as possible. Expose metres in UI where useful, but keep routing, graph, and geometry calculations in millimetres to reduce floating-point tie-break ambiguity and leave a path toward memory/performance gains.
- Regulation: Demo 10.8.1 does not currently include confirmed regulation values. Existing distances should be classified as `FEASIBILITY`, `HEURISTIC`, `SOLVER`, or related areas until a legal/code/project-code source is explicit.
- UI scope: most UI behavior and style can be installation-domain scoped, for example `SALUBRIDAD.UI`. Global controls such as ruler, zoom, weight overlays, and common dwelling-geometry styling should remain under `SYSTEM.UI` or shared UI modules.
- Language: Demo 10.8.1 semantic config keys and new compatibility aliases should stay in English.

## Open Questions

- Should route-route/route-obstacle collision policy be a top-level cross-domain scope (`COLLISIONS`) or live under a shared domain area until multi-installation coordination is implemented?
