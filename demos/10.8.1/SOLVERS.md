# Demo 10.8.1 solver guide

This is the index of the workbench's implemented solvers: what each does, how it
works, and why it is available. These are interactive lab experiments. The
[README](README.md) covers controls and detailed experiment notes;
[IMPLEMENTATION_STEPS](IMPLEMENTATION_STEPS.md) records changes and measured runs.

## Separate the choices

- **Grid** supplies physical nodes and edges. Regular, Hannan, Epsilon and Core
  Simple are graph builders, not alternative objectives.
- **Router** selects the path or tree algorithm.
- **Strategy** selects how Sal combines individual routes and routing orders.
- **Heuristic** controls search guidance. It is separate from the accumulated
  edge, bend and interaction costs.
- **KPI score** evaluates a completed installation. It need not equal the
  internal search objective, particularly for Kou and the fitting perturbation.

Changing a backend can change an early route and therefore the interaction
weights seen by later routes. Equal starting graphs do not imply equal later
weighted graphs. Compare at the same dwelling, machine pose, grid, starts,
weights, minimum setting and installation order.

## Sal path backends

| Router | What and how | Why keep it |
| --- | --- | --- |
| State-expanded A* | Labels arrivals by `(node, incoming direction)`; priority is accumulated cost plus a remaining-cost heuristic. Direction changes incur the bend cost. | Compact bend-aware reference implementation. |
| Line graph L(G) A* | Labels arrivals by incoming directed edge; transitions represent continuation or turns. Uses accumulated cost plus heuristic. | Compare an edge-based representation of the same path problem. |
| Line graph L(G) GBFS | Uses the line-graph target model and accumulated costs, but prioritizes the heuristic alone. It can stop at a more expensive route. | Observe the latency/route-quality tradeoff from greedier expansion. |
| State A* (hard min pieces) | Adds the physical straight-run length and first-piece status to the state-expanded search. Rejects a bend or completion that closes a short piece. | Test hard length feasibility without replacing the grid. |
| Line graph A* (hard min pieces) | Adds the same physical length policy to directed-edge arrivals. | Compare state-expanded and line-graph representations under the same constraint. |

All use the prepared weighted graph. Costs include physical length and applicable
clearance/crossing terms; turning adds the configured bend cost. A* optimality
depends on the selected heuristic and graph model, and does not imply a globally
optimal multi-route installation.

The heuristic selector offers **Pin + bends** (distance to candidate pin access
plus estimated remaining bends), **Pin distance** (distance without bend
estimation), **Machine envelope** (a pin-agnostic bound from the machine center
and access radius), and **Zero** (Dijkstra-style A* diagnostic). These choices
change search guidance; they do not replace the configured bend/crossing costs.

The hard variants merge collinear subdivisions conceptually by accumulating
length until direction changes. They saturate attained length at the largest
relevant minimum and retain nondominated cost/length arrivals. A cheaper arrival
does not dominate a longer arrival if that extra length permits a later turn.
Machine stubs, snap adapters and shaft-entry pieces participate in feasibility.
Immediate reversals cannot manufacture straight-run length.

The existing **Min pieces factor** slider sets:

```text
endpoint minimum = duct diameter * factor
internal minimum = duct diameter * 2 * factor
factor range = 0.50–2.00; default = 1.05
```

Baseline routers use these thresholds for post-route warnings and scoring. Hard
routers also enforce them during search. Lengths are centerline measurements;
there are no fitting deductions or later sanitization in this experiment.

### Sal strategies

| Strategy | Mechanism and reason |
| --- | --- |
| Greedy (Dual-Sort) | Tries the prepared small-room order and its reverse; inexpensive order comparison. |
| First Fit | Tries small-room permutations and stops at the first successful crossing-free candidate; otherwise retains the best scored successful candidate. |
| Best Fit | Evaluates all small-room permutations and retains the best completed score; spends more work on route ordering. |
| Negotiated Congestion | Reroutes with updated congestion costs to resolve interactions. |
| Negotiated Congestion (Favour Large) | Uses the negotiated procedure with a preference for protecting large ducts. |
| Min-Cost Flow (Small Pins) | Keeps shaft/kitchen routing and assigns the small ducts jointly. |
| Min-Cost Flow (Two-Stage) | Evaluates big-first and small-first flow stage orders and keeps the better final score. |

Hard length backends support sequential and negotiated strategies. Flow stages
bypass those A* backends, so their hard-length combinations report unsupported.
These strategies do not exhaust all possible joint route configurations.

Sources: [search dispatch](mep_routing/routing/search.py),
[length search](mep_routing/routing/piece_search.py),
[strategy dispatch](mep_routing/installations/sal/strategy_dispatch.py),
[order policy](mep_routing/installations/sal/orchestration.py).

## Cli supply-tree backends

Cli connects a machine root to one supply grille per target room. Directed
backends use **Wall candidates** to choose the grille location as part of search.
The `Grille center` mode uses the fixed prepared terminal instead; directed
choices without wall candidates currently dispatch to the metric-MST backend.
The three hard-length modes require wall candidates explicitly.

| Router | What and how | Why keep it |
| --- | --- | --- |
| Routing-Core Port: Kou Steiner | Builds terminal-pair paths, a metric-closure MST, expands it, then prunes/simplifies. Tries nine coordinate-heuristic scales. | Adapted routing-core comparison baseline. |
| Kou nonnegative heuristics | Same construction with four nonnegative heuristic scales. | Isolate the effect of signed heuristic trials. |
| Line graph L(G) metric MST | Builds bend-aware terminal-pair paths, connects them by an MST and reduces/orients their physical union. | Include bend costs while keeping a relatively simple approximate tree construction. |
| Directed root-path union (baseline) | Computes turn-aware root paths, chooses the cheapest candidate for each room, then unites the paths. | Cheap reference that does not optimize branch sharing jointly. |
| Directed Steiner arborescence (exact) | Uses terminal-subset DP on turn-aware states. Merges disjoint subsets at a state, then propagates costs with reverse shortest-path search. | Optimize shared branches jointly in the expanded graph model. “Exact” refers to that model, not a physical fitting/fabrication optimum. |
| Steiner: min bends | Adds capped physical straight-run history to the subset solver; turns and endpoints must satisfy the minimum. Straight passage through a tee does not reset a run. | Isolate minimum lengths at bends. |
| Steiner: min bends + tees | Also ends the incoming piece at branches and resets every outgoing piece, including straight continuation. | Model minimum spacing between bends, junctions and endpoints. |
| Steiner: min + fittings | Same bends-and-tees feasibility, plus `1e-6 * physical fitting count` in its scalar objective. | Prefer fewer components for equal or nearly equal primary costs. |

Kou uses weighted paths to build candidates but ranks its final trial trees by
`initial_edge_count + simplified_edge_count`. It does not use the hard solver's
length/bend objective. An MST of pair paths is also an approximation to the
joint tree problem. Do not interpret backend score values as interchangeable.

### Cli hard minimum control

With a hard-length router active, **Hard min** multiplies the largest calculated
supply width: the full-flow trunk width from the existing sizing policy.
The slider initially uses **1.00x**, currently 600 mm for the sourced M71JA inputs.

Dragging the slider sets a multiplier of **0.25–3.00x in 0.05x steps**. The label
shows both the multiplier and resulting minimum in mm. For a 600 mm trunk,
0.50x means 300 mm and 1.50x means 900 mm. Reset restores 1.00x. Every setting
follows subsequent sizing changes automatically. The choice is shared by all
three hard-length variants, saved/restored with solution logs, and retained
across router changes. Baseline routers ignore it. Sal retains its separate
diameter-factor control.

The multiplier changes feasibility only. It does not resize ducts, change graph
weights or relax the minimum automatically. Physical machine/grille adapters
count; collinear graph edges and section changes do not end a piece. Supply
sizing must still be available. Exact machine access is required; a graph that
snaps it reports unsupported. Optional return routing remains the existing
unconstrained metric-MST implementation.

### Fitting perturbation and memory

An elbow, tee or cross counts as one component; a straight continuation counts
as zero. Reducers, straight couplings and endpoint connections are not included.
Candidate-specific departure-direction masks classify a completed junction
once. Turning an existing elbow into a tee does not add a second component at
that node. The solver reuses existing masks and scalar labels without adding a
fitting-history dimension.

The perturbation is a near-tie preference, not lexicographic ordering: sufficiently
small primary-cost differences can be outweighed. Counts are independently
reconstructed from final physical centerlines. The KPI score is unchanged.

Sources: [backend dispatch](mep_routing/installations/cli/solver_backends.py),
[wall preparation and directed DP](mep_routing/installations/cli/directed_wall_graph.py),
[length DP and validation](mep_routing/installations/cli/length_steiner.py),
[fitting classification](mep_routing/installations/cli/fittings.py),
[runtime composition](mep_routing/installations/cli/runtime.py),
[Kou implementation](mep_routing/routing/steiner.py).

## Limits, failures and performance

- Hard Cli solutions undergo reconstructed and materialized physical-tree
  validation. Expanded paths can collapse into an invalid physical union. The
  selected candidate is then blocked without enumerating alternatives.
  **Blocked does not certify infeasibility.**
- Context-obstacle relaxation is a separate runtime retry, identified in status.
  It preserves the requested hard minimum. It does not relax length constraints.
- Capping length bounds history distinctions, not absolute memory. Cli terminal
  subsets grow exponentially with room count. Increasing the minimum can retain
  more unsaturated histories, so slider positions need not have equal latency.
- The recent audit identified unresolved shared wall-preparation issues:
  inconsistent vertical direction labels, repeated cuts on one original edge,
  and zero-length aliases affecting bend costs. Context-retry timing can omit
  the first attempt. The slider does not fix those issues; account for them when
  interpreting route choice and displayed latency.
- Benchmarks measure scene recomputation without rendering. Compare success and
  rejection sets as well as timings. There is no fixed 200 ms guarantee.

Benchmark entrypoints:
[Sal](benchmarks/benchmark_sal_min_pieces.py),
[Cli](benchmarks/benchmark_cli_hard_lengths.py).
Recorded historical comparisons:
[Sal hard lengths](benchmarks/sal_min_pieces_summary.json),
[Cli hard lengths](benchmarks/cli_hard_lengths_summary.json),
[Cli fittings](benchmarks/cli_fittings_summary.json).
Check the recorded catalog size, revisions and weighting corrections before
using older results as current performance claims.

## Other installations

Coc currently uses the two-terminal Kou connection between its exported exhaust
connector and Coc shaft. It has no hard segment-length experiment. San has no
interactive routing implementation in this demo.

Coc can retry a failed Core Simple connection on the Regular grid; successful
fallback is identified in status. See its [runtime](mep_routing/installations/coc/runtime.py).
