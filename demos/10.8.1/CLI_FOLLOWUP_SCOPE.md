# Cli Follow-up Scope

This document defines the boundaries that must be agreed before implementing the remaining Cli validation, diagnostic, and grille-control work.

## Route validation and KPIs

The validation layer should consume a completed `CliRoutingResult`; it must not alter search costs or retry routing. Initial candidate checks are disconnected supply/return grilles, missing machine or grille stubs, segments outside the routing region, supply/return self-crossings, and dimension discontinuities. Metrics should be installation-owned and split by supply and return: routed grille count, length, turns, crossings, segment count, and solver nodes/time.

Before implementation, confirm:

- which conditions are hard failures versus warnings;
- whether supply/return intersections are allowed at different elevations;
- which dimensional transitions require fittings rather than warnings;
- whether KPI values aggregate both networks or remain separate in the plots.

## Debug overlays

Diagnostics should be optional projections of solver artifacts, never inputs to solving. Candidate layers are snapped terminal nodes, metric-closure candidates, selected closure edges, final tree edges, access nodes/stubs, crossing-penalized edges, and rejected/disconnected terminals. Each backend must declare which artifacts it can provide through one Cli diagnostic result; the canvas must tolerate absent layers.

Before implementation, confirm the minimum useful layers, whether overlays are active-installation-only, and whether diagnostic state must be serializable in solution snapshots.

## Grille controls

Grille controls should modify an explicit `CliGrillePlacementConfig` and trigger grille preparation, machine placement, and routing in that order. Potential controls include supply/return dimensions, wall margins, grille separation, routing-region inset, horizontal-return policy, and room eligibility. Return ventilation remains a separate network toggle.

Before implementation, confirm which values are project inputs, machine-family data, hard feasibility rules, or user heuristics. Do not expose a control until the runtime consumes it and its valid unit/range is sourced.

## Non-goals

- No inferred regulation values.
- No debug-only data in shared routing contracts unless another installation needs it.
- No generic control schema until at least Cli and Sal have two concrete matching controls.
- No automatic retry or solution ranking policy hidden inside validation.
