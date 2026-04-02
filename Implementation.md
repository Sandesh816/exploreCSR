# Implementation Notes for `prototype-v7.ipynb`

This document records what was implemented in [prototype-v7.ipynb](/Users/sandesh816/Developer/exploreCSR/prototype-v7.ipynb), what assumptions the implementation currently makes, and what remains open.

`prototype-v6.ipynb` was left unchanged. `prototype-v7.ipynb` was created as the working copy for this round of experiments.

## Goal

The implemented idea is:

1. Start with a relation pool extracted from `C1`.
2. Avoid brute-force reasoning over the full relation universe.
3. Use symbolic heuristics plus an algebraic verifier to produce a shortlist of plausible relation bundles.
4. Prepare only that shortlist for later LLM ranking.

The main design decision in `v7` is that the LLM should not search over all raw relation combinations. The symbolic system should first cut down the search space.

## Problem Framing Used in `v7`

The notebook currently works on the two-object example:

- `pizza`
- `cutter`

Each object is an axis-aligned rectangle with low-level parameters:

- `x`
- `y`
- `w`
- `h`

`C1` is the original canvas.
`C2` is the canvas after the user's edit.
`C3` is the next predicted canvas produced by a selected relation bundle.

The default example in the notebook is:

- `C1`: `cutter.x = -4`
- `C2`: `cutter.x = -5`
- Therefore `delta = {cutter.x}`

## What Was Added in `v7`

The `v7` notebook adds a separate "March 5 Experiment Block" on top of the earlier prototype code.

This block adds:

- a filter-funnel report comparing naive enumeration against heuristic filtering,
- bundle analysis helpers,
- a verifier that searches for the smallest extra set of fixed variables,
- bundle inspection cells for concrete edge cases,
- an LLM ranking prompt builder that operates on already-verified candidates.

It also fixes two inherited execution issues in the copied notebook:

- the duplicate `from __future__ import annotations` line that prevented clean reruns,
- the missing `_num_vars_in_linexpr` dependency by replacing it with a local simple-pin helper used only in the new analysis code.

## Current Pipeline

### 1. Detect the Relation Pool from `C1`

The notebook builds a deduplicated equation pool from `C1`.

In the default two-object example, the pool size is:

- `26` linear relations

The notebook currently uses:

- offsets from `C1`,
- pin equations from `C1`,
- `linear_only=True`

That means the new analysis block works on the linear relation pool only.

### 2. Compute the User-Edited Variables

The notebook computes:

- `delta = changed_params(C1, C2)`

In the default example:

- `delta = {'cutter.x'}`

This is the first signal for relevance.

### 3. Enumerate Candidate Relation Bundles

The analysis block enumerates non-empty subsets of relation indices up to a configured size.

Current default:

- `max_system_size = 3`

Important clarification:

- the current naive count is a subset count, not `25!` or `26!`,
- for the default example, naive subsets up to size 3 equal `2951`,
- all non-empty subsets across every size would be `2^26 - 1 = 67108863`.

### 4. Heuristic Filters

The notebook computes several filters separately instead of merging them into one hard rule.

#### A. Delta-overlap heuristic

Keep only bundles whose union of support variables intersects `delta`.

This implements Jason's set-level heuristic:

- a bundle is relevant if the changed parameter appears somewhere in the bundle,
- not every individual equation must contain the changed variable.

Example:

- `(0, 14)` is allowed by this heuristic because bundle support includes `cutter.x` through relation `014`, even though relation `000` does not mention `cutter.x`.

#### B. Shared-variable heuristic

For multi-equation bundles, require that at least one variable appears in at least two equations.

This is used in `v7` as a measured heuristic, not as the final acceptance rule.

Reason:

- it can remove structurally weak bundles,
- but it can also be too aggressive for some cases you may still want to keep.

#### C. Connected-component heuristic

Build a variable graph for the bundle and ask whether the equations form one connected component.

This is also treated as a measured structural heuristic, not the final acceptance rule.

Reason:

- it is stricter than the shared-variable heuristic,
- it removes many bundles,
- but some disconnected bundles are still algebraically valid under the current solver.

### 5. Verifier and Hyper-Parameter Search

This is the main new selection mechanism.

For each bundle:

- build the linear system,
- treat changed variables in `delta` as constants copied from `C2`,
- search for the smallest extra set of fixed variables from the bundle support,
- compile a repair program if the remaining driven variables become uniquely solvable.

Current default:

- `max_extra_fixed = 2`

The verifier returns the first viable program with the smallest number of extra fixed variables.

This means the notebook prefers bundles that need fewer extra hyper-parameters.

If no viable parameterization is found within the current budget, the bundle is rejected.

### 6. Produce a Deterministic `C3`

If a bundle passes verification:

- it is compiled into a `RepairProgram`,
- the program is applied to `C2`,
- the resulting deterministic next canvas is treated as `C3`.

The analysis block stores:

- relation indices,
- support variables,
- whether the bundle passed the structural heuristics,
- the minimum number of extra fixed variables,
- the chosen fixed and driven variables,
- predicted `C2 -> C3` changes,
- the program text.

### 7. Rank Preparation for an LLM

The current implementation does not call an LLM yet.

Instead it prepares an LLM-ready prompt that contains only the top verified candidates.

The prompt includes:

- changed variables,
- candidate relation indices,
- equation text,
- support variables,
- minimum extra fixed count,
- fixed and driven variables,
- predicted `C2 -> C3` edits,
- optional user-history summary text.

This means the intended future workflow is:

- symbolic shortlist first,
- LLM ranking second.

## Example Output on the Default Scene

When the new analysis cells are run on the default two-object example, the notebook reports:

- equation pool size: `26`
- naive subsets up to size 3: `2951`
- all non-empty subsets across all sizes: `67108863`
- delta-overlap heuristic: `1964`
- delta + shared-variable heuristic: `1438`
- delta + connected-component heuristic: `272`
- delta + verifier: `1115`
- delta + shared-variable + verifier: `811`
- delta + connected-component + verifier: `163`

These counts are useful for the current experiment:

- `delta` overlap alone removes a large part of the naive space,
- the connected-component filter is much more aggressive,
- the verifier is a different kind of filter than the structural heuristics.

## Important Bundle Behaviors Captured in `v7`

The notebook explicitly inspects a few bundles to make the differences visible.

### Bundle `(0, 14)`

Relations:

- `000: cutter.h - 6 = 0`
- `014: cutter.x - pizza.x + 2 = 0`

Observed behavior:

- passes delta-overlap,
- fails the shared-variable heuristic,
- fails the connected-component heuristic,
- still passes the verifier.

Interpretation:

- this is exactly why `v7` keeps structural heuristics separate from final verification.

### Bundle `(14, 24)`

Relations:

- `014: cutter.x - pizza.x + 2 = 0`
- `024: pizza.x + 2 = 0`

Observed behavior:

- passes delta-overlap,
- passes shared-variable,
- passes connectedness,
- fails the verifier.

Interpretation:

- the bundle looks structurally reasonable,
- but becomes contradictory once the user-edited value is fixed.

This is the canonical example of a "bad equation relation" that the verifier is meant to remove.

### Bundle `(14, 25)`

Relations:

- `014: cutter.x - pizza.x + 2 = 0`
- `025: pizza.y - 3 = 0`

Observed behavior:

- passes delta-overlap,
- fails the stricter structural heuristics,
- passes the verifier.

Interpretation:

- algebraic validity does not automatically imply semantic usefulness,
- this is the kind of case a later ranker may need to demote.

## Assumptions in the Current Implementation

The current `v7` implementation makes the following assumptions:

- We are working on only two objects for now.
- Objects are axis-aligned rectangles.
- Rectangle coordinates use `(x, y, w, h)` with `y` treated as the top edge.
- The working relation pool is derived from `C1`.
- The `v7` experiment block uses only linear relations.
- Changed variables in `delta` are treated as constants during `C2 -> C3` repair compilation.
- Extra hyper-parameters are chosen only from variables that appear in the bundle support.
- Hyper-parameter search is greedy by count: use the smallest number of extra fixed variables first.
- The search budget is currently capped at `max_extra_fixed = 2`.
- Bundle enumeration is currently capped at `max_system_size = 3`.
- Shared-variable and connected-component checks are treated as diagnostics and optional filters, not final truth.
- The verifier defines viability as unique solvability under the chosen fixed-variable set.
- The current ranking order inside the notebook is heuristic, not learned.
- The LLM is assumed to rank candidates after symbolic filtering, not generate candidates from scratch.
- User history is currently only a free-text field in the ranking prompt builder.

## What Is Not Implemented Yet

The following pieces are not implemented in `v7`:

- actual open-source LLM inference through Ollama or another runtime,
- learning from past user edit sequences,
- semantic ranking from scene images or text descriptions,
- a formal contradiction explanation beyond pass/fail verification,
- a principled semantic penalty for bundles that are algebraically valid but likely irrelevant,
- scaling beyond the current small two-object experimental setting.

## Recommended Reading Order in the Notebook

If you want to understand `v7` from top to bottom, the most relevant cells are:

- setup and relation-pool generation,
- the "March 5 Experiment Block" markdown cell,
- the helper cell that defines `BundleAnalysis` and the new analysis functions,
- the cell that runs `analyze_relation_bundles(...)`,
- the inspection / prompt-generation cell.

## Short Summary

`prototype-v7.ipynb` implements:

- symbolic relation-bundle enumeration,
- delta-based relevance filtering,
- optional structural heuristics,
- minimal-hyperparameter verification,
- deterministic `C3` preview generation,
- LLM prompt construction over a verified shortlist.

The core idea implemented is:

- do not let the model search the full relation space,
- let the symbolic system produce a smaller, better candidate set first,
- then use a model only for final ranking.
