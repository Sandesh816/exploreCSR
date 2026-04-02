# Plan: LLM-Guided Relation Bundle Search for Canvas Edit Prediction

## Goal

Given a canvas edit C1 → C2, predict plausible next edits C3 by:

1. Detecting geometric relations from C1 (equality, offset, pin).
2. Filtering candidate relation bundles with deterministic heuristics.
3. Using an LLM to propose additional plausible bundles from the full pool.
4. Verifying bundles symbolically: partitioning variables (V/Δ/C/F/D), compiling RepairPrograms, materializing C3.
5. Using a second LLM call to rank verified candidates by likely designer intent.

One-sentence story:

> Deterministic heuristics cut the search space, an LLM proposes bundles it thinks a designer would intend, symbolic verification gates correctness, and a second LLM pass ranks the verified candidates.

---

## Foundational Concepts (from prototype-v6.ipynb)

These concepts are already implemented in Jason's code and must be preserved.

### Canvas and Objects

- **`Rect(x, y, w, h)`**: immutable rectangle, top-left origin. All arithmetic uses Python `Fraction` for exactness.
- **`NamedRect(name, rect)`**: a labeled rectangle (e.g., "pizza", "cutter").
- A canvas is a list of NamedRects.

### Feature Expressions (9 per rectangle)

For each rectangle, the system computes:

| Axis   | Features                        |
|--------|---------------------------------|
| X-axis | left (x), right (x+w), center_x (x+w/2) |
| Y-axis | top (y), bottom (y−h), center_y (y−h/2)  |
| Size   | width (w), height (h), area (w×h)         |

These are the atoms from which all relations are built.

### Symbolic Expressions

- **`LinExpr`**: `Σ coeff_i × var_i + const` — linear combination with Fraction coefficients.
- **`PolyExpr`**: either a LinExpr or a product of two variables (for area).
- **`Equation`**: `PolyExpr == PolyExpr`, canonicalized for deduplication.
- Canonical normalization (pivot, GCD, sign) ensures `pizza.x == cutter.x` and `cutter.x == pizza.x` are the same relation.

### Three Relation Types

1. **Equality**: `pizza.left == cutter.right` — feature-to-feature match.
2. **Offset**: `pizza.left == cutter.right + k` — fixed spacing from C1.
3. **Pin**: `pizza.width == 3` — freeze a feature to its C1 value.

All three types are generated from C1, filtered to only those that hold true in C1, then deduplicated.

### The V/Δ/C/F/D Variable Partitioning

This is the core framework for how bundles become executable programs:

| Symbol | Name             | Definition                        | Role                                    |
|--------|------------------|-----------------------------------|-----------------------------------------|
| **V**  | Support vars     | All variables appearing in bundle E | Universe for this bundle                |
| **Δ**  | Delta            | Params that changed C1 → C2       | The user's edit signal                  |
| **C**  | Constants        | Δ ∩ V                             | Read directly from C2, not solved       |
| **F**  | Fixed vars       | C ∪ F′                            | Become meta-parameters (knobs)          |
| **F′** | Extra fixed      | F \ C                             | Hyper-params beyond delta — overridable |
| **D**  | Driven vars      | V \ (C ∪ F)                       | Solved by the constraint system         |

A bundle is viable when the system of equations over D is consistent and uniquely solvable.

### Hyper-Parameter Strategy

- **Required fixed**: Δ ∩ V — always constants from C2.
- **Extra fixed (F′)**: chosen from remaining V. Controlled by `max_extra_fixed` (typically 1–2).
- The same equation bundle E can produce **multiple programs** depending on the choice of F. The system enumerates parameterizations: different F sets for the same E.
- Hyper-param values default to C2 values but can be overridden.
- Because variables can be promoted to hyper-params, a bundle of disjoint equations can still be valid — one equation's variable gets fixed, the other gets solved.

### RepairProgram and AffineInM

A compiled program expresses every driven variable as an affine function of meta-parameters:

```
pizza.x = 1.0 × m::cutter.x + 2.5
```

- **`AffineInM`**: `Σ coeff_j × m_j + const` — one per driven variable.
- **`RepairProgram`**: stores equations, const_vars, fixed_vars, meta-param names, and output map (var → AffineInM).
- **Application**: plug C2 values into meta-params → evaluate each AffineInM → get C3.

### Census and Provenance

Multiple (E, F) pairs can produce the **same** C3. The census groups unique C3 outcomes:

- Key: tuple of (var, value) pairs in C3
- Value: list of `ProgramProvenance(eq_indices, fixed_vars, program_text)`

A C3 produced by many programs is more robust than one produced by one.

---

## What's Already Built

| Module                  | Status | Contains                                                                 |
|-------------------------|--------|--------------------------------------------------------------------------|
| `milestone1_core.py`    | ✓      | All data structures, relation detection, equation pool generation, linear algebra (Gaussian elimination, determinant, rank), RepairProgram compilation, enumeration, visualization |
| `milestone1_analysis.py`| ✓      | VerifierContext, BundleRecord, bundle verification with minimal parameterization search, filter funnel, LLM ranking prompt builder |
| `milestone1_ollama.py`  | ✓      | Ollama API client, JSON schema for structured output, response parsing, RankingResult |
| `test_milestone1_ollama.py` | ✓  | Unit tests for prompt builder, parser, HTTP client                       |
| `prototype-v6.ipynb`    | ✓      | Jason's original reference notebook                                      |

**Not yet wired**: The Ollama ranking call exists but is not called from any end-to-end flow. There is no LLM proposal stage (only LLM ranking of symbolically-enumerated candidates). No `run_pipeline.py` entry point.

---

## Phase 0: Repo Cleanup

### Task 0.1 — Create `legacy/` and move old notebooks

Move `prototype-v7.ipynb` and `prototype-v8.ipynb` into `legacy/`. Keep `prototype-v6.ipynb` in root as Jason's reference.

### Task 0.2 — Define notebook roles

Create focused demo notebooks in `notebooks/`:

- `01_symbolic_baseline.ipynb` — relation detection from C1, delta, pool inspection.
- `02_llm_bundle_proposal.ipynb` — LLM proposal of bundle IDs, parsing/deduping.
- `03_verification_and_c3.ipynb` — verified bundles, materialized C3s.
- `04_llm_ranking.ipynb` — final ranking of verified candidates.

### Acceptance

- Root has only `prototype-v6.ipynb`, active Python modules, and `run_pipeline.py`.
- Old experiments preserved in `legacy/`.

---

## Phase 1: Freeze the Baseline

### Task 1.1 — Audit `prototype-v6.ipynb` against modules

Confirm that everything actively used from v6 lives in the Python modules (`milestone1_core.py`, `milestone1_analysis.py`), not only in notebooks.

### Task 1.2 — Fix known gaps

- `_num_vars_in_linexpr()` is called by `is_simple_pin()` but never defined. Implement or replace.

### Task 1.3 — Ensure module-level testability

Any behavior that matters for the final submission must be callable from Python files and testable outside Jupyter.

### Acceptance

- All active logic lives in Python modules.
- `prototype-v6.ipynb` is a read-only reference.

---

## Phase 2: Generic Symbolic Layer

### Goal

Keep the symbolic layer generic for N-object canvases, not just the 2-object pizza/cutter example.

### Task 2.1 — Relation pool generation

`detect_c1_equations(c1, ...)` already works. Verify it handles:

- N rectangles (not just 2).
- All three relation types (equality, offset, pin).
- `linear_only` toggle for area relations.
- Stable relation IDs for downstream reference.

For each relation, store: `relation_id`, equation text, support variables, relation type tag.

### Task 2.2 — Scene delta

`changed_params(c1, c2)` already works. Verify it handles:

- Arbitrary object count.
- Multiple simultaneously changed variables.
- No hardcoded object names.

### Task 2.3 — Bundle representation

A bundle is a tuple of relation IDs. But a **candidate** is (E, F) — an equation bundle plus a choice of fixed variables (parameterization). Ensure the data model captures both:

- `BundleRecord` (already exists) tracks equation indices + viable parameterizations.
- Each `ParameterizationRecord` (already exists) tracks fixed_vars, driven_vars, predicted changes, program text.

### Acceptance

- Symbolic layer works for N-object scenes.
- Bundle and parameterization are distinct concepts in the data model.

---

## Phase 3: Deterministic Heuristic Pre-Filters

### Goal

Apply cheap deterministic filters **before** involving any LLM. These cut the relation pool and candidate space without LLM cost.

### Task 3.1 — Delta-overlap filter (already implemented)

**Rule**: A bundle is relevant only if at least one relation in the bundle mentions a variable from Δ.

This is Jason's core heuristic. A bundle like `[000, 014]` passes because relation 014 contains `cutter.x`, even though 000 does not.

Already implemented in `analyze_relation_bundles()` and `verify_bundle()`. Keep as-is.

### Task 3.2 — Structural garbage filter

**Rule**: Reject bundles where equations are individually contradictory given C2 values, before even attempting parameterization.

Examples from Jason's discussion:
- Relations 024 and 025 from the default example — pin equations that contradict the user's edit.
- Any relation that, when combined with the delta constants, produces `0 = nonzero`.

This is a fast pre-check: substitute Δ values as constants, check if any single equation becomes trivially false.

### Task 3.3 — Keep structural heuristics as diagnostics only

The shared-variable and connected-component heuristics (`bundle_has_shared_variable()`, `system_is_connected()`) are already implemented as diagnostics. **Do not use them as hard filters** — because hyper-parameter promotion can make disconnected bundles valid.

Record them in BundleRecord for the LLM ranking prompt (a bundle with shared variables is a soft signal of coherence), but don't reject bundles based on them.

### Acceptance

- Delta-overlap is a hard gate.
- Individual-equation contradiction is a hard gate.
- Structural heuristics are recorded but not used for rejection.

---

## Phase 4: LLM Bundle Proposal

### Goal

Use a reasoning model to explore the combinatorial space intelligently instead of brute-force subset enumeration.

The symbolic enumeration (Phase 3) handles small bundles well (size 1–3 with 26 relations = 2951 subsets). But as relation pools grow (N objects, more relation types), exhaustive enumeration becomes intractable. The LLM proposal stage is the scalability strategy.

### Task 4.1 — Create `instructions.py`

Two prompt families:

- `build_bundle_proposal_messages(c1, c2, delta, relation_pool, ...)` → messages for proposal.
- `build_bundle_ranking_messages(...)` → messages for ranking (refactor out of `milestone1_analysis.py`).

### Task 4.2 — Proposal prompt design

The proposal call receives:

- Scene summary for C1 and C2 (object names, coordinates).
- Changed parameters (Δ) with before/after values.
- Full relation pool with IDs, equation text, support vars, and relation type.
- Heuristic guidance (framed as preferences, not laws):
  - Each proposed bundle must include at least one relation mentioning a Δ variable.
  - Prefer geometrically coherent bundles.
  - Prefer smaller bundles unless a larger one is clearly more coherent.
  - Avoid semantically meaningless cross-axis combinations.
  - Propose diverse hypotheses, not near-duplicates.
  - A bundle may include relations that individually don't mention Δ — the bundle as a whole must.

### Task 4.3 — Bounded proposal strategy

- Multiple proposal rounds (2–4 rounds).
- 16–32 bundles per round.
- Global cap at k ≤ 128 unique bundles.
- Each round's prompt can reference prior rounds' proposals to encourage diversity.

### Task 4.4 — Proposal output schema

Structured JSON only:

```json
{
  "bundles": [
    {
      "candidate_id": "LLM-001",
      "relation_ids": [3, 14, 21],
      "rationale": "These relations maintain the horizontal alignment..."
    }
  ]
}
```

The LLM must only reference existing relation IDs. No invented relations.

### Task 4.5 — Parse, dedupe, normalize

Deterministic post-processing:

- Drop bundles referencing invalid relation IDs.
- Sort relation IDs within each bundle.
- Remove duplicate bundles (same sorted relation set).
- Enforce bundle-size cap.
- Merge with symbolically-enumerated bundles from Phase 3 (union, deduped).

### Task 4.6 — Merge with symbolic enumeration

For small pools (≤30 relations, max_system_size ≤ 3), run both:
1. Exhaustive symbolic enumeration with delta-overlap filter.
2. LLM proposal.

Union the results. This way the LLM can propose larger or more creative bundles that enumeration would miss, while enumeration ensures nothing obvious is lost.

For large pools, skip exhaustive enumeration and rely on LLM proposal + deterministic verification.

### Acceptance

- The system produces a bounded set of candidate bundles from both symbolic enumeration and LLM proposals.
- All candidates pass the delta-overlap hard filter.
- The pipeline is generic across different scenes.

---

## Phase 5: Verification, Parameterization, and C3 Materialization

### Goal

The symbolic verifier is the correctness gate. Every bundle that reaches ranking must have a valid RepairProgram and a deterministic C3.

### Task 5.1 — Verify proposed bundles (already implemented)

For each candidate bundle E:

1. Compute V = support(E).
2. Compute C = Δ ∩ V (constants from C2).
3. Enumerate F sets: start with F = C, then try adding 0..`max_extra_fixed` extra vars from V \ C.
4. For each (E, F): compute D = V \ (C ∪ F). Attempt to compile.

**Compilation fails if** (all already implemented in `compile_repair_program`):
- D is empty (nothing to solve).
- System is underdetermined (equations < driven vars).
- Determinant is zero (singular system).
- Any driven variable has all-zero coefficients.
- System is inconsistent (0 = nonzero row after elimination).
- Rank deficit after elimination.

### Task 5.2 — Enumerate parameterizations (already implemented)

`minimal_parameterizations()` finds all viable (D, F′) pairs with the smallest F′. Store all of them — the same E can yield multiple valid programs.

### Task 5.3 — Materialize C3 (already implemented)

For each viable (E, F):
- Compile → RepairProgram.
- Apply to C2 → C3 canvas.
- Record predicted C2 → C3 changes.

### Task 5.4 — Build census

Group all (E, F) pairs by their C3 output:
- Key: sorted tuple of (var, value) for every variable in C3.
- Value: list of ProgramProvenance.

Track how many distinct programs produce each unique C3. Pass this count to the ranking stage.

### Task 5.5 — Handle hyper-param-dependent contradictions

Beyond simple compilation failure, check for:
- Bundles where different valid F choices produce contradictory values for the same variable.
- Flag these in BundleRecord for the LLM ranker (soft signal of instability, not a hard reject).

### Acceptance

- Every bundle sent to ranking has at least one valid RepairProgram and a deterministic C3.
- Census data is available for ranking.

---

## Phase 6: LLM Ranking of Verified Bundles

### Goal

Use the LLM for the hardest part: infer likely human design intent among symbolically valid candidates.

### Task 6.1 — Ranking prompt design

The ranking call receives:

- C1 and C2 scene summaries.
- Changed parameters (Δ) with before/after values.
- Verified bundles only, each with:
  - Equation text.
  - Support vars.
  - Fixed / driven variable partitions.
  - Predicted C2 → C3 changes.
  - Number of extra hyper-parameters needed (fewer = simpler).
  - Census count: how many distinct programs produce this C3.
- Optional user-history context.

### Task 6.2 — Ranking instructions

Ask the model to rank/prune based on:

- Semantic coherence — do the relations form a meaningful geometric story?
- Relevance to the observed edit — does the bundle explain *why* the user made this change?
- Plausibility of predicted C3 — does the outcome look like something a designer would want?
- Minimality — prefer fewer equations and fewer hyper-parameters.
- Robustness — C3 outcomes produced by many programs are more trustworthy.
- Common design intent patterns.

Complexity is a **preference**, not a law. Don't hardcode that the answer must use max or min relations.

### Task 6.3 — Ranking output schema

```json
{
  "summary": "The most likely intent is...",
  "chosen_candidate_id": "B-003",
  "ranked_candidates": [
    {
      "candidate_id": "B-003",
      "bundle_rank": 1,
      "score": 92,
      "keep": true,
      "rationale": "Maintains horizontal alignment with minimal assumptions."
    }
  ]
}
```

Already implemented in `milestone1_ollama.py` (`RankingResult`, `RankedCandidate`). Extend to include census data in the prompt.

### Task 6.4 — Refactor prompt builder

Move `build_llm_ranking_prompt()` from `milestone1_analysis.py` into `instructions.py` alongside the proposal prompt builder. Keep backward compat during transition.

### Acceptance

- The second LLM call ranks only verified candidates.
- Census count is included as a ranking signal.
- Final outputs are explainable: each ranked option has symbolic support, a program, and a materialized C3.

---

## Phase 7: Ollama Integration

### Task 7.1 — Support both string prompts and message lists

`milestone1_ollama.py` currently works with chat API. Ensure `call_ollama_chat()` can accept:
- A raw string prompt (backward compat).
- A full messages list (system/user split for cleaner prompting).

### Task 7.2 — Add proposal mode

Extend Ollama integration for bundle proposal (not just ranking):
- `build_proposal_schema()` — JSON schema for proposal response.
- `call_ollama_proposal()` — wraps `call_ollama_chat()` with proposal schema.
- `parse_proposal_response()` — validate and extract proposed bundles.

### Task 7.3 — Handle `<think>` tags

If the reasoning model leaks `<think>...</think>` into content, strip before JSON parsing. (Already a known issue.)

### Task 7.4 — Keep config knobs

`OllamaRankerConfig` already has model_name, timeout, temperature, keep_alive. Extend if needed for proposal-specific settings (e.g., different temperature for proposal vs. ranking).

### Acceptance

- Ollama integration supports both proposal and ranking flows.
- Structured JSON output enforced for both.

---

## Phase 8: End-to-End Runner

### Task 8.1 — Create `run_pipeline.py`

Single entry point that:

1. Loads/builds C1 and C2.
2. Detects relation pool from C1 (`detect_c1_equations`).
3. Runs deterministic pre-filters (delta-overlap, contradiction).
4. Runs symbolic enumeration for small pools.
5. Calls LLM proposal stage.
6. Merges and dedupes all candidate bundles.
7. Verifies bundles, enumerates parameterizations, materializes C3.
8. Builds census of unique C3 outcomes.
9. Calls LLM ranking stage.
10. Prints final ranked hypotheses with programs and predicted C3s.

### Task 8.2 — Make caps configurable

Expose as CLI args or config dict:
- `max_system_size` (default 3)
- `max_extra_fixed` (default 2)
- `proposal_rounds` (default 2)
- `bundles_per_round` (default 24)
- `global_bundle_cap` (default 128)
- `ranking_top_k` (default 10)
- `linear_only` (default True)
- `include_offsets`, `include_pins` (default True)

### Acceptance

- A reviewer can run one script and see the whole pipeline from C1 → ranked C3 predictions.

---

## Phase 9: Tests

### Task 9.1 — Parser/client tests

Extend `test_milestone1_ollama.py` for:
- Proposal response parsing (valid, invalid relation IDs, duplicates).
- `<think>` tag stripping.
- Message-list support in `call_ollama_chat`.

### Task 9.2 — Verification tests

Test:
- Delta-overlap filter on known bundles.
- Individual-equation contradiction filter.
- Parameterization enumeration: same E, different F sets.
- Compilation failure conditions (underdetermined, singular, inconsistent).
- Census grouping: two programs producing the same C3.

### Task 9.3 — Pipeline smoke tests

- Default 2-object scene (pizza/cutter) end-to-end.
- At least one 3-object scene.

### Task 9.4 — Prompt-builder tests

- Proposal prompts include relation pool and delta.
- Ranking prompts include verified bundles and census counts.
- No hardcoded 2-object assumptions in prompt text.

### Acceptance

- Tests cover real failure points: parsing, verification edge cases, N-object generality.

---

## Explicit Non-Goals

- Exhaustive subset enumeration for large pools (that's what the LLM proposal replaces).
- Sweeping refactors of `milestone1_core.py` (it works).
- Relying on notebooks as primary implementation.
- Hand-coded semantic scoring rules that try to replace the LLM.
- Forcing all relations in a bundle to individually mention Δ (only the bundle as a whole must).
- Using shared-variable or connected-component as hard rejection filters (they are soft signals because hyper-param promotion can make disconnected bundles valid).

---

## Open Design Questions

### 1. Should the LLM propose parameterizations (F choices), or only equation bundles?

Current approach: LLM proposes E, system enumerates F. Alternative: LLM proposes (E, F) pairs directly. The enumeration approach is safer — the LLM doesn't need to understand the linear algebra, just which relations form a coherent story.

**Recommendation**: LLM proposes E only. System enumerates F.

### 2. How to get hyper-parameter values?

Three options:
- **Default to C2 values** (current approach — simplest).
- **Past user behavior** — if user repeatedly pins a variable, bias toward pinning it.
- **LLM decides** — ask the ranking model whether a hyper-param override makes sense.

**Recommendation**: Start with C2 defaults. Add user-history-based overrides as a later extension.

### 3. When to skip exhaustive enumeration?

If relation pool ≤ 30 and max_system_size ≤ 3: run both enumeration and LLM proposal.  
If pool > 30 or max_system_size > 3: skip enumeration, rely on LLM proposal + verification.

Threshold is configurable.

---

## Final Submission Story

> The system uses deterministic heuristics to pre-filter the relation space, an LLM to propose plausible relation bundles, symbolic code to verify and compile them into executable programs with explicit hyper-parameter partitioning, and a second LLM pass to rank the verified candidates by likely designer intent — with each candidate backed by a materialized C3 prediction.
