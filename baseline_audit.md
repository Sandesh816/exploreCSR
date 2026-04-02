# Phase 1 Baseline Audit

This note records what `prototype-v6.ipynb` already contains and where that logic currently lives in the Python modules.

## Scope

This is a documentation-only checkpoint for Phase 1 Task 1.1:

- audit `prototype-v6.ipynb`,
- confirm active logic is available from Python modules,
- identify concrete gaps before changing behavior.

## What `prototype-v6.ipynb` Contains

The notebook has 8 cells.

- Cell 0 contains the main implementation block:
  - geometry/data structures,
  - symbolic expressions,
  - relation generation and deduplication,
  - repair-program compilation,
  - enumeration of repair programs and `C3` outcomes.
- Cells 1-6 are interactive exploration cells that call the implementation:
  - pick relation IDs,
  - enumerate candidate programs,
  - inspect `RepairProgram` text,
  - inspect census/provenance outputs.
- Cell 7 is empty.

In practice, `prototype-v6.ipynb` is a single large implementation cell plus a small set of exploratory runner cells.

## Mapping from `prototype-v6.ipynb` to Modules

### Logic already present in `milestone1_core.py`

The core notebook implementation has already been lifted into `milestone1_core.py`, including:

- `Rect`, `NamedRect`
- `LinExpr`, `PolyExpr`, `Equation`
- relation generation / filtering / deduplication
- `detect_c1_equations(...)`
- `changed_params(...)`
- `AffineInM`
- `RepairProgram`
- `ProgramProvenance`
- `compile_repair_program(...)`
- `enumerate_repair_programs_and_c3s(...)`

This means the primary symbolic system is no longer notebook-only.

### Logic added after `prototype-v6.ipynb`

The following functionality is not part of the original v6 notebook and now lives in later modules:

- bundle-level verification helpers in `milestone1_analysis.py`
- `BundleRecord` / `ParameterizationRecord`
- shortlist analysis and reporting
- LLM ranking prompt construction
- Ollama client, schema enforcement, and ranking response parsing in `milestone1_ollama.py`

These are post-v6 additions and represent the path toward the final pipeline.

## Concrete Gap Found During Audit

`milestone1_core.py` still references `_num_vars_in_linexpr(...)` from `is_simple_pin(...)`, but that helper is not defined in the module.

Current status:

- the reference exists in the module path,
- the helper is missing,
- this should be fixed in the next Phase 1 step.

This matches the earlier observation that later notebook work had to patch around the missing helper locally.

## Conclusion

Phase 1 Task 1.1 is mostly satisfied:

- the active symbolic implementation from `prototype-v6.ipynb` is already available in Python modules,
- the notebook is mostly acting as a reference/demo artifact,
- there is one concrete inherited gap to fix before treating the module path as fully stable.

## Recommended Next Step

Phase 1 Task 1.2:

- implement the missing `_num_vars_in_linexpr(...)` helper or replace the call in `is_simple_pin(...)` with the simplest equivalent support-size check.
