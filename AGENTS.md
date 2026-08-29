# AGENTS.md

Instructions for AI agents working on this repo.

## Project

Python scripts that transform FAOSTAT CSV data (in `faostat/`) for food-security
risk analysis. Scripts run from the project root and are also copy-pasted into
Google Colab, so Colab compatibility is a hard requirement.

## Colab compatibility (required)

- Target Python 3.10 and the pandas version bundled with Colab (2.x). No walrus-free
  legacy constraints needed, but avoid features newer than 3.10.
- Use only the standard library + pandas/numpy (already in Colab). If another
  package is unavoidable, note the `!pip install` line in a comment at the top.
- Each script must be self-contained: all imports and constants at the top,
  no imports from sibling modules.
- Keep file paths in a single `DATA_DIR` constant at the top (currently
  `"faostat"`) so it's easy to repoint to `/content/...` or Drive in Colab.
- Structure code as small functions plus a `main()` guarded by
  `if __name__ == "__main__":` — this works both as a script and pasted into a cell.
- Output via `print()` and returned DataFrames; no argparse, `input()`, or
  environment-variable configuration.

## Python style

- Follow PEP 8; snake_case names; type hints on function signatures.
- Small, single-purpose functions with short docstrings (what it returns, not how).
- No dead code, no commented-out code, no needless classes.
- Fail loudly: validate assumptions (expected columns, non-empty frames) and
  raise `ValueError` with a clear message rather than silently continuing.

## Pandas best practices

- Never mutate via chained indexing (`df[a][b] = ...`); use `.loc`/`.assign`.
- Prefer method chaining for transforms; avoid `inplace=True`.
- Use vectorized operations; no `iterrows()`/`apply` where a vectorized
  equivalent exists.
- Read CSVs with explicit `usecols=` and `dtype=` where practical; parse
  numerics deliberately (`pd.to_numeric(..., errors="coerce")` only when NaN
  is the intended outcome, and say so).
- Use `merge(..., how=..., validate=...)` and check row counts after joins.
- Handle missing data explicitly — document why values are dropped or filled.
- Don't rely on index side effects; `reset_index()` after groupby when the
  result is used as a plain table.

## Workflow

- Run scripts from the project root: `python3 <script>.py`.
- Tests use the standard `unittest`/`pytest` layout in `test_*.py`; run
  `python3 -m pytest` after changes and keep tests passing.
- Update `README.md` when adding a script or changing how one is run.
