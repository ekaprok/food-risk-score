# Food Security Risk Score

## Setup

You need Python 3.9 or newer and pandas.

**1. Check whether you already have Python.** In a terminal (Terminal on macOS,
PowerShell on Windows), run:

```bash
python3 --version
```

If that prints `Python 3.9.x` or higher, skip to step 3. On Windows, try
`python --version` if `python3` isn't recognised.

**2. Install Python.** Download the installer for your system from
<https://www.python.org/downloads/> and run it.

**3. Install pandas.**

```bash
python3 -m pip install pandas
```

## `load_and_audit.py`

```bash
python3 load_and_audit.py
```

Read-only audit of the five FAOSTAT CSVs.

## `internal_risk_score.py`

```bash
python3 internal_risk_score.py
```

Calculates SSR, IDR, internal risk, external supplier-concentration
risk (the HHI of the import suppliers' shares) and the vulnerability score
that combines them (`SSR x Risk_internal + IDR x Risk_external`). The
following parameters can be configured:

- `DATA_DIR`: path to the FAOSTAT CSVs (default: `"faostat"`)
- `OUT_PATH`: where the output CSV is written (default:
  `"internal_risk_score.csv"`)
- `COUNTRIES`: the countries to score.
- `COMMODITIES`: the commodities to score (default: rice and wheat). Each entry
  maps a display name to the spelling used in each source: `cpc` (`wheat`) for the
  production and trade datasets, `fbs` (`wheat and products`) for the food balance dataset.
- `YEARS`: the first and last year to score, inclusive (default: 2020-2024).
- `RISK_WINDOW`: how many years the internal risk (coefficient of variation of
  production) looks back over, including the current year (default: 5).
- `MISSING_YEAR_POLICY`: how to read a year absent from a source series, one
  setting per series (`production`, `imports`, `exports`, `calories` and
  `suppliers`). `None` raises an error, `FILL_ZERO` reads it as zero, and
  `CARRY_FORWARD` repeats the previous year. A year with no supplier rows in
  the trade matrix therefore scores an external risk of 0.

## Tests

```bash
python3 -m unittest test_internal_risk_score test_load_and_audit
```

## Editor setup (optional)

The scripts only need pandas, but a type checker reading them needs pandas'
type stubs, or it flags every pandas call as an unknown type. A local
environment with the stubs installed keeps the editor quiet:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pandas pandas-stubs pytest
```

`pyrightconfig.json` points the type checker at `.venv` and checks against
Python 3.9, the version this project supports. `ruff.toml` does the same for
the linter. Neither file affects how the scripts run; `.venv` is not committed.
