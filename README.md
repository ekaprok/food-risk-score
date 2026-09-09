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

## `food_score.py`

```bash
python3 food_score.py
```

It writes `food_score.csv`, one row per country, commodity and year,
plus an AVERAGES row and a blank separator line after each block.

### What the columns mean

| column | in plain terms | in the AVERAGES row |
|---|---|---|
| `country` | The country the row is about. | - |
| `commodity` | The crop the row is about: `Rice` or `Wheat`. | - |
| `year` | The year the row covers, or `AVERAGES` for the summary row that closes each block. | - |
| `production (P)` | How much the country grew itself that year, in tonnes. | - |
| `imports (I)` | How much it bought from abroad, in tonnes. | - |
| `exports (E)` | How much it sold abroad, in tonnes. | - |
| `supply (P + I - E)` | - | - |
| `ssr (P / Supply)` | Self-sufficiency ratio: the share of what the country used that it grew itself. 1.0 means it fed itself entirely; 0 means none of it was home-grown. | The plain average of the yearly values over `YEARS`. |
| `idr (I / Supply)` | Import dependency ratio: the share of what the country used that came from abroad. | The plain average of the yearly values over `YEARS`. |
| `w_internal (P / (P + I))` | How much of the score to put on home-grown supply. It is the share of everything coming in — grown plus imported. | The plain average of the yearly values over `YEARS`. |
| `w_external (I / (P + I))` | The same for imports, so the two weights always add up to 1 no matter how much the country exports. | The plain average of the yearly values over `YEARS`. |
| `risk_internal (std(P) / mean(P))` | How unsteady the harvest has been over the last `RISK_WINDOW` years. 0 means production never moves; the bigger the number, the more it swings from year to year. | Not averaged: the last scored year's value is carried down as-is, because it already looks back over the trailing `RISK_WINDOW` years. |
| `risk_external (sum(si^2))` | How concentrated the country's suppliers are, as a Herfindahl-Hirschman index. Near 0 means imports are spread over many countries; 1.0 means a single supplier provides everything, so losing it would cut the whole flow. | The plain average of the yearly values over `YEARS`. |
| `criticality (Kcal_commodity / Kcal_total)` | How much the national diet relies on the crop. | The plain average of the yearly values over `YEARS`. |
| `vulnerability (<formula>)` | If `USE_PROPORTIONAL_WEIGHTS` is set to True, `W_internal x Risk_internal + W_external x Risk_external`. Otherwise, `SSR x Risk_internal + IDR x Risk_external`. | Recalculated from the averaged figures in this row, not averaged from the yearly vulnerabilities. |
| `food_risk (Vulnerability x Criticality)` | The headline figure: vulnerability scaled by how much the diet depends on the crop. | Recalculated too, as this row's vulnerability times this row's criticality. |

Blank cells in the AVERAGES row are deliberate: tonnages are not averaged, only
the ratios are.

### Which dataset feeds which term

| term | source file | element |
|---|---|---|
| P | `Production_WheatRice` | Production |
| I, E | `ImportAndExport_...` | Import/Export quantity |
| Risk_internal | `Production_WheatRice` | Production |
| C_kcal | `Calories_TotalAndWheat` | Food supply (kcal/cap/d) |
| Risk_external | `Trade_ReporterAll_...` | Export quantity |

A year missing from a source file throws an error, unless `MISSING_YEAR_POLICY`
gives that series another reading.

### Parameters

The following parameters can be configured:

- `DATA_DIR`: path to the FAOSTAT CSVs (default: `"faostat"`)
- `OUT_PATH`: where the output CSV is written (default:
  `"food_score.csv"`)
- `COUNTRIES`: the countries to score.
- `COMMODITIES`: the commodities to score. Each entry
  maps a display name to the spelling used in each source: `cpc` (`wheat`) for the
  production and trade datasets, `fbs` (`wheat and products`) for the food balance dataset.
- `YEARS`: the first and last year to score, inclusive.
- `USE_PROPORTIONAL_WEIGHTS`: which weights the vulnerability score uses
  (default: `True`). `True` gives
  `W_internal x Risk_internal + W_external x Risk_external`, splitting the
  score over the inflows (`P + I`), so the weights sum to 1
  whatever it exports. `False` gives the original
  `SSR x Risk_internal + IDR x Risk_external`, taken over apparent supply.
- `RISK_WINDOW`: how many years the internal risk (coefficient of variation of
  production) looks back over, including the current year (default: 5).
- `MISSING_YEAR_POLICY`: how to read a year absent from a source series, one
  setting per series (`production`, `imports`, `exports`, `calories` and
  `suppliers`). `None` raises an error, `FILL_ZERO` reads it as zero, and
  `CARRY_FORWARD` repeats the previous year. A year with no supplier rows in
  the trade matrix therefore scores an external risk of 0.

## Tests

```bash
python3 -m unittest test_food_score test_load_and_audit
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
