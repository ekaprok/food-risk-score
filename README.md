# Food Security Risk Score

Website: https://ekaprok.github.io/food-risk-score/

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
| `vulnerability_weighted (W_internal x Risk_internal + W_external x Risk_external)` | How exposed the country is, splitting the score over the inflows (`P + I`), so the weights sum to 1 whatever it exports. | Recalculated from the averaged figures in this row, not averaged from the yearly vulnerabilities. |
| `food_risk_weighted (Vulnerability_weighted x Criticality)` | The headline figure: vulnerability scaled by how much the diet depends on the crop. | Recalculated too, as this row's vulnerability times this row's criticality. |
| `top_supplier (biggest supplier by tracked volume)` | The country that shipped the most of the crop that year, as recorded in the trade matrix. | - |
| `top_supplier_share (s_max = Volume_max / Volume_trade)` | How much of the year's tracked flows that one supplier accounted for. 0.4 means it shipped 40% of everything the matrix records for that year. | The plain average of the yearly values over `YEARS`. |
| `shock_exposure (W_external x s_max x Shock_magnitude)` | How much of everything coming in (`P + I`) stops arriving if the biggest supplier does. At the default `SHOCK_MAGNITUDE` of 1 the supplier disappears outright. | Recalculated from this row's averaged `w_external` and `s_max`, not averaged from the yearly values. |
| `vulnerability_weighted_sim (V + (1 - V) x Shock_exposure)` | `vulnerability_weighted` topped up by the shock: the exposure takes that fraction of whatever headroom the country had left. A country already fully exposed cannot get worse; one at 0 rises to the exposure itself. Never reads below the standing vulnerability, and never above 1. | Recalculated from this row's `vulnerability_weighted` and `shock_exposure`. |
| `food_risk_weighted_sim (Vulnerability_weighted_sim x Criticality)` | The headline figure with the biggest supplier gone: the simulated vulnerability scaled by how much the diet depends on the crop. Read against `food_risk_weighted` to see what the shock costs. | Recalculated as this row's simulated vulnerability times this row's criticality. |

Two more columns, `vulnerability_ssr_idr` and `food_risk_ssr_idr`, are
commented out in `food_score.py` and so no longer reach the CSV. They weighted
the same two risks over apparent supply (`SSR`/`IDR`) rather than over the
inflows, and were reported beside the `_weighted` pair so the choice of
weighting stayed visible. Uncomment them there to bring both back.

### Which dataset feeds which term

| term | source file | element |
|---|---|---|
| P | `Production_WheatRice` | Production |
| I, E | `ImportAndExport_...` | Import/Export quantity |
| Risk_internal | `Production_WheatRice` | Production |
| C_kcal | `Calories_TotalAndWheat` | Food supply (kcal/cap/d) |
| Risk_external, s_max | `Trade_ReporterAll_...` | Export quantity |
| Shock_exposure | both trade files | see the caveat below |

A year missing from a source file throws an error, unless `MISSING_YEAR_POLICY`
gives that series another reading.

`shock_exposure` is the one term that crosses the two trade files: `s_max` is
the supplier's share of the flows the **trade matrix** tracks, while the
imports behind `w_external` come from the **aggregate** trade file. The two
totals do not have to agree, so the column takes the matrix's supplier mix as
representative of the whole of `I`. It uses `w_external` rather than `idr`
because `I / (P + I)` cannot exceed 1, whereas `I / (P + I - E)` can for a
country that re-exports more than it uses — Qatar and Singapore both do.

`vulnerability_weighted_sim` combines the standing vulnerability and the shock the
way independent risks combine: `V + (1 - V) x E` is the same as
`1 - (1 - V)(1 - E)`, so the country comes through unharmed only if both miss
it. That keeps the result inside 0-1 and monotonic — the shock can only push
the score up, never down.

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
- `RISK_WINDOW`: how many years the internal risk (coefficient of variation of
  production) looks back over, including the current year (default: 5).
- `SHOCK_MAGNITUDE`: how much of the top supplier's flow `shock_exposure`
  assumes is lost, as a fraction (default: `1.0`, the supplier disappearing
  outright).
- `MISSING_YEAR_POLICY`: how to read a year absent from a source series, one
  setting per series (`production`, `imports`, `exports`, `calories` and
  `suppliers`). `None` raises an error, `FILL_ZERO` reads it as zero, and
  `CARRY_FORWARD` repeats the previous year. A year with no supplier rows in
  the trade matrix therefore scores an external risk of 0.
- `FALLBACK_WHEAT_CRITICALITY`: stand-in `criticality` values for countries the
  calories dataset does not cover, as country name -> the crop's share of the
  national calorie supply.

## The web page

`index.html`, `style.css` and `app.js` are a static page that reads
`food_score.csv` in the browser. There is nothing to build and no server code:
GitHub Pages can serve the repository root as it stands (Settings -> Pages ->
Deploy from a branch -> `main` / `/ (root)`).

To look at it locally, serve the folder — opening the file directly will not
work, because the browser refuses to read the CSVs off `file://`:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

Pick a country and a commodity at the top and the whole page follows. The left
column is the country as it is: the latest year's `food_risk_weighted`, the
five-year trend, the five-year average, and a breakdown of who actually shipped
the crop that year, summed live from `faostat/TradeMatrix.csv`. The right column
is the same five years with the biggest supplier removed, from
`food_risk_weighted_sim`. Both trends share one y-axis, so the two columns can
be read against each other. Every score shown is banded: green to 0.10, yellow
to 0.20, red above it.

Re-run `food_score.py` and the page picks up the new CSV on the next reload.

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
