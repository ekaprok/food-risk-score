"""SSR, IDR and internal supply risk.

Builds the internal supply metrics from the FAOSTAT CSVs`:

    Supply        = P + I - E         apparent domestic supply, per year
    SSR           = P / Supply        self-sufficiency ratio, per year
    IDR           = I / Supply        import dependency ratio, per year
    Risk_internal = std(P) / mean(P)  coefficient of variation of production
                                      over the trailing RISK_WINDOW years,
                                      one figure per year

Which dataset feeds which term:

    | term          | source file             | element                |
    |---------------|-------------------------|------------------------|
    | P             | `Production_Wheat`      | Production             |
    | I, E          | `ImportAndExport_Wheat` | Import/Export quantity |
    | Risk_internal | `Production_Wheat`      | Production             |

A year missing from a source file throws an error, unless MISSING_YEAR_POLICY
gives that series another reading.
"""

from typing import Optional

import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "production": f"{DATA_DIR}/AfgThai_Production_WheatRice.csv",
    "trade":      f"{DATA_DIR}/AfgThai_ImportAndExport_WheatRice.csv",
}

COUNTRIES = ("Afghanistan", "Thailand")
COMMODITIES = ("Rice", "Wheat")

# The years the scores cover, inclusive on both ends.
YEARS = (2020, 2024)
# Sliding window for internal risk calculation.
RISK_WINDOW = 5

# What an absent year means for each series.
FILL_ZERO = "fill_zero"
MISSING_YEAR_POLICY = {
    "production": None,
    "imports":    FILL_ZERO,
    "exports":    FILL_ZERO,
}

ROUND_DECIMALS = 2
OUT_PATH = "internal_risk_score.csv"


def load(path: str) -> pd.DataFrame:
    """The CSV as an all-string frame with surrounding whitespace stripped."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.apply(lambda col: col.str.strip())


def rows_for_pair(df: pd.DataFrame, country: str, commodity: str) -> pd.DataFrame:
    """The rows of a FAOSTAT file covering one country and one commodity."""
    missing_columns = {"Area", "Item"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"expected FAOSTAT column(s) missing: {sorted(missing_columns)}")

    selected = df[(df["Area"] == country) & (df["Item"] == commodity)]
    if selected.empty:
        raise ValueError(f"no rows for {country} / {commodity}")
    return selected


def validate_and_get_series(df: pd.DataFrame, dataset: str, element: str,
                            years: tuple[int, int],
                            policy: Optional[str] = None) -> pd.Series:
    """A Series of yearly values for the given element, covering the whole of
    `years`. Throws an error if there are gaps, unless `policy` says how to
    read them."""
    missing_columns = {"Element", "Year", "Value"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"expected FAOSTAT column(s) missing: {sorted(missing_columns)}")

    rows = df[df["Element"] == element]
    year = pd.to_numeric(rows["Year"], errors="coerce")
    unreadable_years = rows.loc[year.isna(), "Year"].tolist()
    if unreadable_years:
        raise ValueError(f"{element}: unreadable year(s) {unreadable_years}")

    index = year.astype(int)
    repeated = sorted(int(y) for y in index[index.duplicated()].unique())
    if repeated:
        raise ValueError(f"{element}: repeated year(s) {repeated}; "
                         "expected one row per year")

    series = pd.Series(rows["Value"].values, index=index.values,
                       name=element).sort_index()

    start, end = years
    missing_rows = [y for y in range(start, end + 1) if y not in series.index]
    if missing_rows and policy != FILL_ZERO:
        raise ValueError(f"{element}: no row for year(s) {missing_rows}; every year "
                         f"in {start}-{end} must be present")
    if missing_rows:
        # Reported rather than tallied, so no filled figure is silent.
        print(f"  filled {dataset} / {element} with zero for "
              f"{', '.join(str(year) for year in missing_rows)}")
        series = pd.concat([series, pd.Series("0", index=missing_rows)]).sort_index()
        series = series.rename(element)

    span = pd.to_numeric(series.loc[start:end], errors="coerce")
    unreadable_values = span.index[span.isna()].tolist()
    if unreadable_values:
        raise ValueError(f"{element}: unreadable value(s) for year(s) "
                         f"{unreadable_values}")
    return span


def build_supply_balance(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Year-indexed production, imports, exports and apparent supply."""
    production_years = (YEARS[0] - RISK_WINDOW + 1, YEARS[1])
    supply_balance = pd.DataFrame({
        "production": validate_and_get_series(
            data["production"], "production", "Production", production_years,
            MISSING_YEAR_POLICY["production"]),
        "imports": validate_and_get_series(
            data["trade"], "trade", "Import quantity", YEARS,
            MISSING_YEAR_POLICY["imports"]),
        "exports": validate_and_get_series(
            data["trade"], "trade", "Export quantity", YEARS,
            MISSING_YEAR_POLICY["exports"]),
    })
    supply_balance["supply"] = (supply_balance["production"]
                                + supply_balance["imports"]
                                - supply_balance["exports"])
    supply_balance.index.name = "year"
    return supply_balance


def ssr_idr_scores(supply_balance: pd.DataFrame) -> pd.DataFrame:
    """The supply balance over YEARS, plus the SSR and IDR columns. The earlier
    years exist only to feed the internal-risk window, so they are dropped here
    rather than scored."""
    start, end = YEARS
    scores = supply_balance.loc[start:end].copy()

    # Supply is the denominator of both ratios, so a non-positive one would
    # give a negative SSR rather than a wrong-but-plausible number.
    bad_supply = scores.index[scores["supply"] <= 0].tolist()
    if bad_supply:
        raise ValueError(f"non-positive apparent supply in {bad_supply}: "
                         "exports exceed production plus imports")

    scores["ssr"] = scores["production"] / scores["supply"]
    scores["idr"] = scores["imports"] / scores["supply"]
    return scores


def internal_risk(supply_balance: pd.DataFrame) -> pd.Series:
    """The coefficient of variation of production over the trailing RISK_WINDOW
    years, one figure for each year in YEARS: how much the harvest moved about
    in the five years up to and including that year."""
    window = supply_balance["production"].rolling(RISK_WINDOW)
    coefficient_of_variation = window.std(ddof=1) / window.mean()
    return coefficient_of_variation.loc[YEARS[0]:YEARS[1]].rename("risk_internal")


COLUMN_FORMATS = [
    ("production",    "Production (t)", "{:>16,.2f}"),
    ("imports",       "Imports (t)",    "{:>14,.2f}"),
    ("exports",       "Exports (t)",    "{:>12,.2f}"),
    ("supply",        "Supply (t)",     "{:>16,.2f}"),
    ("ssr",           "SSR",            "{:>6.2f}"),
    ("idr",           "IDR",            "{:>6.2f}"),
    ("risk_internal", "Risk_internal",  "{:>14.2f}"),
]


def render(df: pd.DataFrame) -> str:
    """Fixed-width table. Column width comes from the format string itself, so
    the header and the numbers under it cannot drift apart."""
    widths = [len(fmt.format(0)) for _, _, fmt in COLUMN_FORMATS]
    header = "  ".join(
        ["Year"] + [title.rjust(w) for (_, title, _), w in zip(COLUMN_FORMATS, widths)])
    lines = [header, "-" * len(header)]
    for year, row in df.iterrows():
        lines.append("  ".join(
            [str(year)] + [fmt.format(row[key]) for key, _, fmt in COLUMN_FORMATS]))
    return "\n".join(lines)


def main() -> pd.DataFrame:
    data = {name: load(path) for name, path in DATASETS.items()}

    # Sorted here rather than after the fact, so the printed tables and the
    # written rows come out in the same order however the constants are listed.
    tables = []
    for country in sorted(COUNTRIES):
        for commodity in sorted(COMMODITIES):
            print("=" * 96)
            print(f"{country.upper()} / {commodity.upper()}  ({YEARS[0]}-{YEARS[1]})")
            print(f"Risk_internal: CV of production over a trailing "
                  f"{RISK_WINDOW}-year window")

            # Built under the heading so that the fills it reports are read
            # against the pair they belong to.
            pair = {name: rows_for_pair(df, country, commodity)
                    for name, df in data.items()}
            supply_balance = build_supply_balance(pair)
            scores = ssr_idr_scores(supply_balance)
            scores["risk_internal"] = internal_risk(supply_balance)
            tables.append(scores.assign(country=country, commodity=commodity))

            print(render(scores))
            print()

    table = pd.concat(tables).reset_index()
    table = table[["country", "commodity", "year"]
                  + [key for key, _, _ in COLUMN_FORMATS]]
    table.round(ROUND_DECIMALS).to_csv(OUT_PATH, index=False)
    print("=" * 96)
    print(f"Wrote {len(table)} rows to {OUT_PATH}")
    return table


if __name__ == "__main__":
    main()
