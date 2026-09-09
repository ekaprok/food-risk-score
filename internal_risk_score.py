"""SSR, IDR and internal supply risk.

Builds the internal supply metrics from the FAOSTAT CSVs`:

    Supply        = P + I - E         apparent domestic supply, per year
    SSR           = P / Supply        self-sufficiency ratio, per year
    IDR           = I / Supply        import dependency ratio, per year
    Risk_internal = std(P) / mean(P)  coefficient of variation of production
                                      over the trailing RISK_WINDOW years,
                                      one figure per year
    C_kcal        = Kcal_commodity / Kcal_total
                                      commodity criticality, per year: how
                                      essential the commodity is to the
                                      national diet, as its share of the
                                      calorie supply
    Risk_external = sum(si^2)         supplier-concentration risk, per year:
                                      the Herfindahl-Hirschman index of the
                                      import suppliers' shares
    V             = SSR * Risk_internal + IDR * Risk_external
                                      vulnerability, per year.
                                      Built from the terms above, not from a
                                      source file of its own.

Which dataset feeds which term:

    | term          | source file                | element                 |
    |---------------|----------------------------|-------------------------|
    | P             | `Production_WheatRice`     | Production              |
    | I, E          | `ImportAndExport_...`      | Import/Export quantity  |
    | Risk_internal | `Production_WheatRice`     | Production              |
    | C_kcal        | `Calories_TotalAndWheat`   | Food supply (kcal/cap/d)|
    | Risk_external | `Trade_ReporterAll_...`    | Export quantity         |

A year missing from a source file throws an error, unless MISSING_YEAR_POLICY
gives that series another reading.
"""

from __future__ import annotations

import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "production": f"{DATA_DIR}/AfgThai_Production_WheatRice.csv",
    "trade":      f"{DATA_DIR}/AfgThai_ImportAndExport_WheatRice.csv",
    "calories":   f"{DATA_DIR}/AfgThai_Calories_TotalRiceWheat.csv",
    "trade_matrix_mirror": f"{DATA_DIR}/AfgThai_Trade_ReporterAll_WheatRice.csv",
}

COUNTRIES = ("Afghanistan", "Thailand")
# What each commodity is called in the files it appears in: the crop and trade
# files go by the CPC name, the food balance sheets by FBS.
COMMODITIES = {
    "Rice":  {"cpc": "Rice",  "fbs": "Rice and products"},
    "Wheat": {"cpc": "Wheat", "fbs": "Wheat and products"},
}

# The years the scores cover, inclusive on both ends.
YEARS = (2020, 2024)
# Sliding window for internal risk calculation.
RISK_WINDOW = 5

# What an absent year means for each series.
FILL_ZERO = "fill_zero"
CARRY_FORWARD = "carry_forward"
MISSING_YEAR_POLICY = {
    "production": None,
    "imports":    FILL_ZERO,
    "exports":    FILL_ZERO,
    "calories":   CARRY_FORWARD,
    "suppliers":  FILL_ZERO,
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


def fill_missing_years(series: pd.Series, dataset: str, element: str,
                       years: tuple[int, int], policy: str | None,
                       zero: str | float) -> pd.Series:
    """`series` with every year of the span present, the absent ones read the
    way `policy` says. Throws an error if it says nothing. `zero` is what
    FILL_ZERO writes: a string into a raw string series, a number into a
    computed one."""
    start, end = years
    missing_rows = [y for y in range(start, end + 1) if y not in series.index]
    if not missing_rows:
        return series
    if policy is None:
        raise ValueError(f"{element}: no row for year(s) {missing_rows}; every year "
                         f"in {start}-{end} must be present")

    reading = "zero" if policy == FILL_ZERO else "the previous year"
    print(f"  filled {dataset} / {element} with {reading} for "
          f"{', '.join(str(year) for year in missing_rows)}")
    if policy == FILL_ZERO:
        filled = pd.concat([series, pd.Series(zero, index=missing_rows)])
    else:
        filled = series.reindex(series.index.union(missing_rows)).ffill()
    filled = filled.sort_index()
    # Concat and reindex both drop the name; put back the one the caller set.
    filled.name = series.name
    return filled


def validate_and_get_series(df: pd.DataFrame, dataset: str, element: str,
                            years: tuple[int, int],
                            policy: str | None = None) -> pd.Series:
    """A Series of yearly values for the given element, covering the whole of
    `years`. Throws an error if there are gaps, unless `policy` says how to
    handle them."""
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

    series = fill_missing_years(series, dataset, element, years, policy, "0")

    start, end = years
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


def external_risk(trade_matrix: pd.DataFrame, country: str,
                  commodity: str) -> pd.Series:
    """How concentrated the country's import suppliers are, per year over
    YEARS: the Herfindahl-Hirschman index of each supplier's share of the
    flows recorded for that year. 1/n when n suppliers ship equal shares,
    1.0 when a single supplier ships everything."""
    supplier, importer = "Reporter Countries", "Partner Countries"
    element, item = "Export quantity", COMMODITIES[commodity]["cpc"]

    missing_columns = ({supplier, importer, "Element", "Item", "Year", "Value"}
                       - set(trade_matrix.columns))
    if missing_columns:
        raise ValueError(f"expected FAOSTAT column(s) missing: {sorted(missing_columns)}")

    rows = trade_matrix[(trade_matrix[importer] == country)
                        & (trade_matrix["Item"] == item)
                        & (trade_matrix["Element"] == element)]
    if rows.empty:
        raise ValueError(f"no supplier rows for {country} / {item}")

    year = pd.to_numeric(rows["Year"], errors="coerce")
    unreadable_years = rows.loc[year.isna(), "Year"].tolist()
    if unreadable_years:
        raise ValueError(f"{element}: unreadable year(s) {unreadable_years}")

    flow = pd.to_numeric(rows["Value"], errors="coerce")
    unreadable_values = rows.loc[flow.isna(), "Value"].tolist()
    if unreadable_values:
        raise ValueError(f"{element}: unreadable value(s) {unreadable_values}")

    # Summed per supplier first: a country listed twice in a year (a re-export,
    # a revision) is one supplier, not two smaller and more diversified ones.
    by_supplier = (pd.DataFrame({"year": year.astype(int),
                                 "supplier": rows[supplier],
                                 "flow": flow})
                   .groupby(["year", "supplier"])["flow"].sum())

    # A year whose recorded flows add up to nothing has no shares to divide
    # out; drop it and let MISSING_YEAR_POLICY say what an absent year reads as.
    yearly_total = by_supplier.groupby(level="year").transform("sum")
    recorded = yearly_total > 0
    share = by_supplier[recorded] / yearly_total[recorded]
    hhi = (share ** 2).groupby(level="year").sum().rename("risk_external")

    hhi = fill_missing_years(hhi, "trade_matrix_mirror", element, YEARS,
                             MISSING_YEAR_POLICY["suppliers"], 0.0)
    return hhi.loc[YEARS[0]:YEARS[1]]


def commodity_criticality(calories: pd.DataFrame, country: str,
                          commodity: str) -> pd.Series:
    """How essential the commodity is to the nation's diet, per year over
    YEARS."""
    element = "Food supply (kcal/capita/day)"
    item = COMMODITIES[commodity]["fbs"]

    def calories_from(of_item: str) -> pd.Series:
        return validate_and_get_series(
            rows_for_pair(calories, country, of_item), f"calories ({of_item})",
            element, YEARS, MISSING_YEAR_POLICY["calories"])

    total = calories_from("Grand Total")
    bad_total = total.index[total <= 0].tolist()
    if bad_total:
        raise ValueError(f"{element}: non-positive total supply in {bad_total}")
    return (calories_from(item) / total).rename("criticality")


def vulnerability_score(scores: pd.DataFrame) -> pd.Series:
    """How exposed the country is on this commodity, per year over YEARS: what
    it grows for itself weighted by how unsteady its harvests are, plus what it
    imports weighted by how concentrated its suppliers are. Every term arrives
    validated from the function that built it, so this only combines them."""
    return (scores["ssr"] * scores["risk_internal"]
            + scores["idr"] * scores["risk_external"]).rename("vulnerability")


COLUMN_FORMATS = [
    ("production",    "Production (t)", "{:>16,.2f}"),
    ("imports",       "Imports (t)",    "{:>14,.2f}"),
    ("exports",       "Exports (t)",    "{:>12,.2f}"),
    ("supply",        "Supply (t)",     "{:>16,.2f}"),
    ("ssr",           "SSR",            "{:>6.2f}"),
    ("idr",           "IDR",            "{:>6.2f}"),
    ("risk_internal", "Risk_internal",  "{:>14.2f}"),
    ("risk_external", "Risk_external",  "{:>14.2f}"),
    ("criticality",   "Criticality",    "{:>12.2f}"),
    ("vulnerability", "Vulnerability",  "{:>14.2f}"),
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

    tables = []
    for country in sorted(COUNTRIES):
        for commodity in sorted(COMMODITIES):
            print("=" * 132)
            print(f"{country.upper()} / {commodity.upper()}  ({YEARS[0]}-{YEARS[1]})")
            print(f"Risk_internal: CV of production over a trailing "
                  f"{RISK_WINDOW}-year window")
            print("Risk_external: HHI of import-supplier concentration "
                  "(1/n spread out, 1.0 a single supplier)")
            print("Criticality:   the commodity's share of the national "
                  "calorie supply")
            print("Vulnerability: SSR x Risk_internal + IDR x Risk_external")

            pair = {name: rows_for_pair(data[name], country,
                                        COMMODITIES[commodity]["cpc"])
                    for name in ("production", "trade")}
            supply_balance = build_supply_balance(pair)
            scores = ssr_idr_scores(supply_balance)
            scores["risk_internal"] = internal_risk(supply_balance)
            scores["risk_external"] = external_risk(
                data["trade_matrix_mirror"], country, commodity)
            scores["criticality"] = commodity_criticality(
                data["calories"], country, commodity)
            scores["vulnerability"] = vulnerability_score(scores)
            tables.append(scores.assign(country=country, commodity=commodity))

            print(render(scores))
            print()

    table = pd.concat(tables).reset_index()
    table = table[["country", "commodity", "year"]
                  + [key for key, _, _ in COLUMN_FORMATS]]
    table.round(ROUND_DECIMALS).to_csv(OUT_PATH, index=False)
    print("=" * 132)
    print(f"Wrote {len(table)} rows to {OUT_PATH}")
    return table


if __name__ == "__main__":
    main()
