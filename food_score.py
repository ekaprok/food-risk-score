"""SSR, IDR and internal supply risk.

Builds the internal supply metrics from the FAOSTAT CSVs. README.md
"`food_score.py`" explains every term and output column.
"""

from __future__ import annotations

import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "production": f"{DATA_DIR}/Production.csv",
    "trade":      f"{DATA_DIR}/ImportExport.csv",
    "calories":   f"{DATA_DIR}/Calories.csv",
    "trade_matrix_mirror": f"{DATA_DIR}/TradeMatrix.csv",
}

COUNTRIES = ("Afghanistan", "Yemen")
# What each commodity is called in the files it appears in: the crop and trade
# files go by the CPC name, the food balance sheets by FBS.
COMMODITIES = {
    # "Rice":  {"cpc": "Rice",  "fbs": "Rice and products"},
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
OUT_PATH = "food_score.csv"

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


def supply_ratios(supply_balance: pd.DataFrame) -> pd.DataFrame:
    """The supply balance over YEARS, plus the SSR and IDR ratios and the
    proportional weights W_internal and W_external, the shares of the inflows
    (P + I). Both pairs are always computed. The earlier years exist only to
    feed the internal-risk window, so they are dropped here rather than
    scored."""
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

    inflows = scores["production"] + scores["imports"]
    scores["w_internal"] = scores["production"] / inflows
    scores["w_external"] = scores["imports"] / inflows
    return scores


def internal_risk(supply_balance: pd.DataFrame) -> pd.Series:
    """The coefficient of variation of production over the trailing RISK_WINDOW
    years, one figure for each year in YEARS: how much the harvest moved about
    in the five years up to and including that year."""
    window = supply_balance["production"].rolling(RISK_WINDOW)
    coefficient_of_variation = window.std(ddof=1) / window.mean()
    return coefficient_of_variation.loc[YEARS[0]:YEARS[1]].rename("risk_internal")


def supplier_flows(trade_matrix: pd.DataFrame, country: str,
                   commodity: str) -> pd.Series:
    """How much each supplier shipped the country, indexed by year and
    supplier."""
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
    return (pd.DataFrame({"year": year.astype(int),
                          "supplier": rows[supplier],
                          "flow": flow})
            .groupby(["year", "supplier"])["flow"].sum())


def external_risk(trade_matrix: pd.DataFrame, country: str, commodity: str,
                  without: pd.Series | None = None) -> pd.Series:
    """How concentrated the country's import suppliers are, per year over
    YEARS: the Herfindahl-Hirschman index of each supplier's share of the
    flows recorded for that year. 1/n when n suppliers ship equal shares,
    1.0 when a single supplier ships everything. `without` names one supplier
    per year to leave out -- the biggest one, say -- and the rest are then
    scored on their own smaller total."""
    flows = supplier_flows(trade_matrix, country, commodity)
    # Names this call in whatever fill_missing_years reports below, so the two
    # runs over the same matrix do not print the same sentence.
    filling = "Export quantity"
    if without is not None:
        # A year the matrix does not cover names no supplier, so ignore the
        # labels that are not there rather than raising on them.
        flows = flows.drop(index=list(without.items()), errors="ignore")
        filling += " minus the biggest supplier"

    # A year whose recorded flows add up to nothing has no shares to divide
    # out; drop it and let MISSING_YEAR_POLICY say what an absent year reads as.
    yearly_total = flows.groupby(level="year").transform("sum")
    recorded = yearly_total > 0
    share = flows[recorded] / yearly_total[recorded]
    hhi = (share ** 2).groupby(level="year").sum().rename("risk_external")

    hhi = fill_missing_years(hhi, "trade_matrix_mirror", filling, YEARS,
                             MISSING_YEAR_POLICY["suppliers"], 0.0)
    return hhi.loc[YEARS[0]:YEARS[1]]


def top_supplier(trade_matrix: pd.DataFrame, country: str,
                 commodity: str) -> pd.DataFrame:
    """The biggest supplier of each year over YEARS and its share s_max of the
    flows tracked that year. A year with nothing tracked has no biggest
    supplier and no shock to simulate, so its share reads as 0."""
    flows = supplier_flows(trade_matrix, country, commodity)
    by_year = flows[flows.groupby(level="year").transform("sum") > 0].groupby(
        level="year")

    # idxmax gives the (year, supplier) label of each year's largest flow.
    biggest = by_year.idxmax()
    span = pd.Index(range(YEARS[0], YEARS[1] + 1), name="year")
    return pd.DataFrame({
        "top_supplier": pd.Series([name for _, name in biggest],
                                  index=biggest.index),
        "top_supplier_share": by_year.max() / by_year.sum(),
    }).reindex(span).fillna({"top_supplier_share": 0.0})

def vulnerability_score(scores: pd.DataFrame, internal: str, external: str,
                        risk_external: str = "risk_external") -> pd.Series:
    """How exposed the country is on this commodity, per year over YEARS: each
    risk weighted by the share named for it, either the proportional weights
    or SSR and IDR. `risk_external` names the external risk to read, so the
    simulation can weigh the suppliers left against the same internal risk."""
    return (scores[internal] * scores["risk_internal"]
            + scores[external] * scores[risk_external])


def shocked_weights(scores: pd.DataFrame) -> pd.DataFrame:
    """The proportional weights once the biggest supplier stops shipping: the
    aggregate imports cut by its share, I_new = I x (1 - s_max), and the two
    shares of the inflows P + I_new that are left."""
    imports = scores["imports"] * (1 - scores["top_supplier_share"])
    inflows = scores["production"] + imports
    return pd.DataFrame({"w_internal_sim": scores["production"] / inflows,
                         "w_external_sim": imports / inflows})


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


def food_risk(scores: pd.DataFrame, vulnerability: str) -> pd.Series:
    """The food security risk, per year over YEARS: the vulnerability in the
    `vulnerability` column weighted by how much of the national diet rides on
    the commodity. A shaky supply of something barely eaten scores low; the
    same shakiness in a staple scores high. Both terms arrive validated, so
    this only multiplies them."""
    return scores[vulnerability] * scores["criticality"]

AVERAGES_LABEL = "AVERAGES"
AVERAGED_COLUMNS = ["ssr", "idr", "w_internal", "w_external", "risk_external",
                    "criticality", "top_supplier_share", "risk_external_sim",
                    "w_internal_sim", "w_external_sim"]
def averaged_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """The one-row AVERAGES table summarising the whole of YEARS. The
    simulation is summarised the same way as the rest: the shocked weights and
    risks averaged over the span, the scores rebuilt from those. Only
    `top_supplier` is left blank -- the biggest supplier can differ from year
    to year, so the span names no one country."""
    summary = scores[AVERAGED_COLUMNS].mean().to_frame().T

    # reads the `risk_internal` value for the last scored year
    summary["risk_internal"] = scores["risk_internal"].loc[YEARS[1]]
    summary["vulnerability_weighted"] = vulnerability_score(
        summary, "w_internal", "w_external")
    summary["vulnerability_ssr_idr"] = vulnerability_score(summary, "ssr", "idr")
    summary["food_risk_weighted"] = food_risk(summary, "vulnerability_weighted")
    summary["food_risk_ssr_idr"] = food_risk(summary, "vulnerability_ssr_idr")
    summary["vulnerability_weighted_sim"] = vulnerability_score(
        summary, "w_internal_sim", "w_external_sim", "risk_external_sim")
    summary["food_risk_weighted_sim"] = food_risk(
        summary, "vulnerability_weighted_sim")
    summary.index = pd.Index([AVERAGES_LABEL], name=scores.index.name)
    return summary.reindex(columns=scores.columns)


COLUMN_FORMATS = [
    ("production",    "Production (t)", "{:>16,.2f}"),
    ("imports",       "Imports (t)",    "{:>14,.2f}"),
    ("exports",       "Exports (t)",    "{:>12,.2f}"),
    ("supply",        "Supply (t)",     "{:>16,.2f}"),
    ("ssr",           "SSR",            "{:>6.2f}"),
    ("idr",           "IDR",            "{:>6.2f}"),
    ("w_internal",    "W_internal",     "{:>11.2f}"),
    ("w_external",    "W_external",     "{:>11.2f}"),
    ("risk_internal", "Risk_internal",  "{:>14.2f}"),
    ("risk_external", "Risk_external",  "{:>14.2f}"),
    ("criticality",   "Criticality",    "{:>12.2f}"),
    ("vulnerability_weighted", "Vulnerability_weighted", "{:>23.2f}"),
    ("vulnerability_ssr_idr",  "Vulnerability_ssr_idr",  "{:>22.2f}"),
    ("food_risk_weighted",     "Food_risk_weighted",     "{:>19.2f}"),
    ("food_risk_ssr_idr",      "Food_risk_ssr_idr",      "{:>18.2f}"),
    ("top_supplier",           "Top_supplier",           "{:>24}"),
    ("top_supplier_share",     "Top_supplier_share",     "{:>19.2f}"),
    ("food_risk_weighted_sim", "Food_risk_weighted_sim", "{:>23.2f}"),
]

CSV_COLUMN_NAMES = {
    "production":    "production (P)",
    "imports":       "imports (I)",
    "exports":       "exports (E)",
    "supply":        "supply (P + I - E)",
    "ssr":           "ssr (P / Supply)",
    "idr":           "idr (I / Supply)",
    "w_internal":    "w_internal (P / (P + I))",
    "w_external":    "w_external (I / (P + I))",
    "risk_internal": "risk_internal (std(P) / mean(P))",
    "risk_external": "risk_external (sum(si^2))",
    "criticality":   "criticality (Kcal_commodity / Kcal_total)",
    "vulnerability_weighted":
        "vulnerability_weighted (W_internal x Risk_internal "
        "+ W_external x Risk_external)",
    "vulnerability_ssr_idr":
        "vulnerability_ssr_idr (SSR x Risk_internal + IDR x Risk_external)",
    "food_risk_weighted":
        "food_risk_weighted (Vulnerability_weighted x Criticality)",
    "food_risk_ssr_idr":
        "food_risk_ssr_idr (Vulnerability_ssr_idr x Criticality)",
    "top_supplier":       "top_supplier (biggest supplier by tracked volume)",
    "top_supplier_share":
        "top_supplier_share (s_max = Volume_max / Volume_trade)",
    "food_risk_weighted_sim":
        "food_risk_weighted_sim (Vulnerability_weighted_sim x Criticality), "
        "the biggest supplier gone",
}


def render(df: pd.DataFrame) -> str:
    """Fixed-width table."""
    widths = [len(fmt.format(0)) for _, _, fmt in COLUMN_FORMATS]
    label_width = max([len("Year")] + [len(str(label)) for label in df.index])
    header = "  ".join(
        ["Year".ljust(label_width)]
        + [title.rjust(w) for (_, title, _), w in zip(COLUMN_FORMATS, widths)])
    lines = [header, "-" * len(header)]
    for label, row in df.iterrows():
        cells = [" " * width if pd.isna(row[key]) else fmt.format(row[key])
                 for (key, _, fmt), width in zip(COLUMN_FORMATS, widths)]
        lines.append("  ".join([str(label).ljust(label_width)] + cells))
    return "\n".join(lines)


def csv_text(table: pd.DataFrame) -> str:
    """`table` as CSV text: rounded, the columns named the way the file names
    them, and a row of empty fields after each AVERAGES row so the
    country/commodity blocks read apart. Empty fields rather than an empty
    line, which a spreadsheet and pd.read_csv both skip; the separators come
    back as all-empty rows, so a reader wanting the scores alone drops them
    with .dropna(how="all")."""
    lines = (table.round(ROUND_DECIMALS)
                  .rename(columns=CSV_COLUMN_NAMES)
                  .to_csv(index=False)
                  .splitlines())
    header, body = lines[0], lines[1:]
    if len(body) != len(table):
        raise ValueError(f"{len(table)} rows came back as {len(body)} CSV lines; "
                         "a value carrying a line break would misplace the breaks")

    separator = "," * (len(table.columns) - 1)
    separated = []
    for line, year in zip(body, table["year"]):
        separated.append(line)
        if year == AVERAGES_LABEL:
            separated.append(separator)
    return "\n".join([header] + separated) + "\n"


def main() -> pd.DataFrame:
    data = {name: load(path) for name, path in DATASETS.items()}

    tables = []
    for country in sorted(COUNTRIES):
        for commodity in sorted(COMMODITIES):
            print("=" * 132)
            print(f"{country.upper()} / {commodity.upper()}  ({YEARS[0]}-{YEARS[1]})")
            print(f"Risk_internal:          CV of production over a trailing "
                  f"{RISK_WINDOW}-year window")
            print("Risk_external:          HHI of import-supplier concentration "
                  "(1/n spread out, 1.0 a single supplier)")
            print("Criticality:            the commodity's share of the national "
                  "calorie supply")
            print("Vulnerability_weighted: W_internal x Risk_internal "
                  "+ W_external x Risk_external")
            print("Vulnerability_ssr_idr:  SSR x Risk_internal "
                  "+ IDR x Risk_external")
            print("Food_risk_weighted:     Vulnerability_weighted x Criticality")
            print("Food_risk_ssr_idr:      Vulnerability_ssr_idr x Criticality")
            print("Top_supplier:           the biggest supplier of the year "
                  "and its share s_max of the tracked flows")
            print("Food_risk_weighted_sim: Food_risk_weighted with that "
                  "supplier gone: imports cut to I x (1 - s_max), the weights "
                  "and Risk_external rebuilt from what is left")
            print(f"{AVERAGES_LABEL}:               the whole span as one row: "
                  "the ratios and weights averaged over it, both Vulnerability "
                  "and both Food_risk columns built from those, the "
                  "simulation included")

            pair = {name: rows_for_pair(data[name], country,
                                        COMMODITIES[commodity]["cpc"])
                    for name in ("production", "trade")}
            supply_balance = build_supply_balance(pair)
            scores = supply_ratios(supply_balance)
            scores["risk_internal"] = internal_risk(supply_balance)
            scores["risk_external"] = external_risk(
                data["trade_matrix_mirror"], country, commodity)
            scores["criticality"] = commodity_criticality(
                data["calories"], country, commodity)
            scores["vulnerability_weighted"] = vulnerability_score(
                scores, "w_internal", "w_external")
            scores["vulnerability_ssr_idr"] = vulnerability_score(scores, "ssr", "idr")
            scores["food_risk_weighted"] = food_risk(scores, "vulnerability_weighted")
            scores["food_risk_ssr_idr"] = food_risk(scores, "vulnerability_ssr_idr")

            # Simulate top supplier loss
            top = top_supplier(data["trade_matrix_mirror"], country, commodity)
            scores["top_supplier"] = top["top_supplier"]
            scores["top_supplier_share"] = top["top_supplier_share"]
            scores["risk_external_sim"] = external_risk(
                data["trade_matrix_mirror"], country, commodity,
                without=scores["top_supplier"])
            shocked = shocked_weights(scores)
            scores["w_internal_sim"] = shocked["w_internal_sim"]
            scores["w_external_sim"] = shocked["w_external_sim"]
            scores["vulnerability_weighted_sim"] = vulnerability_score(
                scores, "w_internal_sim", "w_external_sim", "risk_external_sim")
            scores["food_risk_weighted_sim"] = food_risk(
                scores, "vulnerability_weighted_sim")
            scores = pd.concat([scores, averaged_scores(scores)])
            tables.append(scores.assign(country=country, commodity=commodity))

            print(render(scores))
            print()

    table = pd.concat(tables).reset_index()
    table = table[["country", "commodity", "year"]
                  + [key for key, _, _ in COLUMN_FORMATS]]
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as out:
        out.write(csv_text(table))
    print("=" * 132)
    print(f"Wrote {len(table)} rows to {OUT_PATH}")
    return table


if __name__ == "__main__":
    main()
