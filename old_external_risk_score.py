"""Afghanistan wheat — external supply risk (supplier concentration).

Builds a per-year external-risk score from the FAOSTAT trade matrices audited
in `load_and_audit.py`:

    Risk_external = sum(share_i ** 2)   Herfindahl-Hirschman index (HHI)
                                        over import suppliers

Per the brief, the HHI market shares are computed *within* the trade matrix:
the denominator is the sum of that year's supplier-level flows, not the country
total from `ImportAndExport_Wheat`. Anything else produces shares that do not
sum to 1.

Two trade matrices, one series
------------------------------
Every trade flow is declared twice, once by each end:

* `Trade_PartnerAll` — **self-reported**. Reporter = Afghanistan, suppliers in
  the `Partner` column, element `Import quantity`. Covers 2009-2011, 2017, 2019
  — only the years Afghanistan filed customs declarations.
* `Trade_ReporterAll` — **mirror**. Reporter = each exporter, Afghanistan in
  the `Partner` column, element `Export quantity`. Covers 1986-2024 (no 1988),
  by asking the sellers instead of the buyer.

Never sum the two matrices — each flow both ends reported would be counted
twice. One source is *selected per year* instead, by comparing each against
the country import total in `ImportAndExport_Wheat`:

    coverage = matrix year total
             / `ImportAndExport_Wheat` "Import quantity" for that year

Coverage near 1.0 means the matrix accounts for the whole year's trade, so the
source closest to 1.0 wins. Ties go to the self-reported file, which is the
importer's own customs record and consistently names more small suppliers.

Close enough means 60-140% coverage (`COVERAGE_BAND` below). If neither source
lands in that range, the year gets no score. Many years from 1987-2008 only
cover 1-25% of the trade, and a supplier list that thin says nothing about who
Afghanistan really relied on.

The internal metrics (SSR, IDR, internal risk) live in `internal_risk_score.py`.
"""

from typing import Optional

import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "trade":               f"{DATA_DIR}/Afganistan_ImportAndExport_Wheat.csv",
    "trade_matrix_self":   f"{DATA_DIR}/Afganistan_Trade_PartnerAll_Wheat.csv",
    "trade_matrix_mirror": f"{DATA_DIR}/Afganistan_Trade_ReporterAll_Wheat.csv",
}

SELF, MIRROR = "self-reported", "mirror"

# How far a matrix year total may sit from the country import total before its
# HHI stops being trustworthy.
COVERAGE_BAND = (0.60, 1.40)

# Everything is reported to two decimals. Rounding happens at the point of
# display and on the way out to CSV -- never in the maths -- so no rounding
# error is carried from one term into the next.
DECIMALS = 2

# The score is this project's own output, not FAOSTAT input, so it is written
# beside the script rather than into DATA_DIR.
OUT_PATH = "old_external_risk_score.csv"

# Data-quality notes collected while computing; printed at the end of main().
WARNINGS: list[str] = []


def load(path: str) -> pd.DataFrame:
    """The CSV as an all-string frame with surrounding whitespace stripped."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.apply(lambda col: col.str.strip())


def annual_series(df: pd.DataFrame, element: str) -> pd.Series:
    """A Series of yearly values for the given element."""
    missing = {"Element", "Year", "Value"} - set(df.columns)
    if missing:
        raise ValueError(f"expected FAOSTAT column(s) missing: {sorted(missing)}")
    element_rows = df[df["Element"] == element]
    year = pd.to_numeric(element_rows["Year"], errors="coerce")
    value = pd.to_numeric(element_rows["Value"], errors="coerce")
    usable_mask = year.notna() & value.notna()
    dropped = int((~usable).sum())
    if dropped:
        WARNINGS.append(f"{element}: dropped {dropped} row(s) with no readable year/value")

    # Remove duplicate years
    years = year[usable_mask].astype(int)
    repeated = sorted(int(y) for y in years[years.duplicated()].unique())
    if repeated:
        raise ValueError(f"{element}: repeated year(s) {repeated}; "
                         "expected one row per year")

    return pd.Series(value[usable_mask].values, index=years.values,
                     name=element).sort_index()


def hhi_by_year(df: pd.DataFrame, element: str, supplier_col: str,
                code_col: str, label: str) -> pd.DataFrame:
    """year -> (HHI, supplier count, largest supplier, its share, year total).

    The supplier lives in a different column depending on which way the matrix
    was queried -- `Partner Countries` when Afghanistan is the reporter, and
    `Reporter Countries` in the mirror file -- so the caller names it.

    Shares are supplier flow / total flow *in this dataset* for that year, so
    they sum to 1 by construction. HHI ranges from 1/n (imports split evenly
    across n suppliers) to 1.0 (a single supplier).
    """
    missing = {"Element", "Year", "Value", supplier_col, code_col} - set(df.columns)
    if missing:
        raise ValueError(f"{label}: expected column(s) missing: {sorted(missing)}")
    flows = df[df["Element"] == element].copy()

    # FAOSTAT country lists mix real countries with aggregates ("World",
    # "Europe", "European Union"). An aggregate restates its members' flows, so
    # one leaking in would both double-count tonnage and collapse many suppliers
    # into one -- inflating HHI toward a monopoly that does not exist. M49
    # country codes are exactly 3 digits; 001 is World and aggregates are longer.
    code = flows[code_col].str.strip()
    is_country = code.str.fullmatch(r"\d{3}").fillna(False) & (code != "001")
    if not is_country.all():
        names = sorted(flows.loc[~is_country, supplier_col].unique())
        WARNINGS.append(f"{label}: dropped aggregate supplier(s): {', '.join(names)}")
        flows = flows[is_country]

    flows["year"] = pd.to_numeric(flows["Year"], errors="coerce")
    flows["qty"] = pd.to_numeric(flows["Value"], errors="coerce")
    dropped = int((flows["year"].isna() | flows["qty"].isna()).sum())
    if dropped:
        WARNINGS.append(f"{label}: dropped {dropped} row(s) with no readable year/value")
    flows = flows.dropna(subset=["year", "qty"])
    flows["year"] = flows["year"].astype(int)

    # One supplier can appear twice in a year (re-exports, revisions); sum first
    # so a split entry does not read as two smaller, more diversified suppliers.
    suppliers = flows.groupby(["year", supplier_col])["qty"].sum()

    rows = {}
    for year, group in suppliers.groupby(level="year"):
        total = group.sum()
        if total <= 0:
            WARNINGS.append(f"{label}: {year} has no positive flows; HHI left blank")
            continue
        shares = group / total
        rows[year] = (float((shares ** 2).sum()), len(shares),
                      shares.idxmax()[1], float(shares.max()), float(total))
    return pd.DataFrame.from_dict(
        rows, orient="index",
        columns=["risk_external", "n_partners", "top_partner",
                 "top_partner_share", "imports_from_matrix"],
    ).sort_index()


def pick_source(year: int, country_imports: float,
                hhi: dict[str, pd.DataFrame]) -> Optional[tuple]:
    """(source name, coverage, its HHI row) for one year, or None if unusable."""
    candidates = []
    for name in (SELF, MIRROR):  # SELF first, so it wins an exact tie
        table = hhi[name]
        if year in table.index and country_imports > 0:
            coverage = table.at[year, "imports_from_matrix"] / country_imports
            candidates.append((abs(coverage - 1.0), name, coverage))
    if not candidates:
        return None
    _, name, coverage = min(candidates, key=lambda c: c[0])
    return name, coverage, hhi[name].loc[year]


def build_external(hhi: dict[str, pd.DataFrame], imports: pd.Series) -> pd.DataFrame:
    """Per-year external risk, one trade-matrix source selected per year."""
    lo, hi = COVERAGE_BAND
    rows = {}
    for year in sorted(set(hhi[SELF].index) | set(hhi[MIRROR].index)):
        chosen = pick_source(year, float(imports.get(year, 0.0)), hhi)
        if chosen is None:
            continue  # no country import total to judge coverage against
        name, coverage, row = chosen
        in_band = lo <= coverage <= hi
        rows[year] = {
            # Out-of-band years keep every diagnostic column but surrender the
            # HHI, which drops them out of downstream scores rather than
            # scoring them on partial trade.
            "risk_external": row["risk_external"] if in_band else float("nan"),
            "n_partners": row["n_partners"],
            "top_partner": row["top_partner"],
            "top_partner_share": row["top_partner_share"],
            "hhi_source": name if in_band else f"{name} (rejected)",
            # The tonnage the chosen matrix summed to. Divided by the country
            # import total it gives the coverage, so the ratio is checkable by eye.
            "imports_from_matrix": row["imports_from_matrix"],
            "import_matrix_coverage": coverage,
        }

    external = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    external.index.name = "year"
    if external.empty:
        WARNINGS.append("trade matrix: no year could be scored")
        return external

    rejected = external.index[external["risk_external"].isna()].tolist()
    if rejected:
        WARNINGS.append(
            f"trade matrix: {len(rejected)} year(s) rejected, no source within "
            f"{lo:.0%}-{hi:.0%} of the country import total "
            f"({', '.join(str(y) for y in rejected)})"
        )
    return external


COLUMNS = [
    ("risk_external",       "Ext.risk",   "{:>8.2f}"),
    ("n_partners",          "Partners",   "{:>8.0f}"),  # a count, not a measurement
    ("top_partner_share",   "Top share",  "{:>9.2f}"),
    ("imports_from_matrix", "Matrix (t)", "{:>14,.2f}"),  # what Cover divides
]

# Appended to the table as plain text; these are labels, not numbers.
TEXT_COLUMNS = [
    ("top_partner", "Top partner", 28),
    ("hhi_source", "Source", 24),
    ("import_matrix_coverage", "Cover", 6),
]


def render(df: pd.DataFrame) -> str:
    """Fixed-width table. Column width comes from the format string itself, so
    the header and the numbers under it cannot drift apart."""
    widths = [len(fmt.format(0)) for _, _, fmt in COLUMNS]
    header = "  ".join(
        ["Year"]
        + [title.rjust(w) for (_, title, _), w in zip(COLUMNS, widths)]
        + [title.ljust(w) for _, title, w in TEXT_COLUMNS]
    )
    lines = [header, "-" * len(header)]
    for year, row in df.iterrows():
        cells = [str(year)]
        for (key, _, fmt), width in zip(COLUMNS, widths):
            value = row[key]
            cells.append("-".rjust(width) if pd.isna(value) else fmt.format(value))
        for key, _, width in TEXT_COLUMNS:
            value = row.get(key)
            if isinstance(value, float):
                value = "-" if pd.isna(value) else f"{value:.0%}"
            cells.append(str("-" if value is None else value).ljust(width))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def print_top_partners(external: pd.DataFrame) -> None:
    for year, row in external.iterrows():
        if pd.isna(row["risk_external"]):
            continue
        print(f"  {year}  {row['top_partner']:<28} "
              f"{row['top_partner_share']:.2%} of imports across {row['n_partners']} suppliers"
              f"   [{row['hhi_source']}]")


def print_source_disagreement(hhi: dict[str, pd.DataFrame], imports: pd.Series,
                              external: pd.DataFrame) -> None:
    overlap = sorted(set(hhi[SELF].index) & set(hhi[MIRROR].index) & set(external.index))
    if not overlap:
        print("  (no overlapping years)")
        return
    print("        |------ HHI ------|  |-------- year total, t --------|")
    print("  Year      self    mirror  gap     self       mirror      imports  selected")
    for year in overlap:
        s, m = hhi[SELF].loc[year], hhi[MIRROR].loc[year]
        gap = abs(s["risk_external"] - m["risk_external"])
        print(f"  {year}  {s['risk_external']:>8.2f}  {m['risk_external']:>8.2f}  "
              f"{gap:>4.2f}  {s['imports_from_matrix']:>11,.0f}  {m['imports_from_matrix']:>11,.0f}  "
              f"{imports.at[year]:>11,.0f}  {external.at[year, 'hhi_source']}")


def main() -> pd.DataFrame:
    data = {name: load(path) for name, path in DATASETS.items()}
    imports = annual_series(data["trade"], "Import quantity")
    hhi = {
        SELF:   hhi_by_year(data["trade_matrix_self"], "Import quantity",
                            "Partner Countries", "Partner Country Code (M49)", SELF),
        MIRROR: hhi_by_year(data["trade_matrix_mirror"], "Export quantity",
                            "Reporter Countries", "Reporter Country Code (M49)", MIRROR),
    }
    external = build_external(hhi, imports)

    print("=" * 78)
    print("EXTERNAL RISK  (HHI blank where no matrix covers the year, or none reconciles)")
    print(render(external) if not external.empty else "  (none)")

    print()
    print("=" * 78)
    print("TOP IMPORT PARTNER, BY YEAR  (from the selected source)")
    print_top_partners(external)

    print()
    print("=" * 78)
    print("SOURCE DISAGREEMENT  (years both matrices cover)")
    print_source_disagreement(hhi, imports, external)

    external.round(DECIMALS).to_csv(OUT_PATH)
    print()
    print("=" * 78)
    print(f"Wrote {len(external)} rows to {OUT_PATH}")

    print()
    print("WARNINGS")
    if WARNINGS:
        for warning in WARNINGS:
            print(f"  - {warning}")
    else:
        print("  none")
    return external


if __name__ == "__main__":
    main()
