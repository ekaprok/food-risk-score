"""SSR, IDR and internal supply risk.

Builds the internal supply metrics from the FAOSTAT CSVs`:

    Supply        = P + I - E         apparent domestic supply, per year
    SSR           = P / Supply        self-sufficiency ratio, per year
    IDR           = I / Supply        import dependency ratio, per year
    Risk_internal = std(P) / mean(P)  coefficient of variation, one figure
                                      for the whole of RISK_INTERNAL_YEARS

Which dataset feeds which term:

    | term          | source file             | element                |
    |---------------|-------------------------|------------------------|
    | P             | `Production_Wheat`      | Production             |
    | I, E          | `ImportAndExport_Wheat` | Import/Export quantity |
    | Risk_internal | `Production_Wheat`      | Production             |

Each metric covers its own hard-coded span of years -- SSR_IDR_YEARS and
RISK_INTERNAL_YEARS below.

"""

import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "production": f"{DATA_DIR}/Afganistan_Production_Wheat.csv",
    "trade":      f"{DATA_DIR}/Afganistan_ImportAndExport_Wheat.csv",
}

# What an absent year means. FAOSTAT writes no row in two different situations
# -- the true value was zero, and nobody recorded a value -- and does not mark
# which is which. Nothing in the file resolves it; only knowing the series does.
# So each one declares its own reading:
DROP_YEAR, FILL_ZERO = "drop_year", "fill_zero"
MISSING_YEAR_POLICY = {
    "production": ("production", "Production",      DROP_YEAR),
    "imports":    ("trade",      "Import quantity", DROP_YEAR),
    "exports":    ("trade",      "Export quantity", FILL_ZERO),
}

# The years each metric covers, inclusive on both ends.
SSR_IDR_YEARS = (2010, 2024)
RISK_INTERNAL_YEARS = (2010, 2024)

ROUND_DECIMALS = 2

OUT_PATH = "internal_risk_score.csv"

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
    year: pd.Series = pd.to_numeric(element_rows["Year"], errors="coerce")
    value: pd.Series = pd.to_numeric(element_rows["Value"], errors="coerce")
    valid_mask: pd.Series = year.notna() & value.notna()
    dropped = int((~valid_mask).sum())
    if dropped:
        WARNINGS.append(f"{element}: dropped {dropped} row(s) with no readable year/value")

    # Remove duplicate years
    years = year[valid_mask].astype(int)
    repeated = sorted(int(y) for y in years[years.duplicated()].unique())
    if repeated:
        raise ValueError(f"{element}: repeated year(s) {repeated}; "
                         "expected one row per year")

    return pd.Series(value[valid_mask].values, index=years.values,
                     name=element).sort_index()


def build_supply_balance(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Year-indexed production, imports, exports and apparent supply.
    """
    supply_balance = pd.DataFrame({
        name: annual_series(data[dataset], element)
        for name, (dataset, element, _) in MISSING_YEAR_POLICY.items()
    })

    for name, (_, _, policy) in MISSING_YEAR_POLICY.items():
        if policy == FILL_ZERO:
            supply_balance[name] = supply_balance[name].fillna(0.0)

    datasets_with_drop_policy = [name for name, (_, _, policy) in MISSING_YEAR_POLICY.items()
                                   if policy == DROP_YEAR]
    missing = supply_balance.index[supply_balance[datasets_with_drop_policy].isna().any(axis=1)]
    if len(missing):
        WARNINGS.append(
            f"supply: {len(missing)} year(s) lack {' or '.join(datasets_with_drop_policy)} and are "
            f"skipped ({', '.join(str(y) for y in missing)})"
        )
    supply_balance = supply_balance.dropna(subset=datasets_with_drop_policy)
    if supply_balance.empty:
        raise ValueError("no year has both production and import figures")

    supply_balance["supply"] = (supply_balance["production"]
                                + supply_balance["imports"]
                                - supply_balance["exports"])
    supply_balance.index.name = "year"
    return supply_balance


def ssr_idr_scores(supply_balance: pd.DataFrame) -> pd.DataFrame:
    """The supply balance over SSR_IDR_YEARS, plus the SSR and IDR columns."""
    start, end = SSR_IDR_YEARS
    scores = supply_balance.loc[start:end].copy()
    if scores.empty:
        raise ValueError(f"no supply balance in SSR_IDR_YEARS {start}-{end}")

    # Supply is the denominator of both ratios, so a non-positive one would
    # give a negative SSR rather than a wrong-but-plausible number.
    bad_supply = scores.index[scores["supply"] <= 0].tolist()
    if bad_supply:
        raise ValueError(f"non-positive apparent supply in {bad_supply}: "
                         "exports exceed production plus imports")

    scores["idr"] = scores["imports"] / scores["supply"]
    scores["ssr"] = scores["production"] / scores["supply"]
    return scores


def internal_risk(supply_balance: pd.DataFrame) -> float:
    """The coefficient of variation of production across RISK_INTERNAL_YEARS:
    one figure for how much the harvest moved about over the period."""
    start, end = RISK_INTERNAL_YEARS
    production = supply_balance.loc[start:end, "production"]
    if len(production) < 2:
        raise ValueError(f"internal risk: {len(production)} year(s) of production "
                         f"in RISK_INTERNAL_YEARS {start}-{end}, need at least 2")
    return float(production.std(ddof=1) / production.mean())


COLUMN_FORMATS = [
    ("production",    "Production (t)", "{:>16,.2f}"),
    ("imports",       "Imports (t)",    "{:>14,.2f}"),
    ("exports",       "Exports (t)",    "{:>12,.2f}"),
    ("supply",        "Supply (t)",     "{:>16,.2f}"),
    ("ssr",           "SSR",            "{:>6.2f}"),
    ("idr",           "IDR",            "{:>6.2f}"),
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
    supply_balance = build_supply_balance(data)
    scores = ssr_idr_scores(supply_balance)
    risk = internal_risk(supply_balance)

    print("=" * 78)
    print(f"SUPPLY, SSR AND IDR  ({SSR_IDR_YEARS[0]}-{SSR_IDR_YEARS[1]})")
    print(render(scores))

    print()
    print(f"Risk_internal  {risk:.2f}   (CV of production, "
          f"{RISK_INTERNAL_YEARS[0]}-{RISK_INTERNAL_YEARS[1]})")

    scores.round(ROUND_DECIMALS).to_csv(OUT_PATH)
    print()
    print("=" * 78)
    print(f"Wrote {len(scores)} rows to {OUT_PATH}")

    print()
    print("WARNINGS")
    if WARNINGS:
        for warning in WARNINGS:
            print(f"  - {warning}")
    else:
        print("  none")
    return scores


if __name__ == "__main__":
    main()
