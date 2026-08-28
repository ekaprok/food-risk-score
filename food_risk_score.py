# %% [markdown]
# # Afghanistan Wheat — Vulnerability Score
#
# Builds a per-year Vulnerability Score from the FAOSTAT CSVs audited in
# `load_and_audit.py`:
#
# ```
# Supply = P + I - E            apparent domestic supply
# IDR    = I / Supply           import dependency ratio
# SSR    = P / Supply           self-sufficiency ratio
#
# Risk_internal = std(P) / mean(P)      coefficient of variation, rolling window
# Risk_external = sum(share_i ** 2)     Herfindahl-Hirschman index over partners
#
# V = (SSR x Risk_internal) + (IDR x Risk_external)
#
# C_kcal = kcal_commodity / kcal_total  criticality: share of the diet
#
# Food Security Risk = V x C_kcal
# ```
#
# V measures how fragile the supply is; C_kcal measures how much a disruption
# would hurt. Wheat is 58-67% of Afghan calories, so the two are close in scale
# here — for a marginal crop, criticality would shrink V sharply.
#
# Which dataset feeds which term:
#
# | term          | source file                              | element         |
# |---------------|------------------------------------------|-----------------|
# | P             | `Production_Wheat`                       | Production      |
# | I, E          | `ImportAndExport_Wheat`                  | Import/Export quantity |
# | Risk_internal | `Production_Wheat`                       | Production      |
# | Risk_external | `Trade_PartnerAll_Wheat` *or* `Trade_ReporterAll_Wheat` | Import/Export quantity |
# | C_kcal        | `Calories_TotalAndWheat`                 | Food supply (kcal/capita/day) |
#
# Per the brief, the HHI market shares are computed *within* the trade matrix:
# the denominator is the sum of that year's supplier-level flows, not the country
# total from `ImportAndExport_Wheat`. Anything else produces shares that do not
# sum to 1.
#
# ### Two trade matrices, one series
#
# Every trade flow is declared twice, once by each end:
#
# * `Trade_PartnerAll` — **self-reported**. Reporter = Afghanistan, suppliers in
#   the `Partner` column, element `Import quantity`. Covers 2009-2011, 2017, 2019
#   — only the years Afghanistan filed customs declarations.
# * `Trade_ReporterAll` — **mirror**. Reporter = each exporter, Afghanistan in the
#   `Partner` column, element `Export quantity`. Covers 1986-2024 (no 1988), by
#   asking the sellers instead of the buyer.
#
# One source is *selected per year*, by comparing each against the country
# import total in `ImportAndExport_Wheat` — the same figure the IDR already
# uses as its numerator:
#
#     coverage = matrix year total
#              / `ImportAndExport_Wheat` "Import quantity" for that year
#
# Coverage near 1.0 means the matrix accounts for the whole year's trade, so the
# source closest to 1.0 wins.
#
# Close enough means 60-140% coverage (`COVERAGE_BAND` below). If neither source
# lands in that range, the year gets no score. Many years from 1987-2008 only
# cover 1-25% of the trade, and a supplier list that thin says nothing about who
# Afghanistan really relied on.

# %%
import pandas as pd

# Point this at the folder holding the CSVs (in Colab: "/content/faostat").
DATA_DIR = "faostat"

DATASETS = {
    "production":          f"{DATA_DIR}/Afganistan_Production_Wheat.csv",
    "trade":               f"{DATA_DIR}/Afganistan_ImportAndExport_Wheat.csv",
    "trade_matrix_self":   f"{DATA_DIR}/Afganistan_Trade_PartnerAll_Wheat.csv",
    "trade_matrix_mirror": f"{DATA_DIR}/Afganistan_Trade_ReporterAll_Wheat.csv",
    "calories":            f"{DATA_DIR}/Afganistan_Calories_TotalAndWheat.csv",
}

# The two calorie series share one Element and are told apart by Item instead.
KCAL_ELEMENT = "Food supply (kcal/capita/day)"
KCAL_COMMODITY = "Wheat and products"
KCAL_TOTAL = "Grand Total"

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

# Rolling window for the internal-risk CV. Long enough to see a bad harvest
# cycle, short enough that the 1960s do not describe the 2010s.
CV_WINDOW = 10
CV_MIN_YEARS = 5

# How far a matrix year total may sit from the country import total before its
# HHI stops being trustworthy.
COVERAGE_BAND = (0.60, 1.40)

# The score is this project's own output, not FAOSTAT input, so it is written
# beside the script rather than into DATA_DIR.
OUT_PATH = "food_risk_score.csv"


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.apply(lambda col: col.str.strip())


data = {name: load(path) for name, path in DATASETS.items()}

WARNINGS = []

# %% [markdown]
# ## Supply, IDR and SSR

# %%
def quantity_by_year(df, element, item=None):
    """year -> quantity, for one Element of a FAOSTAT long-format frame.

    Imports and exports live in the same file, told apart only by `Element`,
    so picking one is the whole job. The calorie file needs `item` as well:
    both its series share one Element and differ only by Item. See
    test_food_risk_score.py for worked examples.

    Rows whose Year or Value will not parse as a number are dropped and
    counted; a missing figure is not the same as a zero and must not be
    silently turned into one.
    """
    sub = df[df["Element"] == element]
    if item is not None:
        sub = sub[sub["Item"] == item]
    year = pd.to_numeric(sub["Year"], errors="coerce")
    value = pd.to_numeric(sub["Value"], errors="coerce")
    usable = year.notna() & value.notna()
    dropped = int((~usable).sum())
    if dropped:
        WARNINGS.append(f"{element}: dropped {dropped} row(s) with no readable year/value")
    return pd.Series(value[usable].values, index=year[usable].astype(int).values,
                     name=element).sort_index()


# Building the frame from a dict aligns every series on its year index, so
# 2017 production lands in the same row as 2017 imports. A year one series
# lacks becomes NaN here, and MISSING_YEAR_POLICY decides what happens to
# it next.
balance = pd.DataFrame({
    name: quantity_by_year(data[dataset], element)
    for name, (dataset, element, _) in MISSING_YEAR_POLICY.items()
})

for name, (_, _, policy) in MISSING_YEAR_POLICY.items():
    if policy == FILL_ZERO:
        balance[name] = balance[name].fillna(0.0)

required = [name for name, (_, _, policy) in MISSING_YEAR_POLICY.items()
            if policy == DROP_YEAR]
missing = balance[balance[required].isna().any(axis=1)]
if len(missing):
    WARNINGS.append(
        f"supply: {len(missing)} year(s) lack {' or '.join(required)} and are skipped "
        f"({', '.join(str(y) for y in missing.index)})"
    )
balance = balance.dropna(subset=required)

# =============================================================================
# SUPPLY  —  apparent domestic supply
# =============================================================================
balance["supply"] = balance["production"] + balance["imports"] - balance["exports"]

# Supply is a denominator. A non-positive one means exports exceeded everything
# available, which is a data problem, not a 0%-dependency country.
bad_supply = balance.index[balance["supply"] <= 0].tolist()
if bad_supply:
    WARNINGS.append(
        f"supply: non-positive apparent supply in {', '.join(str(y) for y in bad_supply)}; "
        "IDR/SSR left blank for those years"
    )
denominator = balance["supply"].where(balance["supply"] > 0)

# =============================================================================
# IDR & SSR  —  import dependency and self-sufficiency ratios
# =============================================================================
balance["idr"] = balance["imports"] / denominator
balance["ssr"] = balance["production"] / denominator

# =============================================================================
# INTERNAL RISK
# =============================================================================
rolling = balance["production"].rolling(CV_WINDOW, min_periods=CV_MIN_YEARS)
balance["risk_internal"] = rolling.std(ddof=1) / rolling.mean()

# =========================================================================
# EXTERNAL RISK
# =========================================================================

# %%
def hhi_by_year(df, element, supplier_col, code_col, label):
    """year -> (HHI, supplier count, largest supplier, its share, year total).

    The supplier lives in a different column depending on which way the matrix
    was queried -- `Partner Countries` when Afghanistan is the reporter, and
    `Reporter Countries` in the mirror file -- so the caller names it.

    Shares are supplier flow / total flow *in this dataset* for that year, so
    they sum to 1 by construction. HHI ranges from 1/n (imports split evenly
    across n suppliers) to 1.0 (a single supplier).
    """
    sub = df[df["Element"] == element].copy()

    # FAOSTAT country lists mix real countries with aggregates ("World",
    # "Europe", "European Union"). An aggregate restates its members' flows, so
    # one leaking in would both double-count tonnage and collapse many suppliers
    # into one -- inflating HHI toward a monopoly that does not exist. M49
    # country codes are exactly 3 digits; 001 is World and aggregates are longer.
    code = sub[code_col].str.strip()
    is_country = code.str.fullmatch(r"\d{3}").fillna(False) & (code != "001")
    if not is_country.all():
        names = sorted(sub.loc[~is_country, supplier_col].unique())
        WARNINGS.append(f"{label}: dropped aggregate supplier(s): {', '.join(names)}")
        sub = sub[is_country]

    sub["year"] = pd.to_numeric(sub["Year"], errors="coerce")
    sub["qty"] = pd.to_numeric(sub["Value"], errors="coerce")
    dropped = int((sub["year"].isna() | sub["qty"].isna()).sum())
    if dropped:
        WARNINGS.append(f"{label}: dropped {dropped} row(s) with no readable year/value")
    sub = sub.dropna(subset=["year", "qty"])
    sub["year"] = sub["year"].astype(int)

    # One supplier can appear twice in a year (re-exports, revisions); sum first
    # so a split entry does not read as two smaller, more diversified suppliers.
    suppliers = sub.groupby(["year", supplier_col])["qty"].sum()

    rows = {}
    for year, group in suppliers.groupby(level="year"):
        total = group.sum()
        if total <= 0:
            WARNINGS.append(f"{label}: {year} has no positive flows; HHI left blank")
            continue
        shares = group / total
        rows[year] = (float((shares ** 2).sum()), int(len(shares)),
                      shares.idxmax()[1], float(shares.max()), float(total))
    return pd.DataFrame.from_dict(
        rows, orient="index",
        columns=["risk_external", "n_partners", "top_partner",
                 "top_partner_share", "imports_from_matrix"],
    ).sort_index()


SELF, MIRROR = "self-reported", "mirror"

hhi = {
    SELF:   hhi_by_year(data["trade_matrix_self"], "Import quantity",
                        "Partner Countries", "Partner Country Code (M49)", SELF),
    MIRROR: hhi_by_year(data["trade_matrix_mirror"], "Export quantity",
                        "Reporter Countries", "Reporter Country Code (M49)", MIRROR),
}

# %% [markdown]
# ### Choosing a source per year
#
# Never sum the two matrices — each flow both ends reported would be counted
# twice. Select instead: whichever year total sits closest to the country import
# total is the one describing the same trade the IDR denominator describes.
# Ties go to the self-reported file, which is the importer's own customs record
# and consistently names more small suppliers.

# %%
def pick_source(year, country_imports):
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


lo, hi = COVERAGE_BAND
rows = {}
for year in sorted(set(hhi[SELF].index) | set(hhi[MIRROR].index)):
    if year not in balance.index:
        continue  # no supply figures, so the year can never be scored anyway
    chosen = pick_source(year, balance.at[year, "imports"])
    if chosen is None:
        continue
    name, coverage, row = chosen
    in_band = lo <= coverage <= hi
    rows[year] = {
        # Out-of-band years keep every diagnostic column but surrender the HHI,
        # which drops them out of V rather than scoring them on partial trade.
        "risk_external": row["risk_external"] if in_band else float("nan"),
        "n_partners": row["n_partners"],
        "top_partner": row["top_partner"],
        "top_partner_share": row["top_partner_share"],
        "hhi_source": name if in_band else f"{name} (rejected)",
        # The tonnage the chosen matrix summed to. Divided by the `imports`
        # column it gives import_matrix_coverage, so the ratio is checkable by eye.
        "imports_from_matrix": row["imports_from_matrix"],
        "import_matrix_coverage": coverage,
    }

external = pd.DataFrame.from_dict(rows, orient="index").sort_index()

rejected = external.index[external["risk_external"].isna()].tolist()
if rejected:
    WARNINGS.append(
        f"trade matrix: {len(rejected)} year(s) rejected, no source within "
        f"{lo:.0%}-{hi:.0%} of the country import total "
        f"({', '.join(str(y) for y in rejected)})"
    )

# =========================================================================
# Vulnerability
# =========================================================================

scores = balance.join(external, how="left")

scores["vulnerability"] = (scores["ssr"] * scores["risk_internal"]
                           + scores["idr"] * scores["risk_external"])
scores.index.name = "year"

# %% [markdown]
# ## Criticality and food security risk
#
# Vulnerability says how fragile the wheat supply is. Criticality says how much
# the country would feel it: a commodity nobody eats can collapse without
# causing a food crisis. Multiplying the two turns a supply-chain metric into a
# food-security one.

# %%
# =========================================================================
# CRITICALITY  —  share of the diet this commodity provides
#
#     C_kcal = kcal_commodity / kcal_total
#
# Both series come from the same file and the same Element, so they are told
# apart by Item.
# =========================================================================
kcal_commodity = quantity_by_year(data["calories"], KCAL_ELEMENT, KCAL_COMMODITY)
kcal_total = quantity_by_year(data["calories"], KCAL_ELEMENT, KCAL_TOTAL)

# Total calories is a denominator, and a non-positive one is a broken figure
# rather than a country that eats nothing.
bad_kcal = kcal_total.index[kcal_total <= 0].tolist()
if bad_kcal:
    WARNINGS.append(
        f"calories: non-positive total intake in "
        f"{', '.join(str(y) for y in bad_kcal)}; criticality left blank"
    )

scores["kcal_commodity"] = kcal_commodity
scores["kcal_total"] = kcal_total
scores["criticality"] = scores["kcal_commodity"] / scores["kcal_total"].where(
    scores["kcal_total"] > 0)

# =========================================================================
# FOOD SECURITY RISK  —  the headline number
#
#     Food Security Risk = Vulnerability x Criticality
#
# A fragile supply of something central to the diet scores high; the same
# fragility in a marginal crop scores low.
# =========================================================================
scores["food_security_risk"] = scores["vulnerability"] * scores["criticality"]

scored = scores.dropna(subset=["vulnerability"])
if scored.empty:
    WARNINGS.append("no year has both an internal and an external risk term")

# Calories only run 2010-2023, so some scored years cannot reach a final risk.
no_criticality = scored.index[scored["food_security_risk"].isna()].tolist()
if no_criticality:
    WARNINGS.append(
        f"criticality: {len(no_criticality)} scored year(s) have no calorie data, "
        f"so no food security risk ({', '.join(str(y) for y in no_criticality)})"
    )

# =========================================================================
# Report
# =========================================================================

# %%
# Everything is reported to two decimals. Rounding happens here, at the point
# of display, and again on the way out to CSV -- never in the maths above, so
# no rounding error is carried from one term into the next.
DECIMALS = 2

COLUMNS = [
    ("production",        "Production (t)",  "{:>16,.2f}"),
    ("imports",           "Imports (t)",     "{:>14,.2f}"),
    ("exports",           "Exports (t)",     "{:>12,.2f}"),
    ("supply",            "Supply (t)",      "{:>16,.2f}"),
    ("ssr",               "SSR",             "{:>6.2f}"),
    ("idr",               "IDR",             "{:>6.2f}"),
    ("risk_internal",     "Int.risk",        "{:>8.2f}"),
    ("risk_external",     "Ext.risk",        "{:>8.2f}"),
    ("n_partners",        "Partners",        "{:>8.0f}"),  # a count, not a measurement
    ("imports_from_matrix",      "Matrix (t)",      "{:>14,.2f}"),  # what Cover divides
    ("vulnerability",     "V",               "{:>6.2f}"),
    ("criticality",       "Crit.",           "{:>6.2f}"),
    ("food_security_risk", "FSR",            "{:>6.2f}"),
]

# Appended to the table as plain text; these are labels, not numbers.
TEXT_COLUMNS = [("hhi_source", "Source", 20), ("import_matrix_coverage", "Cover", 6)]


def render(df):
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


print("=" * 78)
print("SCORED YEARS  (both risk terms available)")
print(render(scored) if not scored.empty else "  (none)")

print()
print("=" * 78)
print("FULL BALANCE  (V blank where no matrix covers the year, or none reconciles)")
print(render(scores))

print()
print("=" * 78)
print("TOP IMPORT PARTNER, BY YEAR  (from the selected source)")
for year, row in external.iterrows():
    if pd.isna(row["risk_external"]):
        continue
    print(f"  {year}  {row['top_partner']:<28} "
          f"{row['top_partner_share']:.2%} of imports across {row['n_partners']} suppliers"
          f"   [{row['hhi_source']}]")

print()
print("=" * 78)
print("SOURCE DISAGREEMENT  (years both matrices cover)")
# Read straight from the two HHI tables rather than carrying per-source copies
# in `scores`: this concerns a handful of years and would be blank in the rest.
overlap = sorted(set(hhi[SELF].index) & set(hhi[MIRROR].index) & set(external.index))
if not overlap:
    print("  (no overlapping years)")
else:
    print("        |------ HHI ------|  |-------- year total, t --------|")
    print("  Year      self    mirror  gap     self       mirror      imports  selected")
    for year in overlap:
        s, m = hhi[SELF].loc[year], hhi[MIRROR].loc[year]
        gap = abs(s["risk_external"] - m["risk_external"])
        print(f"  {year}  {s['risk_external']:>8.2f}  {m['risk_external']:>8.2f}  "
              f"{gap:>4.2f}  {s['imports_from_matrix']:>11,.0f}  {m['imports_from_matrix']:>11,.0f}  "
              f"{balance.at[year, 'imports']:>11,.0f}  {external.at[year, 'hhi_source']}")

# %%
scores.round(DECIMALS).to_csv(OUT_PATH)
print()
print("=" * 78)
print(f"Wrote {len(scores)} rows to {OUT_PATH}")

print()
print("WARNINGS")
if WARNINGS:
    for w in WARNINGS:
        print(f"  - {w}")
else:
    print("  none")
