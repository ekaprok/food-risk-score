# %% [markdown]
# # Afghanistan Wheat — Dataset Availability Audit
#
# Loads the five FAOSTAT CSVs and prints which years each series covers.
#
# The files come from an explicit FAOSTAT query, so the requested fields are
# always populated; there is no per-field completeness check here. What the
# query does *not* guarantee is which years came back, which is the whole point
# of the report below.
#
# Nothing is joined, reshaped or combined here; this is a read-only audit.

# %%
import pandas as pd

# Point this at the folder holding the five CSVs (in Colab: "/content").
DATA_DIR = "."

# The two trade matrices are the same flows seen from opposite ends:
#   _self   Afghanistan reports its imports; suppliers sit in `Partner Countries`
#   _mirror each exporter reports its sales; suppliers sit in `Reporter Countries`
# Never summed -- see food_risk_score.py, which selects one per year.
DATASETS = {
    "production":          f"{DATA_DIR}/Afganistan_Production_Wheat.csv",
    "trade":               f"{DATA_DIR}/Afganistan_ImportAndExport_Wheat.csv",
    "calories":            f"{DATA_DIR}/Afganistan_Calories_TotalAndWheat.csv",
    "trade_matrix_self":   f"{DATA_DIR}/Afganistan_Trade_PartnerAll_Wheat.csv",
    "trade_matrix_mirror": f"{DATA_DIR}/Afganistan_Trade_ReporterAll_Wheat.csv",
}

# Which column names the supplier, per trade-matrix orientation.
SUPPLIER_COLUMN = {
    "trade_matrix_self":   "Partner Countries",
    "trade_matrix_mirror": "Reporter Countries",
}

COMMODITIES = [
    # name,       CPC,      FBS,      FBS display name
    ("Wheat",     "0111",   "S2511",  "Wheat and products"),
    ("Maize",     "0112",   "S2514",  "Maize and products"),
    ("Rice",      "0113",   "S2805",  "Rice and products"),
    ("Barley",    "0115",   "S2513",  "Barley and products"),
    ("Soybeans",  "0141",   "S2555",  "Soybeans and products"),
    ("Potatoes",  "01510",  "S2531",  "Potatoes and products"),
]

CPC_TO_FBS = {cpc: fbs for _, cpc, fbs, _ in COMMODITIES}
FBS_TO_CPC = {fbs: cpc for _, cpc, fbs, _ in COMMODITIES}

def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.apply(lambda col: col.str.strip())


data = {name: load(path) for name, path in DATASETS.items()}

# %% [markdown]
# ## Helpers

# %%
def fmt_years(years):
    """Compact a year list into runs: [1961..1970, 1972] -> "1961-1970, 1972"."""
    years = sorted(years)
    if not years:
        return "(none)"
    runs, start, prev = [], years[0], years[0]
    for y in years[1:]:
        if y != prev + 1:
            runs.append((start, prev))
            start = y
        prev = y
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def year_summary(df):
    """(sorted years present, count of rows whose year is blank or unreadable)."""
    parsed = pd.to_numeric(df["Year"], errors="coerce")
    years = sorted(int(y) for y in parsed.dropna().unique())
    return years, int(parsed.isna().sum())


WARNINGS = []

# %% [markdown]
# ## Report

# %%
for name, df in data.items():
    _, undated = year_summary(df)
    print("=" * 78)
    print(f"{name.upper()}  —  {DATASETS[name]}")
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    if undated:
        print(f"  WARN: {undated} row(s) have no readable year")
        WARNINGS.append(f"{name}: {undated} row(s) have a blank or non-numeric Year")

    # --- completeness -------------------------------------------------------
    # One line per series. The year list is the whole answer: consecutive runs
    # collapse, so a hole shows up as a break in the list.
    print("\n  Years by series (Element | Item)")
    for (element, item), sub in df.groupby(["Element", "Item"], sort=True):
        s_years, _ = year_summary(sub)
        label = f"{element} | {item}"
        print(f"    {label:<50} {fmt_years(s_years)}")

    # --- trade matrices: how many suppliers stand behind each year ----------
    # A year present with one supplier is not the same evidence as a year with
    # fifteen, and HHI is computed straight off this count.
    if name in SUPPLIER_COLUMN:
        col = SUPPLIER_COLUMN[name]
        print(f"\n  Suppliers per year (from '{col}')")
        counts = (df.assign(_y=pd.to_numeric(df["Year"], errors="coerce"))
                    .dropna(subset=["_y"])
                    .groupby("_y")[col].nunique().astype(int))
        line = ", ".join(f"{int(y)}:{n}" for y, n in counts.items())
        print(f"    {line}")
        print(f"    {counts.size} year(s), {df[col].nunique()} distinct supplier(s), "
              f"median {int(counts.median())} per year")

    print()

# %% [markdown]
# ## Summary

# %%
print("=" * 78)
print("WARNINGS")
if WARNINGS:
    for w in WARNINGS:
        print(f"  - {w}")
else:
    print("  none")
