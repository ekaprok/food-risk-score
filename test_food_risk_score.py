"""Unit tests for the helpers in food_risk_score.py.

Run with:  python3 -m unittest test_food_risk_score -v

Each test doubles as documentation: read the asserts to see what the helper
does. Covered here — quantity_by_year, hhi_by_year and pick_source.

food_risk_score.py is a flat cell-style script, so importing it would run
the whole calculation against the CSVs. We parse it and exec only the
definitions under test instead, the same trick test_load_and_audit.py uses.
"""

import ast
import pathlib
import unittest

import pandas as pd

SCRIPT = pathlib.Path(__file__).with_name("food_risk_score.py")
WANTED = {"quantity_by_year", "hhi_by_year", "pick_source", "SELF", "MIRROR"}

def _assigned_names(node):
    """Names bound by an assignment, including `A, B = ...` tuple targets."""
    return {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}


_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
_defs = [
    node for node in _tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in WANTED)
    or (isinstance(node, ast.Assign) and _assigned_names(node) & WANTED)
]
# WARNINGS collects data-quality notes; the helpers append to it as they go.
# `hhi` is the table pick_source chooses between — each test supplies its own.
_ns = {"pd": pd, "WARNINGS": [], "hhi": {}}
exec(compile(ast.Module(body=_defs, type_ignores=[]), str(SCRIPT), "exec"), _ns)

quantity_by_year = _ns["quantity_by_year"]
hhi_by_year = _ns["hhi_by_year"]
pick_source = _ns["pick_source"]
SELF, MIRROR = _ns["SELF"], _ns["MIRROR"]
WARNINGS = _ns["WARNINGS"]


def trade_rows(rows):
    """A minimal FAOSTAT long-format frame: (Element, Year, Value) per row."""
    return pd.DataFrame(rows, columns=["Element", "Year", "Value"])


def calorie_rows(rows):
    """A minimal calorie frame: (Item, Year, Value) per row, one Element."""
    return pd.DataFrame(
        [("Food supply (kcal/capita/day)", item, year, value) for item, year, value in rows],
        columns=["Element", "Item", "Year", "Value"],
    )


def matrix_rows(rows):
    """A minimal trade-matrix frame: (supplier, M49 code, Year, Value) per row.

    Element is fixed to "Export quantity" — the mirror file's orientation.
    """
    return pd.DataFrame(
        [(supplier, code, year, value, "Export quantity") for supplier, code, year, value in rows],
        columns=["Reporter Countries", "Reporter Country Code (M49)",
                 "Year", "Value", "Element"],
    )


def hhi(df):
    """Run hhi_by_year over a mirror-shaped frame."""
    return hhi_by_year(df, "Export quantity", "Reporter Countries",
                       "Reporter Country Code (M49)", "test")


class TestQuantityByYear(unittest.TestCase):
    """Pulls one Element out of a long-format file, as a year -> number series."""

    def test_keeps_only_the_requested_element(self):
        # Imports and exports share a file, so this filter is the whole point.
        df = trade_rows([
            ("Import quantity", "2019", "457458.88"),
            ("Export quantity", "2020", "20.68"),
            ("Import quantity", "2020", "598254.11"),
        ])
        result = quantity_by_year(df, "Import quantity")
        self.assertEqual(list(result.index), [2019, 2020])
        self.assertEqual(list(result.values), [457458.88, 598254.11])

    def test_years_come_back_sorted(self):
        # The rolling CV walks this series in order, so order matters.
        df = trade_rows([("Production", y, "100") for y in ("2011", "2009", "2010")])
        self.assertEqual(list(quantity_by_year(df, "Production").index),
                         [2009, 2010, 2011])

    def test_unreadable_rows_are_dropped_not_zeroed(self):
        # A blank harvest figure is missing data. Turning it into 0 would read
        # as total crop failure and wreck the CV.
        df = trade_rows([
            ("Production", "2019", "4890000"),
            ("Production", "2020", ""),
        ])
        result = quantity_by_year(df, "Production")
        self.assertEqual(list(result.index), [2019])


class TestQuantityByYearItemFilter(unittest.TestCase):
    """The calorie file needs Item too: both its series share one Element."""

    ROWS = [
        ("Grand Total", "2020", "2259.95"),
        ("Wheat and products", "2020", "1348.16"),
        ("Grand Total", "2021", "2244.73"),
        ("Wheat and products", "2021", "1304.01"),
    ]

    def test_item_picks_one_of_two_series_sharing_an_element(self):
        result = quantity_by_year(calorie_rows(self.ROWS),
                                  "Food supply (kcal/capita/day)",
                                  item="Wheat and products")
        self.assertEqual(list(result.values), [1348.16, 1304.01])

    def test_criticality_is_the_ratio_of_the_two_series(self):
        # C_kcal = kcal_commodity / kcal_total. Wheat is ~60% of Afghan intake.
        df = calorie_rows(self.ROWS)
        element = "Food supply (kcal/capita/day)"
        wheat = quantity_by_year(df, element, item="Wheat and products")
        total = quantity_by_year(df, element, item="Grand Total")
        self.assertAlmostEqual((wheat / total)[2020], 0.5965, places=4)

    def test_omitting_item_keeps_every_series(self):
        # Without an item filter both series come back, which is right for the
        # trade files and wrong for this one -- hence the explicit argument.
        result = quantity_by_year(calorie_rows(self.ROWS),
                                  "Food supply (kcal/capita/day)")
        self.assertEqual(len(result), 4)


class TestHhiByYear(unittest.TestCase):
    """Measures supplier concentration: 1.0 is one supplier, lower is spread."""

    def test_single_supplier_scores_one(self):
        # Buying 100% from one country: 1.0^2 = 1.0, maximum fragility.
        result = hhi(matrix_rows([("Kazakhstan", "398", "2020", "500")]))
        self.assertEqual(result.at[2020, "risk_external"], 1.0)
        self.assertEqual(result.at[2020, "n_partners"], 1)

    def test_two_equal_suppliers_score_one_half(self):
        # 0.5^2 + 0.5^2 = 0.5
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "500"),
            ("Pakistan", "586", "2020", "500"),
        ]))
        self.assertEqual(result.at[2020, "risk_external"], 0.5)

    def test_spreading_across_more_suppliers_lowers_the_score(self):
        concentrated = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "900"),
            ("Pakistan", "586", "2020", "100"),
        ])).at[2020, "risk_external"]
        diversified = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "400"),
            ("Pakistan", "586", "2020", "300"),
            ("Uzbekistan", "860", "2020", "300"),
        ])).at[2020, "risk_external"]
        self.assertGreater(concentrated, diversified)

    def test_reports_the_largest_supplier_and_its_share(self):
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "750"),
            ("Pakistan", "586", "2020", "250"),
        ]))
        self.assertEqual(result.at[2020, "top_partner"], "Kazakhstan")
        self.assertEqual(result.at[2020, "top_partner_share"], 0.75)

    def test_imports_from_matrix_is_the_year_sum(self):
        # This total is what the coverage check later compares against.
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "750"),
            ("Pakistan", "586", "2020", "250"),
        ]))
        self.assertEqual(result.at[2020, "imports_from_matrix"], 1000.0)

    def test_one_supplier_listed_twice_is_summed_first(self):
        # Two rows for Kazakhstan are one supplier, not two. Left unsummed it
        # would look like a diversified year (0.5) instead of a monopoly (1.0).
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "500"),
            ("Kazakhstan", "398", "2020", "500"),
        ]))
        self.assertEqual(result.at[2020, "n_partners"], 1)
        self.assertEqual(result.at[2020, "risk_external"], 1.0)

    def test_aggregates_are_dropped(self):
        # "World" restates its members' flows. Left in, it would double-count
        # the tonnage and fake a monopoly.
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2020", "500"),
            ("Pakistan", "586", "2020", "500"),
            ("World", "001", "2020", "1000"),
        ]))
        self.assertEqual(result.at[2020, "n_partners"], 2)
        self.assertEqual(result.at[2020, "risk_external"], 0.5)

    def test_each_year_is_scored_separately(self):
        result = hhi(matrix_rows([
            ("Kazakhstan", "398", "2019", "500"),
            ("Pakistan", "586", "2019", "500"),
            ("Kazakhstan", "398", "2020", "500"),
        ]))
        self.assertEqual(result.at[2019, "risk_external"], 0.5)
        self.assertEqual(result.at[2020, "risk_external"], 1.0)


class TestPickSource(unittest.TestCase):
    """Chooses the trade matrix whose year total best matches known imports."""

    def setUp(self):
        # Two candidate sources for 2020. Self-reported saw 300 t, mirror saw
        # 950 t. Against a known import total of 1000 t, mirror covers 95% and
        # self-reported only 30%.
        def table(total):
            return pd.DataFrame({"imports_from_matrix": [total]}, index=[2020])

        _ns["hhi"] = {SELF: table(300.0), MIRROR: table(950.0)}

    def test_picks_the_source_closest_to_the_known_import_total(self):
        name, coverage, _ = pick_source(2020, country_imports=1000.0)
        self.assertEqual(name, MIRROR)
        self.assertEqual(coverage, 0.95)

    def test_coverage_is_imports_from_matrix_over_country_imports(self):
        # One candidate only, so the returned coverage is unambiguously its own:
        # 950 t of supplier detail against a 1000 t import total is 95%.
        _ns["hhi"] = {
            SELF: pd.DataFrame({"imports_from_matrix": []}, index=[]),
            MIRROR: pd.DataFrame({"imports_from_matrix": [950.0]}, index=[2020]),
        }
        _, coverage, _ = pick_source(2020, country_imports=1000.0)
        self.assertEqual(coverage, 0.95)

    def test_over_reporting_can_lose_to_under_reporting(self):
        # Closest to 1.0 wins in either direction: at a 950 t import total the
        # mirror is bang on, but at 150 t it overshoots by more than self
        # undershoots (2.0x vs 0.5x), so self-reported wins.
        _ns["hhi"] = {
            SELF: pd.DataFrame({"imports_from_matrix": [75.0]}, index=[2020]),
            MIRROR: pd.DataFrame({"imports_from_matrix": [300.0]}, index=[2020]),
        }
        name, _, _ = pick_source(2020, country_imports=150.0)
        self.assertEqual(name, SELF)

    def test_falls_back_to_the_only_source_that_covers_the_year(self):
        _ns["hhi"] = {
            SELF: pd.DataFrame({"imports_from_matrix": []}, index=[]),
            MIRROR: pd.DataFrame({"imports_from_matrix": [950.0]}, index=[2020]),
        }
        name, _, _ = pick_source(2020, country_imports=1000.0)
        self.assertEqual(name, MIRROR)

    def test_returns_none_when_no_source_covers_the_year(self):
        self.assertIsNone(pick_source(1975, country_imports=1000.0))

    def test_returns_none_when_there_are_no_imports_to_compare_against(self):
        # Coverage would divide by zero, so the year cannot be judged.
        self.assertIsNone(pick_source(2020, country_imports=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
