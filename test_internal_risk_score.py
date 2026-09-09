"""Unit tests for the helpers in internal_risk_score.py.

Run with:  python3 -m unittest test_internal_risk_score -v

Each test doubles as documentation: read the asserts to see what the helper
does. The script does no work at import time (everything runs under main()),
so it is imported directly.
"""

import contextlib
import io
import unittest
import unittest.mock

import pandas as pd

import internal_risk_score as irs


def trade_rows(rows):
    """A minimal FAOSTAT long-format frame: (Element, Year, Value) per row."""
    return pd.DataFrame(rows, columns=["Element", "Year", "Value"])


def capture(function, *args):
    """The call's result, and whatever it printed while running."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = function(*args)
    return result, stdout.getvalue()


def pair_rows(rows):
    """A FAOSTAT frame carrying more than one country and commodity:
    (Area, Item, Element, Year, Value) per row."""
    return pd.DataFrame(rows, columns=["Area", "Item", "Element", "Year", "Value"])


def matrix_rows(rows):
    """A FAOSTAT trade matrix: mirror data, so the supplier is the reporter and
    the importing country is the partner. (Reporter Countries, Partner
    Countries, Item, Element, Year, Value) per row."""
    return pd.DataFrame(rows, columns=["Reporter Countries", "Partner Countries",
                                       "Item", "Element", "Year", "Value"])


def datasets(production, trade):
    """A data dict shaped like the one main() loads, from row tuples."""
    return {"production": trade_rows(production), "trade": trade_rows(trade)}


def rows_for(element, years, value="100"):
    """One row per year of the inclusive span -- the complete span every
    element has to cover."""
    start, end = years
    return [(element, str(year), value) for year in range(start, end + 1)]


def balance(production, imports=None, exports=None, start=2020):
    """A supply balance shaped like build_supply_balance's, from plain lists."""
    n = len(production)
    frame = pd.DataFrame(
        {"production": production,
         "imports": [0] * n if imports is None else imports,
         "exports": [0] * n if exports is None else exports},
        index=range(start, start + n), dtype=float)
    frame.index.name = "year"
    return frame.assign(
        supply=frame["production"] + frame["imports"] - frame["exports"])


class YearsCase(unittest.TestCase):
    """Repoints the module's year range and risk window at what each fixture
    covers, so a test reads without knowing the real constants -- and so that
    changing them cannot quietly turn a test into a different one."""

    YEARS = (2020, 2024)
    RISK_WINDOW = 5

    def setUp(self):
        self._saved = (irs.YEARS, irs.RISK_WINDOW)
        irs.RISK_WINDOW = self.RISK_WINDOW
        self.set_years(self.YEARS)

    def tearDown(self):
        irs.YEARS, irs.RISK_WINDOW = self._saved

    def set_years(self, years):
        irs.YEARS = years

    def quietly(self, function, *args):
        """The call's result, with the fills it reports kept out of the run."""
        return capture(function, *args)[0]

    @property
    def production_years(self):
        """The wider span production covers, for the first risk window."""
        return (irs.YEARS[0] - irs.RISK_WINDOW + 1, irs.YEARS[1])


class TestRowsForPair(unittest.TestCase):
    """Cuts one country and commodity out of a file holding every pair."""

    FILE = pair_rows([
        ("Afghanistan", "Wheat", "Production", "2020", "80"),
        ("Afghanistan", "Rice",  "Production", "2020", "5"),
        ("Thailand",    "Wheat", "Production", "2020", "1"),
        ("Thailand",    "Rice",  "Production", "2020", "30"),
    ])

    def test_it_keeps_one_pair(self):
        # Both halves of the filter matter: three of these four rows share a
        # country or a commodity with the one wanted.
        selected = irs.rows_for_pair(self.FILE, "Afghanistan", "Wheat")
        self.assertEqual(list(selected["Value"]), ["80"])

    def test_a_pair_the_file_does_not_carry_raises(self):
        # Otherwise it surfaces later as "no row for year(s) [2019, ...]",
        # which reads as a gap in the data rather than a missing country.
        with self.assertRaisesRegex(ValueError, "no rows for Peru / Wheat"):
            irs.rows_for_pair(self.FILE, "Peru", "Wheat")

    def test_missing_columns_raise(self):
        df = trade_rows([("Production", "2020", "100")])
        with self.assertRaisesRegex(ValueError, r"\['Area', 'Item'\]"):
            irs.rows_for_pair(df, "Afghanistan", "Wheat")


class TestValidateAndGetSeries(unittest.TestCase):
    """Pulls one Element out of a long-format file, as a year -> number series
    covering the whole of the requested span."""

    def test_keeps_only_the_requested_element(self):
        # Imports and exports share a file, so this filter is the whole point.
        df = trade_rows([
            ("Import quantity", "2019", "457458.88"),
            ("Export quantity", "2020", "20.68"),
            ("Import quantity", "2020", "598254.11"),
        ])
        result = irs.validate_and_get_series(df, "trade", "Import quantity", (2019, 2020))
        self.assertEqual(list(result.index), [2019, 2020])
        self.assertEqual(list(result.values), [457458.88, 598254.11])

    def test_years_come_back_sorted(self):
        # internal_risk rolls over this series by position, so order matters.
        df = trade_rows([("Production", y, "100") for y in ("2011", "2009", "2010")])
        self.assertEqual(
            list(irs.validate_and_get_series(df, "production", "Production", (2009, 2011)).index),
            [2009, 2010, 2011])

    def test_years_outside_the_span_are_left_out(self):
        df = trade_rows(rows_for("Production", (2018, 2020)))
        self.assertEqual(
            list(irs.validate_and_get_series(df, "production", "Production", (2019, 2020)).index),
            [2019, 2020])

    def test_a_year_with_no_row_raises(self):
        df = trade_rows([("Export quantity", y, "100") for y in ("2019", "2021")])
        with self.assertRaisesRegex(ValueError, r"no row for year\(s\) \[2020\]"):
            irs.validate_and_get_series(df, "trade", "Export quantity", (2019, 2021))

    def test_an_unreadable_value_raises(self):
        # A blank harvest figure is missing data. Turning it into 0 would read
        # as total crop failure and wreck the CV.
        df = trade_rows([
            ("Production", "2019", "4890000"),
            ("Production", "2020", ""),
        ])
        with self.assertRaisesRegex(ValueError, r"unreadable value\(s\) for year\(s\) \[2020\]"):
            irs.validate_and_get_series(df, "production", "Production", (2019, 2020))

    def test_an_unreadable_year_raises(self):
        df = trade_rows([("Production", "twenty-twenty", "100")])
        with self.assertRaisesRegex(ValueError, "unreadable year"):
            irs.validate_and_get_series(df, "production", "Production", (2020, 2020))

    def test_fill_zero_reads_an_absent_year_as_zero(self):
        df = trade_rows([("Export quantity", "2019", "5")])
        result, _printed = capture(irs.validate_and_get_series, df, "trade", "Export quantity",
                                   (2019, 2021), irs.FILL_ZERO)
        self.assertEqual(list(result.index), [2019, 2020, 2021])
        self.assertEqual(list(result.values), [5.0, 0.0, 0.0])

    def test_carry_forward_repeats_the_last_year_on_file(self):
        df = trade_rows([("Production", "2019", "5"), ("Production", "2020", "7")])
        result, printed = capture(irs.validate_and_get_series, df, "calories",
                                  "Production", (2019, 2022), irs.CARRY_FORWARD)
        self.assertEqual(list(result.index), [2019, 2020, 2021, 2022])
        self.assertEqual(list(result.values), [5.0, 7.0, 7.0, 7.0])
        self.assertIn("2021, 2022", printed)

    def test_carry_forward_with_nothing_yet_to_carry_raises(self):
        # 2019 comes before every row on file, so no reading carries into it.
        df = trade_rows([("Production", "2020", "5")])
        with self.assertRaisesRegex(ValueError, r"unreadable value\(s\) for year\(s\) \[2019\]"), \
                contextlib.redirect_stdout(io.StringIO()):
            irs.validate_and_get_series(df, "calories", "Production",
                                        (2019, 2020), irs.CARRY_FORWARD)

    def test_missing_columns_raise(self):
        df = pd.DataFrame({"Year": ["2020"], "Value": ["100"]})
        with self.assertRaises(ValueError):
            irs.validate_and_get_series(df, "production", "Production", (2020, 2020))

    def test_a_repeated_year_raises(self):
        # The year becomes the index, so two rows for 2020 would double the
        # commodity's tonnage. Caught here rather than as a duplicate-label
        # error inside an unrelated join later.
        df = trade_rows([
            ("Production", "2020", "100"),
            ("Production", "2020", "150"),
        ])
        with self.assertRaisesRegex(ValueError, "repeated year"):
            irs.validate_and_get_series(df, "production", "Production", (2020, 2020))


class TestSupplyRatios(YearsCase):

    YEARS = (2023, 2024)
    BALANCE = balance(production=[900, 80, 60],
                      imports=[500, 40, 30],
                      exports=[700, 20, 10],
                      start=2022)

    def test_ssr(self):
        # Supply is production + imports - exports, per year.
        scores = irs.supply_ratios(self.BALANCE)
        self.assertEqual(list(scores.index), [2023, 2024])
        self.assertEqual(scores.at[2023, "ssr"], 80 / (80 + 40 - 20))
        self.assertEqual(scores.at[2024, "ssr"], 60 / (60 + 30 - 10))

    def test_idr(self):
        scores = irs.supply_ratios(self.BALANCE)
        self.assertEqual(scores.at[2023, "idr"], 40 / (80 + 40 - 20))
        self.assertEqual(scores.at[2024, "idr"], 30 / (60 + 30 - 10))
        self.assertEqual(list(scores.columns),
                         ["production", "imports", "exports", "supply", "ssr", "idr",
                          "w_internal", "w_external"])

    def test_the_weights_split_the_inflows(self):
        scores = irs.supply_ratios(self.BALANCE)
        self.assertEqual(scores.at[2023, "w_internal"], 80 / (80 + 40))
        self.assertEqual(scores.at[2023, "w_external"], 40 / (80 + 40))
        self.assertEqual(scores.at[2024, "w_internal"], 60 / (60 + 30))
        self.assertEqual(scores.at[2024, "w_external"], 30 / (60 + 30))
        pairs = scores["w_internal"] + scores["w_external"]
        self.assertEqual(pairs.tolist(), [1.0, 1.0])

    def test_non_positive_supply_raises(self):
        # Exports exceeding production + imports is a broken input, not a
        # 0%-dependency country. A negative SSR should never reach the table.
        self.set_years((2020, 2020))
        with self.assertRaisesRegex(ValueError, "non-positive apparent supply"):
            irs.supply_ratios(balance([10], [0], [50]))

    def test_a_broken_year_outside_years_is_not_scored(self):
        self.set_years((2021, 2021))
        # supply for 2020 = 10 - 50 = -40. Error but we ignore it as it is outside our window.
        scores = irs.supply_ratios(balance(production=[10, 100], imports=[0, 0], exports=[50, 0], start=2020))
        self.assertEqual(list(scores.index), [2021])


class TestInternalRisk(YearsCase):
    """A CV of production per year, over the trailing RISK_WINDOW years."""

    # Small enough to check by hand: a window of two, over two scored years.
    YEARS = (2023, 2024)
    RISK_WINDOW = 2

    def test_it_is_the_cv_of_each_trailing_window(self):
        risk = irs.internal_risk(balance(production=[9000, 100, 300, 300],
                                         imports=[0, 5000, 0, 5000],
                                         exports=[7000, 0, 3000, 0],
                                         start=2021))

        mean_2023 = (100 + 300) / 2
        std_2023 = (((100 - mean_2023) ** 2 + (300 - mean_2023) ** 2) / (2 - 1)) ** 0.5
        mean_2024 = (300 + 300) / 2
        std_2024 = (((300 - mean_2024) ** 2 + (300 - mean_2024) ** 2) / (2 - 1)) ** 0.5

        self.assertEqual(list(risk.index), [2023, 2024])
        self.assertAlmostEqual(risk.at[2023], std_2023 / mean_2023)
        self.assertEqual(risk.at[2024], std_2024 / mean_2024)
        self.assertEqual(risk.name, "risk_internal")


KCAL = "Food supply (kcal/capita/day)"


class TestCommodityCriticality(YearsCase):
    """The commodity's share of the national calorie supply, per year."""

    # Small enough to check by hand: two scored years, both on file.
    YEARS = (2020, 2021)

    FILE = pair_rows([
        ("Afghanistan", "Grand Total",        KCAL, "2020", "2000"),
        ("Afghanistan", "Grand Total",        KCAL, "2021", "2500"),
        ("Afghanistan", "Wheat and products", KCAL, "2020", "1000"),
        ("Afghanistan", "Wheat and products", KCAL, "2021", "500"),
    ])

    def test_commodity_criticality(self):
        score = self.quietly(irs.commodity_criticality, self.FILE,
                             "Afghanistan", "Wheat")

        self.assertEqual(list(score.index), [2020, 2021])
        self.assertAlmostEqual(score.at[2020], 1000 / 2000)
        self.assertAlmostEqual(score.at[2021], 500 / 2500)
        self.assertEqual(score.name, "criticality")

    def test_commodity_criticality_carry_forward(self):
        self.set_years((2020, 2022))

        score, printed = capture(irs.commodity_criticality, self.FILE,
                                 "Afghanistan", "Wheat")

        self.assertEqual(list(score.index), [2020, 2021, 2022])
        self.assertAlmostEqual(float(score.loc[2022]), 500 / 2500)
        self.assertIn("2022", printed)

    def test_a_non_positive_total_raises(self):
        file = pair_rows([
            ("Afghanistan", "Grand Total",        KCAL, "2020", "2000"),
            ("Afghanistan", "Grand Total",        KCAL, "2021", "0"),
            ("Afghanistan", "Wheat and products", KCAL, "2020", "1000"),
            ("Afghanistan", "Wheat and products", KCAL, "2021", "500"),
        ])

        with self.assertRaisesRegex(ValueError, r"non-positive total supply in \[2021\]"):
            irs.commodity_criticality(file, "Afghanistan", "Wheat")


class TestExternalRisk(YearsCase):
    """The HHI of the import suppliers' shares, per year."""

    # Small enough to check by hand: two scored years, two suppliers in the
    # first and one in the second.
    YEARS = (2020, 2021)

    FILE = matrix_rows([
        ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2020", "600"),
        ("Pakistan",   "Afghanistan", "Wheat", "Export quantity", "2020", "400"),
        ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2021", "900"),
        # Distractors:
        ("India",      "Thailand",    "Wheat", "Export quantity", "2020", "50"),
        ("India",      "Afghanistan", "Rice",  "Export quantity", "2020", "50"),
        ("India",      "Afghanistan", "Wheat", "Import quantity", "2020", "50"),
    ])

    def test_it_is_the_sum_of_squared_supplier_shares(self):
        risk = irs.external_risk(self.FILE, "Afghanistan", "Wheat")

        self.assertEqual(list(risk.index), [2020, 2021])
        self.assertAlmostEqual(risk.loc[2020], (600 / 1000) ** 2 + (400 / 1000) ** 2)
        self.assertEqual(risk.name, "risk_external")

    def test_a_lone_supplier_scores_one(self):
        # The top of the range: every tonne comes from one place.
        risk = irs.external_risk(self.FILE, "Afghanistan", "Wheat")
        self.assertAlmostEqual(risk.loc[2021], (900 / 900) ** 2)

    def test_a_supplier_listed_twice_is_one_supplier(self):
        # A re-export or a revision can put a country on two rows for a year.
        # Summed first it is one supplier shipping everything (1.0); read as
        # two it would look half as concentrated (0.5).
        file = matrix_rows([
            ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2020", "500"),
            ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2020", "500"),
        ])
        self.set_years((2020, 2020))
        self.assertAlmostEqual(irs.external_risk(file, "Afghanistan", "Wheat").loc[2020], 1.0)

    def test_a_year_whose_flows_are_all_zero_reads_as_zero(self):
        # Nothing to divide out. Without the guard the shares would be 0/0,
        # and the year would come back NaN rather than 0.
        file = matrix_rows([
            ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2020", "0"),
            ("Pakistan",   "Afghanistan", "Wheat", "Export quantity", "2020", "0"),
        ])
        self.set_years((2020, 2020))
        risk = self.quietly(irs.external_risk, file, "Afghanistan", "Wheat")
        self.assertEqual(risk.loc[2020], 0.0)

    def test_a_year_with_no_supplier_rows_reads_as_zero(self):
        # A data blackout reads as the benign end of the scale, and says so.
        self.set_years((2020, 2022))
        risk, printed = capture(irs.external_risk, self.FILE, "Afghanistan", "Wheat")

        self.assertEqual(list(risk.index), [2020, 2021, 2022])
        self.assertEqual(risk.loc[2022], 0.0)
        self.assertIn("with zero for 2022", printed)

    def test_an_importer_the_matrix_does_not_carry_raises(self):
        with self.assertRaisesRegex(ValueError, "no supplier rows for Peru / Wheat"):
            irs.external_risk(self.FILE, "Peru", "Wheat")

    def test_an_unreadable_value_raises(self):
        # A blank tonnage is missing data; read as 0 it would silently drop a
        # supplier and make the year look more concentrated than it is.
        file = matrix_rows([
            ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "2020", "600"),
            ("Pakistan",   "Afghanistan", "Wheat", "Export quantity", "2020", ""),
        ])
        self.set_years((2020, 2020))
        with self.assertRaisesRegex(ValueError, r"unreadable value\(s\)"):
            irs.external_risk(file, "Afghanistan", "Wheat")

    def test_an_unreadable_year_raises(self):
        file = matrix_rows([
            ("Kazakhstan", "Afghanistan", "Wheat", "Export quantity", "twenty-twenty", "600"),
        ])
        self.set_years((2020, 2020))
        with self.assertRaisesRegex(ValueError, r"unreadable year"):
            irs.external_risk(file, "Afghanistan", "Wheat")

    def test_missing_columns_raise(self):
        # The names, not just the shared prefix: this frame has the columns
        # rows_for_pair wants, so a test on the prefix alone could pass off
        # that helper's copy of the message.
        df = pair_rows([("Afghanistan", "Wheat", "Export quantity", "2020", "600")])
        with self.assertRaisesRegex(
                ValueError, r"\['Partner Countries', 'Reporter Countries'\]"):
            irs.external_risk(df, "Afghanistan", "Wheat")


class TestVulnerabilityScore(unittest.TestCase):
    """Each half of the supply weighted by the risk it carries. It reads only
    USE_PROPORTIONAL_WEIGHTS and no files, so the scores go in by hand and the
    flag is set per test. The two pairs of weights differ, so which pair the
    flag picked is visible in the result."""

    SCORES = pd.DataFrame(
        {"ssr": [0.8, 0.5], "idr": [0.2, 0.5],
         "w_internal": [0.7, 0.3], "w_external": [0.3, 0.7],
         "risk_internal": [0.1, 0.4], "risk_external": [0.5, 0.6]},
        index=[2020, 2021], dtype=float)

    def test_proportional_weights_on_uses_w_internal_and_w_external(self):
        with unittest.mock.patch.object(irs, "USE_PROPORTIONAL_WEIGHTS", True):
            score = irs.vulnerability_score(self.SCORES)

        self.assertEqual(list(score.index), [2020, 2021])
        first, second = score.tolist()
        self.assertAlmostEqual(first, 0.7 * 0.1 + 0.3 * 0.5)
        self.assertAlmostEqual(second, 0.3 * 0.4 + 0.7 * 0.6)
        self.assertEqual(score.name, "vulnerability")

    def test_proportional_weights_off_uses_ssr_and_idr(self):
        with unittest.mock.patch.object(irs, "USE_PROPORTIONAL_WEIGHTS", False):
            score = irs.vulnerability_score(self.SCORES)

        self.assertEqual(list(score.index), [2020, 2021])
        first, second = score.tolist()
        self.assertAlmostEqual(first, 0.8 * 0.1 + 0.2 * 0.5)
        self.assertAlmostEqual(second, 0.5 * 0.4 + 0.5 * 0.6)
        self.assertEqual(score.name, "vulnerability")


class TestFoodRisk(unittest.TestCase):
    """The vulnerability weighted by how much the diet leans on the commodity.
    It reads no module constants and no files, so the scores go in by hand."""

    def test_food_risk(self):
        scores = pd.DataFrame(
            {"vulnerability": [0.2, 0.5], "criticality": [0.6, 0.1]},
            index=[2020, 2021], dtype=float)

        score = irs.food_risk(scores)

        self.assertEqual(list(score.index), [2020, 2021])
        first, second = score.tolist()
        self.assertAlmostEqual(first, 0.2 * 0.6)
        self.assertAlmostEqual(second, 0.5 * 0.1)
        self.assertEqual(score.name, "food_risk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
