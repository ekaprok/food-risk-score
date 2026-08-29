"""Unit tests for the helpers in internal_risk_score.py.

Run with:  python3 -m unittest test_internal_risk_score -v

Each test doubles as documentation: read the asserts to see what the helper
does. The script does no work at import time (everything runs under main()),
so it is imported directly.
"""

import unittest

import pandas as pd

import internal_risk_score as irs


def trade_rows(rows):
    """A minimal FAOSTAT long-format frame: (Element, Year, Value) per row."""
    return pd.DataFrame(rows, columns=["Element", "Year", "Value"])


def datasets(production, trade):
    """A data dict shaped like the one main() loads, from row tuples."""
    return {"production": trade_rows(production), "trade": trade_rows(trade)}


# Wide enough to keep every year the fixtures below use, for the tests that
# are not about the year range.
ALL_YEARS = (1961, 2030)


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


class TestAnnualSeries(unittest.TestCase):
    """Pulls one Element out of a long-format file, as a year -> number series."""

    def setUp(self):
        irs.WARNINGS.clear()

    def test_keeps_only_the_requested_element(self):
        # Imports and exports share a file, so this filter is the whole point.
        df = trade_rows([
            ("Import quantity", "2019", "457458.88"),
            ("Export quantity", "2020", "20.68"),
            ("Import quantity", "2020", "598254.11"),
        ])
        result = irs.annual_series(df, "Import quantity")
        self.assertEqual(list(result.index), [2019, 2020])
        self.assertEqual(list(result.values), [457458.88, 598254.11])

    def test_years_come_back_sorted(self):
        # Both scores slice this by year range, so order matters.
        df = trade_rows([("Production", y, "100") for y in ("2011", "2009", "2010")])
        self.assertEqual(list(irs.annual_series(df, "Production").index),
                         [2009, 2010, 2011])

    def test_unreadable_rows_are_dropped_not_zeroed(self):
        # A blank harvest figure is missing data. Turning it into 0 would read
        # as total crop failure and wreck the CV.
        df = trade_rows([
            ("Production", "2019", "4890000"),
            ("Production", "2020", ""),
        ])
        result = irs.annual_series(df, "Production")
        self.assertEqual(list(result.index), [2019])

    def test_missing_columns_raise(self):
        df = pd.DataFrame({"Year": ["2020"], "Value": ["100"]})
        with self.assertRaises(ValueError):
            irs.annual_series(df, "Production")

    def test_a_repeated_year_raises(self):
        # The year becomes the index, so two rows for 2020 would double the
        # commodity's tonnage. Caught here rather than as a duplicate-label
        # error inside an unrelated join later.
        df = trade_rows([
            ("Production", "2020", "100"),
            ("Production", "2020", "150"),
        ])
        with self.assertRaisesRegex(ValueError, "repeated year"):
            irs.annual_series(df, "Production")


class TestBuildSupplyBalance(unittest.TestCase):
    """The shared tonnage table: aligns the three series and applies
    MISSING_YEAR_POLICY."""

    def setUp(self):
        irs.WARNINGS.clear()

    def test_supply_is_production_plus_imports_less_exports(self):
        supply_balance = irs.build_supply_balance(datasets(
            production=[("Production", "2020", "80")],
            trade=[("Import quantity", "2020", "40"),
                   ("Export quantity", "2020", "20")],
        ))
        self.assertEqual(supply_balance.at[2020, "supply"], 100.0)

    def test_every_year_in_the_files_is_kept(self):
        # Not clipped to either period: the two scores cover different years
        # and both read this table.
        supply_balance = irs.build_supply_balance(datasets(
            production=[("Production", y, "100") for y in ("2019", "2020", "2021")],
            trade=[("Import quantity", y, "40") for y in ("2019", "2020", "2021")],
        ))
        self.assertEqual(list(supply_balance.index), [2019, 2020, 2021])

    def test_missing_exports_read_as_zero(self):
        # The fill is per year, not per file: a year that reported exports
        # keeps its tonnage while the year beside it reads as zero.
        supply_balance = irs.build_supply_balance(datasets(
            production=[("Production", y, "100") for y in ("2019", "2020")],
            trade=[("Import quantity", "2019", "40"),
                   ("Import quantity", "2020", "40"),
                   ("Export quantity", "2019", "5")],
        ))
        self.assertEqual(list(supply_balance["exports"]), [5.0, 0.0])
        self.assertEqual(irs.WARNINGS, [])

    def test_years_missing_production_or_imports_are_dropped(self):
        supply_balance = irs.build_supply_balance(datasets(
            production=[("Production", y, "100") for y in ("2019", "2020")],
            trade=[("Import quantity", "2019", "40")],
        ))
        self.assertEqual(list(supply_balance.index), [2019])
        self.assertTrue(any("2020" in w for w in irs.WARNINGS))

    def test_no_usable_year_raises(self):
        with self.assertRaisesRegex(ValueError, "no year has both"):
            irs.build_supply_balance(datasets(
                production=[("Production", "2019", "100")],
                trade=[("Import quantity", "2020", "40")],
            ))


class TestSsrIdrScores(unittest.TestCase):
    """SSR and IDR on top of the supply balance, over SSR_IDR_YEARS."""

    def setUp(self):
        irs.WARNINGS.clear()
        self._years = irs.SSR_IDR_YEARS
        irs.SSR_IDR_YEARS = ALL_YEARS

    def tearDown(self):
        irs.SSR_IDR_YEARS = self._years

    def test_ssr_and_idr(self):
        # Supply = 80 + 40 - 20 = 100; SSR = 80/100; IDR = 40/100.
        scores = irs.ssr_idr_scores(balance([80], [40], [20]))
        self.assertEqual(scores.at[2020, "ssr"], 0.8)
        self.assertEqual(scores.at[2020, "idr"], 0.4)

    def test_only_the_years_in_ssr_idr_years_are_scored(self):
        irs.SSR_IDR_YEARS = (2021, 2022)
        scores = irs.ssr_idr_scores(balance([100] * 4))
        self.assertEqual(list(scores.index), [2021, 2022])

    def test_the_tonnages_travel_with_the_scores(self):
        # The written table carries the workings, not just the two ratios.
        scores = irs.ssr_idr_scores(balance([80], [40], [20]))
        self.assertEqual(list(scores.columns),
                         ["production", "imports", "exports", "supply", "idr", "ssr"])

    def test_non_positive_supply_raises(self):
        # Exports exceeding production + imports is a broken input, not a
        # 0%-dependency country. A negative SSR should never reach the table.
        with self.assertRaisesRegex(ValueError, "non-positive apparent supply"):
            irs.ssr_idr_scores(balance([10], [0], [50]))

    def test_a_period_with_no_supply_balance_raises(self):
        irs.SSR_IDR_YEARS = (1990, 1999)
        with self.assertRaisesRegex(ValueError, "no supply balance"):
            irs.ssr_idr_scores(balance([100] * 3))


class TestInternalRisk(unittest.TestCase):
    """One CV of production for the whole of RISK_INTERNAL_YEARS, not a column."""

    def setUp(self):
        self._years = irs.RISK_INTERNAL_YEARS
        irs.RISK_INTERNAL_YEARS = ALL_YEARS

    def tearDown(self):
        irs.RISK_INTERNAL_YEARS = self._years

    def test_steady_harvests_score_zero(self):
        self.assertEqual(irs.internal_risk(balance([100] * 5)), 0.0)

    def test_it_is_the_cv_of_the_harvests_in_range(self):
        # mean 300; sample std (ddof=1) sqrt(80000/4) = 141.42.
        self.assertAlmostEqual(irs.internal_risk(balance([100, 300, 500, 300, 300])),
                               141.4213562373095 / 300)

    def test_volatile_production_scores_higher_than_steady(self):
        self.assertGreater(irs.internal_risk(balance([50, 150] * 3)),
                           irs.internal_risk(balance([100] * 6)))

    def test_harvests_outside_the_range_do_not_count(self):
        # A collapse before the period must not show up as risk inside it.
        irs.RISK_INTERNAL_YEARS = (2020, 2030)
        self.assertEqual(
            irs.internal_risk(balance([0, 900] + [100] * 5, start=2018)), 0.0)

    def test_too_few_harvests_raise_rather_than_scoring_nan(self):
        irs.RISK_INTERNAL_YEARS = (2020, 2020)
        with self.assertRaisesRegex(ValueError, "need at least 2"):
            irs.internal_risk(balance([100] * 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
