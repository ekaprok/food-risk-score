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
    """Repoints the module's year range at the span each fixture covers."""

    YEARS = (2020, 2024)

    def setUp(self):
        self._saved = irs.YEARS
        self.set_years(self.YEARS)

    def tearDown(self):
        irs.YEARS = self._saved

    def set_years(self, years):
        irs.YEARS = years

    @property
    def production_years(self):
        """The wider span production covers, for the first risk window."""
        return (irs.YEARS[0] - irs.RISK_WINDOW + 1, irs.YEARS[1])


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
        result = irs.validate_and_get_series(df, "Import quantity", (2019, 2020))
        self.assertEqual(list(result.index), [2019, 2020])
        self.assertEqual(list(result.values), [457458.88, 598254.11])

    def test_years_come_back_sorted(self):
        # internal_risk rolls over this series by position, so order matters.
        df = trade_rows([("Production", y, "100") for y in ("2011", "2009", "2010")])
        self.assertEqual(
            list(irs.validate_and_get_series(df, "Production", (2009, 2011)).index),
            [2009, 2010, 2011])

    def test_years_outside_the_span_are_left_out(self):
        df = trade_rows(rows_for("Production", (2018, 2020)))
        self.assertEqual(
            list(irs.validate_and_get_series(df, "Production", (2019, 2020)).index),
            [2019, 2020])

    def test_a_year_with_no_row_raises(self):
        # FAOSTAT writes no row both when the value was zero and when nobody
        # recorded one, and does not mark which. Neither reading is guessed at.
        df = trade_rows([("Export quantity", y, "100") for y in ("2019", "2021")])
        with self.assertRaisesRegex(ValueError, r"no row for year\(s\) \[2020\]"):
            irs.validate_and_get_series(df, "Export quantity", (2019, 2021))

    def test_an_unreadable_value_raises(self):
        # A blank harvest figure is missing data. Turning it into 0 would read
        # as total crop failure and wreck the CV.
        df = trade_rows([
            ("Production", "2019", "4890000"),
            ("Production", "2020", ""),
        ])
        with self.assertRaisesRegex(ValueError, r"unreadable value\(s\) for year\(s\) \[2020\]"):
            irs.validate_and_get_series(df, "Production", (2019, 2020))

    def test_an_unreadable_year_raises(self):
        df = trade_rows([("Production", "twenty-twenty", "100")])
        with self.assertRaisesRegex(ValueError, "unreadable year"):
            irs.validate_and_get_series(df, "Production", (2020, 2020))

    def test_fill_zero_reads_an_absent_year_as_zero(self):
        # FAOSTAT omits the row in a year nothing was exported.
        df = trade_rows([("Export quantity", "2019", "5")])
        result = irs.validate_and_get_series(df, "Export quantity", (2019, 2021),
                                             irs.FILL_ZERO)
        self.assertEqual(list(result.index), [2019, 2020, 2021])
        self.assertEqual(list(result.values), [5.0, 0.0, 0.0])

    def test_fill_zero_does_not_rescue_an_unreadable_value(self):
        # A row that exists but will not parse is a broken figure, not a zero.
        df = trade_rows([("Export quantity", "2019", "")])
        with self.assertRaisesRegex(ValueError, "unreadable value"):
            irs.validate_and_get_series(df, "Export quantity", (2019, 2019),
                                        irs.FILL_ZERO)

    def test_missing_columns_raise(self):
        df = pd.DataFrame({"Year": ["2020"], "Value": ["100"]})
        with self.assertRaises(ValueError):
            irs.validate_and_get_series(df, "Production", (2020, 2020))

    def test_a_repeated_year_raises(self):
        # The year becomes the index, so two rows for 2020 would double the
        # commodity's tonnage. Caught here rather than as a duplicate-label
        # error inside an unrelated join later.
        df = trade_rows([
            ("Production", "2020", "100"),
            ("Production", "2020", "150"),
        ])
        with self.assertRaisesRegex(ValueError, "repeated year"):
            irs.validate_and_get_series(df, "Production", (2020, 2020))


class TestBuildSupplyBalance(YearsCase):
    """The tonnage table, covering BALANCE_YEARS rather than just YEARS."""

    YEARS = (2020, 2021)

    def full_span(self, production="100", imports="40", exports="20"):
        """A data dict covering each series' own span: YEARS for trade, and the
        wider window span for production."""
        return datasets(
            production=rows_for("Production", self.production_years, production),
            trade=(rows_for("Import quantity", irs.YEARS, imports)
                   + rows_for("Export quantity", irs.YEARS, exports)))

    def test_supply_is_production_plus_imports_less_exports(self):
        supply_balance = irs.build_supply_balance(self.full_span("80", "40", "20"))
        self.assertEqual(supply_balance.at[2020, "supply"], 100.0)

    def test_the_frame_reaches_back_a_full_window_before_years(self):
        # 2020 is the first year scored, so its window needs 2016-2020 -- all
        # of it inside this one frame.
        supply_balance = irs.build_supply_balance(self.full_span())
        self.assertEqual(list(supply_balance.index), [2016, 2017, 2018, 2019, 2020, 2021])

    def test_the_window_years_carry_production_only(self):
        # Nothing scores them, so no trade figure is asked of them -- 2017 and
        # 2018 missing exports is not this script's business.
        supply_balance = irs.build_supply_balance(self.full_span())
        window_only = supply_balance.loc[2016:2019]
        self.assertTrue(window_only["production"].notna().all())
        self.assertTrue(window_only[["imports", "exports"]].isna().all().all())

    def test_absent_exports_read_as_zero(self):
        # The fill is per year, not per file: a year that reported exports
        # keeps its tonnage while the year beside it reads as zero.
        data = self.full_span()
        trade = data["trade"]
        data["trade"] = trade[~((trade["Element"] == "Export quantity")
                                & (trade["Year"] == "2021"))]
        supply_balance = irs.build_supply_balance(data)
        self.assertEqual(list(supply_balance.loc[2020:2021, "exports"]), [20.0, 0.0])

    def test_a_year_missing_imports_raises(self):
        # Imports are reported every year, so a gap is missing data, not a zero.
        data = self.full_span()
        trade = data["trade"]
        data["trade"] = trade[~((trade["Element"] == "Import quantity")
                                & (trade["Year"] == "2021"))]
        with self.assertRaisesRegex(ValueError, r"Import quantity: no row for year\(s\) \[2021\]"):
            irs.build_supply_balance(data)

    def test_a_year_missing_production_raises(self):
        data = self.full_span()
        data["production"] = data["production"][data["production"]["Year"] != "2016"]
        with self.assertRaisesRegex(ValueError, r"Production: no row for year\(s\) \[2016\]"):
            irs.build_supply_balance(data)


class TestSsrIdrScores(YearsCase):
    """SSR and IDR on top of the supply balance, over YEARS."""

    def test_ssr_and_idr(self):
        # Supply = 80 + 40 - 20 = 100; SSR = 80/100; IDR = 40/100.
        self.set_years((2020, 2020))
        scores = irs.ssr_idr_scores(balance([80], [40], [20]))
        self.assertEqual(scores.at[2020, "ssr"], 0.8)
        self.assertEqual(scores.at[2020, "idr"], 0.4)

    def test_the_window_years_are_dropped_rather_than_scored(self):
        # 2020 and 2021 are only in the frame to feed 2022's risk window.
        self.set_years((2022, 2023))
        scores = irs.ssr_idr_scores(balance([100] * 4, start=2020))
        self.assertEqual(list(scores.index), [2022, 2023])

    def test_the_tonnages_travel_with_the_scores(self):
        # The written table carries the workings, not just the two ratios.
        self.set_years((2020, 2020))
        scores = irs.ssr_idr_scores(balance([80], [40], [20]))
        self.assertEqual(list(scores.columns),
                         ["production", "imports", "exports", "supply", "ssr", "idr"])

    def test_non_positive_supply_raises(self):
        # Exports exceeding production + imports is a broken input, not a
        # 0%-dependency country. A negative SSR should never reach the table.
        self.set_years((2020, 2020))
        with self.assertRaisesRegex(ValueError, "non-positive apparent supply"):
            irs.ssr_idr_scores(balance([10], [0], [50]))

    def test_a_broken_year_outside_years_is_not_scored(self):
        # 2020 only feeds a window; its supply never becomes a published ratio.
        self.set_years((2021, 2021))
        scores = irs.ssr_idr_scores(balance([10, 100], [0, 0], [50, 0], start=2020))
        self.assertEqual(list(scores.index), [2021])


class TestInternalRisk(YearsCase):
    """A CV of production per year, over the trailing RISK_WINDOW years."""

    YEARS = (2019, 2024)

    def test_it_scores_every_year_in_years(self):
        risk = irs.internal_risk(balance([100] * 10, start=2015))
        self.assertEqual(list(risk.index), [2019, 2020, 2021, 2022, 2023, 2024])
        self.assertEqual(risk.name, "risk_internal")

    def test_steady_harvests_score_zero(self):
        risk = irs.internal_risk(balance([100] * 10, start=2015))
        self.assertEqual(list(risk.values), [0.0] * 6)

    def test_it_is_the_cv_of_the_window(self):
        # mean 300; sample std (ddof=1) sqrt(80000/4) = 141.42.
        self.set_years((2019, 2019))
        risk = irs.internal_risk(balance([100, 300, 500, 300, 300], start=2015))
        self.assertAlmostEqual(risk.at[2019], 141.4213562373095 / 300)

    def test_a_window_sees_only_its_own_five_years(self):
        # A collapse in 2015 is risk for 2019, which the window still covers,
        # and none at all for 2020, which it no longer reaches.
        risk = irs.internal_risk(balance([0] + [100] * 9, start=2015))
        self.assertGreater(risk.at[2019], 0.0)
        self.assertEqual(risk.at[2020], 0.0)

    def test_only_production_moves_the_score(self):
        # Trade sits in the same frame now; it must not reach the CV.
        steady = balance([100] * 10, start=2015)
        volatile_trade = balance([100] * 10, imports=[0, 900] * 5, start=2015)
        self.assertEqual(list(irs.internal_risk(steady).values),
                         list(irs.internal_risk(volatile_trade).values))

    def test_volatile_production_scores_higher_than_steady(self):
        self.set_years((2019, 2019))
        self.assertGreater(irs.internal_risk(balance([50, 150] * 3, start=2015))[2019],
                           irs.internal_risk(balance([100] * 6, start=2015))[2019])


if __name__ == "__main__":
    unittest.main(verbosity=2)
