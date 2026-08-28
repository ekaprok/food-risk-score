"""Unit tests for the helpers in load_and_audit.py.

Run with:  python3 -m unittest test_load_and_audit -v

Each test is written to double as documentation: read the asserts to see what
the helper does. Covered here — load, fmt_years, year_summary, and
the CPC<->FBS commodity bridge.

load_and_audit.py is a flat cell-style script, so importing it would run the
whole audit against the CSVs. We parse it and exec only the definitions under
test instead.
"""

import ast
import pathlib
import tempfile
import unittest

import pandas as pd

SCRIPT = pathlib.Path(__file__).with_name("load_and_audit.py")
WANTED = {
    "COMMODITIES", "CPC_TO_FBS", "FBS_TO_CPC",
    "load", "fmt_years", "year_summary",
}

_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
_defs = [
    node for node in _tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in WANTED)
    or (isinstance(node, ast.Assign)
        and {t.id for t in node.targets if isinstance(t, ast.Name)} & WANTED)
]
_ns = {"pd": pd}  # load() closes over the module-level pandas import
exec(compile(ast.Module(body=_defs, type_ignores=[]), str(SCRIPT), "exec"), _ns)

load = _ns["load"]
fmt_years = _ns["fmt_years"]
year_summary = _ns["year_summary"]
COMMODITIES = _ns["COMMODITIES"]
CPC_TO_FBS = _ns["CPC_TO_FBS"]
FBS_TO_CPC = _ns["FBS_TO_CPC"]


def frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)


class TestFmtYears(unittest.TestCase):
    """Turns a list of years into the shortest readable string."""

    def test_consecutive_years_collapse_into_a_range(self):
        self.assertEqual(fmt_years([1961, 1962, 1963]), "1961-1963")

    def test_gaps_split_the_ranges_and_lone_years_stay_bare(self):
        self.assertEqual(fmt_years([2009, 2010, 2011, 2017, 2019]),
                         "2009-2011, 2017, 2019")

    def test_input_need_not_be_sorted(self):
        self.assertEqual(fmt_years([1963, 1961, 1962]), "1961-1963")

    def test_empty_reads_as_none_rather_than_blank(self):
        self.assertEqual(fmt_years([]), "(none)")


class TestYearSummary(unittest.TestCase):
    """Returns (years present, count of rows with no usable year)."""

    def test_lists_the_years_present_in_order(self):
        df = frame([["2013"], ["2009"], ["2010"]], ["Year"])
        self.assertEqual(year_summary(df), ([2009, 2010, 2013], 0))

    def test_duplicate_years_count_once(self):
        # Two rows per year is normal: e.g. imports and exports for 1961.
        df = frame([["1961"], ["1961"], ["1962"]], ["Year"])
        years, _ = year_summary(df)
        self.assertEqual(years, [1961, 1962])

    def test_blank_and_unparseable_years_are_counted_not_guessed(self):
        df = frame([["2010"], [""], ["n/a"]], ["Year"])
        years, undated = year_summary(df)
        self.assertEqual(years, [2010])
        self.assertEqual(undated, 2)  # these rows can't be placed on a timeline


class TestLoad(unittest.TestCase):
    """Reads a FAOSTAT CSV without letting pandas reinterpret anything."""

    def _write(self, text):
        path = pathlib.Path(tempfile.mkdtemp()) / "sample.csv"
        path.write_text(text, encoding="utf-8-sig")  # FAOSTAT files carry a BOM
        return path

    def test_strips_the_bom_so_the_first_column_is_named_correctly(self):
        df = load(self._write('Domain Code,Year\n"QCL","1961"\n'))
        self.assertEqual(list(df.columns), ["Domain Code", "Year"])

    def test_codes_keep_their_exact_stored_form(self):
        # Type inference would turn "0111" into 111 and "004" into 4, after
        # which neither joins against anything.
        df = load(self._write('Item Code (CPC),Area Code (M49)\n"0111","004"\n'))
        self.assertEqual(df.at[0, "Item Code (CPC)"], "0111")
        self.assertEqual(df.at[0, "Area Code (M49)"], "004")

    def test_surrounding_whitespace_is_stripped(self):
        df = load(self._write('Item\n"  Wheat  "\n'))
        self.assertEqual(df.at[0, "Item"], "Wheat")

    def test_blank_cells_stay_empty_strings_not_nan(self):
        # The audit tests missingness with `df[col] == ""`, which NaN defeats.
        df = load(self._write('Item,Note\n"Wheat",""\n'))
        self.assertEqual(df.at[0, "Note"], "")
        self.assertFalse(df.isna().any().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
