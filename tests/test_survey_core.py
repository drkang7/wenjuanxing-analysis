"""Synthetic-only tests: no survey files, external APIs or local credentials."""

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import survey_core as s


class MappingTests(unittest.TestCase):
    def test_mapping_uses_real_header_not_position(self):
        actual = s.unique_field_map([{"standard": "time", "raw": "实际秒数字段"}],
                                    {"duration": "time"}, ["标识", "实际秒数字段"],
                                    standard_key="standard", raw_key="raw")
        self.assertEqual(actual, {"duration": "实际秒数字段"})

    def test_missing_duplicate_or_nonexistent_match_stops(self):
        row = {"standard": "time", "raw": "elapsed"}
        for records, headers in [([], ["elapsed"]), ([row, row], ["elapsed"]),
                                 ([row], ["other"]), ([row], ["elapsed", "elapsed"])]:
            with self.subTest(records=records, headers=headers), self.assertRaises(s.SurveyError):
                s.unique_field_map(records, {"duration": "time"}, headers, standard_key="standard", raw_key="raw")


class DurationTests(unittest.TestCase):
    def test_explicit_supported_units(self):
        for value, expected in [("425秒", 425), (" 7分5秒 ", 425), ("1.5分钟", 90),
                                ("2min 5s", 125), ("0秒", 0), ("-1秒", -1)]:
            with self.subTest(value=value):
                self.assertEqual(s.parse_duration(value), expected)

    def test_ambiguous_and_wrong_formats_are_not_silent_missing(self):
        for value in ["1:20", "一分半", "2分75秒", "--5秒", True, 120, "120", "NaN秒"]:
            with self.subTest(value=value), self.assertRaises(s.SurveyError):
                s.parse_duration(value)

    def test_numeric_requires_explicit_unit(self):
        self.assertEqual(s.parse_duration(.5, bare_numeric_unit="minutes"), 30)
        self.assertEqual(s.parse_duration("120", bare_numeric_unit="seconds"), 120)
        self.assertIsNone(s.parse_duration(None))

    def test_audit_reconciles_and_gates_unknown(self):
        result = s.audit_durations(["425秒", "1:20", "", "0秒", "-2秒"], list("abcde"))
        self.assertEqual((result["parsed"], result["nonempty"]), (3, 4))
        self.assertEqual(result["records"][1]["raw"], "1:20")
        self.assertEqual(result["records"][1]["record_id"], "b")
        with self.assertRaises(s.SurveyError):
            s.require_duration_ready(result, allow_missing=True, allow_nonpositive=True)

    def test_missing_nonpositive_require_separate_decision(self):
        result = s.audit_durations([None, "0秒", "-2秒"], [1, 2, 3])
        with self.assertRaises(s.SurveyError):
            s.require_duration_ready(result)
        self.assertEqual(s.require_duration_ready(result, allow_missing=True, allow_nonpositive=True), [None, 0, -2])


class RouteAndDenominatorTests(unittest.TestCase):
    def test_skip_code_is_contextual_not_global(self):
        rows = [{"a": 96, "b": 96}, {"a": 1, "b": 96}, {"a": 1, "b": 96}]
        before = copy.deepcopy(rows)
        issues = s.audit_skip_path(rows, [False, True, False],
            skip_codes_by_field={"a": [96], "b": [96]},
            entered_values_by_field={"a": [1, 2, 96], "b": [1, 2, 96]})
        self.assertEqual([(r["position"], r["field"]) for r in issues], [(3, "a")])
        self.assertEqual(rows, before)

    def test_unresolved_predecessor_stops(self):
        with self.assertRaises(s.SurveyError):
            s.multi_select_summary([{"a": 1}], [None], ["a"])

    def test_multi_select_denominator_is_people_not_choices(self):
        result = s.multi_select_summary([{"a": 96, "b": 96}, {"a": 1, "b": 1}, {"a": 0, "b": 1}],
                                        [False, True, True], ["a", "b"])
        self.assertEqual([r["n"] for r in result], [2, 2])
        self.assertEqual([r["percent"] for r in result], [50, 100])

    def test_missing_active_answer_does_not_change_denominator(self):
        with self.assertRaises(s.SurveyError):
            s.multi_select_summary([{"a": None}, {"a": 1}], [True, True], ["a"])

    def test_ranking_only_confirmed_route_in_denominator(self):
        rows = [{"a": 1, "b": 2, "c": 96}, {"a": 2, "b": 1, "c": 96},
                {"a": 96, "b": 2, "c": 1}, {"a": 96, "b": 96, "c": 96}]
        # Synthetic predecessor codes: the revised route is only 4 or 5.
        predecessor = [3, 4, 5, 1]
        result = s.ranking_summary(rows, [v in (4, 5) for v in predecessor], ["a", "b", "c"],
                                   top_k=2, unranked_values=[96])
        self.assertEqual([r["n"] for r in result], [2, 2, 2])
        self.assertEqual([r["top_percent"] for r in result], [50, 100, 50])
        self.assertEqual([r["first_percent"] for r in result], [0, 50, 50])

    def test_duplicate_and_incomplete_ranks_stop(self):
        for row in [{"a": 1, "b": 1, "c": 96}, {"a": 1, "b": 96, "c": 96}]:
            with self.subTest(row=row), self.assertRaises(s.SurveyError):
                s.ranking_summary([row], [True], ["a", "b", "c"], top_k=2, unranked_values=[96])

    def test_empty_denominator_is_unavailable_not_zero_percent(self):
        result = s.multi_select_summary([{"a": 96}], [False], ["a"])
        self.assertEqual(result[0]["n"], 0)
        self.assertIsNone(result[0]["percent"])


class ScoringAndExclusionTests(unittest.TestCase):
    def test_exact_set_scoring(self):
        self.assertEqual(s.exact_selection_score({"a": 0, "b": 1, "c": 0}, {"b"}), 1)
        self.assertEqual(s.exact_selection_score({"a": 1, "b": 1, "c": 0}, {"b"}), 0)
        self.assertEqual(s.exact_selection_score({"a": 0, "b": 0, "c": 0}, {"b"}), 0)

    def test_missing_score_requires_policy(self):
        with self.assertRaises(s.SurveyError):
            s.exact_selection_score({"a": None, "b": 1}, {"b"})
        self.assertIsNone(s.exact_selection_score({"a": None, "b": 1}, {"b"}, missing_policy="missing"))
        with self.assertRaises(s.SurveyError):
            s.exact_selection_score({"a": 96, "b": 1}, {"b"}, missing_policy="incorrect")

    def test_strict_boundary_and_unique_log(self):
        rows = [{"excel_row": 5, "id": "a", "seconds": 119, "hidden": "do-not-copy"},
                {"excel_row": 3, "id": "b", "seconds": 120, "hidden": "do-not-copy"},
                {"excel_row": 4, "id": "c", "seconds": 118, "hidden": "do-not-copy"}]
        before = copy.deepcopy(rows)
        flags = [{"time": r["seconds"] < 120, "other": r["id"] == "a"} for r in rows]
        result = s.exclusion_log(rows, flags, {"time": "synthetic cutoff", "other": "synthetic eligibility"},
                                 log_fields=["id", "seconds"])
        self.assertEqual((result["before"], result["after"], result["deleted"]), (3, 1, 2))
        self.assertEqual(result["rule_events"], 3)
        self.assertEqual(result["multiple_rules"], 1)
        self.assertEqual([r["excel_row"] for r in result["log"]], [4, 5])
        self.assertEqual(result["invalid"], [True, False, True])
        self.assertNotIn("hidden", result["log"][0])
        self.assertEqual(rows, before)

    def test_unresolved_flags_and_duplicate_rows_stop(self):
        for rows, flags in [([{"excel_row": 2}], [{"r": None}]),
                            ([{"excel_row": 2}, {"excel_row": 2}], [{"r": True}, {"r": False}])]:
            with self.subTest(flags=flags), self.assertRaises(s.SurveyError):
                s.exclusion_log(rows, flags, {"r": "reason"}, log_fields=[])

    def test_distribution_contains_all_scores_and_quartiles(self):
        result = s.score_distribution([0, 1, 1, 4], list(range(5)))
        self.assertEqual([b["count"] for b in result["bins"]], [1, 2, 0, 0, 1])
        self.assertEqual(result["n"], 4)
        self.assertEqual((result["median"], result["q1"], result["q3"]), (1, .75, 1.75))
        self.assertEqual(sum(b["percent"] for b in result["bins"]), 100)


class ChiSquareAndFormatTests(unittest.TestCase):
    def test_2x2_pearson_without_correction(self):
        result = s.chi_square_gate([[20, 20], [20, 20]])
        self.assertEqual((result["status"], result["method"], result["correction"]), ("ok", "Pearson", False))
        self.assertEqual(result["n"], 80)
        self.assertNotIn("p", result)

    def test_2x2_yates_is_programmatic(self):
        result = s.chi_square_gate([[2, 18], [2, 18]])
        self.assertEqual((result["method"], result["correction"], result["lt5_count"]), ("Yates", True, 2))

    def test_small_2x2_and_expected_below_one_pause(self):
        for table in [[[9, 9], [9, 9]], [[0, 1], [20, 19]]]:
            with self.subTest(table=table):
                result = s.chi_square_gate(table)
                self.assertEqual(result["status"], "paused")
                self.assertIsNone(result["method"])

    def test_general_sparse_and_dense_tables(self):
        sparse = s.chi_square_gate([[1, 1, 18], [1, 1, 18]])
        dense = s.chi_square_gate([[10, 12, 18], [12, 8, 20]])
        self.assertEqual(sparse["status"], "paused")
        self.assertEqual(sparse["lt5_count"], 4)
        self.assertEqual(dense["status"], "ok")
        self.assertEqual(dense["df"], 2)
        self.assertAlmostEqual(sum(map(sum, dense["expected"])), dense["n"])

    def test_zero_margins_are_not_silently_dropped(self):
        result = s.chi_square_gate([[0, 0, 0], [10, 10, 10]])
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["cells"], 6)
        self.assertEqual(result["lt1_positions"], [(0, 0), (0, 1), (0, 2)])

    def test_general_twenty_percent_boundary_is_inclusive(self):
        result = s.chi_square_gate([[1, 20, 20, 20, 20], [1, 20, 20, 20, 20]])
        self.assertEqual(result["lt5_fraction"], .2)
        self.assertEqual(result["status"], "ok")

    def test_zero_table_and_single_group_are_not_estimable(self):
        for table in [[[0, 0], [0, 0]], [[10, 10, 10]]]:
            with self.subTest(table=table):
                result = s.chi_square_gate(table)
                self.assertEqual(result["status"], "paused")
                self.assertFalse(result["pearson_rc_conditions_met"])

    def test_invalid_counts_raise(self):
        for table in [[[-1, 1], [2, 3]], [[1.1, 1], [2, 3]], [[True, 1], [2, 3]], [[math.inf, 1], [2, 3]]]:
            with self.subTest(table=table), self.assertRaises(s.SurveyError):
                s.chi_square_gate(table)

    def test_p_value_never_displays_zero(self):
        self.assertEqual(s.format_p(0), "<0.001")
        self.assertEqual(s.format_p(.001), "0.001")
        self.assertEqual(s.format_p(.04), "0.040")
        self.assertEqual(s.format_p(None), "")
        with self.assertRaises(s.SurveyError):
            s.format_p(1.2)

    def test_file_fingerprint_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.txt"
            path.write_bytes(b"synthetic survey")
            first = s.file_sha256(path)
            self.assertEqual(len(first), 64)
            self.assertEqual(s.file_sha256(path), first)
            self.assertEqual(path.read_bytes(), b"synthetic survey")


if __name__ == "__main__":
    unittest.main()
