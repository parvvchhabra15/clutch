import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from clutch.execution import (
    Annotation, EXECUTION_SCHEMA, ExecutionConfig, SequenceHistory,
    annotation_status, collect_execution, combine_scores, crunch_context,
    execution_rating, grade_execution, percentile,
)
from clutch.performance import ParquetLedger


def annotation(style="Yorker", runs=0, extra="None", stroke="Cover Drive", wicket=False, dismissal="", pace="Pace"):
    return Annotation(style, stroke, pace, extra, runs, int(extra != "None"), 0,
                      wicket, "a" if wicket else "", dismissal, "a", "b", runs in {4, 6})


def state(mid="1", day="2024-01-01", total=0, target=None, runs=0, pressure=75):
    return {
        "match_id": mid, "match_date": day, "family": "ipl", "innings": 1 if target is None else 2,
        "sequence": 1, "striker": "a", "bowler": "b", "batting_team": "A", "bowling_team": "B",
        "phase": "death", "runs_before": runs, "wickets_before": 2, "balls_remaining": 30,
        "legal_balls_before": 90, "target": target, "par_before": 180,
        "recent_wickets_before": 0, "free_hit_before": False, "legal_delivery": True,
        "recorded_total_runs": total, "recorded_wicket": False,
        "batting_team_pressure": pressure, "bowling_team_pressure": pressure,
        "batting_prematch_pressure": 80 if pressure > 50 else 45,
        "bowling_prematch_pressure": 80 if pressure > 50 else 45,
    }


class ExecutionControlTests(unittest.TestCase):
    def grade(self, role, event, s=None, sequence=None):
        return grade_execution(role, event, s or state(total=event.runs + event.primary),
                               sequence or SequenceHistory().before("b"))

    def test_wide_yorker_and_no_ball_do_not_earn_technique_credit(self):
        wide = annotation("Wide Yorker", extra="wide")
        score, features, _ = self.grade("bowling", wide, {**state(total=1), "legal_delivery": False})
        self.assertEqual(score, -0.6)
        self.assertIn("recorded_yorker", features)
        self.assertNotIn("effective_yorker", features)
        no_ball = annotation("Full Toss", extra="Overstep")
        score, _, reason = self.grade("bowling", no_ball)
        self.assertEqual(score, -1)
        self.assertEqual(reason, "recorded_no_ball")

    def test_full_toss_penalty_survives_a_lucky_wicket(self):
        event = annotation("Full Toss", wicket=True, dismissal="Caught")
        score, _, _ = self.grade("bowling", event, {**state(), "recorded_wicket": True})
        self.assertEqual(score, -0.35)

    def test_technique_and_recovery_bonuses_do_not_stack(self):
        sequence = {**SequenceHistory().before("b"), "prior_boundary_streak": 2, "four_plus_last_six": 2}
        score, features, _ = self.grade("bowling", annotation(), sequence=sequence)
        self.assertEqual(score, 0.35)
        self.assertIn("post_boundary_recovery", features)
        self.assertIn("effective_yorker", features)
        self.assertNotIn("effective_stock_delivery", features)

    def test_variation_name_alone_does_not_earn_credit(self):
        score, features, _ = self.grade("bowling", annotation("Knuckle", runs=6))
        self.assertEqual(score, 0)
        self.assertIn("recorded_variation", features)
        self.assertNotIn("effective_variation", features)
        effective, _, _ = self.grade("bowling", annotation("Knuckle"))
        repeated = {**SequenceHistory().before("b"), "previous_bowler_style": "Knuckle"}
        same, _, _ = self.grade("bowling", annotation("Knuckle"), sequence=repeated)
        self.assertEqual(effective, same)

    def test_missing_or_suspect_labels_are_not_assumed_successful(self):
        score, _, _ = self.grade("bowling", annotation("None"))
        self.assertIsNone(score)
        suspect = annotation("Googly", pace="Pace")
        self.assertEqual(annotation_status(suspect), "suspect_pace_spin_label")
        self.assertIsNone(self.grade("bowling", suspect)[0])
        self.assertIsNone(self.grade("bowling", annotation("Beamer"))[0])
        self.assertEqual(self.grade("bowling", annotation("None", extra="other"))[0], -1)


class BattingExecutionTests(unittest.TestCase):
    def test_loose_ball_dot_is_context_sensitive_not_always_bad_defence(self):
        event = annotation("Full Toss", stroke="Frontfoot defensive")
        sequence = SequenceHistory().before("b")
        urgent = state(target=181, runs=90)
        easy = state(target=181, runs=175)
        self.assertLess(grade_execution("batting", event, urgent, sequence)[0], 0)
        self.assertEqual(grade_execution("batting", event, easy, sequence)[0], 0)

    def test_winning_single_can_convert_a_loose_ball_without_need_for_boundary(self):
        event = annotation("Full Toss", runs=1)
        s = state(target=181, runs=180, total=1)
        score, features, _ = grade_execution("batting", event, s, SequenceHistory().before("b"))
        self.assertEqual(score, 0.4)
        self.assertIn("loose_ball_punished", features)

    def test_no_automatic_bonus_for_an_innovative_shot_name(self):
        first = annotation("Seamed In", runs=1, stroke="Cover Drive")
        fancy = replace(first, stroke="Reverse Sweep")
        s = state(total=1)
        sequence = SequenceHistory().before("b")
        self.assertEqual(grade_execution("batting", first, s, sequence)[0],
                         grade_execution("batting", fancy, s, sequence)[0])

    def test_recovery_after_dots_and_protective_defence_are_observable_cases(self):
        sequence = {**SequenceHistory().before("b"), "prior_dot_streak": 3}
        score, features, _ = grade_execution("batting", annotation("Seamed In", runs=1), state(total=1), sequence)
        self.assertGreater(score, 0)
        self.assertIn("post_dot_release", features)
        rebuild = {**state(), "wickets_before": 6, "recent_wickets_before": 2, "balls_remaining": 48}
        score, features, _ = grade_execution(
            "batting", annotation(stroke="Frontfoot defensive"), rebuild, SequenceHistory().before("b"))
        self.assertGreater(score, 0)
        self.assertIn("protective_defense", features)

    def test_runout_blame_is_not_invented(self):
        score, _, _ = grade_execution(
            "batting", annotation("Full Toss", wicket=True, dismissal="RunOut"),
            state(), SequenceHistory().before("b"))
        self.assertIsNone(score)


class SequenceAndReferenceTests(unittest.TestCase):
    def test_current_ball_cannot_create_its_own_prior_dot_or_boundary_context(self):
        history = SequenceHistory()
        self.assertEqual(history.before("b")["prior_dot_streak"], 0)
        history.advance(annotation(), state())
        history.advance(annotation(), state())
        before = history.before("b")
        self.assertEqual(before["prior_dot_streak"], 2)
        history.advance(annotation(runs=4), state(total=4))
        self.assertEqual(before["prior_boundary_streak"], 0)
        self.assertEqual(history.before("b")["prior_boundary_streak"], 1)

    def test_parent_runtime_sequences_are_not_added_twice(self):
        sequence = {**SequenceHistory().before("b"), "prior_dot_streak": 3}
        config = ExecutionConfig()
        legacy = crunch_context("batting", state(), sequence, 70, 80, config)[0]
        adaptive = crunch_context("batting", {**state(), "adaptive_runtime": True}, sequence, 70, 80, config)[0]
        self.assertEqual(legacy, 75)
        self.assertEqual(adaptive, 70)

    def collect(self, change_first=False):
        states = [state("1", pressure=35), state("2", pressure=35), state("3", "2024-01-02", pressure=35)]
        annotations = {(s["match_id"], 1, 1): annotation() for s in states}
        if change_first:
            annotations["1", 1, 1] = annotation("Full Toss")
        names = [{"player_id": "a", "player_name": "A"}, {"player_id": "b", "player_name": "B"}]
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "run_manifest.json").write_text(
                json.dumps({"prematch_manifest": str(root / "pre" / "run_manifest.json")}), encoding="utf-8")
            def read(path):
                return names if path.name == "competition_team_player_mapping.csv" else []
            with patch("clutch.execution.read_csv", side_effect=read), \
                    patch("clutch.execution.iter_ball_states", return_value=iter(states)):
                collect_execution(root / "data" / "raw", base, annotations, ExecutionConfig(), captured.append)
        return captured

    def test_reference_learning_excludes_same_day_and_future_observations(self):
        original, changed = self.collect(), self.collect(change_first=True)
        self.assertEqual([r for r in original if r["match_id"] == "2"],
                         [r for r in changed if r["match_id"] == "2"])
        for row in original:
            self.assertEqual(row["ordinary_reference_count_before"], 2 if row["match_id"] == "3" else 0)
        third_a = next(r for r in original if r["match_id"] == "3" and r["role"] == "bowling")
        third_b = next(r for r in changed if r["match_id"] == "3" and r["role"] == "bowling")
        self.assertNotEqual(third_a["peer_reference_before"], third_b["peer_reference_before"])

    def test_execution_ledger_schema_round_trips(self):
        row = self.collect()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.parquet"
            writer = ParquetLedger(path, EXECUTION_SCHEMA)
            writer.add(row)
            writer.close()
            self.assertEqual(pq.read_table(path).to_pylist()[0], row)


class PresentationCalibrationTests(unittest.TestCase):
    def test_percentiles_are_monotonic_tie_aware_and_not_success_probabilities(self):
        values = [55, 55, 50]
        self.assertAlmostEqual(percentile(55, values), 200 / 3)
        self.assertLess(percentile(50, values), percentile(55, values))
        self.assertEqual(percentile(55, [55]), 50)
        self.assertIsNone(percentile(55, []))

    def test_execution_changes_only_its_allocated_share(self):
        config = ExecutionConfig()
        improved = combine_scores(55, 60, 58, config)
        weakened = combine_scores(55, 40, 58, config)
        self.assertAlmostEqual(improved - weakened, 4)
        self.assertAlmostEqual(combine_scores(55, 60, 58, config, 0), .7 * 55 + .3 * 58)
        self.assertEqual(execution_rating([], config), (50, None, 0))


if __name__ == "__main__":
    unittest.main()
