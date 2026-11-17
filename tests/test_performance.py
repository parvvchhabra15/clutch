import copy
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from clutch.data import Dataset, Match, Performance
from clutch.inputs import InputError
from clutch.live import (
    Delivery, InningsState, ParHistory, PerformanceConfig, combined_pressure,
    innings_quality, live_pressure, pressure_weight,
)
from clutch.performance import PLAYER_SCHEMA, ParquetLedger, analyze
from clutch.ratings import (
    HistoricalBaselines, MatchCard, performance_observation, rating_components,
)


def event(sequence=1, innings=1, runs=1, **changes):
    result = Delivery(
        "1", innings, sequence, (sequence - 1) // 6 + 1,
        "A" if innings == 1 else "B", "B" if innings == 1 else "A",
        "a" if innings == 1 else "b", "a2" if innings == 1 else "b2",
        "b" if innings == 1 else "a", runs, 0, 0, "None", False, "", "")
    return replace(result, **changes)


def complete_match_events(mid="1"):
    return [replace(event(i, number), match_id=mid) for number in [1, 2] for i in range(1, 121)]


def model_data(same_day=False):
    matches = [
        Match(str(i), "C", date(2024, 1, 1 if same_day and i <= 2 else i),
              i, "League", "A", "B", "A", "A Won by 1 Run", "venue")
        for i in range(1, 4)]
    data = Dataset(
        matches, {"C": {"comp_name": "Test", "competition_family_id": "ipl", "season": "2024"}},
        {"A": "Team A", "B": "Team B"}, {"a": "Player A", "b": "Player B", "a2": "Partner A", "b2": "Partner B"},
        {}, {}, {m.id: {"a": Performance(appearances=1), "b": Performance(appearances=1)} for m in matches})
    deliveries = {m.id: complete_match_events(m.id) for m in matches}
    metadata = {m.id: {} for m in matches}
    prematch = {(m.id, team): {"team_pressure_score": "60"} for m in matches for team in m.teams}
    return data, deliveries, metadata, prematch


def run_model(inputs, adaptive=False):
    balls, players = [], []
    result = analyze(*inputs, {}, set(), PerformanceConfig(adaptive_runtime=adaptive), balls.append, players.append)
    return balls, players, result


def card(absolute=0.5, lift=0.3, evidence=1.0, status="delivered"):
    return {
        "match_evidence_weight": evidence, "absolute_pressure_contribution": absolute,
        "pressure_contribution_vs_usual": lift, "status": status, "usual_baseline_support": 1.0,
    }


class ScoreboardTests(unittest.TestCase):
    def test_wide_no_ball_byes_and_free_hit_accounting(self):
        state = InningsState(target=10)
        wide = event(runs=0, extra_type="wide", primary=2)
        before = state.before(wide)
        self.assertEqual((before.runs, before.legal_balls), (0, 0))
        state.advance(wide)
        self.assertEqual((state.runs, state.legal_balls), (2, 0))
        no_ball = event(2, runs=4, extra_type="Overstep", primary=1)
        state.advance(no_ball)
        self.assertEqual((state.runs, state.legal_balls, state.free_hit), (7, 0, True))
        state.advance(event(3, runs=0, extra_type="wide", primary=1))
        self.assertTrue(state.free_hit)
        bye = event(4, runs=0, extra_type="byes", primary=2)
        self.assertTrue(state.before(bye).free_hit)
        state.advance(bye)
        self.assertEqual((state.runs, state.legal_balls, state.free_hit), (10, 1, False))
        with self.assertRaisesRegex(InputError, "innings ended"):
            state.before(event(5))

    def test_pressure_snapshot_precedes_wicket_and_new_batter(self):
        state = InningsState()
        wicket = event(runs=0, wicket=True, dismissed="a", dismissal="Bowled")
        before = state.before(wicket)
        state.advance(wicket)
        after = state.before(event(2, striker="a3"))
        self.assertEqual(before.wickets, 0)
        self.assertEqual(after.wickets, 1)
        self.assertEqual(after.recent_wickets, 1)
        self.assertEqual(after.striker_balls, 0)
        self.assertEqual(after.position_band, "top")

    def test_quality_preserves_clean_first_innings_when_chase_is_revised(self):
        match = model_data()[0].matches[0]
        checks = innings_quality(match, complete_match_events(), {"revised_target": "100"})
        self.assertEqual(checks[0]["status"], "included")
        self.assertEqual(checks[1]["status"], "withheld")
        self.assertIn("revised_target", checks[1]["reasons"])

    def test_short_first_innings_does_not_create_an_invented_chase_target(self):
        match = model_data()[0].matches[0]
        events = [row for row in complete_match_events() if row.innings == 2 or row.sequence <= 72]
        checks = innings_quality(match, events, {})
        self.assertTrue(all(row["status"] == "withheld" for row in checks))
        self.assertIsNone(checks[1]["target"])

    def test_illegal_clock_and_unreconciled_margin_are_explicit(self):
        match = model_data()[0].matches[0]
        events = complete_match_events()
        events[0] = replace(events[0], primary=1)
        checks = innings_quality(match, events, {})
        self.assertIn("unclassified_extra", checks[0]["reasons"])
        clean = innings_quality(replace(match, result="A Won by 20 Runs"), complete_match_events(), {})
        self.assertEqual(clean[0]["status"], "included")
        self.assertIn("reported_margin_not_reconciled", clean[1]["reasons"])


class LivePressureTests(unittest.TestCase):
    def test_tight_finish_exceeds_cruise_and_virtually_lost_chase(self):
        state = InningsState(target=181, runs=171, legal_balls=114, wickets=6)
        tight = state.before(event(115))
        cruise = replace(tight, runs=176, legal_balls=90, balls_remaining=30, wickets=2)
        lost = replace(tight, runs=110)
        self.assertGreater(live_pressure(tight, 175)[0], live_pressure(cruise, 175)[0])
        self.assertGreater(live_pressure(tight, 175)[0], live_pressure(lost, 175)[0])

    def test_first_innings_rebuilding_and_defending_are_different_tasks(self):
        snapshot = InningsState(runs=90, legal_balls=60, wickets=1).before(event(61))
        collapse = replace(snapshot, runs=45, wickets=5, recent_wickets=3)
        batting, bowling, _ = live_pressure(snapshot, 170)
        bat_collapse, bowl_collapse, _ = live_pressure(collapse, 170)
        self.assertGreater(bat_collapse, batting)
        self.assertLess(bowl_collapse, bowling)

    def test_layers_blend_instead_of_adding_two_near_hundred_scores(self):
        config = PerformanceConfig()
        team, personal, uncapped = combined_pressure(90, 40, 0.25, config)
        self.assertAlmostEqual(team, 57.5)
        self.assertAlmostEqual(personal, 71.875)
        self.assertEqual(personal, uncapped)
        self.assertEqual(combined_pressure(90, 100, 0.35, config)[1], 100)

    def test_par_updates_only_when_committed_to_history(self):
        history = ParHistory(PerformanceConfig())
        before = history.estimate("ipl", "v")
        history.update("ipl", "v", 230)
        after = history.estimate("ipl", "v")
        self.assertEqual(before[2:], (0, 0))
        self.assertGreater(after[0], before[0])
        self.assertEqual(after[2:], (1, 1))


class AttributionTests(unittest.TestCase):
    def score(self, delivery, role, baseline=None, snapshot=None):
        config = PerformanceConfig()
        if snapshot is None:
            snapshot = InningsState().before(delivery)
        return performance_observation(
            delivery, snapshot, role, "ipl:2024", 165, 165,
            baseline or HistoricalBaselines(config), 65, 80, "v4_known_signal_overlay", config)

    def test_wide_is_a_bowling_failure_not_a_batting_failure(self):
        delivery = event(runs=0, extra_type="wide", primary=2)
        batting, _, _ = self.score(delivery, "batting")
        bowling, _, _ = self.score(delivery, "bowling")
        self.assertIsNone(batting)
        self.assertEqual(bowling["attributed_runs"], 2)
        self.assertEqual(bowling["absolute_contribution"], -2)
        self.assertEqual(bowling["legal_opportunity"], 0)

    def test_stumping_on_wide_is_credited_without_counting_a_legal_ball(self):
        delivery = event(runs=0, extra_type="wide", primary=1,
                         wicket=True, dismissal="Stump", dismissed="a")
        batting, _, _ = self.score(delivery, "batting")
        bowling, _, _ = self.score(delivery, "bowling")
        self.assertEqual(batting["credited_wicket"], 1)
        self.assertEqual(bowling["credited_wicket"], 1)
        self.assertEqual(bowling["legal_opportunity"], 0)
        self.assertEqual(bowling["attributed_runs"], 1)
        self.assertLess(batting["absolute_contribution"], 0)
        self.assertGreater(bowling["absolute_contribution"], 0)

    def test_winning_wide_precludes_stumping_credit(self):
        delivery = event(120, 2, runs=0, extra_type="wide", primary=1,
                         wicket=True, dismissal="Stump", dismissed="b")
        snapshot = InningsState(target=121, runs=120, legal_balls=119).before(delivery)
        batting, _, _ = self.score(delivery, "batting", snapshot=snapshot)
        bowling, _, warnings = self.score(delivery, "bowling", snapshot=snapshot)
        self.assertIsNone(batting)
        self.assertEqual(bowling["credited_wicket"], 0)
        self.assertEqual(bowling["absolute_contribution"], -1)
        self.assertIn("winning_wide_precludes_stumping_credit", warnings)

    def test_no_ball_does_not_consume_a_ball_budget(self):
        delivery = event(runs=6, extra_type="Overstep", primary=1)
        batting, _, _ = self.score(delivery, "batting")
        bowling, _, _ = self.score(delivery, "bowling")
        self.assertEqual(batting["absolute_contribution"], 6)
        self.assertEqual(bowling["absolute_contribution"], -7)

    def test_byes_and_runouts_do_not_become_bowler_runs_or_wickets(self):
        bye, _, _ = self.score(event(runs=0, extra_type="byes", primary=4), "bowling")
        self.assertEqual(bye["attributed_runs"], 0)
        runout = event(runs=0, wicket=True, dismissed="a2", dismissal="RunOut")
        batting, _, warnings = self.score(runout, "batting")
        bowling, _, _ = self.score(runout, "bowling")
        self.assertEqual(bowling["credited_wicket"], 0)
        self.assertEqual(batting["credited_wicket"], 0)
        self.assertAlmostEqual(batting["absolute_contribution"], -batting["expected_runs_per_legal_ball"])
        self.assertIn("non_bowler_wicket_individual_blame_not_inferred", warnings)

    def test_unknown_secondary_extras_are_not_silently_charged_to_bowler(self):
        delivery = event(runs=0, extra_type="Overstep", primary=1, secondary=4)
        row, _, warnings = self.score(delivery, "bowling")
        self.assertIsNone(row)
        self.assertIn("secondary_extra_bowler_attribution_unverified", warnings)

    def test_credited_wicket_and_usual_standard_are_explicit(self):
        baseline = HistoricalBaselines(PerformanceConfig())
        for _ in range(40):
            baseline.update(("ipl:2024", 1, "batting", "powerplay", "top", "a", 1, 2, False, 0.8, True))
        row, _, _ = self.score(event(runs=1), "batting", baseline)
        self.assertGreater(row["usual_reference_before"], 0)
        self.assertAlmostEqual(row["contribution_vs_usual"],
                               row["absolute_contribution"] - row["usual_reference_before"])
        wicket = event(runs=0, wicket=True, dismissed="a", dismissal="Caught")
        batting, _, _ = self.score(wicket, "batting")
        bowling, _, _ = self.score(wicket, "bowling")
        self.assertLess(batting["absolute_contribution"], -5)
        self.assertGreater(bowling["absolute_contribution"], 5)

    def test_winning_boundary_does_not_earn_irrelevant_extra_run_credit(self):
        snapshot = InningsState(target=121, runs=120, legal_balls=119).before(event(120, 2))
        single, _, _ = self.score(event(120, 2, runs=1), "batting", snapshot=snapshot)
        six, _, _ = self.score(event(120, 2, runs=6), "batting", snapshot=snapshot)
        self.assertEqual(single["absolute_contribution"], six["absolute_contribution"])
        self.assertEqual(six["attributed_runs"], 6)
        self.assertEqual(six["useful_attributed_runs"], 1)
        self.assertGreater(single["absolute_contribution"], 0)
        baseline = HistoricalBaselines(PerformanceConfig())
        for _ in range(400):
            baseline.update(("ipl:2024", 2, "batting", "death", "top", "b", 1, 3, False, 2, True))
        maintained, _, _ = self.score(event(120, 2, runs=1), "batting", baseline, snapshot)
        self.assertGreaterEqual(maintained["contribution_vs_usual"], 0)

    def test_last_ball_wicket_has_no_additional_future_resource_reward(self):
        snapshot = InningsState(target=121, runs=120, legal_balls=119).before(event(120, 2))
        dot, _, _ = self.score(event(120, 2, runs=0), "bowling", snapshot=snapshot)
        wicket, _, _ = self.score(
            event(120, 2, runs=0, wicket=True, dismissed="b", dismissal="Bowled"),
            "bowling", snapshot=snapshot)
        self.assertEqual(wicket["wicket_value_before"], 0)
        self.assertEqual(dot["absolute_contribution"], wicket["absolute_contribution"])

    def test_parquet_preserves_nullable_fields_and_explicit_schema(self):
        row, _, _ = self.score(event(), "batting")
        row.update(
            match_date="2024-01-01", family="ipl", player="Player A",
            personal_pressure_upper=87.75, personal_pressure_uncapped=80.0,
            applied_responsibility_uplift=0.0, responsibility_source_flags="")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.parquet"
            writer = ParquetLedger(path, PLAYER_SCHEMA)
            writer.add(row)
            writer.close()
            self.assertEqual(pq.read_table(path).to_pylist()[0], row)


class ChronologyTests(unittest.TestCase):
    def test_adaptive_runtime_still_excludes_current_outcome_and_same_day_history(self):
        original = model_data(same_day=True)
        changed = copy.deepcopy(original)
        changed[1]["1"][19] = replace(changed[1]["1"][19], runs=4)
        changed[0].matches[0] = replace(changed[0].matches[0], result="A Won by 4 Runs")
        balls_a, players_a, _ = run_model(original, adaptive=True)
        balls_b, players_b, _ = run_model(changed, adaptive=True)
        self.assertEqual([r for r in players_a if r["match_id"] == "2"],
                         [r for r in players_b if r["match_id"] == "2"])
        choose = lambda rows: next(r for r in rows if r["match_id"] == "1" and r["innings"] == 1 and r["sequence"] == 20)
        before, after = choose(balls_a), choose(balls_b)
        self.assertTrue(before["adaptive_runtime"])
        for key in before:
            if key != "recorded_total_runs":
                self.assertEqual(before[key], after[key], key)

    def test_current_delivery_outcome_cannot_change_its_pressure_or_benchmark(self):
        original = model_data()
        changed = copy.deepcopy(original)
        changed[1]["1"][19] = replace(changed[1]["1"][19], runs=4)
        changed[0].matches[0] = replace(changed[0].matches[0], result="A Won by 4 Runs")
        balls_a, players_a, _ = run_model(original)
        balls_b, players_b, _ = run_model(changed)
        choose = lambda rows: next(r for r in rows if r["match_id"] == "1" and r["innings"] == 1 and r["sequence"] == 20)
        left, right = choose(balls_a), choose(balls_b)
        for key in left:
            if key != "recorded_total_runs":
                self.assertEqual(left[key], right[key], key)
        left, right = choose(players_a), choose(players_b)
        for key in ["personal_pressure", "usual_reference_before", "expected_runs_per_legal_ball", "wicket_value_before"]:
            self.assertEqual(left[key], right[key], key)
        self.assertNotEqual(left["absolute_contribution"], right["absolute_contribution"])

    def test_same_day_history_is_withheld_from_other_matches(self):
        original = model_data(same_day=True)
        changed = copy.deepcopy(original)
        changed[1]["1"][19] = replace(changed[1]["1"][19], runs=4)
        changed[0].matches[0] = replace(changed[0].matches[0], result="A Won by 4 Runs")
        balls_a, players_a, _ = run_model(original)
        balls_b, players_b, _ = run_model(changed)
        self.assertEqual([r for r in balls_a if r["match_id"] == "2"],
                         [r for r in balls_b if r["match_id"] == "2"])
        self.assertEqual([r for r in players_a if r["match_id"] == "2"],
                         [r for r in players_b if r["match_id"] == "2"])

    def test_future_results_do_not_change_prior_player_contributions(self):
        original = model_data()
        changed = copy.deepcopy(original)
        changed[1]["3"][19] = replace(changed[1]["3"][19], runs=4)
        changed[0].matches[2] = replace(changed[0].matches[2], result="A Won by 4 Runs")
        _, left, _ = run_model(original)
        _, right, _ = run_model(changed)
        self.assertEqual([r for r in left if r["match_id"] != "3"],
                         [r for r in right if r["match_id"] != "3"])


class ReliabilityTests(unittest.TestCase):
    def test_ordinary_deliveries_do_not_become_pressure_observations(self):
        config = PerformanceConfig()
        self.assertEqual(pressure_weight(49, config), 0)
        self.assertGreater(pressure_weight(80, config), 0)
        record = MatchCard("a", "batting", "1", "2024-01-01", "ipl", "A").evidence(config)
        self.assertIsNone(record["absolute_pressure_contribution"])
        self.assertIsNone(record["pressure_contribution_vs_usual"])
        self.assertEqual(record["status"], "insufficient_pressure_exposure")

    def test_pressure_weight_has_no_cliff_at_the_high_pressure_label(self):
        config = PerformanceConfig()
        below = pressure_weight(59.99, config)
        above = pressure_weight(60.01, config)
        self.assertGreater(below, 0)
        self.assertLess(abs(above - below), 0.001)
        self.assertEqual(pressure_weight(70, config), pressure_weight(70, replace(config, high_pressure=65)))

    def test_opportunity_free_appearances_do_not_improve_rating(self):
        config = PerformanceConfig()
        rows = [card() for _ in range(10)]
        before = rating_components(rows, config)
        after = rating_components(rows + [card(0, 0, 0, "insufficient_pressure_exposure")] * 100, config)
        self.assertEqual(before, after)

    def test_tiny_cameo_is_less_convincing_than_repeated_delivery(self):
        config = PerformanceConfig()
        cameo = rating_components([card(6, 6, 0.125)], config)[0]
        repeated = rating_components([card(0.5, 0.3) for _ in range(12)], config)[0]
        self.assertGreater(repeated, cameo)

    def test_more_negative_evidence_increases_confidence_in_failure_not_reward(self):
        config = PerformanceConfig()
        failure = card(-1, -1, 1, "underperformed")
        few = rating_components([failure], config)[0]
        many = rating_components([failure] * 20, config)[0]
        self.assertLess(many, few)
        self.assertLess(few, 50)

    def test_first_ball_pressure_failure_is_not_discarded_as_insufficient_sample(self):
        score = MatchCard("a", "batting", "1", "2024-01-01", "ipl", "A")
        score.events = score.legal_balls = score.pressure_events = score.pressure_legal_balls = 1
        score.weight = 0.5
        score.weighted_absolute = -5
        score.weighted_lift = -5
        score.whole_value = -10
        row = score.evidence(PerformanceConfig())
        self.assertEqual(row["status"], "underperformed")
        self.assertFalse(row["well_supported_pressure_match"])
        self.assertLess(rating_components([row], PerformanceConfig())[0], 50)

    def test_unproven_usual_baseline_cannot_create_resilience_points(self):
        one = {**card(0.5, 0), "usual_baseline_support": 0}
        two = {**card(0.5, 10), "usual_baseline_support": 0}
        self.assertEqual(rating_components([one], PerformanceConfig()),
                         rating_components([two], PerformanceConfig()))

    def test_late_recovery_does_not_erase_negative_whole_innings(self):
        score = MatchCard("a", "batting", "1", "2024-01-01", "ipl", "A")
        score.events = 20
        score.pressure_events = 5
        score.weight = 3
        score.weighted_absolute = 6
        score.weighted_lift = 6
        score.whole_value = -10
        self.assertEqual(score.evidence(PerformanceConfig())["status"], "mixed_recovery_negative_whole_contribution")


if __name__ == "__main__":
    unittest.main()
