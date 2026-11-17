import unittest
from dataclasses import replace
from datetime import date

from clutch.data import Dataset, Match
from clutch.live import Delivery, InningsState, PerformanceConfig, live_pressure
from clutch.runtime import RuntimeState, adaptive_pressure, bowling_time_capacity, runtime_responsibility


def delivery(runs=0, legal=True, wicket=False, dismissed=""):
    return Delivery("m", 2, 1, 1, "A", "B", "p0", "p1", "p0", runs,
                    0 if legal else 1, 0, "None" if legal else "wide",
                    wicket, dismissed, "Caught" if wicket else "")


def dependency_fixture():
    lineup = [f"p{i}" for i in range(11)]
    match = Match("m", "c", date(2024, 1, 1), 1, "League", "A", "B")
    data = Dataset([match], {}, {"A": "A", "B": "B"}, {p: p for p in lineup},
                   {("m", "A"): lineup}, {}, {})
    records = {}
    for player in lineup:
        for role in ["batting", "bowling"]:
            records["m", "A", player, role] = {
                "applied_combined_uplift_0_1": "0.05", "dependency_0_1": "0.2",
                "leadership_uplift_0_1": "0", "projected_contribution": "1",
                "quality_index": "1", "expected_legal_opportunities": "18",
                "prior_legal_balls": "100",
            }
    records["m", "A", "p0", "batting"]["projected_contribution"] = "6"
    records["m", "A", "p1", "batting"]["projected_contribution"] = "5"
    return data, match, records


class PracticalChaseTests(unittest.TestCase):
    def test_thirty_required_per_over_is_not_a_universal_hopelessness_cutoff(self):
        config = PerformanceConfig(adaptive_runtime=True)
        last_over = InningsState(target=201, runs=171, legal_balls=114, wickets=5).before(delivery())
        long_chase = replace(last_over, runs=0, target=300, legal_balls=60, balls_remaining=60)
        short_bat, short_bowl, _, short_info = adaptive_pressure(last_over, 180, RuntimeState(), config)
        long_bat, long_bowl, _, long_info = adaptive_pressure(long_chase, 180, RuntimeState(), config)
        self.assertAlmostEqual(short_info["required_runs_per_over"], 30)
        self.assertAlmostEqual(long_info["required_runs_per_over"], 30)
        self.assertGreater(short_bat, long_bat + 30)
        self.assertGreater(short_bowl, long_bowl + 20)
        self.assertGreater(short_info["practical_liveness"], 0.9)
        self.assertLess(long_info["practical_liveness"], 0.1)
        self.assertGreater(short_bat, live_pressure(last_over, 180)[0])

    def test_twenty_five_per_over_for_two_overs_retains_burst_pressure(self):
        snapshot = InningsState(target=201, runs=151, legal_balls=108, wickets=5).before(delivery())
        batting, _, _, info = adaptive_pressure(snapshot, 180, RuntimeState(),
                                                PerformanceConfig(adaptive_runtime=True))
        self.assertTrue(info["short_burst_context"])
        self.assertGreater(batting, 75)

    def test_free_hit_and_partnership_change_pressure_without_exceeding_budgets(self):
        snapshot = InningsState(runs=80, legal_balls=60, wickets=6).before(delivery())
        config = PerformanceConfig(adaptive_runtime=True)
        normal = adaptive_pressure(snapshot, 170, RuntimeState(), config)
        free = adaptive_pressure(replace(snapshot, free_hit=True), 170, RuntimeState(), config)
        self.assertLess(free[0], normal[0])
        self.assertGreater(free[1], normal[1])
        pair = RuntimeState(partnership_runs=80, partnership_balls=40)
        self.assertGreaterEqual(adaptive_pressure(snapshot, 170, pair, config)[1], normal[1])


class RuntimeResourceTests(unittest.TestCase):
    def test_sequences_and_resources_update_only_after_the_delivery(self):
        runtime = RuntimeState()
        snapshot = InningsState().before(delivery())
        runtime.advance(delivery(), snapshot)
        runtime.advance(delivery(), replace(snapshot, legal_balls=1))
        self.assertEqual(runtime.before()["prior_dot_streak"], 2)
        self.assertEqual(runtime.bowler_balls["p0"], 2)
        runtime.advance(delivery(legal=False), replace(snapshot, legal_balls=2))
        self.assertEqual(runtime.bowler_balls["p0"], 2)
        self.assertEqual(runtime.before()["prior_dot_streak"], 0)
        runtime.advance(delivery(runs=2, wicket=True, dismissed="p1"), replace(snapshot, legal_balls=2))
        self.assertIn("p1", runtime.dismissed)
        self.assertEqual(runtime.before()["partnership_runs_before"], 0)
        self.assertEqual(runtime.before()["partnership_balls_before"], 0)

    def test_dependency_responds_to_losing_a_strong_remaining_partner(self):
        data, match, records = dependency_fixture()
        snapshot = InningsState().before(delivery())
        runtime = RuntimeState()
        config = PerformanceConfig(adaptive_runtime=True)
        before = runtime_responsibility(data, records, match, "A", "p0", "batting", snapshot, runtime, config)
        runtime.dismissed.add("p1")
        after = runtime_responsibility(data, records, match, "A", "p0", "batting", snapshot, runtime, config)
        self.assertGreater(after["dependency"], before["dependency"])
        self.assertEqual(after["remaining_options"], 10)
        self.assertLessEqual(after["uplift"], 0.35)

    def test_bowling_capacity_respects_quota_and_nonconsecutive_overs(self):
        snapshot = InningsState(legal_balls=108).before(delivery())
        runtime = RuntimeState()
        self.assertEqual(bowling_time_capacity("p0", "p0", snapshot, runtime), 6)
        self.assertEqual(bowling_time_capacity("p1", "p0", snapshot, runtime), 6)
        runtime.bowler_balls["p0"] = 24
        self.assertEqual(bowling_time_capacity("p0", "p0", snapshot, runtime), 0)
        data, match, records = dependency_fixture()
        final = replace(snapshot, legal_balls=114, balls_remaining=6)
        result = runtime_responsibility(data, records, match, "A", "p1", "bowling",
                                        final, RuntimeState(), PerformanceConfig(adaptive_runtime=True))
        self.assertEqual(result["remaining_options"], 1)
        self.assertEqual(result["dependency"], 1)

    def test_unknown_dismissal_or_xi_uses_flagged_fallback(self):
        data, match, records = dependency_fixture()
        snapshot = InningsState().before(delivery())
        result = runtime_responsibility(data, records, match, "A", "p0", "batting", snapshot,
                                        RuntimeState(unknown_dismissal=True),
                                        PerformanceConfig(adaptive_runtime=True))
        self.assertIn("fallback", result["status"])
        self.assertEqual(result["uplift"], 0.05)
        data.starting["m", "A"] = data.starting["m", "A"][:-1]
        result = runtime_responsibility(data, records, match, "A", "p0", "bowling", snapshot,
                                        RuntimeState(), PerformanceConfig(adaptive_runtime=True))
        self.assertIn("unverified_xi", result["status"])


if __name__ == "__main__":
    unittest.main()
