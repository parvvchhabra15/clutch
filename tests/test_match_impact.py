import unittest
from clutch.match_impact import HighMatchCard, MatchImpactConfig, delivery_signal, gentle_match_adjustment, high_match_rating, stakes_weight
from clutch.live import PerformanceConfig

def state(index=0, runs=0, target=None, legal=True, total=0):
    return {'innings': 1 if target is None else 2, 'runs_before': runs, 'legal_balls_before': index, 'balls_remaining': 120 - index, 'target': target, 'legal_delivery': legal, 'recorded_total_runs': total}

def contribution(runs=1, innings=1, legal=1, wicket=0, price=10, rate=1.5):
    return {'innings': innings, 'legal_opportunity': legal, 'attributed_runs': runs, 'useful_attributed_runs': runs, 'credited_wicket': wicket, 'wicket_value_before': price, 'expected_credited_wicket_rate': 0.05, 'expected_runs_per_legal_ball': rate}

def card(role='batting', pressure=90, incomplete=False, uncertain=False):
    return HighMatchCard('p', 'Player', role, 'm', '2024-01-01', 'ipl', 'A', pressure, incomplete, uncertain)

def evidence(value, config=None):
    return value.evidence(config or MatchImpactConfig(), {'runs': 180, 'legal_balls': 120})

class HighMatchScopeTests(unittest.TestCase):

    def test_regular_matches_do_not_receive_the_new_channel(self):
        value = card(pressure=55)
        value.add(contribution(runs=6), state(total=6))
        row = evidence(value)
        self.assertEqual(row['status'], 'not_high_stakes')
        self.assertIsNone(row['net_team_impact'])
        self.assertEqual(high_match_rating([row], MatchImpactConfig()), (50, None, 0))

    def test_stakes_weight_is_continuous_and_uses_the_team_not_opponent(self):
        config = MatchImpactConfig()
        self.assertEqual(stakes_weight(22, config), 0)
        self.assertEqual(stakes_weight(70, config), 0)
        self.assertLess(stakes_weight(80, config), stakes_weight(92, config))
        self.assertLess(stakes_weight(70.01, config), 0.001)

    def test_live_ball_pressure_does_not_change_macro_contribution(self):
        left, right = (card(), card())
        row = contribution(runs=3)
        left.add({**row, 'personal_pressure': 20, 'pressure_weight': 0}, state(total=3))
        right.add({**row, 'personal_pressure': 100, 'pressure_weight': 1}, state(total=3))
        self.assertEqual(evidence(left), evidence(right))

    def test_incomplete_rates_are_withheld_not_guessed(self):
        value = card(role='bowling', incomplete=True)
        value.add(contribution(runs=3), state(total=3))
        row = evidence(value)
        self.assertEqual(row['status'], 'withheld_incomplete_attribution')
        self.assertIsNone(row['bowling_economy'])
        self.assertIsNone(row['net_team_impact'])

class PaceAndEconomyTests(unittest.TestCase):

    def test_slow_chase_contribution_is_negative_despite_runs_scored(self):
        value = card()
        for i in range(12):
            value.add(contribution(runs=1, innings=2), state(index=90 + i, runs=120 + i, target=181, total=1))
        row = evidence(value)
        self.assertEqual(row['legal_ball_strike_rate'], 100)
        self.assertLess(row['raw_pace_surplus'], 0)
        self.assertLess(row['net_team_impact'], 0)

    def test_low_required_rate_does_not_demand_unnecessary_acceleration(self):
        value = card()
        for i in range(6):
            value.add(contribution(runs=1, innings=2), state(index=90 + i, runs=165 + i, target=181, total=1))
        self.assertGreater(evidence(value)['net_team_impact'], 0)

    def test_economical_wicketless_spell_helps_and_expensive_wicket_does_not_hide_damage(self):
        economical, expensive = (card('bowling'), card('bowling'))
        for i in range(6):
            economical.add(contribution(runs=0 if i < 3 else 1), state(i, total=1))
            expensive.add(contribution(runs=3, wicket=int(i == 5)), state(i, total=3))
        good, bad = (evidence(economical), evidence(expensive))
        self.assertEqual(good['credited_wickets'], 0)
        self.assertEqual(good['bowling_economy'], 3)
        self.assertGreater(good['net_team_impact'], 0)
        self.assertEqual(bad['credited_wickets'], 1)
        self.assertEqual(bad['bowling_economy'], 18)
        self.assertLess(bad['net_team_impact'], 0)

    def test_defending_budget_tightens_an_otherwise_average_economy(self):
        value = card('bowling')
        for i in range(5):
            value.add(contribution(runs=1, innings=2), state(index=114 + i, runs=176 + i, target=181, total=1))
        row = evidence(value)
        self.assertLess(row['reference_bowling_economy'], 6)
        self.assertLess(row['net_team_impact'], 0)

    def test_over_budget_is_not_rewritten_by_later_partner_progress(self):
        left, right = (card(), card())
        for i in range(6):
            a = state(index=90 + i, runs=140 + i, target=181, total=1)
            b = dict(a)
            if i:
                b['runs_before'] += 10
            row = contribution(runs=1, innings=2)
            left.add(row, a)
            right.add(row, b)
        self.assertEqual(evidence(left)['bounded_pace_impact'], evidence(right)['bounded_pace_impact'])

    def test_impossible_inherited_chase_does_not_create_unlimited_penalty(self):
        value = card()
        value.add(contribution(runs=4, innings=2), state(index=119, runs=100, target=181, total=4))
        row = evidence(value)
        self.assertLess(row['raw_pace_surplus'], -50)
        self.assertGreaterEqual(row['bounded_pace_impact'], -row['context_run_budget'])
        self.assertTrue(row['inherited_chase_shortfall_capped'])

    def test_run_share_and_rrr_endpoints_are_not_extra_points(self):
        value = card()
        value.add(contribution(runs=4), state(total=4))
        first = value.evidence(MatchImpactConfig(), {'runs': 100, 'legal_balls': 120})
        second = value.evidence(MatchImpactConfig(), {'runs': 200, 'legal_balls': 120})
        self.assertNotEqual(first['batter_share_of_team_runs'], second['batter_share_of_team_runs'])
        self.assertEqual(first['net_team_impact'], second['net_team_impact'])

class AggregationTests(unittest.TestCase):

    def test_gentle_marking_uses_individual_delivery_not_blanket_win_or_loss(self):
        config = MatchImpactConfig(gentle_adjustment=True)
        self.assertEqual(delivery_signal(-10, True, True, config), -1)
        self.assertEqual(delivery_signal(15, False, False, config), 0)
        self.assertEqual(delivery_signal(15, True, False, config), 1)
        self.assertEqual(delivery_signal(1, True, True, config), 1)
        self.assertEqual(delivery_signal(-1, True, True, config), 0)
        self.assertEqual(delivery_signal(0, True, False, config), 0)
        cards = [{'net_team_impact': -10, 'high_match_weight': 1, 'gentle_delivery_signal': -1}] * 100
        adjustment = gentle_match_adjustment(cards, config)
        self.assertLess(adjustment, 0)
        self.assertLess(abs(config.match_share * adjustment), 0.6)

    def test_few_matches_are_shrunk_and_negative_trends_are_not_discarded(self):
        config = MatchImpactConfig()
        row = {'net_team_impact': -20, 'high_match_weight': 0.8}
        one, _, support_one = high_match_rating([row], config)
        many, _, support_many = high_match_rating([row] * 8, config)
        self.assertLess(many, one)
        self.assertGreater(support_many, support_one)

    def test_batting_expected_wicket_reward_is_bounded_to_one_dismissal(self):
        value = card()
        for i in range(100):
            value.add(contribution(runs=1, price=10), state(index=i, total=1))
        row = evidence(value)
        self.assertLessEqual(row['expected_wicket_resource_value'], 10)
        unknown = card(uncertain=True)
        unknown.add(contribution(runs=1), state(total=1))
        self.assertEqual(evidence(unknown)['secondary_wicket_impact'], 0)
if __name__ == '__main__':
    unittest.main()
