import copy
import csv
import itertools
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch
from clutch.calibration import StageCalibration, calibrate
from clutch.data import Dataset, Match, Performance, load_dataset
from clutch.engine import applicable, player_adjustment, score_dataset
from clutch.inputs import InputError, Policy, digest, load_policy, numeric
from clutch.qualification import feasible_finish, qualification_bounds
from clutch.rules import RuleBook, series_stakes
from clutch.statistics import Accumulated, StatisticalHistory
ROOT = Path(__file__).resolve().parents[1]

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def test_policy():
    return Policy({('senior_bilateral_t20i', 'ALL_COVERED'): 0.7, ('ipl', 'ALL_COVERED'): 1.0}, {('A', 'B'): 0.8}, 0.25, {'defending_champion_expectation': 0.9, 'strong_favourite_expectation': 0.9, 'documented_home_expectations': 0.7, 'documented_historical_team_expectations': 0.7}, 0.25, 0.1, 0.25, 0.35, {'competition_weight': 30, 'result_consequences_weight': 40, 'expectation_weight': 10, 'recent_underperformance_weight': 5, 'rivalry_weight': 15}, {})

def toy_data(same_day=False):
    teams = {'A': 'Team A', 'B': 'Team B'}
    players = {f'{t}{i}': f'Player {t}{i}' for t in teams for i in range(11)}
    matches = [Match('1', 'C', date(2022, 1, 1), 1, 'League', 'A', 'B', 'A'), Match('2', 'C', date(2022, 1, 1 if same_day else 2), 2, 'League', 'A', 'B', 'B'), Match('3', 'C', date(2022, 1, 3), 3, 'League', 'A', 'B', 'A')]
    starting = {(m.id, t): [f'{t}{i}' for i in range(11)] for m in matches for t in teams}
    captains = {(m.id, t): f'{t}0' for m in matches for t in teams}
    performances = {m.id: {'A0': Performance(batting_balls=20, batting_runs=30, batting_outs=1, appearances=1), 'B0': Performance(bowling_balls=20, bowling_runs=30, bowling_wickets=1, appearances=1)} for m in matches}
    comps = {'C': {'comp_name': 'Test series', 'season': '2022', 'competition_family_id': 'senior_bilateral_t20i'}}
    return Dataset(matches, comps, teams, players, starting, captains, performances)

def qualification_data(recovery_margin=False, already_secured=False):
    data = toy_data()
    data.teams['C'] = 'Team C'
    data.competitions['C']['competition_family_id'] = 'asia_cup_qualifying'
    prior = [('A', 'B', 'A'), ('A', 'C', 'C'), ('B', 'C', 'B')]
    if recovery_margin or already_secured:
        prior = [('A', 'B', 'A'), ('A', 'C', 'A'), ('A', 'B', 'A'), ('B', 'C', 'B')]
    if already_secured:
        prior.append(('A', 'C', 'A'))
    fixtures = prior + [('A', 'B', 'B'), ('A', 'C', 'A'), ('B', 'C', 'B')]
    data.matches = [Match(str(i), 'C', date(2022, 1, i), i, 'League', home, away, winner) for i, (home, away, winner) in enumerate(fixtures, 1)]
    return (data, data.matches[len(prior)])

def rulebook(data, folder, extra=None):
    reference = {'fixture_replacements': [], 'series_lengths': {'C': 3}, 'prior_champions': {}, 'rounds': {}, 'vice_captains': [], 'home_expectation_evidence': []}
    if extra:
        reference.update(extra)
    path = Path(folder) / 'rules.json'
    path.write_text(json.dumps(reference), encoding='utf-8')
    return RuleBook(data, path)

class QualificationTests(unittest.TestCase):

    @staticmethod
    def brute(points, wins, fixtures, target, places, use_wins):
        possible, guaranteed = (False, True)
        for outcomes in itertools.product([0, 1, 2], repeat=len(fixtures)):
            p, w = (dict(points), dict(wins))
            for (home, away), result in zip(fixtures, outcomes):
                p[home] += result
                p[away] += 2 - result
                w[home] += int(result == 2)
                w[away] += int(result == 0)
            own = (p[target], w[target] if use_wins else 0)
            rivals = [(p[t], w[t] if use_wins else 0) for t in p if t != target]
            possible |= sum((value > own for value in rivals)) < places
            guaranteed &= sum((value >= own for value in rivals)) < places
        return (possible, guaranteed)

    def test_solver_matches_exhaustive_small_tournaments(self):
        fixtures = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        for points, wins in [({'A': 0, 'B': 0, 'C': 0}, {'A': 0, 'B': 0, 'C': 0}), ({'A': 2, 'B': 3, 'C': 4}, {'A': 1, 'B': 1, 'C': 2}), ({'A': 4, 'B': 4, 'C': 1}, {'A': 2, 'B': 1, 'C': 0})]:
            for use_wins, places in itertools.product([False, True], [1, 2]):
                with self.subTest(points=points, use_wins=use_wins, places=places):
                    expected = self.brute(points, wins, fixtures, 'A', places, use_wins)
                    result = qualification_bounds(points, wins, fixtures, 'A', places, use_wins)
                    self.assertEqual((result.possible, result.guaranteed), expected)

    def test_equal_points_are_not_arbitrarily_ranked_by_team_name(self):
        result = qualification_bounds({'A': 4, 'B': 4}, {'A': 2, 'B': 2}, [], 'A', 1)
        self.assertTrue(result.possible)
        self.assertFalse(result.guaranteed)

    def test_wins_tiebreak_precedes_unknown_nrr_when_enabled(self):
        result = qualification_bounds({'A': 4, 'B': 4}, {'A': 2, 'B': 1}, [], 'A', 1, True)
        self.assertTrue(result.guaranteed)

    def test_rejects_foreign_fixture(self):
        with self.assertRaises(ValueError):
            feasible_finish({'A': 0, 'B': 0}, {'A': 0, 'B': 0}, [('A', 'C')], 'A', 1, qualify=True)

class StakesTests(unittest.TestCase):

    def test_series_consequences_are_team_specific(self):
        self.assertEqual(series_stakes(1, 0, 1).score, 0.6)
        self.assertEqual(series_stakes(0, 1, 1).score, 0.9)
        self.assertEqual(series_stakes(1, 1, 0).score, 1)
        self.assertEqual(series_stakes(2, 0, 0).score, 0.1)

    def test_replacement_is_announcement_gated(self):
        data = toy_data()
        data.matches = [Match('old', 'C', date(2022, 1, 1), 1, 'League', 'A', 'B', '', 'No Result'), Match('new', 'C', date(2022, 1, 5), 2, 'League', 'A', 'B', 'A')]
        with tempfile.TemporaryDirectory() as folder:
            rules = rulebook(data, folder, {'fixture_replacements': [{'original_match_id': 'old', 'replacement_match_id': 'new', 'effective_date': '2022-01-03'}]})
            self.assertEqual(rules.inactive_ids(date(2022, 1, 2)), {'new'})
            self.assertEqual(rules.inactive_ids(date(2022, 1, 3)), {'old'})

    def test_qualifier_paths_use_structure_not_absolute_match_71(self):
        data = toy_data()
        data.competitions['C']['competition_family_id'] = 'ipl'
        data.matches = [Match('q1', 'C', date(2022, 1, 1), 72, 'Qualifier', 'A', 'B', 'A'), Match('q2', 'C', date(2022, 1, 3), 74, 'Qualifier', 'A', 'B', 'B')]
        with tempfile.TemporaryDirectory() as folder:
            rules = rulebook(data, folder)
            self.assertEqual(rules.assess(data.matches[0], 'A')[0].score, 0.6)
            self.assertEqual(rules.assess(data.matches[1], 'A')[0].score, 0.9)

    def test_standings_ignore_same_day_and_future_results(self):
        data = toy_data(same_day=True)
        table = RuleBook.table(data.matches, date(2022, 1, 1))
        self.assertEqual(table['A']['points'], 0)
        self.assertEqual(table['B']['played'], 0)
        next_day = RuleBook.table(data.matches, date(2022, 1, 2))
        self.assertEqual(next_day['A']['points'], 2)
        self.assertEqual(next_day['B']['points'], 2)

    def test_late_qualification_detects_loss_of_control_not_false_elimination(self):
        data, match = qualification_data()
        with tempfile.TemporaryDirectory() as folder:
            stakes, _ = rulebook(data, folder).assess(match, 'A')
        self.assertEqual(stakes.qualification_context, 'late_loss_of_control')
        self.assertTrue(stakes.possible_after_loss)
        self.assertFalse(stakes.guaranteed_if_lose_then_win_out)
        self.assertTrue(stakes.guaranteed_if_win_out)
        self.assertEqual(stakes.remaining_matches, 2)
        self.assertEqual(stakes.points_before_match, 2)
        self.assertEqual(stakes.max_points_if_win_out, 6)
        self.assertEqual(stakes.score, 0.6)

    def test_late_fixture_with_controlled_recovery_is_not_near_elimination(self):
        data, match = qualification_data(recovery_margin=True)
        with tempfile.TemporaryDirectory() as folder:
            stakes, _ = rulebook(data, folder).assess(match, 'A')
        self.assertEqual(stakes.remaining_matches, 2)
        self.assertTrue(stakes.guaranteed_if_lose_then_win_out)
        self.assertEqual(stakes.qualification_context, '')

    def test_group_opener_secured_and_eliminated_teams_do_not_receive_late_boost(self):
        data, _ = qualification_data()
        with tempfile.TemporaryDirectory() as folder:
            opener, _ = rulebook(data, folder).assess(data.matches[0], 'A')
            settled, match = qualification_data(already_secured=True)
            rules = rulebook(settled, folder)
            secured, _ = rules.assess(match, 'A')
            eliminated, _ = rules.assess(match, 'B')
        self.assertEqual(opener.qualification_context, '')
        for stakes in [secured, eliminated]:
            self.assertEqual(stakes.qualification_context, '')
            self.assertEqual(stakes.score, 0.1)

    def test_qualification_control_does_not_read_current_or_future_results(self):
        data, match = qualification_data()
        altered = copy.deepcopy(data)
        altered.matches = [replace(m, winner=m.away if m.winner == m.home else m.home) if m.day >= match.day else m for m in altered.matches]
        with tempfile.TemporaryDirectory() as folder:
            original, _ = rulebook(data, folder).assess(match, 'A')
            changed, _ = rulebook(altered, folder).assess(match, 'A')
        self.assertEqual(original, changed)

class OppositionTests(unittest.TestCase):

    def test_cold_start_and_sparse_head_to_head_are_neutral_or_shrunk(self):
        history = StatisticalHistory()
        cold = history.opposition_context('A', 'B')
        self.assertEqual((cold.strength_signal, cold.head_to_head_signal), (0, 0))
        self.assertIsNone(cold.head_to_head_result_rate)
        match = toy_data().matches[0]
        history.update_result(match, 0.5)
        favoured = history.opposition_context('A', 'B')
        other_side = history.opposition_context('B', 'A')
        self.assertEqual(favoured.head_to_head_matches, 1)
        self.assertAlmostEqual(favoured.head_to_head_signal, -2 / 11)
        self.assertLess(favoured.strength_signal, 0)
        self.assertAlmostEqual(other_side.strength_signal, -favoured.strength_signal)
        self.assertAlmostEqual(other_side.head_to_head_signal, -favoured.head_to_head_signal)

    def test_head_to_head_does_not_recount_expected_wins(self):
        history = StatisticalHistory()
        match = toy_data().matches[0]
        for number, winner in enumerate(['A', 'A', 'A', 'A', 'B']):
            history.update_result(replace(match, id=str(number), winner=winner), 0.8)
        context = history.opposition_context('A', 'B')
        self.assertAlmostEqual(context.head_to_head_result_rate, 0.8)
        self.assertAlmostEqual(context.head_to_head_expected_rate, 0.8)
        self.assertAlmostEqual(context.head_to_head_signal, 0)

    def test_no_result_is_excluded_and_ties_are_half_results(self):
        history = StatisticalHistory()
        match = toy_data().matches[0]
        history.update_result(replace(match, winner='', result='No Result'), 0.5)
        self.assertEqual(history.opposition_context('A', 'B').head_to_head_matches, 0)
        history.update_result(replace(match, winner='', result='Match tied'), 0.5)
        context = history.opposition_context('A', 'B')
        self.assertEqual(context.head_to_head_matches, 1)
        self.assertEqual(context.head_to_head_result_rate, 0.5)
        self.assertEqual(context.head_to_head_signal, 0)

class DependencyTests(unittest.TestCase):

    def test_equal_unseen_players_have_no_artificial_relative_dependency(self):
        history = StatisticalHistory()
        rows = history.dependency(['a', 'b', 'c'], 'batting')
        self.assertTrue(all((row['ability_rank'] == 1 for row in rows)))
        self.assertTrue(all((abs(row['dependency']) < 1e-12 for row in rows)))
        self.assertTrue(all((row['history_status'] == 'cold_start' for row in rows)))

    def test_dependency_changes_when_alternatives_improve(self):
        history = StatisticalHistory()
        history.players['a'] = Accumulated(batting_balls=200, batting_runs=400, batting_outs=5, appearances=10, batting_expected_runs=250, batting_expected_outs=10)
        history.players['b'] = Accumulated(batting_balls=200, batting_runs=150, batting_outs=15, appearances=10, batting_expected_runs=250, batting_expected_outs=10)
        before = history.dependency(['a', 'b'], 'batting')[0]['dependency']
        history.players['b'] = copy.deepcopy(history.players['a'])
        after = history.dependency(['a', 'b'], 'batting')[0]['dependency']
        self.assertGreater(before, after)
        self.assertAlmostEqual(after, 0)

    def test_voided_match_statistics_can_be_withdrawn(self):
        history = StatisticalHistory()
        match = toy_data().matches[0]
        history.update_performance(match, 'x', {'a': Performance(batting_balls=10, batting_runs=20, appearances=1)}, history.pool_rates('x'))
        self.assertEqual(history.players['a'].batting_runs, 20)
        history.withdraw_performance(match.id)
        self.assertEqual(history.players['a'].batting_runs, 0)
        self.assertEqual(history.pools['x'].batting_balls, 0)

    def test_individual_cap_does_not_change_team_score(self):
        base = 60.0
        score, raw, uplift = player_adjustment(base, 1, 0.25, 0.25, 0.35)
        self.assertEqual(base, 60)
        self.assertAlmostEqual(uplift, 0.35)
        self.assertEqual(score, 81)
        self.assertEqual(raw, 81)
        self.assertEqual(player_adjustment(80, 1, 0.25, 0.25, 0.35)[:2], (100, 108))

    def test_vice_captain_and_dependency_are_capped_together(self):
        score, _, uplift = player_adjustment(60, 1, 0.1, 0.25, 0.35)
        self.assertAlmostEqual(uplift, 0.35)
        self.assertEqual(score, 81)

class LeakageTests(unittest.TestCase):

    def run_model(self, data):
        with tempfile.TemporaryDirectory() as folder:
            rules = rulebook(data, folder)
            return score_dataset(data, test_policy(), rules, calibration=StageCalibration(qualification_aware=True, playoff_aware=True))

    def test_head_to_head_is_snapshotted_before_each_date(self):
        result = self.run_model(toy_data())
        rows = [row for row in result.teams if row['team_id'] == 'A']
        self.assertEqual([row['head_to_head_prior_matches'] for row in rows], [0, 1, 2])
        same_day = self.run_model(toy_data(same_day=True))
        first_date = [row for row in same_day.teams if row['match_id'] in {'1', '2'}]
        self.assertTrue(all((row['head_to_head_prior_matches'] == 0 for row in first_date)))
        self.assertTrue(all((row['head_to_head_points'] == 0 for row in first_date)))

    def test_current_match_outcome_and_performance_do_not_change_pre_match_scores(self):
        baseline = toy_data()
        altered = copy.deepcopy(baseline)
        altered.matches[1] = replace(altered.matches[1], winner='A')
        altered.performances['2']['A0'].batting_runs = 600
        first, second = (self.run_model(baseline), self.run_model(altered))
        self.assertEqual([r for r in first.teams if r['match_id'] == '2'], [r for r in second.teams if r['match_id'] == '2'])
        self.assertEqual([r for r in first.players if r['match_id'] == '2'], [r for r in second.players if r['match_id'] == '2'])
        self.assertNotEqual([r for r in first.players if r['match_id'] == '3'], [r for r in second.players if r['match_id'] == '3'])

    def test_same_day_results_are_all_withheld(self):
        baseline = toy_data(same_day=True)
        altered = copy.deepcopy(baseline)
        altered.matches[0] = replace(altered.matches[0], winner='B')
        altered.performances['1']['A0'].batting_runs = 600
        first, second = (self.run_model(baseline), self.run_model(altered))
        self.assertEqual([r for r in first.players if r['match_id'] == '2'], [r for r in second.players if r['match_id'] == '2'])

    def test_future_results_and_statistics_do_not_change_earlier_outputs(self):
        baseline = toy_data()
        altered = copy.deepcopy(baseline)
        altered.matches[2] = replace(altered.matches[2], winner='B')
        altered.performances['3']['A0'].batting_runs = 600
        first, second = (self.run_model(baseline), self.run_model(altered))
        self.assertEqual([r for r in first.teams if r['match_id'] != '3'], [r for r in second.teams if r['match_id'] != '3'])
        self.assertEqual([r for r in first.players if r['match_id'] != '3'], [r for r in second.players if r['match_id'] != '3'])

    def test_verified_vice_captain_is_applied_only_after_announcement(self):
        data = toy_data()
        entry = {'team_id': 'A', 'player_id': 'A1', 'comp_id': 'C', 'known_from': '2022-01-02', 'valid_from': '2022-01-02', 'valid_to': '2022-01-03'}
        with tempfile.TemporaryDirectory() as folder:
            rules = rulebook(data, folder, {'vice_captains': [entry]})
            result = score_dataset(data, test_policy(), rules)
        first = next((r for r in result.players if r['player_id'] == 'A1' and r['match_id'] == '1'))
        second = next((r for r in result.players if r['player_id'] == 'A1' and r['match_id'] == '2'))
        self.assertEqual(first['leadership_uplift_0_1'], 0)
        self.assertEqual(second['leadership_role'], 'vice_captain')
        self.assertEqual(second['leadership_uplift_0_1'], 0.1)

    def test_ambiguous_xi_withholds_dependency_instead_of_guessing(self):
        data = toy_data()
        data.starting['2', 'A'] = data.starting['2', 'A'][:-1]
        result = self.run_model(data)
        rows = [r for r in result.players if r['match_id'] == '2' and r['team_id'] == 'A']
        self.assertTrue(all((r['dependency_0_1'] is None for r in rows)))
        self.assertTrue(all((r['player_pressure_score'] is None for r in rows)))
        self.assertTrue(all((r['player_context_upper'] is not None for r in rows)))

    def test_context_announcements_do_not_apply_early(self):
        match = toy_data().matches[0]
        entry = {'team_id': 'A', 'known_from': '2022-01-02', 'valid_from': '2022-01-01'}
        self.assertFalse(applicable(entry, match, 'A'))
        self.assertTrue(applicable(entry, replace(match, day=date(2022, 1, 2)), 'A'))
if __name__ == '__main__':
    unittest.main()
