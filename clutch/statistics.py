import math
from collections import defaultdict, deque
from dataclasses import dataclass, fields

from .data import Match, Performance


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


@dataclass
class Accumulated:
    batting_balls: float = 0
    batting_runs: float = 0
    batting_outs: float = 0
    bowling_balls: float = 0
    bowling_runs: float = 0
    bowling_wickets: float = 0
    appearances: float = 0
    batting_expected_runs: float = 0
    batting_expected_outs: float = 0
    bowling_expected_runs: float = 0
    bowling_expected_wickets: float = 0

    def add(self, other: "Accumulated", direction: int = 1) -> None:
        for item in fields(self):
            value = getattr(self, item.name) + direction * getattr(other, item.name)
            if value < -1e-7:
                raise ValueError(f"Negative accumulated statistic: {item.name}")
            setattr(self, item.name, max(0.0, value))


@dataclass(frozen=True)
class OppositionContext:
    strength_signal: float
    strength_support: float
    opponent_prior_results: int
    head_to_head_signal: float
    head_to_head_matches: int
    head_to_head_result_rate: float | None
    head_to_head_expected_rate: float | None
    head_to_head_support: float


class StatisticalHistory:
    def __init__(self):
        self.players = defaultdict(Accumulated)
        self.pools = defaultdict(Accumulated)
        self.contributions: dict[str, tuple[str, dict[str, Accumulated]]] = {}
        self.elo = defaultdict(lambda: 1500.0)
        self.form = defaultdict(lambda: deque(maxlen=5))
        self.results = defaultdict(list)
        self.head_to_head = defaultdict(list)
        self.prior_balls = 60.0
        self.prior_appearances = 5.0
        self.opposition_prior_matches = 10.0
        self.head_to_head_prior_matches = 10.0
        self.opposition_full_signal_gap = 0.25

    def win_probability(self, team: str, opponent: str) -> float:
        return 1 / (1 + 10 ** ((self.elo[opponent] - self.elo[team]) / 400))

    def underperformance(self, team: str) -> float:
        history = self.form[team]
        return clip(sum(expected - actual for expected, actual in history) / (len(history) + 3))

    def historical_strength(self, team: str) -> float:
        # Exclude the recent-form window so this is not another losing-streak bonus.
        earlier = self.results[team][:-5]
        if len(earlier) < 20:
            return 0.0
        win_rate = (sum(earlier) + 5) / (len(earlier) + 10)
        return clip((win_rate - 0.5) / 0.2)

    def opposition_context(self, team: str, opponent: str) -> OppositionContext:
        supported_matches = min(len(self.results[team]), len(self.results[opponent]))
        strength_support = supported_matches / (supported_matches + self.opposition_prior_matches)
        strength = clip(
            (0.5 - self.win_probability(team, opponent)) / self.opposition_full_signal_gap, -1, 1)
        history = self.head_to_head[team, opponent]
        n = len(history)
        # Use excess results over the expectations frozen before those matches.
        # Raw head-to-head wins would recount strength already reflected in Elo.
        surplus = sum(actual - expected for expected, actual in history)
        matchup = clip(
            -surplus / (n + self.head_to_head_prior_matches) / self.opposition_full_signal_gap, -1, 1)
        return OppositionContext(
            strength * strength_support, strength_support, len(self.results[opponent]),
            matchup, n,
            sum(actual for _, actual in history) / n if n else None,
            sum(expected for expected, _ in history) / n if n else None,
            n / (n + self.head_to_head_prior_matches),
        )

    def update_result(self, match: Match, expected_home: float) -> None:
        result = match.result_value(match.home)
        if result is None:
            return
        delta = 20 * (result - expected_home)
        self.elo[match.home] += delta
        self.elo[match.away] -= delta
        for team, expected, actual in [
            (match.home, expected_home, result),
            (match.away, 1 - expected_home, 1 - result),
        ]:
            self.form[team].append((expected, actual))
            self.results[team].append(actual)
            self.head_to_head[team, match.opponent(team)].append((expected, actual))

    def pool_rates(self, family: str) -> tuple[float, float, float]:
        pool = self.pools[family]
        return (
            (pool.batting_runs + 120 * 1.25) / (pool.batting_balls + 120),
            (pool.batting_outs + 120 * 0.05) / (pool.batting_balls + 120),
            (pool.bowling_wickets + 120 * 0.045) / (pool.bowling_balls + 120),
        )

    def update_performance(
        self, match: Match, family: str, performance: dict[str, Performance],
        rates: tuple[float, float, float],
    ) -> None:
        if match.id in self.contributions:
            raise ValueError(f"Match statistics applied twice: {match.id}")
        run_rate, dismissal_rate, wicket_rate = rates
        added = {}
        for player, raw in performance.items():
            values = {item.name: getattr(raw, item.name) for item in fields(raw)}
            contribution = Accumulated(
                **values,
                batting_expected_runs=raw.batting_balls * run_rate,
                batting_expected_outs=raw.batting_balls * dismissal_rate,
                bowling_expected_runs=raw.bowling_balls * run_rate,
                bowling_expected_wickets=raw.bowling_balls * wicket_rate,
            )
            self.players[player].add(contribution)
            self.pools[family].add(contribution)
            added[player] = contribution
        self.contributions[match.id] = (family, added)

    def withdraw_performance(self, match_id: str) -> None:
        if match_id not in self.contributions:
            return
        family, contributions = self.contributions.pop(match_id)
        for player, contribution in contributions.items():
            self.players[player].add(contribution, -1)
            self.pools[family].add(contribution, -1)

    def dependency(self, lineup: list[str], role: str) -> list[dict]:
        if role not in {"batting", "bowling"}:
            raise ValueError(f"Unsupported role: {role}")
        rows = []
        for player in lineup:
            stats = self.players[player]
            balls = getattr(stats, role + "_balls")
            runs = getattr(stats, role + "_runs")
            expected_runs = getattr(stats, role + "_expected_runs")
            run_ratio = (runs + self.prior_balls * 1.25) / (
                expected_runs + self.prior_balls * 1.25)
            if role == "batting":
                wicket_ratio = (stats.batting_outs + self.prior_balls * 0.05) / (
                    stats.batting_expected_outs + self.prior_balls * 0.05)
                quality = run_ratio / math.sqrt(wicket_ratio)
                default_opportunity = 10.0
            else:
                wicket_ratio = (stats.bowling_wickets + self.prior_balls * 0.045) / (
                    stats.bowling_expected_wickets + self.prior_balls * 0.045)
                quality = wicket_ratio / math.sqrt(run_ratio)
                default_opportunity = 10.0
            opportunity = (balls + self.prior_appearances * default_opportunity) / (
                stats.appearances + self.prior_appearances)
            rows.append({
                "player_id": player, "role": role, "prior_legal_balls": balls,
                "prior_appearances": stats.appearances,
                "sample_reliability": balls / (balls + self.prior_balls),
                "quality": quality, "expected_legal_opportunities": opportunity,
                "projected_contribution": quality * opportunity,
            })
        if not rows:
            return []
        total = sum(row["projected_contribution"] for row in rows)
        equal_share = 1 / len(rows)
        for row in rows:
            share = row["projected_contribution"] / total
            row["contribution_share"] = share
            row["dependency"] = (
                clip((share - equal_share) / (1 - equal_share)) if len(rows) > 1 else 1.0
            )
            row["ability_rank"] = 1 + sum(
                other["quality"] > row["quality"] + 1e-12 for other in rows)
            row["dependency_rank"] = 1 + sum(
                other["projected_contribution"] > row["projected_contribution"] + 1e-12
                for other in rows)
            row["history_status"] = "cold_start" if row["prior_appearances"] == 0 else (
                "limited" if row["prior_legal_balls"] < self.prior_balls else "supported")
        return rows
