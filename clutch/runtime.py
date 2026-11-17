import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .inputs import InputError
from .live import live_pressure_components
from .statistics import clip


@dataclass
class RuntimeState:
    dot_streak: int = 0
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=6))
    partnership_runs: int = 0
    partnership_balls: int = 0
    dismissed: set = field(default_factory=set)
    unknown_dismissal: bool = False
    bowler_balls: Counter = field(default_factory=Counter)
    last_bowler_over: dict = field(default_factory=dict)

    def before(self):
        return {
            "prior_dot_streak": self.dot_streak,
            "four_plus_last_six": sum(runs >= 4 for runs in self.recent_scores),
            "partnership_runs_before": self.partnership_runs,
            "partnership_balls_before": self.partnership_balls,
        }

    def advance(self, event, snapshot):
        self.dot_streak = self.dot_streak + 1 if event.legal and event.total_runs == 0 else 0
        self.recent_scores.append(event.total_runs)
        self.partnership_runs += event.total_runs
        self.partnership_balls += int(event.legal)
        self.bowler_balls[event.bowler] += int(event.legal)
        self.last_bowler_over[event.bowler] = snapshot.legal_balls // 6
        if event.wicket:
            if event.dismissed:
                self.dismissed.add(event.dismissed)
            else:
                self.unknown_dismissal = True
            self.partnership_runs = self.partnership_balls = 0


def adaptive_pressure(snapshot, par, runtime, config):
    batting, bowling, proxy, _ = live_pressure_components(snapshot, par)
    sequence = runtime.before()
    balls, wickets = snapshot.balls_remaining, 10 - snapshot.wickets
    if balls <= 0 or wickets <= 0:
        return 0.0, 0.0, proxy, {"practical_liveness": 0.0, "practical_runs_per_ball_limit": None,
                                 "required_runs_per_over": None, "short_burst_context": False}
    rate = par / 120
    scarcity = clip((6 - wickets) / 5)
    old_disruption = max(clip(snapshot.recent_wickets / 3),
                         clip((6 - snapshot.striker_balls) / 6) * scarcity)
    new_disruption = max(old_disruption, clip(runtime.dot_streak / 3))
    batting += 5 * (new_disruption - old_disruption)
    old_threat = clip(snapshot.striker_balls / 20) * clip(
        snapshot.striker_runs / max(1, snapshot.striker_balls) / rate - 0.5)
    partnership_threat = clip(runtime.partnership_runs / 60) * clip(runtime.partnership_balls / 24)
    new_threat = max(old_threat, partnership_threat, clip(sequence["four_plus_last_six"] / 2),
                     float(snapshot.free_hit))
    bowling += 5 * (new_threat - old_threat)
    if snapshot.free_hit:
        batting -= 7.5 * scarcity
    fraction, limit, required_rr = 1.0, None, None
    if snapshot.target is not None:
        required = max(0, snapshot.target - snapshot.runs)
        required_rr = 6 * required / balls
        sustained = clip(rate * config.sustained_pace_multiplier, 2.2, 3.2)
        burst = clip((config.extended_burst_balls - balls) /
                     (config.extended_burst_balls - config.short_burst_balls))
        limit = sustained + burst * (config.short_burst_runs_per_ball - sustained)
        resources = 0.75 + 0.25 * min(1, wickets / 3)
        if wickets < 3:
            resources = min(1, resources + 0.1 * clip(snapshot.striker_balls / 12))
        limit *= resources
        excess = max(0, (required / balls) / limit - 1)
        difficult = math.exp(-config.hopelessness_scale * excess * excess)
        easy = min(1.0, (1 - proxy) / 0.01) ** 0.25
        fraction = min(difficult, easy) if required else 0.0
    return clip(batting * fraction, 0, 100), clip(bowling * fraction, 0, 100), proxy, {
        "practical_liveness": fraction, "practical_runs_per_ball_limit": limit,
        "required_runs_per_over": required_rr,
        "short_burst_context": snapshot.target is not None and balls <= config.short_burst_balls,
    }


def relative_dependency(values, player):
    if player not in values or any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise InputError("Invalid remaining-player contribution set.")
    if len(values) == 1:
        return 1.0
    total = sum(values.values())
    if total <= 0:
        raise InputError("Remaining contribution set has no positive support.")
    equal = 1 / len(values)
    return clip((values[player] / total - equal) / (1 - equal))


def bowling_time_capacity(player, current, snapshot, runtime):
    over = snapshot.legal_balls // 6
    quota = max(0, 24 - runtime.bowler_balls[player])
    if player == current:
        current_balls = min(6 - snapshot.legal_balls % 6, snapshot.balls_remaining)
        next_over = over + 2
    else:
        current_balls = 0
        next_over = over + (2 if runtime.last_bowler_over.get(player) == over else 1)
    future_overs = max(0, (19 - next_over) // 2 + 1)
    return min(quota, current_balls + 6 * future_overs)


def runtime_responsibility(data, prematch_players, match, team, player, role, snapshot, runtime, config):
    source = prematch_players.get((match.id, team, player, role))
    known = source is not None and source["applied_combined_uplift_0_1"] != ""
    base = float(source["applied_combined_uplift_0_1"]) if known else 0.0
    prior_dependency = float(source["dependency_0_1"]) if source and source.get("dependency_0_1") else None
    result = {
        "uplift": base, "dependency": prior_dependency, "remaining_options": None,
        "source": source, "status": "v4_known_signal_overlay" if known else "responsibility_team_context_only",
    }
    if not config.adaptive_runtime:
        return result
    lineup = data.starting.get((match.id, team), [])
    if len(lineup) != 11 or player not in lineup or not known:
        return {**result, "status": "runtime_dependency_fallback_unverified_xi"}
    records = {p: prematch_players.get((match.id, team, p, role)) for p in lineup}
    if any(row is None for row in records.values()):
        return {**result, "status": "runtime_dependency_fallback_missing_prior"}
    if role == "batting":
        if runtime.unknown_dismissal or player in runtime.dismissed:
            return {**result, "status": "runtime_dependency_fallback_unknown_dismissal"}
        values = {p: float(row["projected_contribution"]) for p, row in records.items()
                  if p not in runtime.dismissed}
        status = "runtime_remaining_batting_dependency"
    else:
        candidates = {
            p: row for p, row in records.items()
            if float(row["prior_legal_balls"]) >= config.known_bowler_prior_balls
            or p in runtime.last_bowler_over or p == player
        }
        if len(candidates) < 3 and snapshot.legal_balls // 6 != 19:
            return {**result, "status": "runtime_dependency_fallback_limited_bowling_history"}
        values = {}
        for p, row in candidates.items():
            capacity = bowling_time_capacity(p, player, snapshot, runtime)
            if capacity > 0:
                values[p] = float(row["quality_index"]) * min(
                    capacity, float(row["expected_legal_opportunities"]))
        if player not in values:
            return {**result, "status": "runtime_dependency_fallback_bowler_quota"}
        status = "runtime_known_bowling_capacity_dependency"
    dependency = relative_dependency(values, player)
    leadership = float(source["leadership_uplift_0_1"])
    return {
        "uplift": min(0.35, leadership + 0.25 * dependency), "dependency": dependency,
        "remaining_options": len(values), "source": source, "status": status,
    }
