import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from .live import CREDITED_WICKETS, pressure_weight
from .statistics import clip

@dataclass
class ContextTotals:
    legal: int = 0
    runs: float = 0.0
    wickets: int = 0
    run_counts: Counter = field(default_factory=Counter)

    def update(self, legal, runs, wicket):
        self.legal += int(legal)
        self.runs += runs
        self.wickets += int(wicket)
        self.run_counts[runs] += 1

@dataclass
class NormalTotals:
    events: int = 0
    value: float = 0.0

    def update(self, value):
        self.events += 1
        self.value += value

class HistoricalBaselines:
    def __init__(self, config):
        self.config = config
        self.context = defaultdict(ContextTotals)
        self.normal = defaultdict(NormalTotals)

    def normal_reference(self, player, role, phase):
        prior = self.config.player_prior_events
        parent = self.normal[player, role]
        parent_mean = parent.value / (parent.events + prior)
        local = self.normal[player, role, phase]
        value = (local.value + prior * parent_mean) / (local.events + prior)
        return value, parent.events, parent.events / (parent.events + prior)

    @staticmethod
    def context_keys(family, innings, role, phase, position_band):
        keys = [(role,), (role, family.split(":")[0], innings, phase)]
        if ":" in family:
            keys.append((role, family, innings, phase))
        return keys + [(role, family, innings, phase, position_band)]

    def expectation(self, family, innings, role, snapshot, opponent, venue_ratio):
        rate, wicket_rate = (1.25 if role == "batting" else 1.35), 0.045
        run_prior = {0: 0.37, 1: 0.38, 2: 0.10, 3: 0.01, 4: 0.10, 6: 0.04}
        if role == "bowling":
            run_prior[0], run_prior[1] = 0.27, 0.48
        needed = snapshot.target - snapshot.runs if snapshot.target is not None else None
        useful_rate = (sum(min(run, needed) * probability for run, probability in run_prior.items())
                       if needed is not None else rate)
        prior = self.config.context_prior_balls
        keys = self.context_keys(family, innings, role, snapshot.phase, snapshot.position_band)
        for key in keys:
            values = self.context[key]
            rate = (values.runs + prior * rate) / (values.legal + prior)
            useful_runs = (sum(min(run, needed) * count for run, count in values.run_counts.items())
                           if needed is not None else values.runs)
            useful_rate = (useful_runs + prior * useful_rate) / (values.legal + prior)
            wicket_rate = (values.wickets + prior * wicket_rate) / (values.legal + prior)
        useful_fraction = useful_rate / rate if rate else 1.0
        other_role = "bowling" if role == "batting" else "batting"
        strength, _, _ = self.normal_reference(opponent, other_role, snapshot.phase)
        strength = clip(strength, -1.5, 1.5)
        rate *= clip(venue_ratio, 0.8, 1.2)
        rate += (-0.25 if role == "batting" else 0.25) * strength
        wicket_rate += (0.01 if role == "batting" else -0.01) * strength
        if snapshot.free_hit:
            rate *= 1.2
            wicket_rate = 0.0
        rate = clip(rate, 0.4, 4.0) * useful_fraction
        if needed is not None:
            rate = min(rate, needed)
        return rate, clip(wicket_rate, 0, 0.2), self.context[keys[-1]].legal

    def update(self, observation):
        family, innings, role, phase, band, player, legal, runs, wicket, value, ordinary = observation
        for key in self.context_keys(family, innings, role, phase, band):
            self.context[key].update(legal, runs, wicket)
        if ordinary:
            self.normal[player, role].update(value)
            self.normal[player, role, phase].update(value)

def performance_observation(
    event, snapshot, role, family, par, family_par, baseline, team_pressure,
    personal_pressure, responsibility_status, config,
):
    flags = list(event.attribution_flags)
    credited = event.wicket and event.dismissal in CREDITED_WICKETS
    legal_wide_wicket = event.extra_type == "wide" and event.dismissal in {"Stump", "Hit Wicket"}
    if "bat_runs_on_bye_attribution_unverified" in flags:
        return None, None, flags
    if event.wicket and (
        not event.dismissed
        or (event.dismissal in CREDITED_WICKETS and event.dismissed != event.striker)
        or (credited and ((not event.legal and not legal_wide_wicket) or snapshot.free_hit))
    ):
        return None, None, flags + ["dismissal_attribution_withheld"]
    if (credited and event.extra_type == "wide" and event.dismissal == "Stump"
            and snapshot.target is not None
            and snapshot.runs + event.primary + event.secondary >= snapshot.target):
        credited = False
        flags.append("winning_wide_precludes_stumping_credit")
    if role == "batting" and (event.extra_type == "Penalty" or (event.extra_type == "wide" and not credited)):
        return None, None, flags
    if role == "bowling" and (event.bowler_runs is None or event.extra_type == "Penalty"):
        return None, None, flags
    player = event.striker if role == "batting" else event.bowler
    opponent = event.bowler if role == "batting" else event.striker
    rate, dismissal_rate, context_balls = baseline.expectation(
        family, event.innings, role, snapshot, opponent, par / family_par)
    normal, normal_events, normal_support = baseline.normal_reference(player, role, snapshot.phase)
    batter_strength, _, _ = baseline.normal_reference(event.striker, "batting", snapshot.phase)
    wicket_value = clip(
        par / 120 * (snapshot.balls_remaining - 1) / (10 - snapshot.wickets)
        * clip(1 + 0.1 * batter_strength, 0.75, 1.25), 0, 25)
    needed = snapshot.target - snapshot.runs if snapshot.target is not None else None
    if needed is not None:
        wicket_value = min(wicket_value, needed)
    legal = int(event.legal)
    if role == "batting":
        runs = event.runs
        useful_runs = min(runs, max(0, needed - event.primary - event.secondary)) if needed is not None else runs
        expected_dismissal = 0 if event.wicket and not credited else dismissal_rate * legal
        absolute = useful_runs - rate * legal + wicket_value * (expected_dismissal - int(credited))
    else:
        runs = event.bowler_runs
        useful_runs = min(runs, needed) if needed is not None else runs
        absolute = rate * legal - useful_runs + wicket_value * (int(credited) - dismissal_rate * legal)
    usual = normal * legal
    if needed is not None and legal:
        if role == "batting":
            usual = clip(usual, -rate - wicket_value * (1 - dismissal_rate),
                         needed - rate + wicket_value * dismissal_rate)
        else:
            usual = clip(usual, rate - needed - wicket_value * dismissal_rate,
                         rate + wicket_value * (1 - dismissal_rate))
    lift = absolute - usual
    if normal_events < config.minimum_normal_events:
        flags.append("usual_baseline_limited_history")
    if context_balls < config.context_prior_balls:
        flags.append("context_baseline_shrunk")
    if responsibility_status != "v4_known_signal_overlay":
        flags.append(responsibility_status)
    row = {
        "match_id": event.match_id, "innings": event.innings, "sequence": event.sequence,
        "player_id": player, "role": role,
        "team_id": event.batting_team if role == "batting" else event.bowling_team,
        "phase": snapshot.phase, "position_band": snapshot.position_band,
        "legal_opportunity": legal, "attributed_runs": runs, "useful_attributed_runs": useful_runs,
        "credited_wicket": int(credited),
        "team_pressure": team_pressure, "personal_pressure": personal_pressure,
        "pressure_weight": pressure_weight(personal_pressure, config),
        "high_pressure": personal_pressure >= config.high_pressure,
        "expected_runs_per_legal_ball": rate, "expected_credited_wicket_rate": dismissal_rate,
        "wicket_value_before": wicket_value, "absolute_contribution": absolute,
        "usual_reference_before": usual, "contribution_vs_usual": lift,
        "normal_reference_events_before": normal_events, "normal_reference_support": normal_support,
        "context_prior_legal_balls": context_balls, "flags": ";".join(sorted(set(flags))),
    }
    observation = (
        family, event.innings, role, snapshot.phase, snapshot.position_band, player,
        legal, runs, credited, absolute, team_pressure <= config.normal_pressure_ceiling,
    )
    return row, observation, flags

@dataclass
class MatchCard:
    player_id: str
    role: str
    match_id: str
    match_date: str
    family: str
    team_id: str
    events: int = 0
    legal_balls: int = 0
    attributed_runs: float = 0.0
    credited_wickets: int = 0
    pressure_events: int = 0
    pressure_legal_balls: int = 0
    weight: float = 0.0
    weighted_absolute: float = 0.0
    weighted_lift: float = 0.0
    weighted_normal_support: float = 0.0
    whole_value: float = 0.0
    maximum_pressure: float = 0.0
    responsibility_fallback_events: int = 0

    def add(self, row):
        self.events += 1
        self.legal_balls += row["legal_opportunity"]
        self.attributed_runs += row["attributed_runs"]
        self.credited_wickets += row["credited_wicket"]
        self.pressure_events += int(row["high_pressure"])
        self.pressure_legal_balls += row["legal_opportunity"] * int(row["high_pressure"])
        weight = row["pressure_weight"]
        self.weight += weight
        self.weighted_absolute += weight * row["absolute_contribution"]
        self.weighted_lift += weight * row["contribution_vs_usual"]
        self.weighted_normal_support += weight * row["normal_reference_support"]
        self.whole_value += row["absolute_contribution"]
        self.maximum_pressure = max(self.maximum_pressure, row["personal_pressure"])
        self.responsibility_fallback_events += int(
            "responsibility_team_context_only" in row["flags"] or "runtime_dependency_fallback" in row["flags"])

    def evidence(self, config):
        absolute = self.weighted_absolute / self.weight if self.weight else None
        lift = self.weighted_lift / self.weight if self.weight else None
        normal_support = self.weighted_normal_support / self.weight if self.weight else 0
        margin = (absolute + lift * normal_support) / 2 if self.weight else 0
        if self.weight == 0:
            status = "insufficient_pressure_exposure"
        elif margin < -config.neutral_match_margin:
            status = "underperformed"
        elif margin > config.neutral_match_margin and self.whole_value >= 0:
            status = "delivered"
        elif margin > config.neutral_match_margin:
            status = "mixed_recovery_negative_whole_contribution"
        else:
            status = "near_expectation"
        return {
            "player_id": self.player_id, "role": self.role, "match_id": self.match_id,
            "match_date": self.match_date, "family": self.family, "team_id": self.team_id,
            "attributable_events": self.events, "legal_balls": self.legal_balls,
            "attributed_runs": self.attributed_runs, "credited_wickets": self.credited_wickets,
            "high_pressure_events": self.pressure_events,
            "high_pressure_legal_balls": self.pressure_legal_balls,
            "well_supported_pressure_match": self.pressure_events >= config.minimum_high_events_per_match,
            "pressure_weight": self.weight, "maximum_personal_pressure": self.maximum_pressure,
            "absolute_pressure_contribution": absolute, "pressure_contribution_vs_usual": lift,
            "whole_opportunity_contribution": self.whole_value,
            "usual_baseline_support": normal_support,
            "match_evidence_weight": min(1.0, self.weight / config.match_evidence_scale),
            "status": status, "responsibility_fallback_events": self.responsibility_fallback_events,
        }

def rating_components(cards, config, weights=None):
    meaningful = [row for row in cards if row["match_evidence_weight"] > 0]
    weight = sum(row["match_evidence_weight"] for row in meaningful)
    if not weight:
        return 50.0, 0.0, 0.0, 0.5, 0.0
    absolute = sum(row["match_evidence_weight"] * row["absolute_pressure_contribution"]
                   for row in meaningful) / weight
    lift = sum(row["match_evidence_weight"] * row["usual_baseline_support"] * row["pressure_contribution_vs_usual"]
               for row in meaningful) / weight
    comparable = [row for row in meaningful if row["status"] != "insufficient_pressure_exposure"]
    consistency_weight = sum(row["match_evidence_weight"] for row in comparable)
    success = sum(row["match_evidence_weight"] * (
        1 if row["status"] == "delivered" else 0 if row["status"] == "underperformed" else 0.5
    ) for row in comparable)
    consistency = (success + 2) / (consistency_weight + 4)
    absolute_index = 50 + 30 * math.tanh(absolute / config.contribution_scale)
    resilience_index = 50 + 30 * math.tanh(lift / config.contribution_scale)
    coefficients = weights or (config.absolute_weight, config.resilience_weight, config.consistency_weight)
    unshrunk = sum(a * b for a, b in zip(coefficients, [absolute_index, resilience_index, 100 * consistency]))
    support = weight / (weight + config.career_prior_matches)
    score = 50 + support * (unshrunk - 50)
    return score, absolute, lift, consistency, support
