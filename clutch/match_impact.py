import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from .inputs import InputError, read_csv

@dataclass(frozen=True)
class MatchImpactConfig:
    ball_share: float = 0.70
    match_share: float = 0.30
    stakes_floor: float = 70.0
    run_scale: float = 20.0
    wicket_factor: float = 0.35
    prior_matches: float = 3.0
    minimum_matches: int = 6
    minimum_legal_balls: int = 60
    minimum_normal_events: int = 30
    neutral_run_margin: float = 2.0
    bootstrap_samples: int = 250
    gentle_adjustment: bool = False
    gentle_adjustment_cap: float = 2.0
    material_help_runs: float = 10.0

    def __post_init__(self):
        if not isinstance(self.gentle_adjustment, bool):
            raise InputError("gentle_adjustment must be a boolean.")
        if any(not math.isfinite(value) or value <= 0
               for name, value in asdict(self).items() if name != "gentle_adjustment"):
            raise InputError("Match-impact parameters must be finite and positive.")
        if not math.isclose(self.ball_share + self.match_share, 1):
            raise InputError("Ball/match shares must sum to one.")
        if self.stakes_floor >= 100 or self.wicket_factor > 1:
            raise InputError("Invalid high-stakes floor or secondary wicket factor.")

def stakes_weight(pressure, config):
    if not math.isfinite(pressure) or not 0 <= pressure <= 100:
        raise InputError("Invalid pre-match pressure for the big-match channel.")
    return max(0.0, pressure - config.stakes_floor) / (100 - config.stakes_floor)

def required_rate(state, after=False):
    target = state["target"]
    if target is None:
        return None
    runs = state["runs_before"] + (state["recorded_total_runs"] if after else 0)
    balls = state["balls_remaining"] - (int(state["legal_delivery"]) if after else 0)
    if runs >= target:
        return 0.0
    return 6 * (target - runs) / balls if balls else None

@dataclass
class OverContribution:
    context_rate: float
    required_rate: float | None
    scoring_rate: float
    legal_balls: int = 0
    useful_runs: int = 0

@dataclass
class HighMatchCard:
    player_id: str
    player: str
    role: str
    match_id: str
    match_date: str
    family: str
    team_id: str
    prematch_pressure: float
    incomplete_attribution: bool = False
    uncertain_own_dismissal: bool = False
    blocks: dict = field(default_factory=dict)
    events: int = 0
    legal_balls: int = 0
    runs: int = 0
    useful_runs: int = 0
    wickets: int = 0
    actual_wicket_value: float = 0.0
    expected_wicket_value: float = 0.0
    survival_probability: float = 1.0
    first_state: dict | None = None
    last_state: dict | None = None
    last_useful_runs: int = 0
    last_credited_wicket: bool = False

    def add(self, row, state):
        if self.first_state is None:
            self.first_state = state
        elif row["innings"] != self.first_state["innings"]:
            raise InputError("A player-role match card cannot combine different regular innings.")
        self.last_state = state
        self.last_useful_runs = row["useful_attributed_runs"]
        self.last_credited_wicket = bool(row["credited_wicket"])
        key = (row["innings"], state["legal_balls_before"] // 6)
        if key not in self.blocks:
            context_rate = row["expected_runs_per_legal_ball"]
            required = required_rate(state)
            if required is None:
                rate = context_rate
            elif self.role == "batting":
                rate = required / 6
            else:
                defendable = max(0, state["target"] - state["runs_before"] - 1) / state["balls_remaining"]
                rate = min(context_rate, defendable)
            self.blocks[key] = OverContribution(context_rate, required, rate)
        block = self.blocks[key]
        block.legal_balls += row["legal_opportunity"]
        block.useful_runs += row["useful_attributed_runs"]
        self.events += 1
        self.legal_balls += row["legal_opportunity"]
        self.runs += row["attributed_runs"]
        self.useful_runs += row["useful_attributed_runs"]
        self.wickets += row["credited_wicket"]
        self.actual_wicket_value += row["credited_wicket"] * row["wicket_value_before"]
        probability = row["expected_credited_wicket_rate"] * row["legal_opportunity"]
        if self.role == "batting":
            self.expected_wicket_value += self.survival_probability * probability * row["wicket_value_before"]
            self.survival_probability *= 1 - probability
        else:
            self.expected_wicket_value += probability * row["wicket_value_before"]

    def evidence(self, config, team_innings):
        weight = stakes_weight(self.prematch_pressure, config)
        pace = raw_pace = 0.0
        benchmark_runs = context_runs = 0.0
        capped_shortfall = False
        for block in self.blocks.values():
            quota = block.scoring_rate * block.legal_balls
            context = block.context_rate * block.legal_balls
            benchmark_runs += quota
            context_runs += context
            delta = block.useful_runs - quota if self.role == "batting" else quota - block.useful_runs
            raw_pace += delta
            if self.role == "batting" and block.required_rate is not None:
                # Preserve a negative direction without blaming a player for unlimited inherited RRR.
                bounded = max(delta, -context)
                capped_shortfall |= bounded != delta
                delta = bounded
            pace += delta
        wicket_balance = self.actual_wicket_value - self.expected_wicket_value
        if self.role == "batting":
            wicket_balance = -wicket_balance
            if self.uncertain_own_dismissal:
                wicket_balance = 0.0
        wicket_impact = config.wicket_factor * wicket_balance
        net = pace + wicket_impact
        if weight == 0:
            status, net = "not_high_stakes", None
        elif self.incomplete_attribution:
            status, net = "withheld_incomplete_attribution", None
        else:
            status = ("positive_team_impact" if net > config.neutral_run_margin else
                      "negative_team_impact" if net < -config.neutral_run_margin else "near_neutral")
        first, last = self.first_state, self.last_state
        total_runs, total_balls = team_innings["runs"], team_innings["legal_balls"]
        return {
            "player_id": self.player_id, "player": self.player, "role": self.role,
            "match_id": self.match_id, "match_date": self.match_date, "family": self.family,
            "team_id": self.team_id, "innings": first["innings"],
            "prematch_team_pressure": self.prematch_pressure, "high_match_weight": weight,
            "status": status, "attributable_events": self.events, "legal_balls": self.legal_balls,
            "attributed_runs": self.runs, "useful_attributed_runs": self.useful_runs,
            "credited_wickets": self.wickets,
            "legal_ball_strike_rate": 100 * self.runs / self.legal_balls if self.role == "batting"
                                      and self.legal_balls and not self.incomplete_attribution else None,
            "bowling_economy": 6 * self.runs / self.legal_balls if self.role == "bowling" and self.legal_balls
                               and not self.incomplete_attribution else None,
            "reference_scoring_strike_rate": 100 * benchmark_runs / self.legal_balls if self.role == "batting"
                                            and self.legal_balls else None,
            "reference_bowling_economy": 6 * benchmark_runs / self.legal_balls if self.role == "bowling"
                                       and self.legal_balls else None,
            "innings_team_runs": total_runs,
            "innings_team_run_rate": 6 * total_runs / total_balls if total_balls else None,
            "batter_share_of_team_runs": self.runs / total_runs if self.role == "batting" and total_runs else None,
            "rrr_at_first_opportunity": required_rate(first),
            "rrr_after_last_opportunity": required_rate(last, after=True),
            "reference_run_budget": benchmark_runs, "context_run_budget": context_runs,
            "raw_pace_surplus": raw_pace, "bounded_pace_impact": pace,
            "actual_wicket_resource_value": self.actual_wicket_value,
            "expected_wicket_resource_value": self.expected_wicket_value,
            "secondary_wicket_impact": wicket_impact, "net_team_impact": net,
            "inherited_chase_shortfall_capped": capped_shortfall,
            "wicket_blame_withheld": self.uncertain_own_dismissal,
            "rrr_endpoint_change_is_context_only": True,
        }

def high_match_rating(cards, config):
    used = [row for row in cards if row is not None and row["net_team_impact"] is not None
            and row["high_match_weight"] > 0]
    evidence = sum(row["high_match_weight"] for row in used)
    if not evidence:
        return 50.0, None, 0.0
    mean = sum(row["high_match_weight"] * row["net_team_impact"] for row in used) / evidence
    support = evidence / (evidence + config.prior_matches)
    adjustment = gentle_match_adjustment(used, config)
    return 50 + support * 30 * math.tanh(mean / config.run_scale) + adjustment, mean, support

def delivery_signal(net_impact, verified_win, closing_action, config):
    if net_impact is None:
        return 0.0
    if net_impact < -config.neutral_run_margin:
        return -1.0
    if net_impact >= 0 and verified_win and (closing_action or net_impact >= config.material_help_runs):
        return 1.0
    return 0.0

def gentle_match_adjustment(cards, config):
    if not config.gentle_adjustment:
        return 0.0
    used = [row for row in cards if row is not None and row["net_team_impact"] is not None
            and row["high_match_weight"] > 0]
    weight = sum(row["high_match_weight"] for row in used)
    if not weight:
        return 0.0
    if any("gentle_delivery_signal" not in row for row in used):
        raise InputError("Gentle marking requires explicit individual contribution evidence.")
    mean = sum(row["high_match_weight"] * row["gentle_delivery_signal"] for row in used) / weight
    return config.gentle_adjustment_cap * weight / (weight + config.prior_matches) * mean

def numeric_ball_cards(path):
    grouped = defaultdict(list)
    keys = ["match_evidence_weight", "absolute_pressure_contribution",
            "pressure_contribution_vs_usual", "usual_baseline_support"]
    for row in read_csv(path):
        for key in keys:
            row[key] = float(row[key]) if row[key] else None
        grouped[row["player_id"], row["role"]].append(row)
    return grouped
