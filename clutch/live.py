import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field

from .inputs import InputError, iter_csv
from .statistics import clip


LEGAL = {"None", "byes", "legbyes"}
NO_BALL = {"Overstep", "other"}
CREDITED_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


@dataclass(frozen=True)
class PerformanceConfig:
    prematch_share: float = 0.35
    pressure_floor: float = 50.0
    high_pressure: float = 60.0
    normal_pressure_ceiling: float = 50.0
    context_prior_balls: float = 120.0
    player_prior_events: float = 80.0
    par_prior_innings: float = 8.0
    par_family_window: int = 60
    par_venue_window: int = 20
    match_evidence_scale: float = 8.0
    career_prior_matches: float = 5.0
    minimum_pressure_matches: int = 6
    minimum_pressure_legal_balls: int = 60
    minimum_normal_events: int = 30
    minimum_high_events_per_match: int = 3
    absolute_weight: float = 0.4
    resilience_weight: float = 0.4
    consistency_weight: float = 0.2
    contribution_scale: float = 1.5
    neutral_match_margin: float = 0.1
    bootstrap_samples: int = 250
    adaptive_runtime: bool = False
    short_burst_balls: int = 12
    extended_burst_balls: int = 36
    short_burst_runs_per_ball: float = 5.0
    sustained_pace_multiplier: float = 1.6
    hopelessness_scale: float = 8.0
    known_bowler_prior_balls: int = 12

    def __post_init__(self):
        if not isinstance(self.adaptive_runtime, bool):
            raise InputError("adaptive_runtime must be a boolean.")
        if any(not math.isfinite(value) or value <= 0
               for name, value in asdict(self).items() if name != "adaptive_runtime"):
            raise InputError("Performance parameters must be finite and positive.")
        if not 0 < self.prematch_share < 1:
            raise InputError("Pre-match and live pressure must both contribute.")
        if not (0 < self.pressure_floor < self.high_pressure < 100
                and 0 < self.normal_pressure_ceiling < self.high_pressure):
            raise InputError("Ordinary-pressure and weighting thresholds must be below high pressure.")
        if not math.isclose(self.absolute_weight + self.resilience_weight + self.consistency_weight, 1):
            raise InputError("Balanced rating weights must sum to one.")
        if not self.short_burst_balls < self.extended_burst_balls <= 120:
            raise InputError("Short-burst and sustained-demand horizons must be ordered.")


@dataclass(frozen=True)
class Delivery:
    match_id: str
    innings: int
    sequence: int
    over: int
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    runs: int
    primary: int
    secondary: int
    extra_type: str
    wicket: bool
    dismissed: str
    dismissal: str

    @property
    def legal(self):
        return self.extra_type in LEGAL

    @property
    def total_runs(self):
        return self.runs + self.primary + self.secondary

    @property
    def bowler_runs(self):
        if self.secondary or (self.extra_type == "None" and self.primary):
            return None
        return self.runs + (self.primary if self.extra_type in NO_BALL | {"wide"} else 0)

    @property
    def attribution_flags(self):
        flags = []
        if self.secondary:
            flags.append("secondary_extra_bowler_attribution_unverified")
        if self.extra_type in {"byes", "legbyes"} and self.runs:
            flags.append("bat_runs_on_bye_attribution_unverified")
        if self.wicket and not self.dismissed:
            flags.append("dismissed_player_unknown")
        if self.wicket and self.dismissal not in CREDITED_WICKETS:
            flags.append("non_bowler_wicket_individual_blame_not_inferred")
        if self.wicket and self.dismissal in CREDITED_WICKETS and self.dismissed != self.striker:
            flags.append("credited_wicket_striker_identity_conflict")
        return flags


def load_deliveries(path):
    grouped = defaultdict(list)
    keys = set()
    for row in iter_csv(path):
        event = Delivery(
            row["match_id"], int(row["innings"]), int(row["innings_ball_num"]), int(row["overid"]),
            row["batting_team_id"], row["bowling_team_id"], row["striker_id"],
            row["non_striker_id"], row["bowler_id"], int(row["runs"]),
            int(row["extras1_runs"]), int(row["extras_2_runs"]), row["extras_type"],
            row["is_wicket"] == "true", row["dismissed_player_id"], row["dismissal_type"])
        key = (event.match_id, event.innings, event.sequence)
        if key in keys:
            raise InputError(f"Duplicate delivery identity: {key}")
        keys.add(key)
        if event.extra_type not in LEGAL | NO_BALL | {"wide", "Penalty"}:
            raise InputError(f"Unsupported extras type: {event.extra_type}")
        if min(event.runs, event.primary, event.secondary) < 0 or event.sequence < 1:
            raise InputError(f"Invalid recorded runs/sequence: {key}")
        if row["is_wicket"] not in {"true", "false"}:
            raise InputError(f"Invalid wicket indicator: {key}")
        grouped[event.match_id].append(event)
    for events in grouped.values():
        events.sort(key=lambda event: (event.innings, event.sequence))
    return dict(grouped)


@dataclass(frozen=True)
class Snapshot:
    runs: int
    wickets: int
    legal_balls: int
    balls_remaining: int
    target: int | None
    phase: str
    position_band: str
    striker_balls: int
    striker_runs: int
    recent_wickets: int
    free_hit: bool


@dataclass
class InningsState:
    target: int | None = None
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    free_hit: bool = False
    entrants: dict = field(default_factory=dict)
    batter_balls: Counter = field(default_factory=Counter)
    batter_runs: Counter = field(default_factory=Counter)
    wicket_positions: deque = field(default_factory=deque)

    def before(self, event):
        if self.legal_balls >= 120 or self.wickets >= 10 or (
            self.target is not None and self.runs >= self.target
        ):
            raise InputError(f"Delivery after innings ended: {event.match_id}/{event.innings}/{event.sequence}")
        for player in [event.striker, event.non_striker]:
            if player not in self.entrants:
                self.entrants[player] = len(self.entrants) + 1
        position = self.entrants[event.striker]
        return Snapshot(
            self.runs, self.wickets, self.legal_balls, 120 - self.legal_balls, self.target,
            "powerplay" if self.legal_balls < 36 else "middle" if self.legal_balls < 90 else "death",
            "top" if position <= 3 else "middle" if position <= 6 else "lower",
            self.batter_balls[event.striker], self.batter_runs[event.striker],
            sum(self.legal_balls - at <= 12 for at in self.wicket_positions), self.free_hit)

    def advance(self, event):
        self.runs += event.total_runs
        self.legal_balls += int(event.legal)
        self.batter_balls[event.striker] += int(event.legal)
        self.batter_runs[event.striker] += event.runs
        if event.wicket:
            self.wickets += 1
            self.wicket_positions.append(self.legal_balls)
        while self.wicket_positions and self.legal_balls - self.wicket_positions[0] > 12:
            self.wicket_positions.popleft()
        if event.extra_type in NO_BALL:
            self.free_hit = True
        elif event.legal:
            self.free_hit = False


def innings_quality(match, events, metadata, voided=False):
    groups = {number: [event for event in events if event.innings == number] for number in [1, 2]}
    output = []
    first_valid = False
    first_total = sum(event.total_runs for event in groups[1])
    for number in [1, 2]:
        innings = groups[number]
        reasons = []
        legal = sum(event.legal for event in innings)
        wickets = sum(event.wicket for event in innings)
        runs = sum(event.total_runs for event in innings)
        if not innings:
            reasons.append("no_recorded_innings")
        if voided:
            reasons.append("voided_superseded_fixture")
        if metadata.get("first_inn_revised_overs") and number == 1:
            reasons.append("revised_budget_timing_unverified")
        if number == 2 and any(metadata.get(key) for key in [
            "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
        ]):
            reasons.append("revised_target_or_budget_timing_unverified")
        if innings:
            if [event.sequence for event in innings] != list(range(1, len(innings) + 1)):
                reasons.append("noncontiguous_delivery_sequence")
            if len({(event.batting_team, event.bowling_team) for event in innings}) != 1:
                reasons.append("inconsistent_innings_teams")
            if any(set([event.batting_team, event.bowling_team]) != set(match.teams) for event in innings):
                reasons.append("innings_team_identity_conflict")
            over_counts = Counter()
            for event in innings:
                over_counts[event.over] += int(event.legal)
            if any(count != 6 for over, count in over_counts.items() if over != innings[-1].over):
                reasons.append("irregular_completed_over_legal_count")
            if any(event.extra_type == "None" and (event.primary or event.secondary) for event in innings):
                reasons.append("unclassified_extra_may_change_legal_clock")
            if legal > 120 or wickets > 10:
                reasons.append("innings_resource_count_out_of_range")
            complete = legal == 120 or wickets == 10
            if number == 1 and not complete:
                reasons.append("first_innings_short_or_incomplete_budget_unknown")
            if number == 2:
                if not first_valid:
                    reasons.append("first_innings_target_not_trusted")
                if not complete and runs < first_total + 1:
                    reasons.append("chase_ended_early_target_or_coverage_unknown")
                margin = re.search(r"won\s+(?:by\s+)?(\d+)\s+runs?", match.result, re.I)
                if margin and first_total - runs != int(margin.group(1)):
                    reasons.append("reported_margin_not_reconciled")
            if not reasons:
                state = InningsState(first_total + 1 if number == 2 else None)
                try:
                    for event in innings:
                        state.before(event)
                        state.advance(event)
                except InputError:
                    reasons.append("events_after_innings_termination")
        if number == 1:
            first_valid = not reasons
        output.append({
            "match_id": match.id, "match_date": match.day.isoformat(), "innings": number,
            "delivery_rows": len(innings), "legal_balls": legal, "wickets": wickets, "runs": runs,
            "target": first_total + 1 if number == 2 and first_valid else None,
            "status": "included" if not reasons else "withheld",
            "reasons": ";".join(sorted(set(reasons))),
        })
    return output


class ParHistory:
    def __init__(self, config):
        self.config = config
        self.families = defaultdict(lambda: deque(maxlen=config.par_family_window))
        self.venues = defaultdict(lambda: deque(maxlen=config.par_venue_window))

    def estimate(self, family, venue):
        values = self.families[family]
        prior = self.config.par_prior_innings
        family_par = (sum(values) + prior * 165.0) / (len(values) + prior)
        local = self.venues[family, venue]
        par = (sum(local) + prior * family_par) / (len(local) + prior)
        return clip(par, 90, 260), family_par, len(local), len(values)

    def update(self, family, venue, runs):
        self.families[family].append(runs)
        self.venues[family, venue].append(runs)


def live_pressure_components(snapshot, par):
    balls = snapshot.balls_remaining
    wickets = 10 - snapshot.wickets
    if balls <= 0 or wickets <= 0:
        return 0.0, 0.0, 0.5, 0.0
    rate = par / 120
    urgency = (1 - balls / 120) ** 1.5
    scarcity = clip((6 - wickets) / 5)
    remaining_capacity = rate * min(balls, wickets / 0.05)
    if snapshot.target is not None:
        required = snapshot.target - snapshot.runs
        if required <= 0:
            return 0.0, 0.0, 1.0, 0.0
        rate_ratio = (required / balls) / rate
        demand_bat = clip(0.5 + 0.55 * (rate_ratio - 1))
        demand_bowl = clip(0.5 + 0.55 * (1 / max(0.05, rate_ratio) - 1))
        z = (remaining_capacity - required) / max(3.0, 2.2 * math.sqrt(balls))
        contest_proxy = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        # Only near-certain outcomes taper strongly; a last-ball boundary can remain demanding.
        live_fraction = min(1.0, min(contest_proxy, 1 - contest_proxy) / 0.01) ** 0.25
    else:
        projected = snapshot.runs + remaining_capacity
        demand_bat = clip(0.5 + (par - projected) / 60)
        demand_bowl = clip(0.5 + (projected - par) / 60)
        contest_proxy = 0.5
        live_fraction = 1.0
    new_batter = clip((6 - snapshot.striker_balls) / 6) * scarcity
    batting_disruption = max(clip(snapshot.recent_wickets / 3), new_batter)
    established_threat = clip(snapshot.striker_balls / 20) * clip(
        snapshot.striker_runs / max(1, snapshot.striker_balls) / rate - 0.5)
    batting = 20 + 35 * demand_bat + 25 * urgency + 15 * scarcity + 5 * batting_disruption
    bowling = 20 + 35 * demand_bowl + 25 * urgency + 15 * (wickets / 10) + 5 * established_threat
    return batting, bowling, contest_proxy, live_fraction


def live_pressure(snapshot, par):
    batting, bowling, proxy, fraction = live_pressure_components(snapshot, par)
    return clip(batting * fraction, 0, 100), clip(bowling * fraction, 0, 100), proxy


def combined_pressure(prematch, live, uplift, config):
    if not 0 <= prematch <= 100 or not 0 <= live <= 100 or not 0 <= uplift <= 0.35 + 1e-9:
        raise InputError("Invalid pressure layer or approved responsibility uplift.")
    team = config.prematch_share * prematch + (1 - config.prematch_share) * live
    raw = team * (1 + uplift)
    return team, min(100.0, raw), raw


def pressure_weight(pressure, config):
    return (max(0.0, pressure - config.pressure_floor) / (100 - config.pressure_floor)) ** 2
