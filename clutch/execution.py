import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .inputs import InputError, iter_csv, read_csv
from .live import CREDITED_WICKETS, NO_BALL
from .match_impact import high_match_rating
from .performance import ledger_schema
from .ratings import rating_components
from .statistics import clip

# Preserve the published match-resampling draws across packaging changes.
BOOTSTRAP_NAMESPACE = "performance-v3"

YORKERS = {"Yorker", "Wide Yorker"}

VARIATIONS = {
    "Off Cutter", "Leg Cut", "Back of Hand", "Knuckle", "Slow Ball", "Slow Bouncer",
    "Googly", "Carrom", "Arm ball", "Flipper", "Slider", "Quicker Ball", "Top Spin", "Doosra",
}

PACE_MOVEMENT = {
    "Seamed In", "Seamed Out", "Inswing", "Out Swing", "Reverse Swing",
    "Reverse Swing In", "Reverse Swing Out", "Back of Hand", "Knuckle", "Off Cutter", "Leg Cut",
}

SPIN_MOVEMENT = {"Orthodox", "Leg Break", "Off Break", "Googly", "Carrom", "Arm ball",
                 "Chinaman", "Flipper", "Slider", "Top Spin", "Doosra"}

KNOWN_STYLES = YORKERS | VARIATIONS | PACE_MOVEMENT | SPIN_MOVEMENT | {
    "Straight", "Angled In", "Angled Away", "Bouncer", "Full Toss", "Beamer", "Flick",
}

DEFENSIVE_SHOTS = {"Frontfoot defensive", "Backfoot defensive", "Defense", "Intentional Padding"}

INNOVATIVE_SHOTS = {"Paddle Scoop", "Reverse Sweep", "UpperCut", "Paddle Sweep",
                    "Reverse Paddle Scoop", "Switch", "Inside out"}

FEATURES = [
    "recorded_yorker", "recorded_variation", "effective_yorker", "effective_variation",
    "effective_stock_delivery", "effective_scoring_response",
    "full_toss", "wide", "no_ball", "confirmed_beamer", "post_boundary_recovery",
    "post_dot_release", "loose_ball_punished", "loose_ball_missed",
    "difficult_delivery_managed", "protective_defense", "wide_stumping_loss",
]

@dataclass(frozen=True)
class ExecutionConfig:
    outcome_share: float = 0.50
    execution_share: float = 0.20
    high_match_share: float = 0.30
    crunch_floor: float = 60.0
    sequence_bonus_limit: float = 5.0
    ordinary_ceiling: float = 50.0
    context_prior_events: float = 80.0
    player_prior_events: float = 80.0
    usual_reference_share: float = 0.40
    signal_scale: float = 0.20
    match_evidence_scale: float = 4.0
    prior_matches: float = 5.0
    bootstrap_samples: int = 250

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise InputError("Execution parameters must be finite and positive.")
        if not math.isclose(self.outcome_share + self.execution_share + self.high_match_share, 1):
            raise InputError("Outcome/execution/high-match shares must sum to one.")
        if not 0 < self.ordinary_ceiling < self.crunch_floor < 100 or self.usual_reference_share > 1:
            raise InputError("Invalid execution thresholds or usual-reference share.")

@dataclass(frozen=True, slots=True)
class Annotation:
    style: str
    stroke: str
    pace: str
    extra_type: str
    runs: int
    primary: int
    secondary: int
    wicket: bool
    dismissed: str
    dismissal: str
    striker: str
    bowler: str
    boundary: bool

def annotation_status(annotation):
    if annotation.style in {"", "None"}:
        return "missing_style"
    if annotation.style not in KNOWN_STYLES or annotation.style == "Flick":
        return "unverified_style_semantics"
    if annotation.style == "Beamer" and annotation.extra_type not in NO_BALL:
        return "beamer_without_recorded_no_ball"
    if ((annotation.pace == "Pace" and annotation.style in SPIN_MOVEMENT)
            or (annotation.pace == "Spin" and annotation.style in PACE_MOVEMENT)):
        return "suspect_pace_spin_label"
    if annotation.pace not in {"Pace", "Spin"}:
        return "unknown_bowling_kind"
    return "usable_label"

def delivery_group(annotation):
    if annotation_status(annotation) != "usable_label":
        return "unverified"
    if annotation.style in YORKERS:
        return "yorker"
    if annotation.style in VARIATIONS:
        return "variation"
    if annotation.style == "Full Toss":
        return "full_toss"
    if annotation.style == "Beamer":
        return "unsafe_no_ball"
    return "stock"

def load_annotations(path):
    annotations = {}
    audit = {
        "delivery_styles": Counter(), "strokes": Counter(), "annotation_status": Counter(),
        "rows": 0,
    }
    for row in iter_csv(path):
        key = (row["match_id"], int(row["innings"]), int(row["innings_ball_num"]))
        if key in annotations:
            raise InputError(f"Duplicate execution annotation key: {key}")
        annotation = Annotation(
            row["delivery_style"].strip(), row["stroke"].strip(), row["pace_or_spin"],
            row["extras_type"], int(row["runs"]), int(row["extras1_runs"]), int(row["extras_2_runs"]),
            row["is_wicket"] == "true", row["dismissed_player_id"], row["dismissal_type"],
            row["striker_id"], row["bowler_id"], row["is_four"] == "true" or row["is_six"] == "true")
        annotations[key] = annotation
        audit["rows"] += 1
        audit["delivery_styles"][annotation.style] += 1
        audit["strokes"][annotation.stroke] += 1
        audit["annotation_status"][annotation_status(annotation)] += 1
    return annotations, audit

@dataclass
class SequenceHistory:
    dot_streak: int = 0
    boundary_streak: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=6))
    previous_styles: dict = field(default_factory=dict)

    def before(self, bowler):
        return {
            "prior_dot_streak": self.dot_streak, "prior_boundary_streak": self.boundary_streak,
            "four_plus_last_six": sum(runs >= 4 for runs, _ in self.recent),
            "dots_last_six": sum(dot for _, dot in self.recent),
            "previous_bowler_style": self.previous_styles.get(bowler, ""),
        }

    def advance(self, annotation, state):
        total = state["recorded_total_runs"]
        dot = state["legal_delivery"] and total == 0
        self.dot_streak = self.dot_streak + 1 if dot else 0
        self.boundary_streak = self.boundary_streak + 1 if annotation.boundary else 0
        self.recent.append((total, dot))
        self.previous_styles[annotation.bowler] = annotation.style

def crunch_context(role, state, sequence, personal_pressure, prematch_pressure, config):
    sequence_strength = (
        clip(sequence["prior_dot_streak"] / 3) if role == "batting"
        else clip(sequence["four_plus_last_six"] / 2))
    sequence_bonus = 0.0 if state.get("adaptive_runtime", False) else config.sequence_bonus_limit * sequence_strength
    pressure = min(100.0, personal_pressure + sequence_bonus)
    contexts = []
    if prematch_pressure > 70:
        contexts.append("high_stakes_match")
    if state["phase"] == "death":
        contexts.append("death_overs")
    if role == "batting" and sequence["prior_dot_streak"] >= 2:
        contexts.append("after_dot_chain")
    if role == "bowling" and (sequence["prior_boundary_streak"] >= 1 or sequence["four_plus_last_six"] >= 2):
        contexts.append("after_boundary_or_run_leak")
    if role == "batting" and state["recent_wickets_before"] >= 2:
        contexts.append("recent_collapse")
    if personal_pressure >= 80:
        contexts.append("severe_live_pressure")
    weight = ((max(0, pressure - config.crunch_floor)) / (100 - config.crunch_floor)) ** 2 if contexts else 0.0
    return pressure, weight, ";".join(contexts)

def grade_execution(role, annotation, state, sequence):
    style = annotation.style
    quality = annotation_status(annotation)
    usable = quality == "usable_label"
    legal = state["legal_delivery"]
    total = state["recorded_total_runs"]
    required = state["target"] - state["runs_before"] if state["target"] is not None else None
    useful_runs = (min(annotation.runs, max(0, required - annotation.primary - annotation.secondary))
                   if required is not None else annotation.runs)
    credited = (annotation.wicket and annotation.dismissal in CREDITED_WICKETS
                and annotation.dismissed == annotation.striker
                and not state["free_hit_before"] and annotation.extra_type not in NO_BALL)
    if annotation.extra_type == "wide" and annotation.dismissal not in {"Stump", "Hit Wicket"}:
        credited = False
    if (annotation.extra_type == "wide" and annotation.dismissal == "Stump"
            and required is not None and annotation.primary + annotation.secondary >= required):
        credited = False
    features = set()
    if style in YORKERS:
        features.add("recorded_yorker")
    if style in VARIATIONS:
        features.add("recorded_variation")
    if style == "Full Toss":
        features.add("full_toss")
    if role == "bowling":
        if annotation.extra_type in NO_BALL:
            features.add("no_ball")
            if style == "Beamer":
                features.add("confirmed_beamer")
            return -1.0, features, "recorded_no_ball"
        if annotation.extra_type == "wide":
            features.add("wide")
            return -0.6, features, "recorded_wide"
        if not usable:
            return None, features, quality
        if style == "Full Toss":
            return -0.35, features, "labelled_full_toss_control_error"
        score, reason = 0.0, "routine_recorded_delivery"
        affordable = (max(0, required - 1) / state["balls_remaining"]
                      if required is not None else state["par_before"] / 120)
        effective = legal and (total <= min(2, affordable) or credited)
        if effective:
            if style not in YORKERS | VARIATIONS:
                features.add("effective_stock_delivery")
            score, reason = 0.1, "effective_legal_delivery"
        if style in YORKERS and effective:
            features.add("effective_yorker")
            score, reason = (0.35 if state["phase"] == "death" else 0.25), "effective_recorded_yorker"
        if style in VARIATIONS and effective:
            features.add("effective_variation")
            candidate = 0.20 if state["phase"] == "death" else 0.15
            if candidate > score:
                score, reason = candidate, "effective_recorded_variation"
        if effective and (sequence["prior_boundary_streak"] >= 1 or sequence["four_plus_last_six"] >= 2):
            features.add("post_boundary_recovery")
            if 0.15 > score:
                score, reason = 0.15, "disciplined_recovery_after_run_leak"
        return score, features, reason

    if annotation.wicket and annotation.dismissal not in CREDITED_WICKETS:
        return None, features, "running_or_retirement_blame_unobserved"
    if annotation.extra_type == "wide":
        if credited:
            features.add("wide_stumping_loss")
            return -0.6, features, "dismissal_on_wide"
        return None, features, "no_batting_opportunity_on_wide"
    if style == "Beamer" or annotation.extra_type in NO_BALL:
        return None, features, "unsafe_or_free_delivery_not_graded_as_batting_technique"
    demand = (required / state["balls_remaining"]) if required is not None else None
    high_demand = (
        (state["phase"] == "death" and required is None)
        or (demand is not None and demand >= state["par_before"] / 120)
        or (required is not None and state["balls_remaining"] <= 6 and required >= state["balls_remaining"]))
    score, reason = (0.0, "routine_recorded_response") if usable else (None, quality)
    scoring_demand = demand if demand is not None else state["par_before"] / 120
    if usable and legal and not credited and useful_runs >= clip(scoring_demand, 1, 2):
        features.add("effective_scoring_response")
        score, reason = 0.1, "useful_scoring_response"
    if usable and style == "Full Toss" and legal:
        if credited:
            features.add("loose_ball_missed")
            return -0.6, features, "dismissed_on_labelled_loose_ball"
        needed_for_conversion = min(4, required) if required is not None else 4
        if useful_runs >= needed_for_conversion:
            features.add("loose_ball_punished")
            score, reason = 0.4, "converted_labelled_loose_ball"
        elif useful_runs >= 1:
            score, reason = 0.1, "scored_from_labelled_loose_ball"
        elif high_demand:
            features.add("loose_ball_missed")
            return -0.3, features, "loose_ball_dot_with_scoring_demand"
    if usable and style in YORKERS | VARIATIONS and legal and not credited:
        if useful_runs >= 1:
            features.add("difficult_delivery_managed")
            if score is None or score < 0.2:
                score, reason = 0.2, "useful_response_to_recorded_difficult_delivery"
        else:
            rebuilding = (state["wickets_before"] >= 4 and state["recent_wickets_before"] > 0
                          and state["balls_remaining"] >= 30
                          and (demand is None or demand <= state["par_before"] / 120))
            if rebuilding and annotation.stroke in DEFENSIVE_SHOTS:
                features.add("protective_defense")
                if score is None or score < 0.1:
                    score, reason = 0.1, "contextual_protective_response"
    if legal and not credited and sequence["prior_dot_streak"] >= 2 and useful_runs >= 1:
        features.add("post_dot_release")
        candidate = 0.2 if usable else 0.15
        if score is None or score < candidate:
            score, reason = candidate, "scoring_release_after_dot_chain"
    return score, features, reason

@dataclass
class MeanHistory:
    count: int = 0
    total: float = 0.0

class ExecutionHistory:
    def __init__(self, config):
        self.config = config
        self.context = defaultdict(MeanHistory)
        self.players = defaultdict(MeanHistory)

    def keys(self, role, family, phase, pace, group):
        return [(role, phase, pace), (role, family, phase, pace, group)]

    def references(self, player, role, family, phase, pace, group):
        peer = 0.0
        for key in self.keys(role, family, phase, pace, group):
            cell = self.context[key]
            peer = (cell.total + self.config.context_prior_events * peer) / (
                cell.count + self.config.context_prior_events)
        own = self.players[player, role, phase]
        offset = own.total / (own.count + self.config.player_prior_events)
        return peer, offset, own.count

    def update(self, observation):
        player, role, family, phase, pace, group, signal, peer = observation
        for key in self.keys(role, family, phase, pace, group):
            cell = self.context[key]
            cell.count += 1
            cell.total += signal
        own = self.players[player, role, phase]
        own.count += 1
        own.total += signal - peer

@dataclass
class ExecutionCard:
    player_id: str
    player: str
    role: str
    match_id: str
    match_date: str
    family: str
    opportunities: int = 0
    graded: int = 0
    total_weight: float = 0.0
    graded_weight: float = 0.0
    weighted_signal: float = 0.0
    features: Counter = field(default_factory=Counter)

    def add(self, row):
        if row["execution_weight"] <= 0:
            return
        self.opportunities += 1
        self.total_weight += row["execution_weight"]
        self.features.update(filter(None, row["features"].split(";")))
        if row["scored_signal"] is not None:
            self.graded += 1
            self.graded_weight += row["execution_weight"]
            self.weighted_signal += row["execution_weight"] * row["scored_signal"]

    def evidence(self, config):
        coverage = self.graded_weight / self.total_weight if self.total_weight else 0.0
        return {
            "player_id": self.player_id, "player": self.player, "role": self.role,
            "match_id": self.match_id, "match_date": self.match_date, "family": self.family,
            "crunch_opportunities": self.opportunities, "graded_crunch_opportunities": self.graded,
            "execution_weight": self.total_weight, "graded_execution_weight": self.graded_weight,
            "annotation_evidence_coverage": coverage,
            "mean_execution_signal": self.weighted_signal / self.graded_weight if self.graded_weight else None,
            "match_evidence_weight": min(1.0, self.graded_weight / config.match_evidence_scale) * coverage,
            **{name: self.features[name] for name in FEATURES},
        }

def execution_rating(cards, config):
    available = [row for row in cards if row is not None and row["mean_execution_signal"] is not None
                 and row["match_evidence_weight"] > 0]
    evidence = sum(row["match_evidence_weight"] for row in available)
    if not evidence:
        return 50.0, None, 0.0
    mean = sum(row["match_evidence_weight"] * row["mean_execution_signal"] for row in available) / evidence
    support = evidence / (evidence + config.prior_matches)
    return 50 + support * 25 * math.tanh(mean / config.signal_scale), mean, support

def percentile(value, reference):
    if not reference:
        return None
    lower = sum(score < value - 1e-12 for score in reference)
    tied = sum(abs(score - value) <= 1e-12 for score in reference)
    return 100 * (lower + 0.5 * tied) / len(reference)

EXECUTION_SCHEMA = ledger_schema(
    ["match_id", "match_date", "family", "player_id", "player", "role", "phase", "pace_or_spin",
     "delivery_style", "stroke", "previous_bowler_style", "annotation_status", "crunch_context",
     "features", "scoring_reason", "responsibility_basis"],
    ["innings", "sequence", "prior_dot_streak", "prior_boundary_streak", "four_plus_last_six",
     "dots_last_six", "ordinary_reference_count_before", "legal_opportunity", "recorded_bat_runs",
     "recorded_team_runs"],
    ["team_pressure", "base_personal_pressure", "execution_pressure", "execution_weight",
     "raw_signal", "peer_reference_before", "personal_reference_offset_before", "scored_signal"],
    ["training_eligible", "eligible_execution_opportunity", "recorded_wicket", "runtime_context_in_parent"])

def task_group(role, state):
    rate = ((state["target"] - state["runs_before"]) / state["balls_remaining"]
            if state["target"] is not None else None)
    if role == "batting":
        if (state["wickets_before"] >= 4 and state["balls_remaining"] >= 30
                and (rate is None or rate <= state["par_before"] / 120)):
            return "rebuild"
        if state["phase"] == "death" or (rate is not None and rate > state["par_before"] / 120):
            return "accelerate"
        return "normal"
    if rate is None:
        return "restrict_first_innings"
    return "tight_defence" if rate <= state["par_before"] / 120 else "protect_run_budget"

def iter_ball_states(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()

def collect_execution(raw, ball_base, annotations, config, emit):
    manifest = json.loads((ball_base / "run_manifest.json").read_text(encoding="utf-8"))
    prematch = Path(manifest["prematch_manifest"]).parent
    overlays = {
        (row["match_id"], row["team_id"], row["player_id"], row["role"]): row
        for row in read_csv(prematch / "player_pressure.csv")
    }
    names = {row["player_id"]: row["player_name"]
             for row in read_csv(raw / "competition_team_player_mapping.csv")}
    history = ExecutionHistory(config)
    sequences = defaultdict(SequenceHistory)
    cards, profiles, action_profiles = {}, {}, {}
    diagnostics = Counter()
    previous_day = ""
    for day, states in groupby(iter_ball_states(ball_base / "ball_pressure.parquet"), key=lambda row: row["match_date"]):
        if day < previous_day:
            raise InputError("The frozen ball ledger is not chronologically ordered.")
        previous_day = day
        pending = []
        for state in states:
            key = (state["match_id"], state["innings"], state["sequence"])
            annotation = annotations[key]
            if annotation.striker != state["striker"] or annotation.bowler != state["bowler"]:
                raise InputError(f"Annotation actor mismatch: {key}")
            sequence = sequences[state["match_id"], state["innings"]]
            before = sequence.before(annotation.bowler)
            if annotation.extra_type == "Penalty":
                sequence.advance(annotation, state)
                continue
            for role, player, team in [
                ("batting", state["striker"], state["batting_team"]),
                ("bowling", state["bowler"], state["bowling_team"]),
            ]:
                if role == "batting" and annotation.extra_type == "wide" and not annotation.wicket:
                    continue
                if player not in names:
                    raise InputError(f"Execution actor has no source identity: {player}")
                common = state["batting_team_pressure" if role == "batting" else "bowling_team_pressure"]
                prematch_pressure = state["batting_prematch_pressure" if role == "batting" else "bowling_prematch_pressure"]
                overlay = overlays.get((state["match_id"], team, player, role))
                known = overlay is not None and overlay["applied_combined_uplift_0_1"] != ""
                if state.get("adaptive_runtime", False):
                    personal = state["batting_personal_pressure" if role == "batting" else "bowling_personal_pressure"]
                    responsibility_basis = state["batting_dependency_status" if role == "batting" else "bowling_dependency_status"]
                else:
                    uplift = float(overlay["applied_combined_uplift_0_1"]) if known else 0.0
                    personal = min(100.0, common * (1 + uplift))
                    responsibility_basis = "known_prematch_overlay" if known else "team_context_only"
                pressure, weight, context = crunch_context(role, state, before, personal, prematch_pressure, config)
                raw_signal, features, reason = grade_execution(role, annotation, state, before)
                quality = annotation_status(annotation)
                opportunity = not (role == "batting" and (
                    annotation.extra_type in NO_BALL or annotation.style == "Beamer"
                    or (annotation.wicket and annotation.dismissal not in CREDITED_WICKETS)))
                group = (delivery_group(annotation) if role == "batting" else quality) + ":" + task_group(role, state)
                peer, offset, prior_count = history.references(
                    player, role, state["family"], state["phase"], annotation.pace, group)
                scored = None
                if raw_signal is not None and opportunity:
                    reference = clip(peer + offset, -1, 0.4)
                    scored = raw_signal - config.usual_reference_share * reference
                    if raw_signal < 0:
                        scored = min(0.5 * raw_signal, scored)
                ordinary = scored is not None and common <= config.ordinary_ceiling and weight == 0
                row = {
                    "match_id": state["match_id"], "match_date": day, "family": state["family"],
                    "player_id": player, "player": names[player], "role": role, "phase": state["phase"],
                    "pace_or_spin": annotation.pace, "delivery_style": annotation.style, "stroke": annotation.stroke,
                    "previous_bowler_style": before["previous_bowler_style"], "annotation_status": quality,
                    "crunch_context": context, "features": ";".join(sorted(features)), "scoring_reason": reason,
                    "responsibility_basis": responsibility_basis,
                    "innings": state["innings"], "sequence": state["sequence"],
                    "prior_dot_streak": before["prior_dot_streak"],
                    "prior_boundary_streak": before["prior_boundary_streak"],
                    "four_plus_last_six": before["four_plus_last_six"], "dots_last_six": before["dots_last_six"],
                    "ordinary_reference_count_before": prior_count,
                    "legal_opportunity": int(state["legal_delivery"]), "recorded_bat_runs": annotation.runs,
                    "recorded_team_runs": state["recorded_total_runs"], "recorded_wicket": annotation.wicket,
                    "team_pressure": common, "base_personal_pressure": personal,
                    "execution_pressure": pressure, "execution_weight": weight,
                    "raw_signal": raw_signal, "peer_reference_before": peer,
                    "personal_reference_offset_before": offset, "scored_signal": scored,
                    "training_eligible": ordinary, "eligible_execution_opportunity": opportunity,
                    "runtime_context_in_parent": state.get("adaptive_runtime", False),
                }
                emit(row)
                diagnostics["ledger_rows"] += 1
                if weight > 0 and opportunity:
                    diagnostics["crunch_opportunities"] += 1
                    diagnostics["crunch_" + quality] += 1
                    if scored is None:
                        diagnostics["ungraded_crunch_opportunities"] += 1
                    card_key = (player, role, state["match_id"])
                    if card_key not in cards:
                        cards[card_key] = ExecutionCard(
                            player, names[player], role, state["match_id"], day, state["family"])
                    cards[card_key].add(row)
                    profile_key = (player, role, state["phase"])
                    if profile_key not in profiles:
                        profiles[profile_key] = {
                            "player_id": player, "player": names[player], "role": role, "phase": state["phase"],
                            "crunch_opportunities": 0, "graded_opportunities": 0,
                            **{name: 0 for name in FEATURES},
                        }
                    profile = profiles[profile_key]
                    profile["crunch_opportunities"] += 1
                    profile["graded_opportunities"] += int(scored is not None)
                    for feature in features:
                        profile[feature] += 1
                    action = annotation.style if role == "bowling" else annotation.stroke
                    action_key = (player, role, state["phase"], action)
                    if action_key not in action_profiles:
                        action_profiles[action_key] = {
                            "player_id": player, "player": names[player], "role": role, "phase": state["phase"],
                            "action_type": "delivery_style" if role == "bowling" else "stroke",
                            "action_label": action, "crunch_opportunities": 0, "legal_opportunities": 0,
                            "off_bat_runs": 0, "team_runs_from_deliveries": 0, "recorded_wicket_events": 0,
                            "wides": 0, "no_balls": 0, "graded_execution_weight": 0.0,
                            "weighted_execution_signal": 0.0,
                        }
                    action_profile = action_profiles[action_key]
                    action_profile["crunch_opportunities"] += 1
                    action_profile["legal_opportunities"] += int(state["legal_delivery"])
                    action_profile["off_bat_runs"] += annotation.runs
                    action_profile["team_runs_from_deliveries"] += state["recorded_total_runs"]
                    action_profile["recorded_wicket_events"] += int(annotation.wicket)
                    action_profile["wides"] += int(annotation.extra_type == "wide")
                    action_profile["no_balls"] += int(annotation.extra_type in NO_BALL)
                    if scored is not None:
                        action_profile["graded_execution_weight"] += weight
                        action_profile["weighted_execution_signal"] += weight * scored
                if ordinary:
                    pending.append((player, role, state["family"], state["phase"],
                                    annotation.pace, group, raw_signal, peer))
            sequence.advance(annotation, state)
        for observation in pending:
            history.update(observation)
    evidence = [card.evidence(config) for card in cards.values()]
    evidence.sort(key=lambda row: (row["match_date"], row["match_id"], row["role"], row["player_id"]))
    profile_rows = sorted(profiles.values(), key=lambda row: (row["role"], row["player_id"], row["phase"]))
    for row in profile_rows:
        n = row["crunch_opportunities"]
        row["wide_rate_per_crunch_opportunity"] = row["wide"] / n if row["role"] == "bowling" else None
        row["no_ball_rate_per_crunch_opportunity"] = row["no_ball"] / n if row["role"] == "bowling" else None
        row["annotation_grade_coverage"] = row["graded_opportunities"] / n
    action_rows = sorted(action_profiles.values(), key=lambda row: (
        row["role"], row["player_id"], row["phase"], row["action_label"]))
    return evidence, profile_rows, action_rows, dict(diagnostics)

def combine_scores(outcome, execution, high_match, config, execution_share=None):
    share = config.execution_share if execution_share is None else execution_share
    outcome_share = 1 - config.high_match_share - share
    if not 0 <= share <= 1 or outcome_share < 0:
        raise InputError("Invalid execution sensitivity allocation.")
    return outcome_share * outcome + share * execution + config.high_match_share * high_match

def final_ratings(base_profiles, execution_cards, ball_cards, high_cards, config, ball_config, high_config):
    grouped_execution, grouped_high = defaultdict(list), defaultdict(list)
    for row in execution_cards:
        grouped_execution[row["player_id"], row["role"]].append(row)
    for row in high_cards:
        grouped_high[row["player_id"], row["role"]].append(row)
    result = []
    for original in base_profiles:
        key = original["player_id"], original["role"]
        cards = grouped_execution[key]
        score, mean, support = execution_rating(cards, config)
        outcome, high_match = float(original["ball_channel_score"]), float(original["high_match_channel_score"])
        raw_index = combine_scores(outcome, score, high_match, config)
        eligible = original["eligible"] == "True"
        lower = upper = None
        if eligible:
            old = {row["match_id"]: row for row in ball_cards[key]}
            high = {row["match_id"]: row for row in grouped_high[key]}
            execution = {row["match_id"]: row for row in cards}
            matches = sorted(set(old) | set(high) | set(execution))
            seed = int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_NAMESPACE}:{key}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            samples = []
            for _ in range(config.bootstrap_samples):
                selected = [matches[int(i)] for i in rng.integers(0, len(matches), len(matches))]
                b = rating_components([old[mid] for mid in selected if mid in old], ball_config)[0]
                m = high_match_rating([high.get(mid) for mid in selected], high_config)[0]
                e = execution_rating([execution.get(mid) for mid in selected], config)[0]
                samples.append(combine_scores(b, e, m, config))
            lower, upper = (float(value) for value in np.quantile(samples, [0.025, 0.975]))
        opportunity_count = sum(row["crunch_opportunities"] for row in cards)
        graded_count = sum(row["graded_crunch_opportunities"] for row in cards)
        result.append({
            "rank": None, "player_id": key[0], "player": original["player"], "role": key[1],
            "performance_percentile": None, "raw_performance_index": raw_index,
            "outcome_score": outcome, "execution_score": score, "high_match_score": high_match,
            "outcome_points": config.outcome_share * outcome,
            "execution_points": config.execution_share * score,
            "high_match_points": config.high_match_share * high_match,
            "gentle_high_match_adjustment": float(original.get("gentle_high_match_adjustment", 0)),
            "gentle_final_rating_adjustment": config.high_match_share * float(original.get("gentle_high_match_adjustment", 0)),
            "helped_win_mark_matches": int(original.get("helped_win_mark_matches", 0)),
            "negative_delivery_mark_matches": int(original.get("negative_delivery_mark_matches", 0)),
            "eligible": eligible, "eligibility_route": original["eligibility_route"],
            "execution_evidence_status": "observed" if graded_count else "no_graded_crunch_evidence",
            "crunch_matches": len(cards), "crunch_opportunities": opportunity_count,
            "graded_crunch_opportunities": graded_count,
            "execution_annotation_coverage": graded_count / opportunity_count if opportunity_count else None,
            "execution_evidence_support": support, "mean_execution_signal": mean,
            "raw_index_interval_low": lower, "raw_index_interval_high": upper,
            "percentile_interval_low": None, "percentile_interval_high": None,
            "eligible_role_reference_count": None,
            "raw_index_execution10": combine_scores(outcome, score, high_match, config, 0.10),
            "raw_index_execution30": combine_scores(outcome, score, high_match, config, 0.30),
            "raw_index_no_execution": combine_scores(outcome, score, high_match, config, 0.0),
            **{feature: sum(row[feature] for row in cards) for feature in FEATURES},
        })
    for role in ["batting", "bowling"]:
        pool = sorted([row for row in result if row["role"] == role and row["eligible"]],
                      key=lambda row: (-row["raw_performance_index"], row["player_id"]))
        reference = [row["raw_performance_index"] for row in pool]
        for rank, row in enumerate(pool, 1):
            row["rank"] = rank
            row["performance_percentile"] = percentile(row["raw_performance_index"], reference)
            row["percentile_interval_low"] = percentile(row["raw_index_interval_low"], reference)
            row["percentile_interval_high"] = percentile(row["raw_index_interval_high"], reference)
            row["eligible_role_reference_count"] = len(pool)
        for name in ["execution10", "execution30", "no_execution"]:
            for rank, row in enumerate(sorted(pool, key=lambda row: (-row["raw_index_" + name], row["player_id"])), 1):
                row["rank_" + name] = rank
    for row in result:
        for name in ["execution10", "execution30", "no_execution"]:
            row.setdefault("rank_" + name, None)
    return sorted(result, key=lambda row: (row["role"], row["rank"] is None, row["rank"] or 0, row["player_id"]))
