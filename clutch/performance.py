from collections import Counter
from itertools import groupby
import pyarrow as pa
import pyarrow.parquet as pq
from .inputs import InputError
from .live import InningsState, ParHistory, combined_pressure, innings_quality, live_pressure, pressure_weight
from .ratings import HistoricalBaselines, MatchCard, performance_observation
from .runtime import RuntimeState, adaptive_pressure, runtime_responsibility

def ledger_schema(strings, integers, floats, booleans=()):
    return pa.schema(
        [(name, pa.string()) for name in strings]
        + [(name, pa.int64()) for name in integers]
        + [(name, pa.float64()) for name in floats]
        + [(name, pa.bool_()) for name in booleans])

BALL_SCHEMA = ledger_schema(
    ["match_id", "match_date", "family", "batting_team", "bowling_team", "striker", "bowler", "phase",
     "batting_dependency_status", "bowling_dependency_status"],
    ["innings", "sequence", "runs_before", "wickets_before", "legal_balls_before", "balls_remaining",
     "target", "recent_wickets_before", "striker_balls_before", "striker_runs_before",
     "recorded_total_runs", "prior_venue_innings", "prior_family_innings",
     "prior_dot_streak", "four_plus_last_six", "partnership_runs_before", "partnership_balls_before",
     "bowler_legal_balls_before", "bowler_quota_balls_remaining",
     "remaining_batting_options", "remaining_bowling_options"],
    ["par_before", "batting_live_pressure", "bowling_live_pressure", "batting_team_pressure",
     "bowling_team_pressure", "batting_prematch_pressure", "bowling_prematch_pressure", "contest_balance_proxy",
     "batting_runtime_dependency", "bowling_runtime_dependency",
     "batting_responsibility_uplift", "bowling_responsibility_uplift",
     "batting_personal_pressure", "bowling_personal_pressure",
     "practical_liveness", "practical_runs_per_ball_limit", "required_runs_per_over"],
    ["free_hit_before", "legal_delivery", "recorded_wicket", "adaptive_runtime", "short_burst_context"])

PLAYER_SCHEMA = ledger_schema(
    ["match_id", "match_date", "family", "player_id", "player", "role", "team_id",
     "phase", "position_band", "flags", "responsibility_source_flags"],
    ["innings", "sequence", "legal_opportunity", "attributed_runs", "useful_attributed_runs", "credited_wicket",
     "normal_reference_events_before", "context_prior_legal_balls"],
    ["team_pressure", "personal_pressure", "personal_pressure_upper", "personal_pressure_uncapped",
     "applied_responsibility_uplift", "pressure_weight", "expected_runs_per_legal_ball",
     "expected_credited_wicket_rate", "wicket_value_before", "absolute_contribution",
     "usual_reference_before", "contribution_vs_usual", "normal_reference_support"],
    ["high_pressure"])

class ParquetLedger:
    def __init__(self, path, schema):
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.rows = []
        self.count = 0

    def add(self, row):
        if set(row) != set(self.schema.names):
            raise InputError(f"Ledger schema mismatch: {set(row) ^ set(self.schema.names)}")
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= 8192:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.writer.close()

def collect_card(cards, row):
    key = (row["player_id"], row["role"], row["match_id"])
    if key not in cards:
        cards[key] = MatchCard(
            row["player_id"], row["role"], row["match_id"], row["match_date"], row["family"], row["team_id"])
    cards[key].add(row)

def analyze(data, deliveries, metadata, prematch_teams, prematch_players, voided, config, emit_ball, emit_player):
    baselines, pars = HistoricalBaselines(config), ParHistory(config)
    quality, attribution, cards = [], [], {}
    flags = Counter()
    ball_count = player_count = 0
    for day, group in groupby(sorted(data.matches, key=lambda m: (m.day, m.id)), key=lambda m: m.day):
        pending, par_updates = [], []
        for match in group:
            events = deliveries.get(match.id, [])
            checks = innings_quality(match, events, metadata[match.id], match.id in voided)
            if any(prematch_teams[match.id, team]["team_pressure_score"] == "" for team in match.teams):
                for check in checks:
                    check["status"] = "withheld"
                    check["reasons"] = ";".join(filter(None, [check["reasons"], "prematch_pressure_unresolved"]))
            quality.extend(checks)
            family = data.competitions[match.comp]["competition_family_id"]
            context_family = family + ":" + data.competitions[match.comp]["season"]
            par, family_par, local_history, family_history = pars.estimate(family, match.venue)
            for check in checks:
                if check["status"] != "included":
                    continue
                number = check["innings"]
                state = InningsState(check["target"])
                runtime = RuntimeState()
                for event in (event for event in events if event.innings == number):
                    snapshot = state.before(event)
                    if config.adaptive_runtime:
                        batting_live, bowling_live, contest, runtime_info = adaptive_pressure(snapshot, par, runtime, config)
                    else:
                        batting_live, bowling_live, contest = live_pressure(snapshot, par)
                        runtime_info = {
                            "practical_liveness": None, "practical_runs_per_ball_limit": None,
                            "required_runs_per_over": (
                                6 * (snapshot.target - snapshot.runs) / snapshot.balls_remaining
                                if snapshot.target is not None else None),
                            "short_burst_context": False,
                        }
                    pre_bat = float(prematch_teams[match.id, event.batting_team]["team_pressure_score"])
                    pre_bowl = float(prematch_teams[match.id, event.bowling_team]["team_pressure_score"])
                    batting_team, _, _ = combined_pressure(pre_bat, batting_live, 0, config)
                    bowling_team, _, _ = combined_pressure(pre_bowl, bowling_live, 0, config)
                    responsibilities = {
                        role: runtime_responsibility(
                            data, prematch_players, match, team, player, role, snapshot, runtime, config)
                        for role, player, team in [
                            ("batting", event.striker, event.batting_team),
                            ("bowling", event.bowler, event.bowling_team)]
                    }
                    personal_bat = min(100, batting_team * (1 + responsibilities["batting"]["uplift"]))
                    personal_bowl = min(100, bowling_team * (1 + responsibilities["bowling"]["uplift"]))
                    emit_ball({
                        "match_id": match.id, "match_date": day.isoformat(), "family": family,
                        "innings": number, "sequence": event.sequence,
                        "batting_team": event.batting_team, "bowling_team": event.bowling_team,
                        "striker": event.striker, "bowler": event.bowler, "phase": snapshot.phase,
                        "runs_before": snapshot.runs, "wickets_before": snapshot.wickets,
                        "legal_balls_before": snapshot.legal_balls, "balls_remaining": snapshot.balls_remaining,
                        "target": snapshot.target, "recent_wickets_before": snapshot.recent_wickets,
                        "striker_balls_before": snapshot.striker_balls, "striker_runs_before": snapshot.striker_runs,
                        "free_hit_before": snapshot.free_hit, "par_before": par,
                        "prior_venue_innings": local_history, "prior_family_innings": family_history,
                        "batting_live_pressure": batting_live, "bowling_live_pressure": bowling_live,
                        "batting_team_pressure": batting_team, "bowling_team_pressure": bowling_team,
                        "batting_prematch_pressure": pre_bat, "bowling_prematch_pressure": pre_bowl,
                        "contest_balance_proxy": contest, "legal_delivery": event.legal,
                        "recorded_total_runs": event.total_runs, "recorded_wicket": event.wicket,
                        "adaptive_runtime": config.adaptive_runtime, **runtime.before(), **runtime_info,
                        "bowler_legal_balls_before": runtime.bowler_balls[event.bowler],
                        "bowler_quota_balls_remaining": max(0, 24 - runtime.bowler_balls[event.bowler]),
                        "remaining_batting_options": responsibilities["batting"]["remaining_options"],
                        "remaining_bowling_options": responsibilities["bowling"]["remaining_options"],
                        "batting_runtime_dependency": responsibilities["batting"]["dependency"],
                        "bowling_runtime_dependency": responsibilities["bowling"]["dependency"],
                        "batting_responsibility_uplift": responsibilities["batting"]["uplift"],
                        "bowling_responsibility_uplift": responsibilities["bowling"]["uplift"],
                        "batting_personal_pressure": personal_bat, "bowling_personal_pressure": personal_bowl,
                        "batting_dependency_status": responsibilities["batting"]["status"],
                        "bowling_dependency_status": responsibilities["bowling"]["status"],
                    })
                    ball_count += 1
                    for role, player, team, live, prematch in [
                        ("batting", event.striker, event.batting_team, batting_live, pre_bat),
                        ("bowling", event.bowler, event.bowling_team, bowling_live, pre_bowl),
                    ]:
                        if player not in data.players:
                            attribution.append({
                                "match_id": match.id, "innings": number, "sequence": event.sequence,
                                "role": role, "player_id": player, "reason": "player_identity_unresolved"})
                            continue
                        responsibility = responsibilities[role]
                        source = responsibility["source"]
                        uplift = responsibility["uplift"]
                        responsibility_status = responsibility["status"]
                        common, personal, uncapped = combined_pressure(prematch, live, uplift, config)
                        row, observation, warnings = performance_observation(
                            event, snapshot, role, context_family, par, family_par, baselines,
                            common, personal, responsibility_status, config)
                        flags.update(warnings)
                        if row is None:
                            if warnings:
                                attribution.append({
                                    "match_id": match.id, "innings": number, "sequence": event.sequence,
                                    "role": role, "player_id": player, "reason": ";".join(warnings)})
                            continue
                        row.update({
                            "match_date": day.isoformat(), "family": family, "player": data.players[player],
                            "personal_pressure_upper": min(100.0, common * 1.35),
                            "personal_pressure_uncapped": uncapped,
                            "applied_responsibility_uplift": uplift,
                            "responsibility_source_flags": ";".join(filter(None, [
                                source["flags"] if source else "no_reconstructed_starting_xi_overlay",
                                responsibility_status if config.adaptive_runtime else "",
                            ])),
                        })
                        if not all(0 <= row[key] <= 100 for key in ["personal_pressure", "team_pressure"]):
                            raise InputError("Invalid composed pressure.")
                        emit_player(row)
                        player_count += 1
                        pending.append(observation)
                        collect_card(cards, row)
                    runtime.advance(event, snapshot)
                    state.advance(event)
                if number == 1:
                    par_updates.append((family, match.venue, state.runs))
        for observation in pending:
            baselines.update(observation)
        for observation in par_updates:
            pars.update(*observation)
    match_rows = [card.evidence(config) for card in cards.values()]
    match_rows.sort(key=lambda row: (row["match_date"], row["match_id"], row["role"], row["player_id"]))
    for row in match_rows:
        row["player"] = data.players[row["player_id"]]
        row["team"] = data.teams[row["team_id"]]
    keys = sorted({(row["player_id"], row["role"]) for row in match_rows})
    profiles = [
        {"player_id": player, "player": data.players[player], "role": role,
         "ordinary_reference_events": baselines.normal[player, role].events}
        for player, role in keys
    ]
    summary = {
        "matches_in_source": len(data.matches), "included_innings": sum(row["status"] == "included" for row in quality),
        "withheld_innings": sum(row["status"] == "withheld" for row in quality),
        "delivery_states": ball_count, "attributed_player_delivery_rows": player_count,
        "player_match_role_rows": len(match_rows), "individual_flag_counts": dict(flags),
        "withheld_innings_reason_counts": dict(Counter(
            reason for row in quality for reason in row["reasons"].split(";") if reason)),
        "super_over_rows_not_modelled": sum(event.innings > 2 for events in deliveries.values() for event in events),
    }
    if ball_count != sum(row["delivery_rows"] for row in quality if row["status"] == "included"):
        raise InputError("Included innings do not match the delivery ledger.")
    if len({(row["player_id"], row["role"], row["match_id"]) for row in match_rows}) != len(match_rows):
        raise InputError("Duplicate player-match contribution cards.")
    return quality, attribution, match_rows, profiles, summary
