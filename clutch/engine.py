import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from itertools import groupby

from .calibration import StageCalibration, calibrate
from .data import Dataset, Match
from .inputs import Policy
from .qualification import QualificationUnresolved
from .rules import RuleBook
from .statistics import StatisticalHistory, clip


LOG = logging.getLogger(__name__)


@dataclass
class Results:
    teams: list[dict]
    players: list[dict]
    standings: list[dict]
    exceptions: list[dict]
    summary: dict


def player_adjustment(
    team_score: float, dependency: float, leadership: float,
    dependency_limit: float, combined_cap: float,
) -> tuple[float, float, float]:
    if not 0 <= team_score <= 100 or not 0 <= dependency <= 1:
        raise ValueError("Invalid team score or dependency.")
    if any(not 0 <= value <= 1 for value in [leadership, dependency_limit, combined_cap]):
        raise ValueError("Uplift proportions must be in [0,1].")
    uplift = min(combined_cap, leadership + dependency_limit * dependency)
    uncapped = team_score * (1 + uplift)
    return min(100.0, uncapped), uncapped, uplift


def applicable(entry: dict, match: Match, team: str) -> bool:
    if entry["team_id"] != team:
        return False
    if entry.get("comp_id") and entry["comp_id"] != match.comp:
        return False
    if entry.get("match_id") and entry["match_id"] != match.id:
        return False
    return (
        date.fromisoformat(entry["known_from"]) <= match.day
        and date.fromisoformat(entry["valid_from"]) <= match.day
        and (not entry.get("valid_to") or match.day <= date.fromisoformat(entry["valid_to"]))
    )


def score_dataset(
    data: Dataset, policy: Policy, rules: RuleBook, until: date | None = None,
    calibration: StageCalibration = StageCalibration(),
) -> Results:
    history = StatisticalHistory()
    team_rows, player_rows, standings = [], [], []
    issues = list(data.issues)
    previous_ipl_champion = None
    pending = [m for m in data.matches if until is None or m.day <= until]
    for day, day_group in groupby(pending, key=lambda match: match.day):
        day_matches = list(day_group)
        inactive = rules.inactive_ids(day)
        for mid in inactive:
            history.withdraw_performance(mid)
        expected = {m.id: history.win_probability(m.home, m.away) for m in day_matches}
        rates = {data.competitions[m.comp]["competition_family_id"]:
                 history.pool_rates(data.competitions[m.comp]["competition_family_id"])
                 for m in day_matches}
        for match in day_matches:
            if match.id in inactive:
                raise ValueError(f"A match is inactive at its own start: {match.id}")
            comp = data.competitions[match.comp]
            family = comp["competition_family_id"]
            snapshots_seen = set()
            for team in match.teams:
                opponent = match.opponent(team)
                probability = expected[match.id] if team == match.home else 1 - expected[match.id]
                flags = []
                try:
                    stakes, snapshots = rules.assess(match, team)
                    stakes_value, reason = stakes.score, stakes.reason
                    flags.extend(stakes.flags)
                    loss_possible = stakes.possible_after_loss
                    win_guaranteed = stakes.guaranteed_after_win
                    qualification_context = stakes.qualification_context
                    remaining_matches = stakes.remaining_matches
                    loss_recovery_guaranteed = stakes.guaranteed_if_lose_then_win_out
                    win_out_guaranteed = stakes.guaranteed_if_win_out
                    points_before_match = stakes.points_before_match
                    max_qualification_points = stakes.max_points_if_win_out
                except QualificationUnresolved as exc:
                    LOG.error("Match %s team %s: %s", match.id, team, exc)
                    stakes_value, reason, snapshots = None, "solver_unresolved", []
                    loss_possible, win_guaranteed = None, None
                    qualification_context = ""
                    remaining_matches = loss_recovery_guaranteed = win_out_guaranteed = None
                    points_before_match = max_qualification_points = None
                    flags.append("solver_unresolved")
                    issues.append({"match_id": match.id, "team_id": team,
                                   "issue": "solver_unresolved", "detail": str(exc)})
                for snapshot in snapshots:
                    key = (snapshot["phase"], snapshot["team_id"])
                    if key not in snapshots_seen:
                        snapshots_seen.add(key)
                        standings.append(snapshot)

                importance = policy.competition_value(match.comp, family, comp["season"])
                rivalry = policy.rivalry_value(team, opponent)
                champion = rules.reference["prior_champions"].get(match.comp)
                if family == "ipl" and champion is None:
                    champion = previous_ipl_champion
                defending = float(champion == team)
                favourite = clip((probability - 0.5) / 0.25)
                historical = history.historical_strength(team)
                home_entries = [entry for entry in rules.reference["home_expectation_evidence"]
                                if applicable(entry, match, team)]
                if len(home_entries) > 1:
                    raise ValueError(f"Overlapping home-expectation evidence: {match.id}/{team}")
                home_signal = float(home_entries[0]["active"]) if home_entries else 0.0
                signals = {
                    "champion": defending * policy.expectation["defending_champion_expectation"],
                    "favourite": favourite * policy.expectation["strong_favourite_expectation"],
                    "historical": historical * policy.expectation["documented_historical_team_expectations"],
                    "home": home_signal * policy.expectation["documented_home_expectations"],
                }
                expectation = max(signals.values())
                expectation_upper = expectation
                if not home_entries:
                    flags.append("home_expectation_unverified")
                    expectation_upper = max(expectation_upper,
                                            policy.expectation["documented_home_expectations"])
                if champion is None and family in {"ipl", "senior_asia_cup", "emerging_t20"}:
                    flags.append("defending_champion_unverified")
                    expectation_upper = max(expectation_upper,
                                            policy.expectation["defending_champion_expectation"])
                if len(history.results[team]) < 25:
                    flags.append("historical_expectation_limited_history")
                    expectation_upper = max(
                        expectation_upper,
                        policy.expectation["documented_historical_team_expectations"])
                if len(history.results[team]) < 5 or len(history.results[opponent]) < 5:
                    flags.append("elo_cold_start")
                if not history.form[team]:
                    flags.append("form_prior_no_completed_results")
                form = history.underperformance(team)
                opposition = history.opposition_context(team, opponent)
                if calibration is not None:
                    if opposition.strength_support < 0.5:
                        flags.append("opposition_strength_limited_history")
                    if opposition.head_to_head_matches == 0:
                        flags.append("head_to_head_cold_start")
                    elif opposition.head_to_head_support < 0.5:
                        flags.append("head_to_head_limited_history")
                phase = rules.phase(match)
                win_path_guaranteed = win_guaranteed is True or win_out_guaranteed is True
                calibrated = calibrate(
                    policy, importance, stakes_value, expectation, form, rivalry,
                    family, phase, reason, calibration,
                    opposition_strength=opposition.strength_signal,
                    head_to_head=opposition.head_to_head_signal,
                    qualification_context=qualification_context,
                    max_qualification_points=max_qualification_points,
                    qualification_win_path_guaranteed=win_path_guaranteed)
                possible = calibrate(
                    policy, importance, stakes_value, expectation_upper, form, rivalry,
                    family, phase, reason, calibration,
                    opposition_strength=opposition.strength_signal,
                    head_to_head=opposition.head_to_head_signal,
                    qualification_context=qualification_context,
                    max_qualification_points=max_qualification_points,
                    qualification_win_path_guaranteed=win_path_guaranteed)
                if calibrated.practical_qualification_status in {
                    "below_practical_target", "must_win_for_practical_target",
                }:
                    flags.append("ipl_practical_target_policy_not_official_cutoff")
                team_score = calibrated.score
                lower, upper = calibrated.known_lower, possible.possible_upper
                row = {
                    "match_id": match.id, "match_date": day.isoformat(),
                    "comp_id": match.comp, "competition": comp["comp_name"],
                    "family": family, "phase": phase,
                    "team_id": team, "team": data.teams[team],
                    "opponent_id": opponent, "opponent": data.teams[opponent],
                    "team_pressure_score": team_score,
                    "known_signal_lower": lower, "context_upper": min(100, upper),
                    "competition_0_1": importance, "stakes_0_1": stakes_value,
                    "expectation_0_1": expectation, "form_0_1": form, "rivalry_0_1": rivalry,
                    "calibration_version": calibrated.model,
                    "calibration_tier": calibrated.tier,
                    "baseline_pressure_points": calibrated.baseline,
                    "expectation_bonus_points": calibrated.expectation_bonus,
                    "form_bonus_points": calibrated.form_bonus,
                    "rivalry_bonus_points": calibrated.rivalry_bonus,
                    "opposition_strength_points": calibrated.opposition_strength_points,
                    "head_to_head_points": calibrated.head_to_head_points,
                    "opposition_strength_signal": opposition.strength_signal,
                    "opposition_strength_support": opposition.strength_support,
                    "opponent_prior_completed_results": opposition.opponent_prior_results,
                    "head_to_head_signal": opposition.head_to_head_signal,
                    "head_to_head_prior_matches": opposition.head_to_head_matches,
                    "head_to_head_prior_result_rate": opposition.head_to_head_result_rate,
                    "head_to_head_prior_expected_rate": opposition.head_to_head_expected_rate,
                    "head_to_head_support": opposition.head_to_head_support,
                    "stakes_reason": reason,
                    "can_qualify_after_loss": loss_possible,
                    "qualification_guaranteed_after_win": win_guaranteed,
                    "qualification_context": qualification_context,
                    "qualification_points_before_match": points_before_match,
                    "max_qualification_points_if_win_out": max_qualification_points,
                    "ipl_practical_points_target": (
                        calibration.ipl_practical_points_target
                        if calibration is not None and calibration.playoff_aware
                        and family == "ipl" and phase == "league" else None),
                    "practical_qualification_status": calibrated.practical_qualification_status,
                    "objective_fixtures_remaining": remaining_matches,
                    "qualification_guaranteed_if_lose_then_win_out": loss_recovery_guaranteed,
                    "qualification_guaranteed_if_win_out": win_out_guaranteed,
                    "pre_match_win_probability": probability,
                    "pre_match_elo": history.elo[team],
                    "prior_completed_results": len(history.results[team]),
                    "recent_result_count": len(history.form[team]),
                    "defending_champion_signal": signals["champion"],
                    "favourite_signal": signals["favourite"],
                    "historical_success_proxy_signal": signals["historical"],
                    "documented_home_signal": signals["home"],
                    "score_basis": "known_signals_and_declared_proxies" if team_score is not None
                                   else "unresolved",
                    "flags": ";".join(sorted(set(flags))),
                    "history_cutoff": "strictly_before_match_date",
                }
                team_rows.append(row)

                lineup = data.starting.get((match.id, team), [])
                captain = data.captains.get((match.id, team))
                vice_entries = [entry for entry in rules.reference["vice_captains"]
                                if applicable(entry, match, team)]
                if len(vice_entries) > 1:
                    raise ValueError(f"Overlapping vice-captain evidence: {match.id}/{team}")
                vice = vice_entries[0]["player_id"] if vice_entries else None
                if vice not in lineup:
                    vice = None
                for role in ["batting", "bowling"]:
                    for stats in history.dependency(lineup, role):
                        player = stats["player_id"]
                        personal_flags = []
                        if player == captain:
                            leadership_role, leadership = "captain", policy.captain
                        elif player == vice:
                            leadership_role, leadership = "vice_captain", policy.vice_captain
                            if vice_entries[0].get("evidence_status") == "season_scoped_secondary_source":
                                personal_flags.append("vice_captain_season_scope_proxy")
                        else:
                            leadership_role, leadership = "no_verified_leadership_role", 0.0
                            if vice is None:
                                personal_flags.append("vice_captain_unverified")
                            if captain is None:
                                personal_flags.append("captain_unresolved")
                        dependency = stats["dependency"]
                        valid_lineup = len(lineup) == 11
                        if not valid_lineup:
                            personal_flags.append("dependency_withheld_ambiguous_xi")
                        score, uncapped, uplift = None, None, None
                        if valid_lineup and team_score is not None:
                            score, uncapped, uplift = player_adjustment(
                                team_score, dependency, leadership,
                                policy.dependency, policy.combined_cap)
                        leadership_upper = leadership
                        if "captain_unresolved" in personal_flags:
                            leadership_upper = max(leadership_upper, policy.captain)
                        elif "vice_captain_unverified" in personal_flags:
                            leadership_upper = max(leadership_upper, policy.vice_captain)
                        upper_score, _, _ = player_adjustment(
                            min(100, upper), dependency if valid_lineup else 1,
                            leadership_upper, policy.dependency, policy.combined_cap)
                        player_rows.append({
                            "match_id": match.id, "match_date": day.isoformat(),
                            "comp_id": match.comp, "team_id": team, "team": data.teams[team],
                            "player_id": player, "player": data.players[player], "role": role,
                            "team_pressure_score": team_score,
                            "player_pressure_score": score, "player_pressure_uncapped": uncapped,
                            "player_context_upper": upper_score,
                            "leadership_role": leadership_role,
                            "leadership_uplift_0_1": leadership,
                            "dependency_0_1": dependency if valid_lineup else None,
                            "applied_combined_uplift_0_1": uplift,
                            "ability_rank": stats["ability_rank"] if valid_lineup else None,
                            "dependency_rank": stats["dependency_rank"] if valid_lineup else None,
                            "quality_index": stats["quality"],
                            "projected_contribution": stats["projected_contribution"],
                            "contribution_share": stats["contribution_share"] if valid_lineup else None,
                            "expected_legal_opportunities": stats["expected_legal_opportunities"],
                            "prior_legal_balls": stats["prior_legal_balls"],
                            "prior_appearances": stats["prior_appearances"],
                            "sample_reliability": stats["sample_reliability"],
                            "history_status": stats["history_status"],
                            "lineup_status": "retrospective_starting_xi_proxy",
                            "saturated_at_100": uncapped is not None and uncapped > 100,
                            "flags": ";".join(personal_flags),
                        })
        # All snapshots on this date precede every result/statistics update on it.
        for match in day_matches:
            family = data.competitions[match.comp]["competition_family_id"]
            history.update_result(match, expected[match.id])
            if match.stage == "Final" and family == "ipl" and match.winner:
                previous_ipl_champion = match.winner
            history.update_performance(
                match, family, data.performances.get(match.id, {}), rates[family])
        LOG.info("Scored through %s (%d match-team rows)", day, len(team_rows))

    flags = Counter(flag for row in team_rows for flag in row["flags"].split(";") if flag)
    player_flags = Counter(flag for row in player_rows for flag in row["flags"].split(";") if flag)
    summary = {
        "matches_scored": len(pending), "team_rows": len(team_rows),
        "player_role_rows": len(player_rows), "standings_rows": len(standings),
        "team_flag_counts": dict(sorted(flags.items())),
        "player_flag_counts": dict(sorted(player_flags.items())),
        "stakes_reason_counts": dict(sorted(Counter(r["stakes_reason"] for r in team_rows).items())),
        "player_rows_withheld": sum(r["player_pressure_score"] is None for r in player_rows),
        "saturated_player_rows": sum(r["saturated_at_100"] for r in player_rows),
        "vice_captain_rows_with_evidence": sum(r["leadership_role"] == "vice_captain" for r in player_rows),
        "dependency_scope": "separate batting/bowling; prior legal-delivery statistics; no current-match outcomes",
        "calibration_version": calibration.version,
        "calibration_parameters": asdict(calibration),
        "qualification_context_counts": dict(sorted(Counter(
            row["qualification_context"] for row in team_rows if row["qualification_context"]).items())),
        "practical_qualification_status_counts": dict(sorted(Counter(
            row["practical_qualification_status"] for row in team_rows
            if row["practical_qualification_status"] != "not_applied").items())),
        "formula": "stage and competition base + limited context + opponent adjustments",
        "individual_uplifts": {
            "captain": policy.captain, "vice_captain": policy.vice_captain,
            "max_dependency": policy.dependency, "combined_cap": policy.combined_cap,
        },
        "limitations": [
            "Scores are contextual indices, not measured psychological stress.",
            "Date-level cutoff withholds same-day results; exact start timestamps are unavailable.",
            "Qualification bounds do not manufacture NRR; future fixture metadata is a declared schedule proxy.",
            "Unverified vice-captain and home-expectation facts are flagged, not inferred from reputation.",
            "The starting XI is retrospectively reconstructed from participation/impact flags; ambiguous sizes withhold dependency.",
            "Dependency excludes illegal deliveries and Super Overs; bowling uses off-bat runs, not full official economy.",
            "Historical success and Elo favourite status are explicit statistical proxies, not observed public expectations.",
            "Stage anchors are expert priors: no final-score percentage gap or protected match ranking is imposed.",
            "Late qualification control is a feasibility bound, not a qualification probability or proof that a loss eliminates.",
            "The IPL points target is a practical calibration policy, not an official cutoff or qualification guarantee.",
            "Practical-target downgrades preserve mathematical status and known controlled qualification paths.",
            "Opposition and head-to-head cover only earlier results in this archive, not complete career histories.",
            "Head-to-head adjusts residual performance beyond historical Elo expectations; raw pair win rates remain visible.",
            "Context bounds hold the scoring policy and opposition estimates fixed; they do not capture all model uncertainty.",
        ],
    }
    return Results(team_rows, player_rows, standings, issues, summary)
