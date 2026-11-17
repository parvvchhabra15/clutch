import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .data import Dataset, Match
from .inputs import InputError
from .qualification import qualification_bounds


@dataclass(frozen=True)
class Stakes:
    score: float
    reason: str
    flags: tuple[str, ...] = ()
    possible_after_loss: bool | None = None
    guaranteed_after_win: bool | None = None
    qualification_context: str = ""
    remaining_matches: int | None = None
    guaranteed_if_lose_then_win_out: bool | None = None
    guaranteed_if_win_out: bool | None = None
    points_before_match: int | None = None
    max_points_if_win_out: int | None = None


def series_stakes(team_wins: int, opponent_wins: int, remaining_after: int) -> Stakes:
    if remaining_after < 0:
        raise ValueError("Negative remaining series fixtures.")
    if abs(team_wins - opponent_wins) > remaining_after + 1:
        return Stakes(0.10, "series_already_decided")
    if remaining_after == 0 and team_wins == opponent_wins:
        return Stakes(1.0, "series_decider")
    if team_wins + remaining_after < opponent_wins + 1:
        return Stakes(0.90, "loss_ends_series_survival")
    if team_wins + 1 > opponent_wins + remaining_after:
        return Stakes(0.60, "opportunity_to_clinch")
    return Stakes(0.35, "ordinary_live_series")


class RuleBook:
    def __init__(self, dataset: Dataset, path: Path):
        self.data = dataset
        self.reference = json.loads(path.read_text(encoding="utf-8"))
        self.by_comp = defaultdict(list)
        self.by_id = {match.id: match for match in dataset.matches}
        for match in dataset.matches:
            self.by_comp[match.comp].append(match)
        self.replacements = self.reference["fixture_replacements"]
        for replacement in self.replacements:
            old = self.by_id[replacement["original_match_id"]]
            new = self.by_id[replacement["replacement_match_id"]]
            if old.comp != new.comp or set(old.teams) != set(new.teams):
                raise InputError("Fixture replacement does not match the same teams/competition.")
            if old.winner:
                raise InputError("Voiding a completed result needs an explicit rating-history correction.")
        self.groups = {}
        self.catalogue_rows = []
        for comp_id, matches in self.by_comp.items():
            family = dataset.competitions[comp_id]["competition_family_id"]
            if family in {"senior_asia_cup", "emerging_t20"}:
                rule = self.reference["rounds"].get(comp_id)
                if rule is None:
                    raise InputError(f"Missing group-stage rules for {comp_id}")
                initial = [m for m in matches if m.number <= rule["initial_group_matches"]]
                self.groups[comp_id] = self._components(initial)
                sizes = sorted(len(group) for group in self.groups[comp_id])
                if sizes != [rule["group_size"], rule["group_size"]]:
                    raise InputError(f"Unexpected initial group graph for {comp_id}: {sizes}")
            numbers = {match.number for match in matches}
            length = max(numbers)
            self.catalogue_rows.append({
                "comp_id": comp_id, "comp_name": dataset.competitions[comp_id]["comp_name"],
                "family": family, "recorded_matches": len(matches),
                "max_fixture_slot": length,
                "missing_slots": ";".join(str(n) for n in range(1, length + 1) if n not in numbers),
                "series_length_status": (
                    "public_source_confirmed" if comp_id in self.reference["series_lengths"]
                    else "archive_slot_max_provisional" if family == "senior_bilateral_t20i"
                    else "not_bilateral"
                ),
                "fixture_schedule_status": "retrospective_fixture_metadata_no_future_outcomes",
            })

    @staticmethod
    def _components(fixtures: list[Match]) -> list[set[str]]:
        adjacency = defaultdict(set)
        for match in fixtures:
            adjacency[match.home].add(match.away)
            adjacency[match.away].add(match.home)
        remaining = set(adjacency)
        components = []
        while remaining:
            queue = [min(remaining)]
            group = set()
            while queue:
                team = queue.pop()
                if team in group:
                    continue
                group.add(team)
                queue.extend(adjacency[team] - group)
            remaining -= group
            components.append(group)
        return components

    def inactive_ids(self, as_of: date) -> set[str]:
        inactive = set()
        for item in self.replacements:
            if as_of >= date.fromisoformat(item["effective_date"]):
                inactive.add(item["original_match_id"])
            else:
                inactive.add(item["replacement_match_id"])
        return inactive

    def phase(self, match: Match) -> str:
        family = self.data.competitions[match.comp]["competition_family_id"]
        if match.stage in {"Final", "Semi Final", "Eliminator"}:
            return match.stage
        if family == "ipl" and match.stage == "Qualifier":
            slots = sorted(m.number for m in self.by_comp[match.comp] if m.stage == "Qualifier")
            if len(slots) != 2:
                raise InputError(f"Cannot identify qualifier pathway for {match.comp}")
            return "Qualifier 1" if match.number == slots[0] else "Qualifier 2"
        if family in {"senior_asia_cup", "emerging_t20"}:
            limits = self.reference["rounds"][match.comp]
            if match.number <= limits["initial_group_matches"]:
                return "group"
            if family == "senior_asia_cup" and match.number <= limits["super_four_end"]:
                return "super_four"
            raise InputError(f"Unexpected stage/slot: {match.id}")
        return "league"

    def pool(self, match: Match, team: str) -> tuple[list[Match], int, str]:
        family = self.data.competitions[match.comp]["competition_family_id"]
        phase = self.phase(match)
        candidates = [m for m in self.by_comp[match.comp]
                      if m.id not in self.inactive_ids(match.day) and self.phase(m) == phase]
        if phase == "group":
            group = next(group for group in self.groups[match.comp] if team in group)
            candidates = [m for m in candidates if set(m.teams).issubset(group)]
            return candidates, 2, "group_progression"
        if phase == "super_four":
            return candidates, 2, "reach_final"
        if family == "ipl":
            return candidates, 4, "playoff_qualification"
        if family == "asia_cup_qualifying":
            return candidates, 1, "qualify_for_main_event"
        if family == "senior_tri_series":
            has_final = any(m.stage == "Final" for m in self.by_comp[match.comp])
            return candidates, 2 if has_final else 1, "reach_final" if has_final else "league_title"
        raise InputError(f"No league rule for {match.comp}")

    @staticmethod
    def table(fixtures: list[Match], as_of: date) -> dict[str, dict[str, int]]:
        teams = {team for match in fixtures for team in match.teams}
        table = {team: {"points": 0, "wins": 0, "played": 0} for team in teams}
        for match in fixtures:
            if match.day >= as_of:
                continue
            for team in match.teams:
                result = match.result_value(team)
                table[team]["played"] += 1
                table[team]["points"] += 1 if result is None else int(2 * result)
                table[team]["wins"] += int(result == 1)
        return table

    def assess(self, match: Match, team: str) -> tuple[Stakes, list[dict]]:
        family = self.data.competitions[match.comp]["competition_family_id"]
        phase = self.phase(match)
        if phase == "Final":
            return Stakes(1.0, "final"), []
        if phase in {"Semi Final", "Eliminator", "Qualifier 2"}:
            return Stakes(0.90, "direct_elimination"), []
        if phase == "Qualifier 1":
            return Stakes(0.60, "qualifier_with_second_chance"), []
        if family == "standalone_t20i":
            return Stakes(0.35, "standalone_match"), []
        if family == "senior_bilateral_t20i":
            prior = [m for m in self.by_comp[match.comp] if m.day < match.day]
            wins = Counter(m.winner for m in prior if m.winner)
            verified = self.reference["series_lengths"].get(match.comp)
            length = verified or max(m.number for m in self.by_comp[match.comp])
            result = series_stakes(wins[team], wins[match.opponent(team)], length - match.number)
            flags = [] if verified else ["series_length_inferred"]
            known_prior_slots = {m.number for m in prior}
            if any(n not in known_prior_slots for n in range(1, match.number)):
                flags.append("missing_prior_fixture_or_same_day_result")
            return Stakes(result.score, result.reason, tuple(flags)), [{
                "snapshot_match_id": match.id, "comp_id": match.comp, "phase": "series",
                "team_id": t, "played": sum(t in m.teams for m in prior),
                "wins": wins[t], "points": "", "rank_best": "", "rank_worst": "",
                "nrr": "", "nrr_status": "not_applicable",
            } for t in match.teams]

        fixtures, places, objective = self.pool(match, team)
        table = self.table(fixtures, match.day)
        future = [m for m in fixtures if m.day >= match.day]
        remaining_for_team = sum(team in m.teams for m in future)
        if match.id not in {m.id for m in future}:
            raise InputError(f"Current match absent from its active fixture pool: {match.id}")
        points = {key: row["points"] for key, row in table.items()}
        wins = {key: row["wins"] for key, row in table.items()}
        use_wins = family == "ipl"
        snapshots = []
        for key, row in table.items():
            own = (row["points"], row["wins"] if use_wins else 0)
            others = [(r["points"], r["wins"] if use_wins else 0)
                      for opponent, r in table.items() if opponent != key]
            snapshots.append({
                "snapshot_match_id": match.id, "comp_id": match.comp, "phase": phase,
                "team_id": key, **row,
                "rank_best": 1 + sum(value > own for value in others),
                "rank_worst": 1 + sum(value >= own for value in others),
                "nrr": "", "nrr_status": "not_reconstructed_use_tie_bounds",
            })
        flags = ["fixture_schedule_proxy", "nrr_tie_bounds"]
        if family != "ipl":
            flags.append("points_rule_assumed_2_1_0")
        elif self.data.competitions[match.comp]["season"] != "2026":
            flags.append("historical_ipl_tiebreak_inherited")
        if any(m.day == match.day and m.id != match.id for m in fixtures):
            flags.append("same_day_results_withheld")

        def bounds(fixed: str | None, target_places: int, win_remaining: bool = False):
            p, w = dict(points), dict(wins)
            remaining = future
            if fixed:
                winner = team if fixed == "win" else match.opponent(team)
                p[winner] += 2
                w[winner] += 1
                remaining = [m for m in future if m.id != match.id]
            if win_remaining:
                own_games = sum(team in m.teams for m in remaining)
                p[team] += 2 * own_games
                w[team] += own_games
                remaining = [m for m in remaining if team not in m.teams]
            return qualification_bounds(
                p, w, [m.teams for m in remaining], team, target_places, use_wins,
            )

        details = {
            "remaining_matches": remaining_for_team,
            "points_before_match": points[team],
            "max_points_if_win_out": points[team] + 2 * remaining_for_team,
        }
        current = bounds(None, places)
        secondary = False
        if current.guaranteed and family == "ipl":
            places = 2
            objective = "top_two_second_chance"
            secondary = True
            current = bounds(None, places)
        if current.guaranteed:
            return Stakes(0.10, objective + "_already_secured", tuple(flags),
                          **details), snapshots
        if not current.possible:
            return Stakes(0.10, objective + "_unreachable", tuple(flags),
                          **details), snapshots
        lost, won = bounds("loss", places), bounds("win", places)
        qualification_objectives = {
            "playoff_qualification", "group_progression", "reach_final", "qualify_for_main_event",
        }
        if not secondary and objective in qualification_objectives:
            if not lost.possible and won.possible:
                details.update(
                    qualification_context="loss_eliminates",
                    guaranteed_if_lose_then_win_out=False,
                    guaranteed_if_win_out=won.guaranteed if remaining_for_team == 1 else None)
            elif remaining_for_team <= 2 and table[team]["played"] > 0:
                # Remaining own wins are hypothetical, never the recorded future results.
                loss_recovery = lost if remaining_for_team == 1 else bounds("loss", places, True)
                win_out = won if remaining_for_team == 1 else bounds("win", places, True)
                details.update(
                    guaranteed_if_lose_then_win_out=loss_recovery.guaranteed,
                    guaranteed_if_win_out=win_out.guaranteed)
                if not loss_recovery.guaranteed:
                    details["qualification_context"] = (
                        "late_loss_of_control" if win_out.guaranteed else "late_external_dependency")
        if not lost.possible and won.possible:
            decisive = objective == "league_title" and len(future) == 1 and won.guaranteed
            score = 1.0 if decisive else 0.90
            if secondary:
                score = 0.60
            return Stakes(score, objective + "_must_win", tuple(flags),
                          lost.possible, won.guaranteed, **details), snapshots
        if won.guaranteed:
            return Stakes(0.60, objective + "_can_clinch", tuple(flags),
                          lost.possible, won.guaranteed, **details), snapshots
        score = 0.60 if remaining_for_team <= 3 or secondary else 0.35
        return Stakes(score, objective + "_live", tuple(flags),
                      lost.possible, won.guaranteed, **details), snapshots
