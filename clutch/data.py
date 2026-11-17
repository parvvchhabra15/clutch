from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .inputs import InputError, iter_csv, read_csv


@dataclass(frozen=True)
class Match:
    id: str
    comp: str
    day: date
    number: int
    stage: str
    home: str
    away: str
    winner: str = ""
    result: str = ""
    venue: str = ""
    venue_country: str = ""
    home_captain: str = ""
    away_captain: str = ""

    @property
    def teams(self) -> tuple[str, str]:
        return self.home, self.away

    def opponent(self, team: str) -> str:
        if team not in self.teams:
            raise InputError(f"{team} is not in match {self.id}")
        return self.away if team == self.home else self.home

    def result_value(self, team: str) -> float | None:
        if self.winner:
            return float(self.winner == team)
        if "tied" in self.result.casefold():
            return 0.5
        if self.result.casefold() in {"no result", "match abandoned"}:
            return None
        raise InputError(f"Uninterpretable outcome in match {self.id}: {self.result!r}")


@dataclass
class Performance:
    batting_balls: float = 0
    batting_runs: float = 0
    batting_outs: float = 0
    bowling_balls: float = 0
    bowling_runs: float = 0
    bowling_wickets: float = 0
    appearances: float = 0


@dataclass
class Dataset:
    matches: list[Match]
    competitions: dict[str, dict[str, str]]
    teams: dict[str, str]
    players: dict[str, str]
    starting: dict[tuple[str, str], list[str]]
    captains: dict[tuple[str, str], str]
    performances: dict[str, dict[str, Performance]]
    issues: list[dict] = field(default_factory=list)
    delivery_counts: dict[str, int] = field(default_factory=dict)


def load_dataset(raw: Path, config: Path) -> Dataset:
    raw = Path(raw)
    config = Path(config)
    source_competitions = read_csv(raw / "competitions.csv")
    catalogue = read_csv(config / "reference" / "competition_catalogue.csv")
    competitions = {row["comp_id"]: row for row in catalogue}
    if len(competitions) != len(catalogue) or set(competitions) != {
            row["comp_id"] for row in source_competitions}:
        raise InputError("Competition catalogue differs from the source IDs.")
    for row in source_competitions:
        if any(competitions[row["comp_id"]][key] != row[key]
               for key in ["comp_name", "season", "match_format"]):
            raise InputError(f"Stale competition catalogue: {row['comp_id']}")
    teams = {}
    for row in read_csv(raw / "teams.csv"):
        team_id, name = row["team_id"], row["team_name"]
        if team_id in teams and teams[team_id] != name:
            raise InputError(f"Conflicting team names: {team_id}")
        teams[team_id] = name
    players = {}
    for row in read_csv(raw / "competition_team_player_mapping.csv"):
        player_id, name = row["player_id"], row["player_name"]
        if player_id in players and players[player_id] != name:
            raise InputError(f"Conflicting names for player ID {player_id}")
        players[player_id] = name

    matches = []
    for row in read_csv(raw / "matches.csv"):
        match = Match(
            row["match_id"], row["comp_id"], date.fromisoformat(row["match_date"]),
            int(row["match_number"]), row["stage"], row["home_team_id"],
            row["away_team_id"], row["win_team_id"], row["result"], row["venue_id"],
            row["venue_country"], row["home_team_captain_id"], row["away_team_captain_id"],
        )
        if match.comp not in competitions or any(team not in teams for team in match.teams):
            raise InputError(f"Unresolved match identity: {match.id}")
        if match.home == match.away or (match.winner and match.winner not in match.teams):
            raise InputError(f"Invalid teams/winner in match {match.id}")
        matches.append(match)
    by_id = {match.id: match for match in matches}
    if len(by_id) != len(matches):
        raise InputError("Duplicate match IDs.")
    matches.sort(key=lambda match: (match.day, match.id))

    starting_sets = defaultdict(set)
    played_sets = defaultdict(set)
    captain_sets = defaultdict(set)
    for row in read_csv(raw / "playing_eleven.csv"):
        mid, team, player = row["match_id"], row["team_id"], row["player_id"]
        if mid not in by_id or team not in by_id[mid].teams:
            raise InputError(f"Invalid lineup membership: {mid}/{team}")
        if row["is_played"] == "true":
            if player not in players:
                raise InputError(f"Played player has no identity: {player}")
            played_sets[mid].add(player)
            if row["is_impact"] != "1":
                starting_sets[mid, team].add(player)
            if row["is_captain"] == "true":
                captain_sets[mid, team].add(player)

    issues = []
    captains = {}
    for match in matches:
        for team, stated in [(match.home, match.home_captain),
                             (match.away, match.away_captain)]:
            candidates = captain_sets[match.id, team] | ({stated} if stated else set())
            starters = starting_sets[match.id, team]
            if len(starters) != 11:
                issues.append({"match_id": match.id, "team_id": team,
                               "issue": "starting_xi_not_eleven", "detail": str(len(starters))})
            if len(candidates) == 1 and next(iter(candidates)) in starters:
                captains[match.id, team] = next(iter(candidates))
            else:
                issues.append({"match_id": match.id, "team_id": team,
                               "issue": "captain_unresolved",
                               "detail": ";".join(sorted(candidates))})

    performances = defaultdict(lambda: defaultdict(Performance))
    delivery_keys = set()
    legal_types = {"None", "byes", "legbyes"}
    illegal_types = {"wide", "Overstep", "other"}
    credited = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
    matches_with_balls = set()
    delivery_counts = Counter()
    for row in iter_csv(raw / "ball_by_ball.csv"):
        mid = row["match_id"]
        if mid not in by_id:
            raise InputError(f"Delivery references unknown match {mid}")
        delivery_counts[mid] += 1
        key = (mid, row["innings"], row["innings_ball_num"])
        if key in delivery_keys:
            raise InputError(f"Duplicate delivery key: {key}")
        delivery_keys.add(key)
        if int(row["innings"]) > 2:
            continue
        matches_with_balls.add(mid)
        extra_type = row["extras_type"]
        if extra_type not in legal_types | illegal_types | {"Penalty"}:
            raise InputError(f"Unknown extra type: {extra_type}")
        if extra_type not in legal_types:
            continue
        runs = int(row["runs"])
        if extra_type in {"byes", "legbyes"} and runs:
            issues.append({"match_id": mid, "team_id": row["batting_team_id"],
                           "issue": "positive_bat_runs_on_bye_excluded", "detail": str(key)})
            continue
        striker, bowler = row["striker_id"], row["bowler_id"]
        if striker not in players or bowler not in players:
            raise InputError(f"Unresolved delivery player identity: {key}")
        bat, bowl = performances[mid][striker], performances[mid][bowler]
        bat.batting_balls += 1
        bat.batting_runs += runs
        bowl.bowling_balls += 1
        bowl.bowling_runs += runs
        if row["is_wicket"] == "true":
            dismissed = row["dismissed_player_id"]
            if dismissed:
                if dismissed not in players:
                    raise InputError(f"Unknown dismissed player: {dismissed}")
                performances[mid][dismissed].batting_outs += 1
            else:
                issues.append({"match_id": mid, "team_id": row["batting_team_id"],
                               "issue": "dismissed_player_missing", "detail": str(key)})
            if row["dismissal_type"] in credited:
                bowl.bowling_wickets += 1
    for mid in matches_with_balls:
        observed = set(performances[mid])
        for player in played_sets[mid] | observed:
            performances[mid][player].appearances = 1
    return Dataset(
        matches, competitions, teams, players,
        {key: sorted(value) for key, value in starting_sets.items()}, captains,
        {mid: dict(values) for mid, values in performances.items()}, issues, dict(delivery_counts),
    )
