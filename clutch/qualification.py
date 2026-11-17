from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class QualificationUnresolved(RuntimeError):
    pass


@dataclass(frozen=True)
class QualificationBounds:
    possible: bool
    guaranteed: bool


def feasible_finish(
    points: dict[str, int],
    wins: dict[str, int],
    fixtures: list[tuple[str, str]],
    team: str,
    places: int,
    *,
    qualify: bool,
    wins_before_nrr: bool = False,
) -> bool:
    """Bounds over all 2/0, 0/2 and 1/1 outcomes, without inventing NRR."""
    opponents = sorted(set(points) - {team})
    if team not in points or not 1 <= places <= len(points):
        raise ValueError("Invalid qualification team or number of places.")
    if places == len(points):
        return qualify
    lex = sum(wins.values()) + len(fixtures) + 1 if wins_before_nrr else 1
    base = {key: value * lex + (wins[key] if wins_before_nrr else 0)
            for key, value in points.items()}
    if not fixtures:
        above = sum(base[key] > base[team] for key in opponents)
        above_or_equal = sum(base[key] >= base[team] for key in opponents)
        return above < places if qualify else above_or_equal >= places

    n_games = len(fixtures)
    n_vars = 2 * n_games + len(opponents)
    coefficients = {key: np.zeros(n_vars) for key in points}
    constraints = []
    lower = []
    upper = []
    for index, (home, away) in enumerate(fixtures):
        if home == away or home not in points or away not in points:
            raise ValueError("Fixture does not belong to this qualification pool.")
        # Two binaries: home win, away win; both zero means shared points.
        base[home] += lex
        base[away] += lex
        coefficients[home][2 * index] += lex + int(wins_before_nrr)
        coefficients[home][2 * index + 1] -= lex
        coefficients[away][2 * index] -= lex
        coefficients[away][2 * index + 1] += lex + int(wins_before_nrr)
        row = np.zeros(n_vars)
        row[2 * index:2 * index + 2] = 1
        constraints.append(row)
        lower.append(-np.inf)
        upper.append(1)
    big_m = max(base.values()) - min(base.values()) + 4 * n_games * (lex + 1) + 2
    for index, opponent in enumerate(opponents):
        difference = coefficients[opponent] - coefficients[team]
        offset = base[opponent] - base[team]
        row = difference.copy()
        if qualify:
            # At most places-1 opponents must strictly exceed the team's score.
            row[2 * n_games + index] = -big_m
            lower.append(-np.inf)
            upper.append(-offset)
        else:
            # A tie may put the opponent ahead: pessimistic unknown-NRR bound.
            row[2 * n_games + index] = -big_m
            lower.append(-offset - big_m)
            upper.append(np.inf)
        constraints.append(row)
    row = np.zeros(n_vars)
    row[2 * n_games:] = 1
    constraints.append(row)
    lower.append(-np.inf if qualify else places)
    upper.append(places - 1 if qualify else np.inf)
    result = milp(
        np.zeros(n_vars), integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(np.asarray(constraints), lower, upper),
        options={"time_limit": 3.0, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.status == 0:
        return True
    if result.status == 2:
        return False
    raise QualificationUnresolved(f"Qualification solver did not resolve: {result.message}")


def qualification_bounds(
    points: dict[str, int], wins: dict[str, int],
    fixtures: list[tuple[str, str]], team: str, places: int,
    wins_before_nrr: bool = False,
) -> QualificationBounds:
    possible = feasible_finish(points, wins, fixtures, team, places, qualify=True,
                               wins_before_nrr=wins_before_nrr)
    guaranteed = not feasible_finish(points, wins, fixtures, team, places, qualify=False,
                                    wins_before_nrr=wins_before_nrr)
    return QualificationBounds(possible, guaranteed)
