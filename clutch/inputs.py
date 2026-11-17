import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator


class InputError(ValueError):
    """An input cannot be interpreted without changing its meaning."""


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise InputError(f"Missing or duplicate CSV headers: {path}")
        for line, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise InputError(f"Inconsistent column count: {path}:{line}")
            yield row


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(iter_csv(path))


def numeric(value: str, maximum: float, label: str) -> float:
    try:
        result = float(value.strip())
    except ValueError as exc:
        raise InputError(f"{label}: expected a number, got {value!r}") from exc
    if not math.isfinite(result) or not 0 <= result <= maximum:
        raise InputError(f"{label}: must be between 0 and {maximum}")
    return result


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise InputError(f"Specify headers when writing an empty table: {path}")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)


ALIASES = {
    "IND": "India", "INDIA": "India", "PAK": "Pakistan",
    "AUS": "Australia", "ENG": "England", "SOUTH AFRICA": "South Africa",
    "NZ": "New Zealand", "RCB": "Royal Challengers Bengaluru",
    "CSK": "Chennai Super Kings", "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders", "RR": "Rajasthan Royals", "PBKS": "Punjab Kings",
}


@dataclass(frozen=True)
class Policy:
    competition: dict[tuple[str, str], float]
    rivalry: dict[tuple[str, str], float]
    rivalry_default: float
    expectation: dict[str, float]
    captain: float
    vice_captain: float
    dependency: float
    combined_cap: float
    weights: dict[str, float]
    normalized_rows: dict[str, list[dict]]

    def competition_value(self, comp_id: str, family: str, season: str) -> float:
        for key in [(comp_id, season), (comp_id, "ALL_COVERED"),
                    (family, season), (family, "ALL_COVERED")]:
            if key in self.competition:
                return self.competition[key]
        raise InputError(f"No competition rating for {comp_id}/{family}/{season}")

    def rivalry_value(self, team: str, opponent: str) -> float:
        return self.rivalry.get(tuple(sorted((team, opponent))), self.rivalry_default)


def load_policy(folder: Path, team_names: dict[str, str]) -> Policy:
    folder = Path(folder)
    competition = {}
    competition_rows = []
    for row in read_csv(folder / "01_competition_importance.csv"):
        key = (row["comp_id_override"] or row["competition_family_id"], row["scope"])
        if key in competition:
            raise InputError(f"Duplicate competition override: {key}")
        competition[key] = numeric(row["importance_0_10"], 10, str(key)) / 10
        competition_rows.append({"entity": key[0], "scope": key[1],
                                 "importance_0_1": competition[key]})

    ids_by_name = {name.casefold(): team_id for team_id, name in team_names.items()}
    if len(ids_by_name) != len(team_names):
        raise InputError("Ambiguous team names; rivalry resolution requires IDs.")
    pairs = {}
    rivalry_rows = []
    default = None
    for row in read_csv(folder / "02_rivalry_scale.csv"):
        if not any(value.strip() for value in row.values()):
            continue
        text = row["_how_to_fill"].strip()
        pair = re.fullmatch(r"(.+?)\s+VS\s+(.+)", text, re.IGNORECASE)
        if pair:
            names = [ALIASES.get(name.strip().upper(), name.strip())
                     for name in pair.groups()]
            if any(name.casefold() not in ids_by_name for name in names):
                raise InputError(f"Unresolved rivalry: {text}")
            ids = tuple(sorted(ids_by_name[name.casefold()] for name in names))
            if ids[0] == ids[1] or ids in pairs:
                raise InputError(f"Duplicate or self rivalry: {text}")
            value = numeric(row["rivalry_level"], 10, text) / 10
            pairs[ids] = value
            rivalry_rows.append({"team_a_id": ids[0], "team_b_id": ids[1],
                                 "rivalry_0_1": value, "rule": "explicit_pair"})
        elif text.upper() == "OTHERS":
            if default is not None:
                raise InputError("Multiple OTHERS rivalry rules.")
            default = numeric(row["rivalry_level"], 10, "OTHERS") / 10
        elif row["rivalry_level"] in {"none", "established", "major", "exceptional"}:
            if row["rivalry_level"] != "none" and row["rivalry_0_10"].strip():
                raise InputError("Tier scores need a pair-to-tier mapping; cannot ignore them.")
        else:
            raise InputError(f"Unrecognised rivalry input row: {row}")
    if default is None:
        raise InputError("An explicit OTHERS rule is required; missing is not zero.")
    rivalry_rows.append({"team_a_id": "", "team_b_id": "", "rivalry_0_1": default,
                         "rule": "declared_default"})

    values = {}
    policy_rows = []
    for row in read_csv(folder / "03_expectation_and_responsibility.csv"):
        first, second = row["value"].strip(), row.get("SCORE", "").strip()
        if first and second and numeric(first, 100, row["setting"]) != numeric(
                second, 100, row["setting"]):
            raise InputError(f"Conflicting value and SCORE: {row['setting']}")
        if row["unit"] not in {"score_0_to_10", "percent_of_team_score"}:
            raise InputError(f"Unknown policy unit: {row['unit']}")
        scale = 10 if row["unit"] == "score_0_to_10" else 100
        if row["setting"] in values:
            raise InputError(f"Duplicate policy setting: {row['setting']}")
        values[row["setting"]] = numeric(first or second, scale, row["setting"]) / scale
        policy_rows.append({"setting": row["setting"], "value_0_1": values[row["setting"]],
                            "source_column": "value" if first else "SCORE"})
    expectation_keys = [
        "defending_champion_expectation", "strong_favourite_expectation",
        "documented_home_expectations", "documented_historical_team_expectations",
    ]
    required = expectation_keys + [
        "captain_uplift_pct", "vice_captain_uplift_pct", "max_dependency_uplift_pct",
        "combined_player_uplift_cap_pct",
    ]
    if set(values) != set(required):
        raise InputError(f"Unexpected/missing policy keys: {set(values) ^ set(required)}")
    settings = {row["setting"]: row["value"] for row in
                read_csv(folder / "reference" / "agreed_settings.csv")}
    weights = {key: numeric(settings[key], 100, key) for key in [
        "expectation_weight",
        "recent_underperformance_weight", "rivalry_weight",
    ]}
    if sum(weights.values()) <= 0:
        raise InputError("At least one context weight must be positive.")
    for key in ["early_clinch_stakes_score", "second_chance_qualifier_stakes_score"]:
        if numeric(settings[key], 1, key) != 0.6:
            raise InputError(f"{key} differs from the supported approved 0.60 contract.")
    return Policy(
        competition, pairs, default, {key: values[key] for key in expectation_keys},
        values["captain_uplift_pct"], values["vice_captain_uplift_pct"],
        values["max_dependency_uplift_pct"], values["combined_player_uplift_cap_pct"],
        weights, {"competition.csv": competition_rows, "rivalries.csv": rivalry_rows,
                  "responsibility.csv": policy_rows},
    )
