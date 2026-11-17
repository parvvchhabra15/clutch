import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from .calibration import StageCalibration
from .competitive import RefinementConfig
from .data import load_dataset
from .engine import score_dataset
from .execution import EXECUTION_SCHEMA, ExecutionConfig, collect_execution, load_annotations
from .full_toss import FullTossPolicy, regrade_execution
from .inputs import InputError, digest, load_policy, read_csv, write_csv, write_json
from .live import PerformanceConfig, load_deliveries
from .match_impact import MatchImpactConfig
from .performance import BALL_SCHEMA, PLAYER_SCHEMA, ParquetLedger, analyze
from .rules import RuleBook
from .scoring import (
    context_states, high_match_rows, parquet_rows, rate_players,
    refine_deliveries, refine_execution,
)


LOG = logging.getLogger(__name__)
ATTRIBUTION_FIELDS = ["match_id", "innings", "sequence", "role", "player_id", "reason"]
COMPETITIVE_PLAYER_FIELDS = [
    ("source_pressure_weight", pa.float64()), ("source_absolute_contribution", pa.float64()),
    ("source_credited_wicket", pa.int64()), ("source_high_pressure", pa.bool_()),
    ("scoring_pressure_anchor", pa.float64()), ("pressure_anchor_sequence", pa.int64()),
    ("contest_before", pa.string()), ("competitive_scoring_active", pa.bool_()),
]
ASSIGNED_EXECUTION_FIELDS = [
    ("source_execution_weight", pa.float64()), ("scoring_pressure_anchor", pa.float64()),
    ("pressure_anchor_sequence", pa.int64()), ("scoring_crunch_context", pa.string()),
    ("contest_before", pa.string()), ("competitive_scoring_active", pa.bool_()),
]
FULL_TOSS_FIELDS = [
    ("before_grace_raw_signal", pa.float64()), ("before_grace_scored_signal", pa.float64()),
    ("before_grace_execution_weight", pa.float64()),
    ("before_grace_training_eligible", pa.bool_()),
    ("full_toss_policy_applied", pa.bool_()),
]


def run(data_dir, config_dir, output):
    data_dir, config_dir, output = (Path(p).resolve() for p in (data_dir, config_dir, output))
    if output.exists():
        raise InputError(f"Output already exists: {output}")
    model = json.loads((config_dir / "model.json").read_text(encoding="utf-8"))
    if model["model"] != "CLUTCH":
        raise InputError("Unrecognised model configuration.")
    input_hashes = {}
    for name, expected in model["dataset_sha256"].items():
        observed = digest(data_dir / name)
        if observed != expected:
            raise InputError(f"The supplied data pack does not match the current data contract: {name}")
        input_hashes[name] = observed
    config_files = [
        config_dir / "model.json", config_dir / "rules.json",
        config_dir / "01_competition_importance.csv", config_dir / "02_rivalry_scale.csv",
        config_dir / "03_expectation_and_responsibility.csv",
        config_dir / "reference" / "competition_catalogue.csv",
        config_dir / "reference" / "agreed_settings.csv",
    ]
    config_hashes = {str(p.relative_to(config_dir)): digest(p) for p in config_files}
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    parameters = model["parameters"]
    ball_config = PerformanceConfig(**parameters["outcomes"])
    execution_config = ExecutionConfig(**parameters["execution"])
    high_config = MatchImpactConfig(**parameters["high_match"])
    competitive_config = RefinementConfig(**parameters["refinements"])
    full_toss_policy = FullTossPolicy(**parameters["execution_grace"])

    LOG.info("Reading the supplied data and calculating pre-match pressure")
    data = load_dataset(data_dir, config_dir)
    policy = load_policy(config_dir, data.teams)
    rules = RuleBook(data, config_dir / "rules.json")
    prematch = score_dataset(data, policy, rules, calibration=StageCalibration())
    output.mkdir(parents=True)
    pre_dir = output / "prematch"
    write_csv(pre_dir / "team_pressure.csv", prematch.teams)
    write_csv(pre_dir / "player_pressure.csv", prematch.players)
    write_csv(pre_dir / "standings_before_match.csv", prematch.standings)
    write_csv(pre_dir / "exceptions.csv", prematch.exceptions,
              ["match_id", "team_id", "issue", "detail"])
    write_json(pre_dir / "run_manifest.json", {
        "model": "CLUTCH", "summary": prematch.summary,
        "calibration": asdict(StageCalibration()),
    })
    teams = {(r["match_id"], r["team_id"]): r for r in read_csv(pre_dir / "team_pressure.csv")}
    players = {(r["match_id"], r["team_id"], r["player_id"], r["role"]): r
               for r in read_csv(pre_dir / "player_pressure.csv")}

    LOG.info("Building chronological delivery states and contextual run/wicket values")
    metadata = {r["match_id"]: r for r in read_csv(data_dir / "matches.csv")}
    deliveries = load_deliveries(data_dir / "ball_by_ball.csv")
    if set(deliveries) - set(metadata):
        raise InputError("Deliveries reference an unknown match.")
    as_of = max(m.day.isoformat() for m in data.matches)
    voided = {r["original_match_id"] for r in rules.reference["fixture_replacements"]
              if r["effective_date"] <= as_of}
    base = output / "_work" / "deliveries"
    base.mkdir(parents=True)
    ball_writer = ParquetLedger(base / "ball_pressure.parquet", BALL_SCHEMA)
    player_writer = ParquetLedger(base / "player_delivery_contributions.parquet", PLAYER_SCHEMA)
    try:
        quality, attribution, _, profiles, scope = analyze(
            data, deliveries, metadata, teams, players, voided, ball_config,
            ball_writer.add, player_writer.add)
    finally:
        ball_writer.close()
        player_writer.close()
    write_csv(base / "attribution_exceptions.csv", attribution, ATTRIBUTION_FIELDS)
    write_json(base / "run_manifest.json", {"prematch_manifest": str(pre_dir / "run_manifest.json")})
    write_csv(output / "innings_quality.csv", quality)
    write_csv(output / "attribution_exceptions.csv", attribution, ATTRIBUTION_FIELDS)
    del deliveries

    LOG.info("Applying pressure ownership, late-play rules and rebuild/defence credit")
    states, endings, clinches, uncertain = context_states(base, data_dir)
    writer = ParquetLedger(output / "player_delivery_contributions.parquet",
                           pa.schema(list(PLAYER_SCHEMA) + COMPETITIVE_PLAYER_FIELDS))
    try:
        cards, raw_high, scored_high, rescue, totals = refine_deliveries(
            base, states, uncertain, high_config, ball_config, competitive_config, writer.add)
    finally:
        writer.close()
    ball_rows = sorted([card.evidence(ball_config) for card in cards.values()],
                       key=lambda r: (r["match_date"], r["match_id"], r["role"], r["player_id"]))
    high_rows = high_match_rows(
        raw_high, scored_high, rescue, endings, clinches, data_dir, high_config, competitive_config)
    match_lookup = {m.id: m for m in data.matches}
    for row in ball_rows:
        match = match_lookup[row["match_id"]]
        row.update({
            "player": data.players[row["player_id"]],
            "team": data.teams[row["team_id"]],
            "opponent": data.teams[match.opponent(row["team_id"])],
            "competition": data.competitions[match.comp]["comp_name"],
            "stage": match.stage, "result": match.result,
        })

    LOG.info("Scoring observed skill, errors and full-toss uncertainty")
    annotations, annotation_counts = load_annotations(data_dir / "ball_by_ball.csv")
    labelled = output / "_work" / "labelled_actions"
    labelled.mkdir()
    writer = ParquetLedger(labelled / "execution_ledger.parquet", EXECUTION_SCHEMA)
    try:
        collect_execution(data_dir, base, annotations, execution_config, writer.add)
    finally:
        writer.close()
    assigned = output / "_work" / "assigned_actions"
    assigned.mkdir()
    assigned_schema = pa.schema(list(EXECUTION_SCHEMA) + ASSIGNED_EXECUTION_FIELDS)
    writer = ParquetLedger(assigned / "execution_ledger.parquet", assigned_schema)
    try:
        refine_execution(labelled, states, execution_config, writer.add)
    finally:
        writer.close()
    writer = ParquetLedger(output / "execution_ledger.parquet",
                           pa.schema(list(assigned_schema) + FULL_TOSS_FIELDS))
    try:
        execution_rows, diagnostics = regrade_execution(
            parquet_rows(assigned / "execution_ledger.parquet"), states, annotations,
            execution_config, full_toss_policy, writer.add)
    finally:
        writer.close()

    LOG.info("Combining the three score parts and calculating score intervals")
    ratings = rate_players(
        profiles, ball_rows, high_rows, execution_rows, totals,
        execution_config, ball_config, high_config, competitive_config)
    for row in ratings:
        key = row["player_id"], row["role"]
        row.update({name: int(diagnostics[key][name]) for name in (
            "policy_affected_crunch_events", "controlled_legal_full_tosses",
            "boundary_debits", "neutral_events", "ungraded_events")})
    for filename, rows in (
        ("player_ratings.csv", ratings), ("player_match_contributions.csv", ball_rows),
        ("high_match_contributions.csv", high_rows), ("execution_match_cards.csv", execution_rows),
    ):
        write_csv(output / filename, rows)
    for role, name in (("batting", "batters"), ("bowling", "bowlers")):
        write_csv(output / f"top10_{name}.csv",
                  [r for r in ratings if r["role"] == role and r["rank"] is not None and r["rank"] <= 10],
                  list(ratings[0]))
    if any(digest(data_dir / name) != expected for name, expected in input_hashes.items()):
        raise InputError("The data changed during scoring; no completion manifest was written.")
    if any(digest(config_dir / name) != expected for name, expected in config_hashes.items()):
        raise InputError("Configuration changed during scoring.")
    if any(digest(Path(__file__).parent / name) != expected for name, expected in code.items()):
        raise InputError("Scoring code changed during execution.")
    files = [
        "player_ratings.csv", "player_match_contributions.csv", "high_match_contributions.csv",
        "execution_match_cards.csv", "player_delivery_contributions.parquet", "execution_ledger.parquet",
    ]
    summary = {
        "supplied_matches": len(data.matches),
        "supplied_deliveries": annotation_counts["rows"],
        "scored_matches": len({s["match_id"] for s in states.values()}),
        "scored_innings": scope["included_innings"],
        "delivery_states": len(states),
        "player_role_ratings": len(ratings),
        "eligible_batters": sum(r["eligible"] and r["role"] == "batting" for r in ratings),
        "eligible_bowlers": sum(r["eligible"] and r["role"] == "bowling" for r in ratings),
        "data_start": min(m.day.isoformat() for m in data.matches), "data_end": as_of,
    }
    write_json(output / "run_manifest.json", {
        "model": "CLUTCH", "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters, "summary": summary,
        "data_sha256": input_hashes, "config_sha256": config_hashes, "code_sha256": code,
        "artifacts": {name: digest(output / name) for name in files},
        "interpretation": "Historical pressure performance, not measured anxiety or a prediction of future form.",
    })
    LOG.info("Completed: %s", json.dumps(summary, sort_keys=True))
    return ratings
