"""Build the offline dashboard from a completed score run and its reliability checks."""

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clutch.assurance import SCORE_FIELDS, evaluate, load_evidence
from clutch.inputs import digest, read_csv


COLORS = ("#3864a8", "#c02f74", "#70436b")


def number(value):
    if value == "" or value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite dashboard value.")
    return result


def integer(value):
    result = number(value)
    if result is None or result != int(result):
        raise ValueError("Expected an integer dashboard count.")
    return int(result)


def safe(value):
    return html.escape(str(value), quote=True)


def fmt(value, places=2):
    return "Not available" if value is None else f"{value:,.{places}f}"


def collect(run, checks, data_dir):
    evidence = load_evidence(run)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    check_manifest = json.loads((checks / "checks_manifest.json").read_text(encoding="utf-8"))
    if check_manifest["scoring_manifest_sha256"] != digest(run / "run_manifest.json"):
        raise ValueError("The score and reliability runs do not match.")
    if digest(data_dir / "matches.csv") != manifest["data_sha256"]["matches.csv"]:
        raise ValueError("Match metadata differs from the scored data.")
    weights = [manifest["parameters"]["execution"][key]
               for key in ("outcome_share", "execution_share", "high_match_share")]
    if weights != [.5, .2, .3]:
        raise ValueError("The method description needs updating for these weights.")
    profiles = [r for r in read_csv(checks / "player_ratings_with_assurance.csv") if r["eligible"] == "True"]
    metadata = {r["match_id"]: r for r in read_csv(data_dir / "matches.csv")}
    ball_cards = {(r["player_id"], r["role"], r["match_id"]): r
                  for r in read_csv(run / "player_match_contributions.csv")}
    high_cards = {(r["player_id"], r["role"], r["match_id"]): r
                  for r in read_csv(run / "high_match_contributions.csv")}
    action_cards = {(r["player_id"], r["role"], r["match_id"]): r
                    for r in read_csv(run / "execution_match_cards.csv")}
    with duckdb.connect(config={"threads": 2, "memory_limit": "512MB"}) as con:
        con.read_parquet(str(run / "player_delivery_contributions.parquet")).create_view("events")
        query = con.execute("""
            SELECT player_id,role,match_id,min(team_id) AS team_id,
                sum(attributed_runs) AS runs,sum(legal_opportunity) AS balls,
                sum(credited_wicket) AS wickets,
                sum(CASE WHEN high_pressure AND competitive_scoring_active THEN attributed_runs ELSE 0 END) AS pressure_runs,
                sum(CASE WHEN high_pressure AND competitive_scoring_active THEN legal_opportunity ELSE 0 END) AS pressure_balls,
                sum(CASE WHEN high_pressure AND competitive_scoring_active THEN credited_wicket ELSE 0 END) AS pressure_wickets
            FROM events GROUP BY player_id,role,match_id
        """)
        columns = [c[0] for c in query.description]
        stats = {(r["player_id"], r["role"], r["match_id"]): r
                 for r in (dict(zip(columns, values)) for values in query.fetchall())}
    pressure_totals = {}
    for (pid, role, _), row in stats.items():
        total = pressure_totals.setdefault((pid, role), {"runs": 0, "balls": 0, "wickets": 0})
        for field in total:
            total[field] += row["pressure_"+field]
    indices = {(r["player_id"], r["role"]): i for i, r in enumerate(evidence.players)}
    players = []
    for profile in profiles:
        pid, role = profile["player_id"], profile["role"]
        total = pressure_totals[pid, role]
        if total["balls"] != integer(profile["high_pressure_legal_balls"]):
            raise ValueError("Pressure totals do not reconcile.")
        row = {
            "key": role+"-"+pid, "id": pid, "name": profile["player"], "role": role,
            "rank": integer(profile["rank"]), "score": number(profile["raw_performance_index"]),
            "points": [number(profile[k]) for k in ("outcome_points", "execution_points", "high_match_points")],
            "component_scores": [number(profile[k]) for k in ("outcome_score", "execution_score", "high_match_score")],
            "score_interval": [number(profile[k]) for k in ("raw_index_interval_low", "raw_index_interval_high")],
            "repeat_rate": 100*number(profile["joint_top10_resample_share"]),
            "rank_interval": [number(profile[k]) for k in ("joint_rank_p05", "joint_rank_p95")],
            "pressure_balls": total["balls"], "pressure_runs": total["runs"],
            "pressure_wickets": total["wickets"],
            "pressure_rate": (100 if role == "batting" else 6)*total["runs"]/total["balls"] if total["balls"] else None,
            "pressure_matches": integer(profile["distinct_pressure_matches"]),
            "big_balls": integer(profile["high_stakes_legal_balls"]),
            "big_matches": integer(profile["high_stakes_role_matches"]),
            "role_matches": integer(profile["matches_with_role_opportunities"]),
            "ordinary_events": integer(profile["ordinary_reference_events"]),
            "effective_pressure_matches": number(profile["effective_pressure_matches"]),
            "effective_big_matches": number(profile["effective_high_stakes_matches"]),
            "route": profile["eligibility_route"],
            "families": profile["observed_families"].split(";"),
            "matches": [],
        }
        if not math.isclose(sum(row["points"]), row["score"], abs_tol=1e-9, rel_tol=0):
            raise ValueError("Weighted points do not add to the score.")
        if row["rank"] <= 10:
            i = indices[pid, role]
            vectors = {field: evidence.matrices[field].getrow(i).toarray().ravel() for field in SCORE_FIELDS}
            totals = {field: vector.sum() for field, vector in vectors.items()}
            without = evaluate({field: totals[field]-values for field, values in vectors.items()}, evidence.parameters)
            for j, mid in enumerate(evidence.matches):
                key = pid, role, mid
                if key not in ball_cards and key not in high_cards and key not in action_cards:
                    continue
                stat = stats.get(key)
                base = ball_cards.get(key) or high_cards.get(key)
                team_id = stat["team_id"] if stat else base["team_id"] if base else None
                match = metadata[mid]
                if team_id == match["home_team_id"]:
                    opponent = match["away_team"]
                elif team_id == match["away_team_id"]:
                    opponent = match["home_team"]
                elif stat is not None:
                    raise ValueError("Unresolved team in a match summary.")
                else:
                    opponent = "Unavailable"
                row["matches"].append({
                    "id": mid, "date": match["match_date"], "opponent": opponent,
                    "stage": match["stage"], "runs": stat["runs"] if stat else None,
                    "balls": stat["balls"] if stat else None, "wickets": stat["wickets"] if stat else None,
                    "pressure_balls": stat["pressure_balls"] if stat else None,
                    "effect": row["score"]-float(without["score"][j]),
                })
            row["matches"].sort(key=lambda m: (m["date"], m["id"]))
        players.append(row)
    players.sort(key=lambda p: (p["role"], p["rank"]))
    for role in ("batting", "bowling"):
        if [p["rank"] for p in players if p["role"] == role and p["rank"] <= 10] != list(range(1, 11)):
            raise ValueError("An ordered top ten is missing.")
    factors = json.loads((ROOT / "assets" / "data" / "factors.json").read_text(encoding="utf-8"))
    identifiers = [item[0] for group in factors for item in group["items"]]
    expected = [f"{prefix}{i:02}" for prefix, count in
                (("D", 11), ("P", 21), ("R", 8), ("L", 18), ("O", 14),
                 ("X", 10), ("M", 14), ("Q", 14), ("N", 25)) for i in range(1, count+1)]
    if sorted(identifiers) != sorted(expected):
        raise ValueError("The complete factor checklist must contain each of the 135 items once.")
    return {
        "scope": manifest["summary"], "weights": weights, "players": players, "factors": factors,
        "checks": {k: check_manifest[k] for k in
                   ("draws", "eligible_player_roles", "scored_matches", "sensitivity_scenarios", "chronological_player_years")},
        "provenance": {"score_manifest_sha256": digest(run / "run_manifest.json"),
                       "checks_manifest_sha256": digest(checks / "checks_manifest.json")},
    }


def table_rows(players):
    return "\n".join(
        f'<tr data-player="{p["key"]}" data-rank="{p["rank"]}">'
        f'<td>{p["rank"]:02}</td><th scope="row" class="player-name">{safe(p["name"])}</th>'
        f'<td class="score">{p["score"]:.2f}</td><td>{p["pressure_balls"]:,}</td>'
        f'<td>{p["big_matches"]}</td><td>{p["repeat_rate"]:.1f}%</td></tr>'
        for p in players)


def match_rows(player, limit=None):
    matches = sorted(player["matches"], key=lambda m: abs(m["effect"]), reverse=True)
    if limit is not None:
        matches = matches[:limit]
    result = []
    for match in matches:
        if match["runs"] is None or match["balls"] is None:
            figures = "Unavailable"
        elif player["role"] == "batting":
            figures = f'{match["runs"]:g} ({match["balls"]} legal balls)'
        else:
            figures = f'{match["wickets"]}/{match["runs"]:g} ({match["balls"]} legal balls)'
        result.append(
            f'<tr><td>{safe(match["date"])}</td><td>{safe(match["opponent"])}</td>'
            f'<td>{safe(match["stage"])}</td><td>{safe(figures)}</td>'
            f'<td>{match["effect"]:+.2f}</td></tr>')
    return "\n".join(result)


def component_svg(players):
    width, height, left, right, top, step = 850, 495, 208, 38, 22, 40
    span = width-left-right
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Weighted score parts, out of 100">',
            '<rect width="100%" height="100%" fill="#fffafd"/>']
    for tick in range(0, 101, 25):
        x = left+span*tick/100
        body.append(f'<path d="M{x} 12V425" stroke="#efdde7"/><text x="{x}" y="449" text-anchor="middle" fill="#705b6a" font-family="Segoe UI,sans-serif" font-size="15">{tick}</text>')
    for i, player in enumerate(players[:10]):
        y = top+i*step
        body.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" fill="#352338" font-family="Segoe UI,sans-serif" font-size="15">{safe(player["name"])}</text>')
        x = left
        for amount, color in zip(player["points"], COLORS):
            w = amount/100*span
            body.append(f'<rect x="{x:.3f}" y="{y}" width="{w:.3f}" height="27" fill="{color}"/>')
            x += w
        body.append(f'<text x="{x+8:.3f}" y="{y+19}" fill="#352338" font-family="Segoe UI,sans-serif" font-size="15">{player["score"]:.2f}</text>')
    body.append('<text x="510" y="483" text-anchor="middle" fill="#705b6a" font-family="Segoe UI,sans-serif" font-size="15">Final score / 100</text></svg>')
    return "".join(body)


def match_svg(player, extent):
    matches = player["matches"]
    width, height, left, top, bottom = 920, 300, 58, 22, 252
    lower, upper = extent
    y = lambda value: bottom-(value-lower)/(upper-lower)*(bottom-top)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Change in score from each match">',
            '<rect width="100%" height="100%" fill="#fffafd"/>']
    for value in (lower, 0, upper):
        yy = y(value)
        body.append(f'<path d="M{left} {yy:.2f}H900" stroke="#e4cad8"/><text x="{left-9}" y="{yy+5:.2f}" text-anchor="end" fill="#705b6a" font-family="Segoe UI,sans-serif" font-size="15">{value:+.1f}</text>')
    step = (900-left)/max(1, len(matches))
    for i, match in enumerate(matches):
        x = left+(i+.5)*step
        yy, zero = y(match["effect"]), y(0)
        color = COLORS[0] if match["effect"] >= 0 else COLORS[1]
        body.append(f'<rect x="{x-step*.32:.2f}" y="{min(yy,zero):.2f}" width="{max(.7,step*.64):.2f}" height="{max(.5,abs(yy-zero)):.2f}" fill="{color}"/>')
    if matches:
        body.append(f'<text x="{left}" y="283" fill="#705b6a" font-family="Segoe UI,sans-serif" font-size="15">{matches[0]["date"]}</text>')
        body.append(f'<text x="900" y="283" text-anchor="end" fill="#705b6a" font-family="Segoe UI,sans-serif" font-size="15">{matches[-1]["date"]}</text>')
    body.append("</svg>")
    return "".join(body)


def build_html(data):
    batting = [p for p in data["players"] if p["role"] == "batting"]
    bowling = [p for p in data["players"] if p["role"] == "bowling"]
    first = batting[0]
    tables = []
    for role, players in (("batting", batting), ("bowling", bowling)):
        title = "Batters" if role == "batting" else "Bowlers"
        tables.append(f'''
        <div class="role-table" data-role="{role}" id="{role}-rankings">
          <h3 class="static-role-title">{title}</h3>
          <div class="table-scroll"><table>
            <caption class="sr-only">Ranked {role} performances under the current model</caption>
            <thead><tr><th scope="col">Rank</th><th scope="col">Player</th><th scope="col">Score</th><th scope="col">Pressure<br>balls</th><th scope="col">Big-match<br>games</th><th scope="col">Top-10<br>repeat rate</th></tr></thead>
            <tbody>{table_rows(players)}</tbody>
          </table></div>
        </div>''')
    factor_groups = []
    for group in data["factors"]:
        items = "".join(f'<li data-factor="{safe(code)}">{safe(label)}</li>' for code, label in group["items"])
        factor_groups.append(
            f'<details class="factor-group" data-kind="{safe(group["kind"])}">'
            f'<summary>{safe(group["title"])} <span>{len(group["items"])} items</span></summary>'
            f'<ul class="factor-list">{items}</ul></details>')
    scope = data["scope"]
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="CLUTCH: batting and bowling performance under pressure, with transparent match evidence and a complete factor checklist.">
  <title>CLUTCH | Cricket under pressure</title>
  <link rel="stylesheet" href="styles.css">
  <script defer src="snapshot.js"></script>
  <script defer src="plotly-basic.min.js"></script>
  <script defer src="app.js"></script>
</head>
<body>
<a class="skip-link" href="#rankings">Skip to rankings</a>
<header class="masthead">
  <a class="wordmark" href="#top" aria-label="CLUTCH home"><span class="wickets" aria-hidden="true"><i></i><i></i><i></i></span>CLUTCH</a>
  <nav aria-label="Page sections"><a href="#rankings">Rankings</a><a href="#player">Player evidence</a><a href="#factors">Every factor</a><a href="#method">Method</a></nav>
</header>
<main id="top">
  <section class="hero" aria-labelledby="hero-title">
    <div><p class="eyebrow">SUPERR SELECTOR / SEASON 2</p><h1 id="hero-title">Who delivers<br><em>when it matters.</em></h1>
      <p class="hero-copy">Runs and wickets are the start, not the whole story. Explore the situation, the player's contribution and the evidence behind each selection.</p>
      <p class="scope-line">Historical data: {scope["data_start"]} to {scope["data_end"]}. No live match feed.</p></div>
    <div class="hero-facts" aria-label="Analysis scope">
      <div><strong>{scope["supplied_matches"]:,}</strong><span>supplied matches</span></div>
      <div><strong>{scope["scored_matches"]:,}</strong><span>matches with usable scoring evidence</span></div>
      <div><strong>{scope["eligible_batters"]} / {scope["eligible_bowlers"]}</strong><span>eligible batting / bowling entries</span></div>
      <p>Separate lists. Same rules for everyone.</p>
    </div>
  </section>
  <p id="notice" class="notice" role="status" hidden></p>
  <section id="rankings" aria-labelledby="rankings-title">
    <div class="section-heading"><div><p class="eyebrow">THE SELECTIONS</p><h2 id="rankings-title">Performance under pressure</h2></div><p>Current model scores, not predictions of future form.</p></div>
    <div class="toolbar" aria-label="Ranking filters">
      <div class="segmented" role="group" aria-label="Choose batting or bowling"><button type="button" data-role-button="batting" aria-pressed="true" disabled>Batting</button><button type="button" data-role-button="bowling" aria-pressed="false" disabled>Bowling</button></div>
      <label class="search-label">Find a player<input id="player-search" type="search" placeholder="Search ranked players" disabled></label>
      <label>Show<select id="ranking-scope" disabled><option value="top">Top 10</option><option value="all">All ranked players</option></select></label>
      <button id="download" type="button" class="quiet-button" disabled>Download table</button>
    </div>
    <noscript><p class="notice">Both ranked lists and the full factor checklist are available below. Enable JavaScript for filters and player selection.</p></noscript>
    <div class="ranking-grid">
      <div class="panel ranking-panel"><p id="result-count" class="table-note">All eligible entries are listed. Ranks 1-10 are the selections.</p>{"".join(tables)}<p id="empty-ranking" class="empty" hidden>No ranked player matches this search.</p>
        <p class="fineprint">Top-10 repeat rate: how often a player remains in the top ten across 1,000 shared match redraws. It is not a probability of true ability.</p></div>
      <figure class="panel score-chart"><figcaption><h3 id="component-title">How the top ten scores add up</h3><p>Weighted points. Neutral total: 50.</p></figcaption>
        <div class="legend"><span><i class="swatch result"></i>Runs / wickets 50%</span><span><i class="swatch skill"></i>Skill / errors 20%</span><span><i class="swatch big"></i>Big matches 30%</span></div>
        <div id="component-chart" class="plot" aria-label="Interactive weighted-score comparison" hidden></div>
        <img id="component-fallback" src="batting-scores.svg" alt="Batting top ten, with the three weighted score parts" width="850" height="495">
        <p class="fineprint">A close score gap is not a clear separation in ability. Read the match evidence and uncertainty together.</p>
      </figure>
    </div>
  </section>
  <section id="player" aria-labelledby="player-title">
    <div class="section-heading"><div><p class="eyebrow">LOOK BEHIND THE RANK</p><h2 id="player-title">Player evidence</h2></div>
      <label>Choose a player<select id="player-select" disabled><option value="{first["key"]}">{safe(first["name"])}</option></select></label></div>
    <article class="panel profile">
      <div class="profile-heading"><div><span id="profile-rank" class="rank-label">BATTING / #{first["rank"]}</span><h3 id="profile-name">{safe(first["name"])}</h3><p id="profile-route">Qualifies through pressure-ball and big-match evidence.</p></div>
        <div class="profile-score"><strong id="profile-score">{first["score"]:.2f}</strong><span>current score / 100</span></div></div>
      <div id="profile-metrics" class="profile-metrics">
        <div><strong>{first["pressure_balls"]:,}</strong><span>pressure balls</span></div><div><strong>{first["pressure_matches"]}</strong><span>pressure matches</span></div>
        <div><strong>{first["big_matches"]}</strong><span>big-match games</span></div><div><strong>{first["repeat_rate"]:.1f}%</strong><span>top-ten repeat rate</span></div>
      </div>
      <div id="profile-parts" class="profile-parts"><span>Runs / wickets: {first["points"][0]:.2f}</span><span>Skill / errors: {first["points"][1]:.2f}</span><span>Big matches: {first["points"][2]:.2f}</span></div>
      <p id="profile-uncertainty" class="uncertainty">95% score range: {first["score_interval"][0]:.2f} to {first["score_interval"][1]:.2f}. Comparison averages stay fixed.</p>
      <p id="profile-warning" class="fineprint">A rank reflects this dataset and these rules, not a certain ordering of players.</p>
    </article>
    <div id="match-evidence" class="panel match-evidence">
      <div class="section-heading compact"><div><h3>Which matches moved this score?</h3><p>Including a match can raise or lower the final score.</p></div><label>Season<select id="match-season" disabled><option value="all">All seasons</option></select></label></div>
      <div id="match-chart" class="plot match-plot" aria-label="Change in score from each match" hidden></div>
      <img id="match-fallback" src="{first["key"]}-matches.svg" alt="Match-by-match score influence for {safe(first["name"])}" width="920" height="300">
      <div class="table-scroll"><table><caption class="sr-only">Largest match influences for the selected player</caption><thead><tr><th scope="col">Date</th><th scope="col">Opponent</th><th scope="col">Stage</th><th scope="col">Figures</th><th scope="col">Score effect</th></tr></thead><tbody id="match-rows">{match_rows(first, 8)}</tbody></table></div>
      <button type="button" id="more-matches" class="quiet-button" disabled>Show all matches</button>
      <p class="fineprint">Effect = current score minus score without that match, with earlier comparison averages unchanged. A negative effect need not mean a bad performance. Figures use attributable totals and legal balls, which may differ from official balls faced.</p>
    </div>
    <p id="match-unavailable" class="notice" hidden>Match-by-match details are packaged for the selected top ten in each role. Aggregate figures remain available for other ranked players.</p>
  </section>
  <section id="factors" aria-labelledby="factors-title">
    <div class="section-heading"><div><p class="eyebrow">THE COMPLETE CHECKLIST</p><h2 id="factors-title">Every factor considered</h2></div><button id="expand-factors" type="button" class="quiet-button" disabled>Expand all groups</button></div>
    <p class="section-intro">Some items set pressure, some measure performance, and some check reliability. They do not all add points. Unmeasured and unused ideas are listed separately.</p>
    <div class="factor-groups">{"".join(factor_groups)}</div>
  </section>
  <section id="method" aria-labelledby="method-title">
    <div class="section-heading"><div><p class="eyebrow">HOW TO READ THE RESULT</p><h2 id="method-title">Three parts. One transparent score.</h2></div></div>
    <div class="method-grid">
      <article class="panel"><span class="method-weight">50%</span><h3>Runs and wickets</h3><p>How good was the contribution for that situation? Was it above the player's usual level? Did it happen repeatedly?</p></article>
      <article class="panel"><span class="method-weight">20%</span><h3>Skill and errors</h3><p>Effective deliveries and batting responses, recovery after setbacks, and discipline. A label alone is not proof of good or poor technique.</p></article>
      <article class="panel"><span class="method-weight">30%</span><h3>Big matches</h3><p>The full competitive innings or spell when the team's pre-match stakes are high, including earned rebuilding and defensive credit.</p></article>
    </div>
    <div class="limits panel"><h3>What the score does not claim</h3><p>This is a historical, situation-based index, not measured anxiety or a calibrated chance of winning. Small samples and different competition mixes matter. The archive is not complete career history and contains no T20 World Cup data. The checks are not a blind external test or proof of predictive superiority.</p></div>
    <p class="report-note">For the detailed algorithm, refer to the report.</p>
  </section>
</main>
<footer><span>CLUTCH / SupeRR Selector</span><span>Derived summaries only. Raw competition data is not included.</span><a href="https://github.com/parvvchhabra15/clutch">Repository (access required)</a><a href="plotly-LICENSE.txt">Chart library license</a></footer>
</body>
</html>'''


def build(run, checks, data_dir):
    data = collect(run, checks, data_dir)
    site = ROOT / "site"
    charts = ROOT / "assets" / "charts"
    site.mkdir(exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    effects = [m["effect"] for p in data["players"] for m in p["matches"]]
    data["match_effect_extent"] = [min(-.25, min(effects)*1.15), max(.25, max(effects)*1.15)]
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > 1024*1024:
        raise ValueError("The browser summary exceeds the 1 MiB budget.")
    (ROOT / "assets" / "data" / "dashboard-summary.json").write_text(payload+"\n", encoding="utf-8")
    (site / "snapshot.js").write_text("window.CLUTCH_DATA = "+payload+";\n", encoding="utf-8")
    for role in ("batting", "bowling"):
        name = f"{role}-scores.svg"
        (charts / name).write_text(component_svg([p for p in data["players"] if p["role"] == role]), encoding="utf-8")
        shutil.copyfile(charts / name, site / name)
    for player in data["players"]:
        if player["matches"]:
            name = player["key"]+"-matches.svg"
            (charts / name).write_text(match_svg(player, data["match_effect_extent"]), encoding="utf-8")
            shutil.copyfile(charts / name, site / name)
    (site / "index.html").write_text(build_html(data), encoding="utf-8")
    export_fields = ["role", "rank", "name", "score", "pressure_balls", "pressure_matches",
                     "big_matches", "repeat_rate"]
    with (site / "rankings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=export_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["players"])
    print(json.dumps({"site": str(site), "ranked_entries": len(data["players"]),
                      "selected_profiles": sum(bool(p["matches"]) for p in data["players"]),
                      "match_summaries": len(effects), "factor_items": sum(len(g["items"]) for g in data["factors"]),
                      "browser_data_bytes": len(payload.encode("utf-8"))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.run.resolve(), args.checks.resolve(), args.data_dir.resolve())
