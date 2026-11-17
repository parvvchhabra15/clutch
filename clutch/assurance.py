import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import duckdb
import numpy as np
from scipy import sparse, stats
from .inputs import InputError, digest, read_csv, write_csv, write_json

SCORE_FIELDS=("bw","ba","bl","bs","hw","hv","hg","ew","ev")

FIELDS=SCORE_FIELDS+("quality","pballs","pmatches","pcap","pcap2",
                     "hballs","hmatches","hcap","hcap2","normal","runs","legal")

def fractional_top_probability(values, count):
    order = np.sort(values)[::-1]
    cutoff = order[min(count, len(order)) - 1]
    better = values > cutoff + 1e-10
    equal = abs(values - cutoff) <= 1e-10
    places = min(count, len(values)) - int(better.sum())
    return better.astype(float) + equal * places / max(1, int(equal.sum()))


def divide(numerator, denominator):
    return np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=float),where=denominator>0)

def evaluate(totals, parameters):
    b,h,e=(parameters[name] for name in ("outcomes","high_match","execution"))
    bw,hw,ew=(totals[name] for name in ("bw","hw","ew"))
    absolute=divide(totals["ba"],bw)
    lift=divide(totals["bl"],bw)
    consistency=(totals["bs"]+2)/(bw+4)
    unshrunk=(b["absolute_weight"]*(50+30*np.tanh(absolute/b["contribution_scale"]))
              +b["resilience_weight"]*(50+30*np.tanh(lift/b["contribution_scale"]))
              +b["consistency_weight"]*100*consistency)
    outcome=50+bw/(bw+b["career_prior_matches"])*(unshrunk-50)
    execution=50+ew/(ew+e["prior_matches"])*25*np.tanh(divide(totals["ev"],ew)/e["signal_scale"])
    high_mean=divide(totals["hv"],hw)
    high=50+hw/(hw+h["prior_matches"])*30*np.tanh(high_mean/h["run_scale"])
    if h["gentle_adjustment"]:
        high+=h["gentle_adjustment_cap"]*totals["hg"]/(hw+h["prior_matches"])
    score=e["outcome_share"]*outcome+e["execution_share"]*execution+e["high_match_share"]*high
    return {"score":score,"outcome":outcome,"execution":execution,"high_match":high,
            "absolute":absolute,"lift":lift,"consistency":consistency,"high_mean":high_mean}

def eligible(totals, result, parameters):
    b,h,r=(parameters[name] for name in ("outcomes","high_match","refinements"))
    peff=divide(totals["pcap"]**2,totals["pcap2"])
    heff=divide(totals["hcap"]**2,totals["hcap2"])
    ball=((totals["pmatches"]>=b["minimum_pressure_matches"])
          &(totals["pballs"]>=b["minimum_pressure_legal_balls"])
          &(peff>=r["minimum_effective_matches"])&(result["absolute"]>0))
    high=((totals["hmatches"]>=h["minimum_matches"])
          &(totals["hballs"]>=h["minimum_legal_balls"])
          &(heff>=r["minimum_effective_matches"])&(result["high_mean"]>0))
    return (totals["normal"]>=b["minimum_normal_events"])&(totals["quality"]>0)&(ball|high)

def midranks(values):
    differences=values[None,:]-values[:,None]
    return (differences>1e-10).sum(axis=1)+((abs(differences)<=1e-10).sum(axis=1)+1)/2

@dataclass
class Evidence:
    players: list
    matches: list
    dates: np.ndarray
    families: np.ndarray
    innings: np.ndarray
    matrices: dict
    parameters: dict
    hashes: dict
    base: Path
    omitted_execution_cards: int = 0

    def totals(self, counts=None, player_indices=None, cell_mask=None, high_power=1.0, score_only=False):
        counts=np.ones(len(self.matches)) if counts is None else np.asarray(counts,dtype=float)
        if counts.ndim not in (1,2) or counts.shape[0]!=len(self.matches) or not np.isfinite(counts).all() or (counts<0).any():
            raise InputError("Match counts must be finite, nonnegative and aligned with the match axis.")
        if cell_mask is not None and np.shape(cell_mask)!=self.innings.shape:
            raise InputError("The innings scope must match the player-by-match axes.")
        select=slice(None) if player_indices is None else player_indices
        ratio=None
        if high_power!=1:
            if not np.isfinite(high_power) or high_power<=0:
                raise InputError("High-match sensitivity exponent must be positive.")
            ratio=self.matrices["hw"].copy()
            ratio.data=np.power(ratio.data,high_power-1)
            ratio=ratio[select]
        answer={}
        for field in SCORE_FIELDS if score_only else FIELDS:
            matrix=self.matrices[field][select]
            if ratio is not None and field in {"hw","hv","hg"}:
                matrix=matrix.multiply(ratio)
            if cell_mask is not None:
                matrix=matrix.multiply(cell_mask[select])
            answer[field]=np.asarray(matrix@counts)
        return answer

def load_evidence(base):
    base=Path(base).resolve()
    manifest=json.loads((base/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != "CLUTCH" or manifest.get("status") != "complete":
        raise InputError("Assurance requires a completed CLUTCH run.")
    hashes = {}
    for name, expected in manifest["artifacts"].items():
        path = base / name
        if digest(path) != expected:
            raise InputError(f"Scoring artifact changed: {name}")
        hashes[str(path)] = expected
    for name, expected in manifest["code_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise InputError(f"Scoring implementation changed: {name}")
    players=read_csv(base/"player_ratings.csv")
    ball=read_csv(base/"player_match_contributions.csv")
    high=read_csv(base/"high_match_contributions.csv")
    execution=read_csv(base/"execution_match_cards.csv")
    pindex={(r["player_id"],r["role"]):i for i,r in enumerate(players)}
    metadata={}
    for record in ball+high+execution:
        value=record["match_date"],record["family"]
        if record["match_id"] in metadata and metadata[record["match_id"]]!=value:
            raise InputError("Conflicting date/family for one match.")
        metadata[record["match_id"]]=value
    matches=sorted(metadata)
    mindex={mid:i for i,mid in enumerate(matches)}
    shape=len(players),len(matches)
    entries={name:([],[],[]) for name in FIELDS}
    def add(field,key,mid,value):
        if not np.isfinite(value):
            raise InputError(f"Non-finite sufficient statistic: {field}/{key}/{mid}")
        if value:
            pi,mi,values=entries[field]
            pi.append(pindex[key]); mi.append(mindex[mid]); values.append(value)
    for table in (ball,high,execution):
        keys=[(r["player_id"],r["role"],r["match_id"]) for r in table]
        if len(keys)!=len(set(keys)):
            raise InputError("Duplicate player-role match cards.")
    for row in ball:
        key=row["player_id"],row["role"]
        mid=row["match_id"]
        w=float(row["match_evidence_weight"])
        add("bw",key,mid,w)
        if w:
            add("ba",key,mid,w*float(row["absolute_pressure_contribution"]))
            add("bl",key,mid,w*float(row["usual_baseline_support"])*float(row["pressure_contribution_vs_usual"]))
            success=1 if row["status"]=="delivered" else 0 if row["status"]=="underperformed" else .5
            add("bs",key,mid,w*success)
        n=int(row["high_pressure_legal_balls"])
        for field,value in {"quality":float(row["whole_opportunity_contribution"]),"pballs":n,
                            "pmatches":int(n>0),"pcap":min(1,n/8),"pcap2":min(1,n/8)**2,
                            "runs":float(row["attributed_runs"]),"legal":int(row["legal_balls"])}.items():
            add(field,key,mid,value)
    for row in high:
        if row["net_team_impact"]=="" or float(row["high_match_weight"])<=0:
            continue
        key=row["player_id"],row["role"]
        mid=row["match_id"]
        w=float(row["high_match_weight"])
        n=int(row["legal_balls"])
        for field,value in {"hw":w,"hv":w*float(row["net_team_impact"]),
                            "hg":w*float(row["gentle_delivery_signal"]),"hballs":n,"hmatches":int(n>0),
                            "hcap":min(1,n/8),"hcap2":min(1,n/8)**2}.items():
            add(field,key,mid,value)
    omitted_execution_cards=0
    for row in execution:
        key=row["player_id"],row["role"]
        if key not in pindex:
            omitted_execution_cards+=1
            continue
        if row["mean_execution_signal"]=="":
            continue
        w=float(row["match_evidence_weight"])
        add("ew",key,row["match_id"],w)
        add("ev",key,row["match_id"],w*float(row["mean_execution_signal"]))
    innings=np.zeros(shape,dtype=np.int8)
    with duckdb.connect(config={"threads":2,"memory_limit":"512MB"}) as con:
        con.read_parquet(str(base/"player_delivery_contributions.parquet")).create_view("e")
        con.read_parquet(str(base/"execution_ledger.parquet")).create_view("x")
        ceiling=manifest["parameters"]["outcomes"]["normal_pressure_ceiling"]
        for pid,role,mid,n in con.execute(f"""
            SELECT player_id,role,match_id,sum(CAST(team_pressure<={ceiling} AS INTEGER))
            FROM e GROUP BY ALL""").fetchall():
            add("normal",(pid,role),mid,n)
        for pid,role,mid,low,high_innings in con.execute("""
            SELECT player_id,role,match_id,min(innings),max(innings) FROM (
                SELECT player_id,role,match_id,innings FROM e
                UNION ALL SELECT player_id,role,match_id,innings FROM x) GROUP BY ALL""").fetchall():
            if (pid,role) not in pindex:
                continue
            if low!=high_innings or low not in (1,2):
                raise InputError("A role/match combines unsupported innings.")
            innings[pindex[pid,role],mindex[mid]]=low
    matrices={name:sparse.csr_matrix((values,(pi,mi)),shape=shape)
              for name,(pi,mi,values) in entries.items()}
    evidence=Evidence(players,matches,np.array([metadata[m][0] for m in matches]),
                      np.array([metadata[m][1] for m in matches]),innings,matrices,
                      manifest["parameters"],hashes,base,omitted_execution_cards)
    full=evidence.totals()
    point=evaluate(full,evidence.parameters)
    for field,column in [("score","raw_performance_index"),("outcome","outcome_score"),
                         ("execution","execution_score"),("high_match","high_match_score")]:
        expected=np.array([float(r[column]) for r in players])
        if not np.allclose(point[field],expected,atol=1e-9,rtol=0):
            raise InputError(f"Full-score assurance kernel does not reproduce {column}.")
    if not np.array_equal(eligible(full,point,evidence.parameters),[r["eligible"]=="True" for r in players]):
        raise InputError("Assurance eligibility does not reproduce the frozen population.")
    if not np.array_equal(full["normal"],[int(r["ordinary_reference_events"]) for r in players]):
        raise InputError("Ordinary-reference counts do not reconcile.")
    return evidence

def joint_bootstrap(evidence, draws=1000, seed=618407):
    if not isinstance(draws,int) or draws<2:
        raise InputError("At least two bootstrap draws are required.")
    indices=np.array([i for i,p in enumerate(evidence.players) if p["eligible"]=="True"],dtype=int)
    if not len(indices):
        raise InputError("There are no eligible players to compare.")
    rng=np.random.default_rng(seed)
    count=len(evidence.matches)
    samples=rng.multinomial(count,np.full(count,1/count),size=draws).T
    scores=evaluate(evidence.totals(samples,indices,score_only=True),evidence.parameters)["score"].T
    ranks=np.zeros_like(scores)
    memberships=np.zeros_like(scores)
    roles=np.array([evidence.players[i]["role"] for i in indices])
    for role in ("batting","bowling"):
        positions=np.flatnonzero(roles==role)
        if not len(positions):
            continue
        for draw in range(draws):
            values=scores[draw,positions]
            ranks[draw,positions]=midranks(values)
            memberships[draw,positions]=fractional_top_probability(values,10)
        if not np.allclose(memberships[:,positions].sum(axis=1),min(10,len(positions))):
            raise InputError("Fractional top-ten places do not sum correctly.")
    result=[]
    for j,i in enumerate(indices):
        player=evidence.players[i]
        lo,med,hi=np.quantile(ranks[:,j],[.05,.5,.95])
        slow,shigh=np.quantile(scores[:,j],[.025,.975])
        result.append({
            "player_id":player["player_id"],"player":player["player"],"role":player["role"],
            "current_rank":int(player["rank"]),"current_score":float(player["raw_performance_index"]),
            "top10_resample_share":float(memberships[:,j].mean()),
            "rank_p05":float(lo),"rank_median":float(med),"rank_p95":float(hi),
            "score_bootstrap_mean":float(scores[:,j].mean()),"score_bootstrap_sd":float(scores[:,j].std()),
            "score_p025":float(slow),"score_p975":float(shigh),
            "draws":draws,"eligible_pool":"fixed_original_role_pool",
        })
    return sorted(result,key=lambda r:(r["role"],r["current_rank"])),indices,samples,scores

def sensitivity_definitions(evidence):
    ones=np.ones(len(evidence.matches))
    scenarios=[("control",{},ones,None,1.0,False)]
    for label,section,updates in [
        ("execution_10","execution",{"execution_share":.1,"outcome_share":.6}),
        ("execution_30","execution",{"execution_share":.3,"outcome_share":.4}),
        ("usual_share_20","outcomes",{"absolute_weight":.6,"resilience_weight":.2}),
        ("usual_share_40","outcomes",{"absolute_weight":.4,"resilience_weight":.4}),
        ("outcome_prior_3","outcomes",{"career_prior_matches":3.0}),
        ("outcome_prior_8","outcomes",{"career_prior_matches":8.0}),
        ("high_match_prior_1p5","high_match",{"prior_matches":1.5}),
        ("high_match_prior_6","high_match",{"prior_matches":6.0}),
    ]:
        scenarios.append((label,{section:updates},ones,None,1.0,False))
    scenarios.extend([
        ("stakes_curve_sqrt",{},ones,None,.5,False),
        ("stakes_curve_squared",{},ones,None,2.0,False),
        ("ipl_only",{},(evidence.families=="ipl").astype(float),None,1.0,False),
        ("senior_non_ipl_only",{},(~np.isin(evidence.families,["ipl","emerging_t20"])).astype(float),None,1.0,False),
        ("exclude_emerging",{},(evidence.families!="emerging_t20").astype(float),None,1.0,False),
        ("first_innings_only",{},ones,evidence.innings==1,1.0,False),
        ("chase_innings_only",{},ones,evidence.innings==2,1.0,False),
        ("since_2025",{},(evidence.dates>="2025-01-01").astype(float),None,1.0,False),
    ])
    for year in sorted(set(day[:4] for day in evidence.dates)):
        scenarios.append((f"without_{year}",{},np.array([not d.startswith(year) for d in evidence.dates],dtype=float),None,1.0,False))
    ages=(np.datetime64(str(max(evidence.dates)),"D")-evidence.dates.astype("datetime64[D]")).astype(float)
    for half_life in (1,2):
        scenarios.append((f"recency_half_life_{half_life}y",{},.5**(ages/(365.25*half_life)),None,1.0,True))
    return scenarios

def sensitivity(evidence):
    scenarios=sensitivity_definitions(evidence)
    original=np.array([p["eligible"]=="True" for p in evidence.players])
    summary,detail=[],[]
    for name,changes,counts,mask,power,fixed_pool in scenarios:
        parameters=copy.deepcopy(evidence.parameters)
        for section,updates in changes.items():
            parameters[section].update(updates)
        totals=evidence.totals(counts,cell_mask=mask,high_power=power)
        values=evaluate(totals,parameters)
        admitted=original if fixed_pool else eligible(totals,values,parameters)
        for role in ("batting","bowling"):
            role_indices=[i for i,p in enumerate(evidence.players) if p["role"]==role]
            pool=sorted([i for i in role_indices if admitted[i]],
                        key=lambda i:(-values["score"][i],evidence.players[i]["player_id"]))
            ranks={i:n for n,i in enumerate(pool,1)}
            old_top={i for i in role_indices if evidence.players[i]["rank"] and int(evidence.players[i]["rank"])<=10}
            summary.append({
                "scenario":name,"role":role,"eligible_players":len(pool),"top10_retained":len(old_top&set(pool[:10])),
                "original_top10_evaluable":int(sum(admitted[i] for i in old_top)),
                "eligibility_mode":"fixed_original_pool" if fixed_pool else "recomputed_from_scoped_evidence",
                "parameter_changes":json.dumps(changes,sort_keys=True),"stakes_exponent":power,
                "reference_histories":"frozen_parent_references",
            })
            for i in role_indices:
                player=evidence.players[i]
                detail.append({
                    "scenario":name,"role":role,"player_id":player["player_id"],"player":player["player"],
                    "current_rank":int(player["rank"]) if player["rank"] else None,"scenario_rank":ranks.get(i),
                    "current_score":float(player["raw_performance_index"]),"scenario_score":float(values["score"][i]),
                    "eligible_in_scenario":bool(admitted[i]),
                })
    return summary,detail

def correlation(x,y,minimum=10):
    if len(x)<minimum:
        return None,"insufficient_player_pairs"
    if np.ptp(x)==0 or np.ptp(y)==0:
        return None,"constant_metric"
    value=float(stats.spearmanr(x,y).statistic)
    if not np.isfinite(value):
        raise InputError("Unexpected non-finite rank correlation.")
    return value,"observed"

def walk_forward(evidence):
    summary,predictions=[],[]
    for year in (2024,2025,2026):
        cutoff=f"{year}-01-01"
        end=f"{year+1}-01-01"
        train=(evidence.dates<cutoff).astype(float)
        test=((evidence.dates>=cutoff)&(evidence.dates<end)).astype(float)
        train_totals=evidence.totals(train)
        test_totals=evidence.totals(test)
        before=evaluate(train_totals,evidence.parameters)
        after=evaluate(test_totals,evidence.parameters)
        admitted=eligible(train_totals,before,evidence.parameters)
        supported=(test_totals["pmatches"]>=3)&(test_totals["pballs"]>=24)&(test_totals["bw"]>0)
        for role in ("batting","bowling"):
            selected=np.array([i for i,p in enumerate(evidence.players)
                               if p["role"]==role and admitted[i] and supported[i]],dtype=int)
            raw=divide(train_totals["runs"],train_totals["legal"])
            raw*=100 if role=="batting" else -6
            predictors={"full_index":before["score"],"prior_pressure_quality":before["absolute"],"simple_rate":raw}
            for target_name,target in [("future_pressure_quality",after["absolute"]),("future_full_index",after["score"])]:
                record={"role":role,"test_year":year,"player_pairs":len(selected),"target":target_name,
                        "train_before":cutoff,"test_end_exclusive":end,
                        "future_support_rule":"3 pressure appearances and 24 pressure legal balls"}
                for label,predictor in predictors.items():
                    value,status=correlation(predictor[selected],target[selected])
                    record[label+"_spearman"]=value
                    record[label+"_status"]=status
                summary.append(record)
            for i in selected:
                player=evidence.players[i]
                predictions.append({
                    "role":role,"player_id":player["player_id"],"player":player["player"],"test_year":year,
                    "train_score":float(before["score"][i]),"train_pressure_quality":float(before["absolute"][i]),
                    "train_simple_rate":float(raw[i]),"train_ordinary_events":int(train_totals["normal"][i]),
                    "train_pressure_balls":int(train_totals["pballs"][i]),
                    "test_pressure_balls":int(test_totals["pballs"][i]),
                    "test_pressure_matches":int(test_totals["pmatches"][i]),
                    "test_pressure_quality":float(after["absolute"][i]),"test_full_index":float(after["score"][i]),
                })
    return summary,predictions


def run_checks(base, output, draws=1000):
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise InputError(f"Output already exists: {output}")
    evidence = load_evidence(base)
    uncertainty, _, _, _ = joint_bootstrap(evidence, draws=draws)
    scenarios, scenario_players = sensitivity(evidence)
    chronology, chronological_players = walk_forward(evidence)
    by_player = {(r["player_id"], r["role"]): r for r in uncertainty}
    profiles = []
    for player in evidence.players:
        row = dict(player)
        item = by_player.get((row["player_id"], row["role"]))
        row.update({
            "joint_top10_resample_share": item["top10_resample_share"] if item else None,
            "joint_rank_p05": item["rank_p05"] if item else None,
            "joint_rank_median": item["rank_median"] if item else None,
            "joint_rank_p95": item["rank_p95"] if item else None,
            "joint_score_mean": item["score_bootstrap_mean"] if item else None,
            "joint_score_sd": item["score_bootstrap_sd"] if item else None,
            "joint_score_p025": item["score_p025"] if item else None,
            "joint_score_p975": item["score_p975"] if item else None,
            "assurance_status": "fixed_pool_joint_resampling" if item else "not_eligible",
        })
        profiles.append(row)
    for path, expected in evidence.hashes.items():
        if digest(Path(path)) != expected:
            raise InputError("Scoring artifacts changed during the checks.")
    output.mkdir(parents=True)
    for name, rows in (
        ("player_ratings_with_assurance.csv", profiles),
        ("joint_rank_uncertainty.csv", uncertainty),
        ("sensitivity_summary.csv", scenarios),
        ("sensitivity_rankings.csv", scenario_players),
        ("walk_forward_summary.csv", chronology),
        ("walk_forward_player_pairs.csv", chronological_players),
    ):
        write_csv(output / name, rows)
    write_json(output / "checks_manifest.json", {
        "model": "CLUTCH", "status": "complete",
        "scoring_manifest_sha256": digest(base / "run_manifest.json"),
        "draws": draws, "seed": 618407,
        "eligible_player_roles": len(uncertainty),
        "scored_matches": len(evidence.matches),
        "sensitivity_scenarios": len({r["scenario"] for r in scenarios}),
        "chronological_player_years": len(chronological_players),
        "interpretation": "Conditional stability with fixed references; not a probability of true ability.",
    })
    logging.info("Completed reliability checks for %d eligible player-role entries.", len(uncertainty))


def main():
    parser = argparse.ArgumentParser(description="Check the stability of a completed CLUTCH run.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_checks(args.run, args.output)
    except (InputError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
