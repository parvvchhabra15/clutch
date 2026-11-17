"""Descriptive selection challenges; these functions never change player ratings."""

import math
from dataclasses import dataclass

import numpy as np

from .inputs import InputError


@dataclass(frozen=True)
class ComparisonPolicy:
    minimum_legal_per_cell: int = 20
    minimum_common_legal: int = 60
    broad_coverage: float = .5


def compare_cells(incumbent, challenger, role, policy=ComparisonPolicy(), incumbent_total=None, challenger_total=None):
    if role not in {"batting","bowling"}:
        raise InputError("Unknown comparison role.")
    common=sorted(key for key in incumbent.keys() & challenger.keys()
                  if incumbent[key]["legal"]>=policy.minimum_legal_per_cell
                  and challenger[key]["legal"]>=policy.minimum_legal_per_cell)
    a_total=sum(c["legal"] for c in incumbent.values()) if incumbent_total is None else incumbent_total
    b_total=sum(c["legal"] for c in challenger.values()) if challenger_total is None else challenger_total
    na=sum(incumbent[k]["legal"] for k in common)
    nb=sum(challenger[k]["legal"] for k in common)
    result={
        "common_cells":len(common),"common_contexts":";".join("|".join(map(str,k)) for k in common),
        "incumbent_common_legal":na,"challenger_common_legal":nb,
        "incumbent_coverage":na/a_total if a_total else None,
        "challenger_coverage":nb/b_total if b_total else None,
        "sufficient_common_sample":na>=policy.minimum_common_legal and nb>=policy.minimum_common_legal,
        "broad_common_coverage":bool(a_total and b_total and na/a_total>=policy.broad_coverage and nb/b_total>=policy.broad_coverage),
        "incumbent_standardized_rate":None,"challenger_standardized_rate":None,
        "incumbent_standardized_wickets_per100":None,"challenger_standardized_wickets_per100":None,
        "incumbent_standardized_value_per100":None,"challenger_standardized_value_per100":None,
        "incumbent_illegal_events_per100_legal":None,"challenger_illegal_events_per100_legal":None,
        "challenger_run_rate_advantage":None,"challenger_wicket_rate_advantage":None,
        "matched_outcome_finding":"no_common_supported_context",
    }
    if not common:
        return result
    weights={k:.5*(incumbent[k]["legal"]/na+challenger[k]["legal"]/nb) for k in common}
    rate_scale=100 if role=="batting" else 6
    for label,cells in (("incumbent",incumbent),("challenger",challenger)):
        result[label+"_standardized_rate"]=rate_scale*sum(weights[k]*cells[k]["runs"]/cells[k]["legal"] for k in common)
        result[label+"_standardized_wickets_per100"]=100*sum(weights[k]*cells[k]["wickets"]/cells[k]["legal"] for k in common)
        result[label+"_standardized_value_per100"]=100*sum(weights[k]*cells[k]["value"]/cells[k]["legal"] for k in common)
        result[label+"_illegal_events_per100_legal"]=100*sum(weights[k]*cells[k]["illegal"]/cells[k]["legal"] for k in common)
    run_diff=result["challenger_standardized_rate"]-result["incumbent_standardized_rate"]
    wicket_diff=result["challenger_standardized_wickets_per100"]-result["incumbent_standardized_wickets_per100"]
    if role=="bowling":
        run_diff=-run_diff
    else:
        wicket_diff=-wicket_diff
    result["challenger_run_rate_advantage"]=run_diff
    result["challenger_wicket_rate_advantage"]=wicket_diff
    if not result["sufficient_common_sample"]:
        result["matched_outcome_finding"]="insufficient_common_sample"
    elif run_diff>=-1e-10 and wicket_diff>=-1e-10 and (run_diff>1e-10 or wicket_diff>1e-10):
        result["matched_outcome_finding"]="challenger_stronger_observed_runs_and_wickets"
    elif run_diff<=1e-10 and wicket_diff<=1e-10 and (run_diff < -1e-10 or wicket_diff < -1e-10):
        result["matched_outcome_finding"]="incumbent_stronger_observed_runs_and_wickets"
    elif abs(run_diff)<=1e-10 and abs(wicket_diff)<=1e-10:
        result["matched_outcome_finding"]="same_observed_rates"
    else:
        result["matched_outcome_finding"]="observed_run_wicket_tradeoff"
    return result


def allocated_index_points(per_match,parameters):
    """One exact, proportional allocation of centered scores, not causal effects."""
    b,h,e=(parameters[name] for name in ("outcomes","high_match","execution"))
    arrays={name:np.asarray(per_match[name],dtype=float) for name in ("bw","ba","bl","bs","hw","hv","hg","ew","ev")}
    shape=arrays["bw"].shape
    if len(shape)!=1 or any(a.shape!=shape or not np.isfinite(a).all() for a in arrays.values()):
        raise InputError("Match allocation requires aligned finite vectors.")
    if any((arrays[name]<0).any() for name in ("bw","hw","ew")):
        raise InputError("Evidence weights must be nonnegative.")
    def value_part(weight,value,amplitude,scale,prior):
        total=float(weight.sum())
        if total==0:
            if np.any(value!=0):
                raise InputError("Weighted values exist without evidence.")
            return np.zeros(shape)
        mean=float(value.sum())/total
        response=math.tanh(mean/scale)/mean if abs(mean)>1e-12 else 1/scale
        return amplitude*response/(total+prior)*value
    bw=arrays["bw"].sum()
    result={
        "absolute_quality_points":value_part(arrays["bw"],arrays["ba"],e["outcome_share"]*b["absolute_weight"]*30,
                                             b["contribution_scale"],b["career_prior_matches"]),
        "usual_comparison_points":value_part(arrays["bw"],arrays["bl"],e["outcome_share"]*b["resilience_weight"]*30,
                                             b["contribution_scale"],b["career_prior_matches"]),
        "consistency_points":e["outcome_share"]*b["consistency_weight"]*100*bw/(bw+b["career_prior_matches"])/(bw+4)
                             *(arrays["bs"]-.5*arrays["bw"]),
        "execution_points":value_part(arrays["ew"],arrays["ev"],e["execution_share"]*25,
                                       e["signal_scale"],e["prior_matches"]),
        "high_match_impact_points":value_part(arrays["hw"],arrays["hv"],e["high_match_share"]*30,
                                               h["run_scale"],h["prior_matches"]),
        "gentle_marking_points":e["high_match_share"]*h["gentle_adjustment_cap"]*arrays["hg"]/
                                (arrays["hw"].sum()+h["prior_matches"]) if h["gentle_adjustment"] else np.zeros(shape),
    }
    result["allocated_index_points"]=sum(result.values())
    return result


def paired_score_evidence(incumbent_scores,challenger_scores):
    a=np.asarray(incumbent_scores,dtype=float)
    b=np.asarray(challenger_scores,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise InputError("Paired scores require aligned finite shared draws.")
    differences=a-b
    lo,hi=np.quantile(differences,[.05,.95])
    return {
        "incumbent_higher_score_resample_share":float(np.mean((differences>1e-10)+.5*(abs(differences)<=1e-10))),
        "paired_score_gap_p05":float(lo),"paired_score_gap_p95":float(hi),
        "resample_separation":"incumbent_ahead_through_middle_90pct" if lo>0 else (
            "challenger_ahead_through_middle_90pct" if hi<0 else "score_order_overlaps"),
    }
