"""Bounded V3 refinements; no state-value replacement or named-player rules."""

import math
from dataclasses import dataclass

from .inputs import InputError
from .ratings import MatchCard


@dataclass(frozen=True)
class RefinementConfig:
    absolute_weight: float = .50
    resilience_weight: float = .30
    consistency_weight: float = .20
    minimum_effective_matches: float = 4.0
    rescue_min_runs: int = 20
    rescue_min_balls: int = 12
    rescue_min_balls_remaining: int = 24
    rescue_min_partnership: int = 25
    rescue_min_run_share: float = .35
    rescue_credit_fraction: float = .5
    rescue_credit_cap: float = 10.0

    def __post_init__(self):
        if not math.isclose(self.absolute_weight+self.resilience_weight+self.consistency_weight,1):
            raise InputError("Outcome shares must sum to one.")
        if any(not math.isfinite(x) or x<=0 for x in self.__dict__.values()):
            raise InputError("Refinement parameters must be finite and positive.")


def contest_status(target, runs, balls, wickets):
    if target is None:
        return "first_innings"
    if runs>=target:
        return "won"
    if balls<=0 or wickets>=10:
        return "tied" if runs==target-1 else "lost"
    # A six can still force a regulation tie when seven are needed from one.
    return "extras_dependent" if target-runs>6*balls+1 else "competitive"


def delivery_activity(state):
    before=contest_status(state["target"],state["runs_before"],state["balls_remaining"],state["wickets_before"])
    after=contest_status(state["target"],state["runs_before"]+state["recorded_total_runs"],
                         state["balls_remaining"]-int(state["legal_delivery"]),
                         state["wickets_before"]+int(state["recorded_wicket"]))
    exceptional=not state["legal_delivery"] or state["recorded_total_runs"]>6
    reopens=before=="extras_dependent" and after in {"competitive","won","tied"}
    ongoing=before in {"first_innings","competitive"}
    return {
        "contest_before":before,"contest_after":after,
        "batting_active":ongoing or (exceptional and reopens),
        "bowling_active":ongoing or (before=="extras_dependent" and exceptional),
        "reopens_contest":reopens,
        "bowling_clinch":before=="competitive" and after in {"extras_dependent","lost"},
        "bowling_extra_risk_only":before=="extras_dependent" and exceptional,
    }


def scoring_observation(row, activity):
    result=dict(row)
    active=activity[row["role"]+"_active"]
    if row["role"]=="bowling" and activity["bowling_extra_risk_only"]:
        # Keep the cost of extras, but do not award a cheap wicket after a boundary-only clinch.
        removed=row["credited_wicket"]*row["wicket_value_before"]
        result["absolute_contribution"]-=removed
        result["contribution_vs_usual"]-=removed
        result["credited_wicket"]=0
    result["high_pressure"]=bool(row["high_pressure"] and active)
    return result,active


class CompetitiveMatchCard(MatchCard):
    def add_competitive(self, row, active):
        previous=self.whole_value
        super().add(row)
        if not active:
            self.whole_value=previous


class AssignmentAnchors:
    def __init__(self):
        self.previous={}
        self.current={}

    def observe(self, state, role, opportunity=True):
        if not opportunity:
            return None
        mid,innings=state["match_id"],state["innings"]
        over=state["legal_balls_before"]//6
        person=state["striker"] if role=="batting" else state["bowler"]
        status=delivery_activity(state)["contest_before"]
        signature=(over,person,status)
        if role=="batting":
            signature+=(state["wickets_before"],state["free_hit_before"])
        key=mid,innings,role
        if self.previous.get(key)!=signature:
            self.previous[key]=signature
            self.current[key]=(float(state[role+"_personal_pressure"]),state["sequence"])
        return self.current[key]


def effective_matches(exposures, scale=8.0):
    weights=[min(1.0,max(0,x)/scale) for x in exposures]
    return sum(weights)**2/sum(x*x for x in weights) if any(weights) else 0.0


def sufficient_support(exposures, minimum_matches, minimum_balls, minimum_effective):
    observed=[x for x in exposures if x>0]
    return (len(observed)>=minimum_matches and sum(observed)>=minimum_balls
            and effective_matches(observed)>=minimum_effective)


class RescueEvidence:
    def __init__(self):
        self.entry=None
        self.collapse=None
        self.runs_after=0
        self.balls_after=0
        self.maximum_partnership=0
        self.contextual_value_after=0.0
        self.team_runs_after_last_opportunity=0

    def add(self, row, state, active, config):
        if self.entry is None:
            self.entry=state
        if not active:
            return
        collapse=(state["wickets_before"]>=4 and (
            state["recent_wickets_before"]>=1 or state["partnership_runs_before"]<config.rescue_min_partnership) or (
            state["wickets_before"]>=3 and state["recent_wickets_before"]>=2))
        liveness=state.get("practical_liveness")
        still_live=liveness is None or liveness>=.2
        if self.collapse is None and collapse and still_live and state["balls_remaining"]>=config.rescue_min_balls_remaining:
            self.collapse=state
        if self.collapse is not None:
            self.runs_after+=row["useful_attributed_runs"]
            self.balls_after+=row["legal_opportunity"]
            self.contextual_value_after+=row["absolute_contribution"]
            self.team_runs_after_last_opportunity=state["runs_before"]+state["recorded_total_runs"]
            self.maximum_partnership=max(
                self.maximum_partnership,
                state["partnership_runs_before"]+state["recorded_total_runs"])


def rescue_adjustment(card, evidence, config):
    if card["net_team_impact"] is None or evidence is None or evidence.entry is None:
        return 0.0,"unobserved"
    if card["net_team_impact"]>=0:
        return 0.0,"already_nonnegative"
    if evidence.entry["target"] is None:
        return 0.0,"first_innings_already_context_priced"
    runs=card["useful_attributed_runs"]
    context=card["context_run_budget"]
    if card["role"]=="batting":
        contextual=runs-context+card["secondary_wicket_impact"]
        collapse=evidence.collapse
        if collapse is None or evidence.runs_after<config.rescue_min_runs or evidence.balls_after<config.rescue_min_balls:
            return 0.0,"no_substantial_live_collapse_rescue"
        added_partnership=evidence.maximum_partnership-collapse["partnership_runs_before"]
        # A later partner's innings must not rewrite an already-completed rescue.
        remaining_team_runs=max(1,evidence.team_runs_after_last_opportunity-collapse["runs_before"])
        share=evidence.runs_after/remaining_team_runs
        if added_partnership<config.rescue_min_partnership and share<config.rescue_min_run_share:
            return 0.0,"insufficient_stabilization_evidence"
        contextual=min(contextual,evidence.contextual_value_after)
    else:
        contextual=context-runs+card["secondary_wicket_impact"]
        inherited=(evidence.entry["four_plus_last_six"]>=2
                   or evidence.entry["target"]<.85*evidence.entry["par_before"])
        if (card["legal_balls"]<config.rescue_min_balls or not inherited or runs>context
                or card["reference_run_budget"]>=.8*context):
            return 0.0,"no_contextually_good_inherited_defence"
    if contextual<=0:
        return 0.0,"contextual_performance_not_positive"
    floor=min(config.rescue_credit_cap,config.rescue_credit_fraction*contextual)
    return max(0,floor-card["net_team_impact"]),"contextual_rescue_floor"
