import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import groupby
from .execution import (
    ExecutionCard, ExecutionHistory, annotation_status, delivery_group,
    grade_execution, task_group,
)
from .inputs import InputError
from .live import NO_BALL
from .statistics import clip

@dataclass(frozen=True)
class FullTossPolicy:
    effective_credit: float = .10
    boundary_penalty: float = .15

    def __post_init__(self):
        if (not math.isfinite(self.effective_credit) or not math.isfinite(self.boundary_penalty)
                or not 0<=self.effective_credit<=.10 or not 0<self.boundary_penalty<=.35):
            raise InputError("Full-toss grace is capped at ordinary effectiveness; boundary penalties must remain negative.")

def grade_with_grace(role,annotation,state,sequence,policy=FullTossPolicy()):
    if (annotation.style!="Full Toss" or annotation.extra_type in NO_BALL | {"wide"}
            or not state["legal_delivery"] or annotation_status(annotation)!="usable_label"):
        return grade_execution(role,annotation,state,sequence)
    # The tag identifies neither height nor intent. Use ordinary response/effectiveness,
    # never inferred yorker credit or an assumed easy batting opportunity.
    raw,features,reason=grade_execution(role,replace(annotation,style="Straight"),state,sequence)
    features.add("full_toss")
    if role=="batting":
        return raw,features,"full_toss_height_unknown:"+reason
    if (annotation.secondary or (annotation.extra_type=="None" and annotation.primary)
            or (annotation.extra_type in {"byes","legbyes"} and annotation.runs)):
        return None,{"full_toss"},"full_toss_charge_attribution_unverified"
    features.discard("effective_stock_delivery")
    if raw is not None and raw>0 and state["recorded_total_runs"]<=2:
        return policy.effective_credit,features,"effective_legal_full_toss_height_unknown"
    if annotation.runs>=4:
        return -policy.boundary_penalty,{"full_toss"},"legal_full_toss_boundary_cost_height_unknown"
    return 0.0,{"full_toss"},"legal_full_toss_no_confirmed_control_failure"

def regrade_execution(source_rows,states,annotations,config,policy,emit):
    history=ExecutionHistory(config)
    cards={}
    diagnostics=defaultdict(Counter)
    previous_day=""
    for day,rows in groupby(source_rows,key=lambda r:r["match_date"]):
        if day<previous_day:
            raise InputError("The parent execution ledger is not chronologically ordered.")
        previous_day=day
        pending=[]
        for source in rows:
            row=dict(source)
            key=row["match_id"],row["innings"],row["sequence"]
            state=states[key]
            annotation=annotations[key]
            role,player=row["role"],row["player_id"]
            if player!=(annotation.striker if role=="batting" else annotation.bowler):
                raise InputError(f"Execution actor mismatch: {key}/{role}")
            before={name:row[name] for name in ("prior_dot_streak","prior_boundary_streak",
                                               "four_plus_last_six","dots_last_six","previous_bowler_style")}
            raw,features,reason=grade_with_grace(role,annotation,state,before,policy)
            group=(delivery_group(annotation) if role=="batting" else annotation_status(annotation))+":"+task_group(role,state)
            peer,offset,count=history.references(player,role,row["family"],row["phase"],annotation.pace,group)
            scored=None
            if raw is not None and row["eligible_execution_opportunity"]:
                scored=raw-config.usual_reference_share*clip(peer+offset,-1,.4)
                if raw<0:
                    scored=min(.5*raw,scored)
            ordinary=(scored is not None and row["team_pressure"]<=config.ordinary_ceiling
                      and source["source_execution_weight"]==0)
            weight=source["execution_weight"]
            risk=(role=="bowling" and row["contest_before"]=="extras_dependent"
                  and (not state["legal_delivery"] or state["recorded_total_runs"]>6))
            if risk:
                if raw is None or raw>=0:
                    weight=0.0
                else:
                    pre=state["bowling_prematch_pressure"]
                    weight=max(weight,(max(0,pre-config.crunch_floor)/(100-config.crunch_floor))**2)
            affected=(annotation.style=="Full Toss" and annotation.extra_type not in NO_BALL | {"wide"}
                      and state["legal_delivery"] and annotation_status(annotation)=="usable_label")
            row.update({
                "raw_signal":raw,"scored_signal":scored,"features":";".join(sorted(features)),
                "scoring_reason":reason,"peer_reference_before":peer,"personal_reference_offset_before":offset,
                "ordinary_reference_count_before":count,"training_eligible":ordinary,"execution_weight":weight,
                "before_grace_raw_signal":source["raw_signal"],"before_grace_scored_signal":source["scored_signal"],
                "before_grace_execution_weight":source["execution_weight"],"before_grace_training_eligible":source["training_eligible"],
                "full_toss_policy_applied":affected,
            })
            emit(row)
            if weight>0 and row["eligible_execution_opportunity"]:
                card_key=player,role,row["match_id"]
                if card_key not in cards:
                    cards[card_key]=ExecutionCard(player,row["player"],role,row["match_id"],day,row["family"])
                cards[card_key].add(row)
                if affected:
                    counter=diagnostics[player,role]
                    counter["policy_affected_crunch_events"]+=1
                    counter["controlled_legal_full_tosses"]+=int(reason=="effective_legal_full_toss_height_unknown")
                    counter["boundary_debits"]+=int(reason=="legal_full_toss_boundary_cost_height_unknown")
                    counter["neutral_events"]+=int(raw==0)
                    counter["ungraded_events"]+=int(raw is None)
            if ordinary:
                pending.append((player,role,row["family"],row["phase"],annotation.pace,group,raw,peer))
        for record in pending:
            history.update(record)
    result=sorted([c.evidence(config) for c in cards.values()],
                  key=lambda r:(r["match_date"],r["match_id"],r["role"],r["player_id"]))
    return result,diagnostics
