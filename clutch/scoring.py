import math
from collections import defaultdict
import pyarrow.parquet as pq
from .execution import ExecutionCard, final_ratings
from .inputs import InputError, iter_csv, read_csv
from .live import CREDITED_WICKETS, pressure_weight
from .match_impact import HighMatchCard, delivery_signal, gentle_match_adjustment, high_match_rating
from .ratings import rating_components
from .competitive import (
    AssignmentAnchors, CompetitiveMatchCard, RescueEvidence, delivery_activity,
    effective_matches, rescue_adjustment, scoring_observation, sufficient_support,
)

def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()

def context_states(ball_base, raw_dir):
    penalty=set()
    no_batting=set()
    uncertain_dismissals=set()
    for row in iter_csv(raw_dir/"ball_by_ball.csv"):
        key=row["match_id"],int(row["innings"]),int(row["innings_ball_num"])
        if row["extras_type"]=="Penalty":
            penalty.add(key)
        if row["extras_type"]=="wide" and row["is_wicket"]!="true":
            no_batting.add(key)
        if (int(row["innings"])<=2 and row["is_wicket"]=="true" and row["dismissed_player_id"]
                and row["dismissal_type"] not in CREDITED_WICKETS):
            uncertain_dismissals.add((row["dismissed_player_id"],row["match_id"]))
    states={}
    anchors=AssignmentAnchors()
    endings={}
    clinches={}
    for original in parquet_rows(ball_base/"ball_pressure.parquet"):
        state=dict(original)
        key=state["match_id"],state["innings"],state["sequence"]
        activity=delivery_activity(state)
        state.update(activity)
        for role in ["batting","bowling"]:
            opportunity=key not in penalty and not (role=="batting" and key in no_batting)
            anchor=anchors.observe(state,role,opportunity)
            state[role+"_anchor_pressure"]=anchor[0] if anchor else None
            state[role+"_anchor_sequence"]=anchor[1] if anchor else None
        states[key]=state
        endings[key[:2]]=state
        if state["bowling_clinch"]:
            clinches[key[:2]]=state
    return states,endings,clinches,uncertain_dismissals

def high_match_rows(raw_cards, scored_cards, rescue, endings, clinches, raw_dir, config, refinement):
    matches={r["match_id"]:r for r in read_csv(raw_dir/"matches.csv")}
    competitions={r["comp_id"]:r["comp_name"] for r in read_csv(raw_dir/"competitions.csv")}
    output=[]
    for key,raw_card in raw_cards.items():
        ending=endings[raw_card.match_id,raw_card.first_state["innings"]]
        team_innings={"runs":ending["runs_before"]+ending["recorded_total_runs"],
                      "legal_balls":ending["legal_balls_before"]+int(ending["legal_delivery"])}
        official=raw_card.evidence(config,team_innings)
        scored=scored_cards.get(key)
        row=scored.evidence(config,team_innings) if scored else dict(official)
        if scored is None:
            row.update({"status":"no_competitive_opportunities","net_team_impact":None,
                        "high_match_weight":0.0,"attributable_events":0,"legal_balls":0,
                        "attributed_runs":0,"useful_attributed_runs":0,"credited_wickets":0,
                        "legal_ball_strike_rate":None,"bowling_economy":None,
                        "reference_scoring_strike_rate":None,"reference_bowling_economy":None,
                        "reference_run_budget":0.0,"context_run_budget":0.0,
                        "raw_pace_surplus":0.0,"bounded_pace_impact":0.0,
                        "actual_wicket_resource_value":0.0,"expected_wicket_resource_value":0.0,
                        "secondary_wicket_impact":0.0,"batter_share_of_team_runs":0.0,
                        "rrr_at_first_opportunity":None,"rrr_after_last_opportunity":None})
        original_net=row["net_team_impact"]
        adjustment,reason=rescue_adjustment(row,rescue.get(key),refinement) if scored else (0.0,"no_competitive_opportunities")
        if original_net is not None:
            row["net_team_impact"]+=adjustment
            net=row["net_team_impact"]
            row["status"]="positive_team_impact" if net>config.neutral_run_margin else (
                "negative_team_impact" if net < -config.neutral_run_margin else "near_neutral")
        match=matches[raw_card.match_id]
        first,second=endings.get((raw_card.match_id,1)),endings.get((raw_card.match_id,2))
        winner=None
        if first and second:
            first_runs=first["runs_before"]+first["recorded_total_runs"]
            second_runs=second["runs_before"]+second["recorded_total_runs"]
            if second_runs>first_runs:
                winner=second["batting_team"]
            elif second_runs<first_runs:
                winner=first["batting_team"]
            if winner!=match["win_team_id"]:
                winner=None
        won=winner is not None and raw_card.team_id==winner
        closing=False
        if won and scored is not None and row["innings"]==2:
            if row["role"]=="batting":
                closing=(scored.last_state["sequence"]==second["sequence"]
                         and second["balls_remaining"]<=12 and scored.last_useful_runs>0
                         and second["batting_team_pressure"]>=60)
            else:
                clinch=clinches.get((raw_card.match_id,2))
                closing=bool(clinch and clinch["bowler"]==row["player_id"]
                             and clinch["bowling_team_pressure"]>=60
                             and (clinch["contest_after"]=="extras_dependent"
                                  or (clinch["balls_remaining"]==1 and clinch["legal_delivery"])
                                  or scored.last_credited_wicket)
                             and (clinch["balls_remaining"]<=12
                                  or clinch["wickets_before"]+int(clinch["recorded_wicket"])==10))
        row.update({
            "competition":competitions[match["comp_id"]],
            "team":match["home_team"] if row["team_id"]==match["home_team_id"] else match["away_team"],
            "opponent":match["away_team"] if row["team_id"]==match["home_team_id"] else match["home_team"],
            "match_result":match["result"],"verified_regulation_win":won,
            "closing_action_observed":closing,
            "gentle_delivery_signal":delivery_signal(row["net_team_impact"],won,closing,config),
            "pre_rescue_net_team_impact":original_net,"rescue_adjustment":adjustment,"rescue_reason":reason,
            "raw_attributed_runs":official["attributed_runs"],"raw_legal_balls":official["legal_balls"],
            "raw_credited_wickets":official["credited_wickets"],
            "raw_legal_ball_strike_rate":official["legal_ball_strike_rate"],
            "raw_bowling_economy":official["bowling_economy"],
            "raw_net_team_impact":official["net_team_impact"],
            "post_clinch_runs_not_scored":official["attributed_runs"]-row["attributed_runs"],
            "post_clinch_legal_balls_not_scored":official["legal_balls"]-row["legal_balls"],
        })
        evidence=rescue[key]
        collapse=evidence.collapse
        row.update({
            "entry_runs_before":evidence.entry["runs_before"],
            "entry_balls_remaining":evidence.entry["balls_remaining"],
            "entry_wickets_before":evidence.entry["wickets_before"],
            "entry_four_plus_last_six":evidence.entry["four_plus_last_six"],
            "entry_target":evidence.entry["target"],"entry_par":evidence.entry["par_before"],
            "collapse_sequence":collapse["sequence"] if collapse else None,
            "collapse_runs_before":collapse["runs_before"] if collapse else None,
            "collapse_wickets_before":collapse["wickets_before"] if collapse else None,
            "collapse_balls_remaining":collapse["balls_remaining"] if collapse else None,
            "post_collapse_runs":evidence.runs_after,"post_collapse_legal_balls":evidence.balls_after,
            "post_collapse_contextual_contribution":evidence.contextual_value_after,
            "team_runs_after_last_rescue_opportunity":evidence.team_runs_after_last_opportunity,
            "partnership_growth_after_collapse":(
                evidence.maximum_partnership-collapse["partnership_runs_before"] if collapse else None),
        })
        output.append(row)
    return sorted(output,key=lambda r:(r["match_date"],r["match_id"],r["role"],r["player_id"]))

def refine_deliveries(ball_base, states, uncertain_dismissals, high_config, config, refinement, emit):
    cards={}
    raw_high={}
    scored_high={}
    rescue={}
    totals=defaultdict(lambda:{"runs":0,"balls":0,"wickets":0,"raw_quality":0.0,"pressure_runs":0,"pressure_balls":0})
    incomplete={(r["player_id"],r["role"],r["match_id"]) for r in read_csv(ball_base/"attribution_exceptions.csv")}
    for raw in parquet_rows(ball_base/"player_delivery_contributions.parquet"):
        state=states[raw["match_id"],raw["innings"],raw["sequence"]]
        role=raw["role"]
        row,active=scoring_observation(raw,state)
        anchor=state[role+"_anchor_pressure"]
        if anchor is None:
            raise InputError("An attributed delivery has no pre-action assignment anchor.")
        row["source_pressure_weight"]=raw["pressure_weight"]
        row["source_absolute_contribution"]=raw["absolute_contribution"]
        row["source_credited_wicket"]=raw["credited_wicket"]
        row["source_high_pressure"]=raw["high_pressure"]
        row["scoring_pressure_anchor"]=anchor
        row["pressure_anchor_sequence"]=state[role+"_anchor_sequence"]
        row["contest_before"]=state["contest_before"]
        row["competitive_scoring_active"]=bool(active)
        row["pressure_weight"]=pressure_weight(anchor,config) if active else 0.0
        row["high_pressure"]=bool(active and anchor>=config.high_pressure)
        emit(row)
        key=raw["player_id"],role,raw["match_id"]
        if key not in cards:
            cards[key]=CompetitiveMatchCard(raw["player_id"],role,raw["match_id"],
                                            raw["match_date"],raw["family"],raw["team_id"])
        cards[key].add_competitive(row,active)
        aggregate=totals[raw["player_id"],role]
        aggregate["runs"]+=raw["attributed_runs"]
        aggregate["balls"]+=raw["legal_opportunity"]
        aggregate["wickets"]+=raw["credited_wicket"]
        aggregate["raw_quality"]+=raw["absolute_contribution"]
        if row["high_pressure"]:
            aggregate["pressure_runs"]+=raw["attributed_runs"]
            aggregate["pressure_balls"]+=raw["legal_opportunity"]
        pre=state[role+"_prematch_pressure"]
        if pre<=high_config.stakes_floor:
            continue
        def new_card():
            return HighMatchCard(raw["player_id"],raw["player"],role,raw["match_id"],raw["match_date"],
                                 raw["family"],raw["team_id"],pre,key in incomplete,
                                 role=="batting" and (raw["player_id"],raw["match_id"]) in uncertain_dismissals)
        if key not in raw_high:
            raw_high[key]=new_card()
            rescue[key]=RescueEvidence()
        raw_high[key].add(raw,state)
        rescue[key].add(row,state,active,refinement)
        if active:
            if key not in scored_high:
                scored_high[key]=new_card()
            scored_high[key].add(row,state)
    return cards,raw_high,scored_high,rescue,totals

def refine_execution(base, states, config, emit):
    anchors={}
    cards={}
    for original in parquet_rows(base/"execution_ledger.parquet"):
        row=dict(original)
        state=states[row["match_id"],row["innings"],row["sequence"]]
        role=row["role"]
        anchor_sequence=state[role+"_anchor_sequence"]
        anchor_pressure=state[role+"_anchor_pressure"]
        if anchor_sequence is None:
            # An annotated winning-wide dismissal can remain with no attributable batting action.
            anchor_sequence=row["sequence"]
            anchor_pressure=row["base_personal_pressure"]
        group=row["match_id"],row["innings"],role,anchor_sequence
        if group not in anchors:
            anchors[group]=row["crunch_context"]
        context=anchors[group]
        active=state[role+"_active"]
        weight=((max(0,anchor_pressure-config.crunch_floor))/(100-config.crunch_floor))**2 if context and active else 0.0
        risk=role=="bowling" and state["bowling_extra_risk_only"]
        if risk:
            if row["raw_signal"] is not None and row["raw_signal"]<0:
                pre=state["bowling_prematch_pressure"]
                weight=max(weight,(max(0,pre-config.crunch_floor)/(100-config.crunch_floor))**2)
                context="post_clinch_extra_risk"
            else:
                weight=0.0
        row.update({"source_execution_weight":original["execution_weight"],
                    "execution_weight":weight,"scoring_pressure_anchor":anchor_pressure,
                    "pressure_anchor_sequence":anchor_sequence,"scoring_crunch_context":context,
                    "contest_before":state["contest_before"],"competitive_scoring_active":bool(active)})
        emit(row)
        if weight>0 and row["eligible_execution_opportunity"]:
            key=row["player_id"],role,row["match_id"]
            if key not in cards:
                cards[key]=ExecutionCard(row["player_id"],row["player"],role,row["match_id"],row["match_date"],row["family"])
            cards[key].add(row)
    return sorted([card.evidence(config) for card in cards.values()],
                  key=lambda row:(row["match_date"],row["match_id"],row["role"],row["player_id"]))

def group_cards(rows):
    result=defaultdict(list)
    for row in rows:
        result[row["player_id"],row["role"]].append(row)
    return result

def rate_players(base_profiles, ball_rows, high_rows, execution_rows, totals,
                 config, ball_config, high_config, refinement):
    balls=group_cards(ball_rows)
    highs=group_cards(high_rows)
    synthetic_parents=[]
    details={}
    for source in base_profiles:
        key=source["player_id"],source["role"]
        history=balls[key]
        high=highs[key]
        outcome,absolute,lift,consistency,support=rating_components(history,ball_config)
        macro,mean,macro_support=high_match_rating(high,high_config)
        normal=int(source["ordinary_reference_events"])
        whole=sum(r["whole_opportunity_contribution"] for r in history)
        pressure_exposures=[r["high_pressure_legal_balls"] for r in history]
        high_exposures=[r["legal_balls"] for r in high if r["net_team_impact"] is not None and r["high_match_weight"]>0]
        enough_ball=sufficient_support(pressure_exposures,ball_config.minimum_pressure_matches,
                                      ball_config.minimum_pressure_legal_balls,refinement.minimum_effective_matches)
        enough_high=sufficient_support(high_exposures,high_config.minimum_matches,
                                      high_config.minimum_legal_balls,refinement.minimum_effective_matches)
        usual_ok=normal>=ball_config.minimum_normal_events
        ball_eligible=enough_ball and usual_ok and whole>0 and absolute>0
        high_eligible=enough_high and usual_ok and whole>0 and mean is not None and mean>0
        gentle=gentle_match_adjustment(high,high_config)
        reasons=[]
        if not usual_ok:
            reasons.append("insufficient_usual_baseline_evidence")
        if whole<=0:
            reasons.append("below_competitive_whole_contextual_quality_floor")
        if not enough_ball and not enough_high:
            reasons.append("insufficient_cumulative_or_effective_pressure_exposure")
        if not ball_eligible and not high_eligible and not reasons:
            reasons.append("below_route_contextual_quality_floor")
        synthetic_parents.append({
            "player_id":key[0],"player":source["player"],"role":key[1],
            "ball_channel_score":outcome,"high_match_channel_score":macro,
            "performance_score":.7*outcome+.3*macro,"rank":"",
            "eligible":str(ball_eligible or high_eligible),
            "eligibility_route":"both_channels" if ball_eligible and high_eligible else (
                "ball_channel" if ball_eligible else "high_match_channel" if high_eligible
                else "insufficient_evidence_or_quality"),
            "gentle_high_match_adjustment":gentle,
            "helped_win_mark_matches":sum(r["gentle_delivery_signal"]==1 for r in high),
            "negative_delivery_mark_matches":sum(r["gentle_delivery_signal"]==-1 for r in high),
        })
        details[key]={
            "eligibility_reasons":";".join(reasons),
            "ordinary_reference_events":normal,
            "absolute_pressure_contribution":absolute,"reliability_adjusted_pressure_lift":lift,
            "smoothed_match_consistency":consistency,"outcome_evidence_support":support,
            "original_whole_contextual_quality":totals[key]["raw_quality"],
            "competitive_whole_contextual_quality":whole,
            "matches_with_role_opportunities":len(history),
            "distinct_pressure_matches":sum(x>0 for x in pressure_exposures),
            "effective_pressure_matches":effective_matches(pressure_exposures),
            "high_pressure_legal_balls":sum(pressure_exposures),
            "high_stakes_role_matches":len(high_exposures),
            "effective_high_stakes_matches":effective_matches(high_exposures),
            "high_stakes_legal_balls":sum(high_exposures),
            "mean_high_match_team_impact":mean,"high_match_evidence_support":macro_support,
            "rescue_floor_matches":sum(r["rescue_adjustment"]>0 for r in high),
            "post_clinch_high_match_runs_not_scored":sum(r["post_clinch_runs_not_scored"] for r in high),
            "raw_runs":totals[key]["runs"],"raw_legal_balls":totals[key]["balls"],
            "raw_credited_wickets":totals[key]["wickets"],
            "observed_families":";".join(sorted({r["family"] for r in history})),
        }
    ratings=final_ratings(synthetic_parents,execution_rows,balls,high_rows,config,ball_config,high_config)
    for row in ratings:
        key=row["player_id"],row["role"]
        row.update(details[key])
    return ratings
