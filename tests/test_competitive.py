import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from clutch.execution import ExecutionConfig
from clutch.live import PerformanceConfig
from clutch.match_impact import MatchImpactConfig
from clutch.scoring import rate_players, refine_execution
from clutch.competitive import (
    AssignmentAnchors, CompetitiveMatchCard, RefinementConfig, RescueEvidence,
    contest_status, delivery_activity, rescue_adjustment, scoring_observation, sufficient_support,
)


def state(needed=29, balls=6, runs=0, legal=True, **changes):
    result={
        "match_id":"m","innings":2,"sequence":1,"target":191,"runs_before":191-needed,
        "balls_remaining":balls,"legal_balls_before":120-balls,"wickets_before":5,
        "recorded_total_runs":runs,"legal_delivery":legal,"recorded_wicket":False,
        "striker":"batter","bowler":"bowler","free_hit_before":False,
        "batting_personal_pressure":85.0,"bowling_personal_pressure":85.0,
        "batting_prematch_pressure":90.0,"bowling_prematch_pressure":90.0,
        "recent_wickets_before":0,"partnership_runs_before":0,
        "four_plus_last_six":0,"par_before":170.0,"practical_liveness":1.0,
    }
    result.update(changes)
    return result


def observation(role="bowling", value=2.0, **changes):
    result={
        "role":role,"absolute_contribution":value,"contribution_vs_usual":value,
        "credited_wicket":0,"wicket_value_before":20.0,"high_pressure":True,
        "legal_opportunity":1,"attributed_runs":0,"useful_attributed_runs":0,
        "pressure_weight":.5,"normal_reference_support":.5,"personal_pressure":85.0,"flags":"",
    }
    result.update(changes)
    return result


class ClinchTests(unittest.TestCase):
    def test_two_clinching_dots_remain_but_later_twenty_two_do_not(self):
        needed=29
        activities=[]
        competitive_runs=0
        for index,runs in enumerate([0,0,6,4,6,6]):
            current=state(needed,6-index,runs,sequence=index+1)
            activity=delivery_activity(current)
            activities.append(activity)
            competitive_runs+=runs*activity["bowling_active"]
            needed-=runs
        self.assertEqual([a["batting_active"] for a in activities],[True,True,False,False,False,False])
        self.assertEqual([a["bowling_active"] for a in activities],[True,True,False,False,False,False])
        self.assertTrue(activities[1]["bowling_clinch"])
        self.assertEqual(competitive_runs,0)

    def test_seven_from_one_still_has_a_tie_at_stake(self):
        current=state(7,1,6)
        activity=delivery_activity(current)
        self.assertEqual(activity["contest_before"],"competitive")
        self.assertEqual(activity["contest_after"],"tied")
        self.assertTrue(activity["batting_active"])
        self.assertFalse(activity["bowling_clinch"])
        self.assertEqual(contest_status(191,183,1,5),"extras_dependent")

    def test_a_no_ball_reopens_the_tie_then_normal_scoring(self):
        first=delivery_activity(state(8,1,1,False))
        self.assertTrue(first["reopens_contest"])
        self.assertTrue(first["bowling_active"])
        second=delivery_activity(state(7,1,6))
        self.assertTrue(second["batting_active"])
        self.assertTrue(second["bowling_active"])

    def test_non_reopening_wide_is_not_forgiven_or_batting_merit(self):
        activity=delivery_activity(state(29,4,1,False))
        self.assertFalse(activity["reopens_contest"])
        self.assertFalse(activity["batting_active"])
        row,active=scoring_observation(observation(value=-1,legal_opportunity=0,attributed_runs=1),activity)
        self.assertTrue(active)
        self.assertEqual(row["absolute_contribution"],-1)

    def test_rare_seven_run_legal_event_can_force_a_tie(self):
        activity=delivery_activity(state(8,1,7))
        self.assertTrue(activity["reopens_contest"])
        self.assertEqual(activity["contest_after"],"tied")
        self.assertTrue(activity["batting_active"])

    def test_no_cheap_late_wicket_credit_even_on_a_wide(self):
        original=observation(value=19,credited_wicket=1,legal_opportunity=0,attributed_runs=1)
        updated,active=scoring_observation(original,delivery_activity(state(14,2,1,False,recorded_wicket=True)))
        self.assertTrue(active)
        self.assertEqual(updated["absolute_contribution"],-1)
        self.assertEqual(updated["credited_wicket"],0)
        self.assertEqual(original["credited_wicket"],1)
        _,active=scoring_observation(original,delivery_activity(state(14,2,0,recorded_wicket=True)))
        self.assertFalse(active)

    def test_first_innings_expensive_spells_cannot_be_excused_by_the_result(self):
        activity=delivery_activity(state(100,12,6,target=None))
        self.assertTrue(activity["batting_active"])
        self.assertTrue(activity["bowling_active"])
        self.assertFalse(activity["bowling_clinch"])

    def test_post_clinch_runs_do_not_rewrite_whole_match_consistency(self):
        card=CompetitiveMatchCard("p","bowling","m","2025-01-01","test","t")
        card.add_competitive(observation(value=4),True)
        card.add_competitive(observation(value=-50,attributed_runs=50,pressure_weight=0,high_pressure=False),False)
        self.assertEqual(card.whole_value,4)
        self.assertEqual(card.attributed_runs,50)
        self.assertEqual(card.evidence(PerformanceConfig())["status"],"delivered")

    def test_execution_retains_extra_risk_but_not_late_yorker_bonus(self):
        row={
            "match_id":"m","innings":2,"sequence":1,"role":"bowling","player_id":"p","player":"P",
            "match_date":"2025-01-01","family":"test","crunch_context":"high_stakes_match",
            "execution_weight":0.0,"base_personal_pressure":40.0,
            "raw_signal":-1.0,"scored_signal":-1.0,"eligible_execution_opportunity":True,"features":"no_ball",
        }
        s=state(29,4,1,False)
        s.update(delivery_activity(s))
        s.update({"bowling_anchor_pressure":40.0,"bowling_anchor_sequence":1})
        next_state=state(28,4,0,sequence=2)
        next_state.update(delivery_activity(next_state))
        next_state.update({"bowling_anchor_pressure":40.0,"bowling_anchor_sequence":1})
        emitted=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            pq.write_table(pa.Table.from_pylist([row,{**row,"sequence":2,"raw_signal":.35,"scored_signal":.35,"features":"effective_yorker"}]),
                           path/"execution_ledger.parquet")
            cards=refine_execution(path,{("m",2,1):s,("m",2,2):next_state},ExecutionConfig(),emitted.append)
        self.assertLess(cards[0]["mean_execution_signal"],0)
        self.assertGreater(emitted[0]["execution_weight"],0)
        self.assertEqual(emitted[1]["execution_weight"],0)


class AnchorAndRescueTests(unittest.TestCase):
    def test_assignment_cannot_gain_weight_from_its_own_dots(self):
        anchors=AssignmentAnchors()
        first=anchors.observe(state(12,6,0,batting_personal_pressure=65),"batting")
        after=anchors.observe(state(12,5,0,sequence=2,batting_personal_pressure=95),"batting")
        self.assertEqual(first,after)
        partner=anchors.observe(state(12,4,0,sequence=3,striker="partner",batting_personal_pressure=95),"batting")
        self.assertEqual(partner,(95.0,3))

    def test_clinch_and_reopening_override_the_assignment(self):
        anchors=AssignmentAnchors()
        anchors.observe(state(29,5,0),"bowling")
        dead=state(29,4,1,False,sequence=2,bowling_personal_pressure=40)
        self.assertEqual(anchors.observe(dead,"bowling"),(40.0,2))
        reopened=state(25,4,0,sequence=3,bowling_personal_pressure=80)
        self.assertEqual(anchors.observe(reopened,"bowling"),(80.0,3))

    def test_cumulative_short_finishes_and_one_flash(self):
        self.assertTrue(sufficient_support([2]*30,6,60,4))
        self.assertFalse(sufficient_support([2],6,60,4))
        self.assertFalse(sufficient_support([80,1,1,1,1,1],6,60,4))

    def test_positive_collapse_rescue_has_a_capped_floor(self):
        evidence=RescueEvidence()
        evidence.entry=evidence.collapse=state(177,107,wickets_before=3,recent_wickets_before=3,runs_before=12)
        evidence.runs_after=64
        evidence.balls_after=44
        evidence.maximum_partnership=68
        evidence.contextual_value_after=41.0
        evidence.team_runs_after_last_opportunity=154
        card={"net_team_impact":-18.0,"role":"batting","useful_attributed_runs":64,
              "context_run_budget":50.0,"secondary_wicket_impact":2.0,"innings_team_runs":154}
        gain,reason=rescue_adjustment(card,evidence,RefinementConfig())
        self.assertEqual(reason,"contextual_rescue_floor")
        self.assertEqual(card["net_team_impact"]+gain,8.0)
        evidence.contextual_value_after=-1
        self.assertEqual(rescue_adjustment(card,evidence,RefinementConfig())[0],0)

    def test_slow_padding_and_late_hopeless_batting_do_not_qualify_as_rescue(self):
        evidence=RescueEvidence()
        evidence.add(observation("batting",value=1,useful_attributed_runs=30),
                     state(100,18,wickets_before=6,recent_wickets_before=3),True,RefinementConfig())
        self.assertIsNone(evidence.collapse)
        evidence=RescueEvidence()
        evidence.entry=evidence.collapse=state(120,60,wickets_before=4,recent_wickets_before=2)
        evidence.runs_after=20
        evidence.balls_after=30
        evidence.maximum_partnership=30
        evidence.contextual_value_after=-10
        evidence.team_runs_after_last_opportunity=100
        card={"net_team_impact":-20,"role":"batting","useful_attributed_runs":20,
              "context_run_budget":45,"secondary_wicket_impact":3,"innings_team_runs":100}
        self.assertEqual(rescue_adjustment(card,evidence,RefinementConfig())[0],0)

    def test_good_inherited_containment_not_expensive_bowling_gets_protection(self):
        evidence=RescueEvidence()
        evidence.entry=state(30,30,target=120,runs_before=90,par_before=180)
        card={"net_team_impact":-4,"role":"bowling","useful_attributed_runs":12,
              "context_run_budget":20,"reference_run_budget":10,"secondary_wicket_impact":-2,"legal_balls":12}
        self.assertEqual(rescue_adjustment(card,evidence,RefinementConfig())[0],7)
        card["useful_attributed_runs"]=64
        self.assertEqual(rescue_adjustment(card,evidence,RefinementConfig())[0],0)

    def test_rescue_credit_ignores_partner_runs_after_the_batters_last_opportunity(self):
        evidence=RescueEvidence()
        evidence.entry=evidence.collapse=state(179,90,wickets_before=4,recent_wickets_before=2,runs_before=12)
        evidence.runs_after=25
        evidence.balls_after=18
        evidence.maximum_partnership=20
        evidence.contextual_value_after=5
        evidence.team_runs_after_last_opportunity=45
        card={"net_team_impact":-10,"role":"batting","useful_attributed_runs":25,
              "context_run_budget":20,"secondary_wicket_impact":0,"innings_team_runs":80}
        before=rescue_adjustment(card,evidence,RefinementConfig())[0]
        card["innings_team_runs"]=172
        self.assertGreater(before,0)
        self.assertEqual(rescue_adjustment(card,evidence,RefinementConfig())[0],before)


class EligibilityTests(unittest.TestCase):
    def rating(self, ordinary=30, whole=2.0, count=30, legal=2):
        history=[{
            "player_id":"p","role":"batting","match_id":str(i),"family":"test",
            "match_evidence_weight":.2,"absolute_pressure_contribution":1.0,
            "pressure_contribution_vs_usual":.5,"usual_baseline_support":.5,
            "status":"delivered","whole_opportunity_contribution":whole,"high_pressure_legal_balls":legal,
        } for i in range(count)]
        return rate_players(
            [{"player_id":"p","player":"P","role":"batting","ordinary_reference_events":str(ordinary)}],
            history,[],[],{("p","batting"):{"raw_quality":-100,"runs":60,"balls":60,"wickets":0}},
            ExecutionConfig(bootstrap_samples=5),PerformanceConfig(),MatchImpactConfig(),RefinementConfig())[0]

    def test_short_finisher_qualifies_but_consolation_cannot_trigger_quality_failure(self):
        self.assertTrue(self.rating()["eligible"])

    def test_weak_contextual_batting_stays_ineligible(self):
        self.assertFalse(self.rating(whole=-2)["eligible"])

    def test_ordinary_history_and_distinct_evidence_guards_remain(self):
        self.assertFalse(self.rating(ordinary=29)["eligible"])
        self.assertFalse(self.rating(count=1)["eligible"])


if __name__=="__main__":
    unittest.main()
