import unittest
from dataclasses import replace

from clutch.execution import Annotation,ExecutionConfig,grade_execution
from clutch.full_toss import FullTossPolicy,grade_with_grace,regrade_execution


def annotation(**changes):
    value=Annotation("Full Toss","Drive","Pace","None",0,0,0,False,"","","batter","bowler",False)
    return replace(value,**changes)


def state(a,**changes):
    value={"legal_delivery":a.extra_type not in {"wide","Overstep","other"},
           "recorded_total_runs":a.runs+a.primary+a.secondary,"target":180,"runs_before":150,
           "balls_remaining":12,"par_before":180.0,"phase":"death","wickets_before":4,
           "recent_wickets_before":0,"free_hit_before":False,"bowling_prematch_pressure":90.0}
    value.update(changes)
    return value


SEQUENCE={"prior_dot_streak":0,"prior_boundary_streak":0,"four_plus_last_six":0,
          "dots_last_six":0,"previous_bowler_style":""}


class FullTossGraceTests(unittest.TestCase):
    def test_effective_legal_full_toss_receives_ordinary_credit_not_yorker_credit(self):
        a=annotation()
        raw,features,reason=grade_with_grace("bowling",a,state(a),SEQUENCE)
        self.assertEqual(raw,.1)
        self.assertIn("full_toss",features)
        self.assertNotIn("recorded_yorker",features)
        self.assertNotIn("effective_yorker",features)
        self.assertNotIn("effective_stock_delivery",features)
        self.assertIn("height_unknown",reason)
        self.assertEqual(grade_execution("bowling",a,state(a),SEQUENCE)[0],-.35)

    def test_costly_legal_full_toss_remains_negative(self):
        a=annotation(runs=6,boundary=True)
        self.assertEqual(grade_with_grace("bowling",a,state(a),SEQUENCE)[0],-.15)

    def test_a_single_that_loses_a_one_run_defence_is_not_rewarded(self):
        a=annotation(runs=1)
        raw,_,_=grade_with_grace("bowling",a,state(a,target=151,balls_remaining=1),SEQUENCE)
        self.assertEqual(raw,0.0)

    def test_hard_illegal_errors_are_not_excused_by_the_full_toss_tag(self):
        for extra,penalty in (("Overstep",-1.0),("other",-1.0),("wide",-.6)):
            a=annotation(extra_type=extra,primary=1)
            self.assertEqual(grade_with_grace("bowling",a,state(a),SEQUENCE)[0],penalty)
            self.assertEqual(grade_with_grace("bowling",a,state(a,legal_delivery=True),SEQUENCE)[0],penalty)

    def test_batting_does_not_assume_an_easy_ball_from_unknown_height(self):
        a=annotation(runs=4,boundary=True)
        raw,features,_=grade_with_grace("batting",a,state(a),SEQUENCE)
        self.assertEqual(raw,.1)
        self.assertNotIn("loose_ball_punished",features)
        a=annotation(wicket=True,dismissed="batter",dismissal="Bowled")
        raw,features,_=grade_with_grace("batting",a,state(a),SEQUENCE)
        self.assertEqual(raw,0.0)
        self.assertNotIn("loose_ball_missed",features)

    def test_unknown_charges_are_ungraded_not_a_measured_success(self):
        a=annotation(secondary=1)
        raw,_,reason=grade_with_grace("bowling",a,state(a),SEQUENCE)
        self.assertIsNone(raw)
        self.assertIn("unverified",reason)

    def test_existing_yorker_grading_is_unchanged(self):
        a=annotation(style="Yorker")
        self.assertEqual(grade_with_grace("bowling",a,state(a),SEQUENCE),
                         grade_execution("bowling",a,state(a),SEQUENCE))

    def source_row(self,mid,day,weight=0.0,contest="first_innings"):
        return {"match_id":mid,"innings":1,"sequence":1,"match_date":day,"role":"bowling",
                "player_id":"bowler","player":"Anonymous bowler","family":"ipl","phase":"middle",
                "eligible_execution_opportunity":True,"team_pressure":40.0,"source_execution_weight":0.0,
                "execution_weight":weight,"contest_before":contest,"raw_signal":-.35,"scored_signal":-.35,
                "training_eligible":True,**SEQUENCE}

    def test_changed_grades_rebuild_references_only_after_the_date_finishes(self):
        rows=[self.source_row("a","2025-01-01"),self.source_row("b","2025-01-01"),
              self.source_row("c","2025-01-02")]
        a=annotation()
        states={(r["match_id"],1,1):state(a,target=None,phase="middle") for r in rows}
        annotations={key:a for key in states}
        emitted=[]
        regrade_execution(rows,states,annotations,ExecutionConfig(),FullTossPolicy(),emitted.append)
        self.assertEqual([r["ordinary_reference_count_before"] for r in emitted],[0,0,2])
        self.assertEqual([r["raw_signal"] for r in emitted],[.1,.1,.1])
        self.assertEqual(emitted[0]["peer_reference_before"],0)
        self.assertEqual(emitted[1]["peer_reference_before"],0)
        self.assertGreater(emitted[2]["peer_reference_before"],0)
        self.assertEqual(rows[0]["raw_signal"],-.35)
        self.assertEqual(emitted[0]["before_grace_raw_signal"],-.35)

    def test_ordinary_post_clinch_balls_do_not_regain_weight_through_grace(self):
        source=self.source_row("m","2025-01-01",contest="extras_dependent")
        source["innings"]=2
        a=annotation()
        states={("m",2,1):state(a,target=191,runs_before=162,balls_remaining=4)}
        emitted=[]
        cards,_=regrade_execution([source],states,{("m",2,1):a},ExecutionConfig(),FullTossPolicy(),emitted.append)
        self.assertEqual(emitted[0]["raw_signal"],.1)
        self.assertEqual(emitted[0]["execution_weight"],0)
        self.assertEqual(cards,[])


if __name__=="__main__":
    unittest.main()
