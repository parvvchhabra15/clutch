import unittest
from dataclasses import asdict

import numpy as np

from clutch.execution import ExecutionConfig
from clutch.live import PerformanceConfig
from clutch.match_impact import MatchImpactConfig
from clutch.evidence import compare_cells,allocated_index_points,paired_score_evidence
from clutch.assurance import evaluate


def cell(legal,runs,wickets=0,value=0,illegal=0):
    return {"legal":legal,"runs":runs,"wickets":wickets,"value":value,"illegal":illegal}


class SelectionEvidenceTests(unittest.TestCase):
    def test_context_standardization_prevents_a_misleading_aggregate_strike_rate(self):
        pp=("ipl",1,"powerplay")
        death=("ipl",1,"death")
        a={pp:cell(80,80),death:cell(20,60)}
        b={pp:cell(20,18),death:cell(80,200)}
        self.assertGreater(sum(v["runs"] for v in b.values()),sum(v["runs"] for v in a.values()))
        result=compare_cells(a,b,"batting")
        self.assertAlmostEqual(result["incumbent_standardized_rate"],200)
        self.assertAlmostEqual(result["challenger_standardized_rate"],170)
        self.assertEqual(result["matched_outcome_finding"],"incumbent_stronger_observed_runs_and_wickets")

    def test_bowling_can_be_a_genuine_economy_wicket_tradeoff(self):
        key=("ipl",2,"death")
        result=compare_cells({key:cell(60,60,2)},{key:cell(60,80,5)},"bowling")
        self.assertEqual(result["matched_outcome_finding"],"observed_run_wicket_tradeoff")
        self.assertLess(result["challenger_run_rate_advantage"],0)
        self.assertGreater(result["challenger_wicket_rate_advantage"],0)

    def test_illegal_costs_stay_in_run_numerator_not_legal_ball_denominator(self):
        key=("ipl",2,"death")
        result=compare_cells({key:cell(60,60,2)},{key:cell(60,64,2,illegal=4)},"bowling")
        self.assertAlmostEqual(result["challenger_standardized_rate"],6.4)
        self.assertAlmostEqual(result["challenger_illegal_events_per100_legal"],100*4/60)
        self.assertEqual(result["matched_outcome_finding"],"incumbent_stronger_observed_runs_and_wickets")

    def test_small_or_nonoverlapping_samples_do_not_become_superiority_claims(self):
        key=("ipl",1,"middle")
        result=compare_cells({key:cell(30,20)},{key:cell(30,60)},"batting")
        self.assertEqual(result["matched_outcome_finding"],"insufficient_common_sample")
        result=compare_cells({key:cell(90,100)},{("other",1,"middle"):cell(90,120)},"batting")
        self.assertEqual(result["matched_outcome_finding"],"no_common_supported_context")
        self.assertIsNone(result["challenger_standardized_rate"])

    def test_narrow_common_coverage_is_disclosed(self):
        key=("ipl",1,"middle")
        result=compare_cells({key:cell(60,60)},{key:cell(60,90)},"batting",
                             incumbent_total=300,challenger_total=60)
        self.assertTrue(result["sufficient_common_sample"])
        self.assertFalse(result["broad_common_coverage"])
        self.assertAlmostEqual(result["incumbent_coverage"],.2)

    def test_signed_match_allocations_reconcile_to_the_full_native_index(self):
        parameters={
            "outcomes":asdict(PerformanceConfig(absolute_weight=.5,resilience_weight=.3,consistency_weight=.2)),
            "execution":asdict(ExecutionConfig()),"high_match":asdict(MatchImpactConfig(gentle_adjustment=True)),
        }
        values={"bw":np.array([.2,.8,.5]),"ba":np.array([.4,-.3,.6]),"bl":np.array([-.1,.3,-.1]),
                "bs":np.array([.2,0,.25]),"hw":np.array([.1,.6,.8]),"hv":np.array([.5,-4,10]),
                "hg":np.array([.1,-.6,.8]),"ew":np.array([.3,.8,.4]),"ev":np.array([.03,-.1,.07])}
        parts=allocated_index_points(values,parameters)
        score=evaluate({k:np.array([v.sum()]) for k,v in values.items()},parameters)["score"][0]
        self.assertAlmostEqual(50+parts["allocated_index_points"].sum(),score,places=12)
        self.assertLess(parts["allocated_index_points"][1],0)

    def test_zero_net_components_keep_opposing_match_contributions(self):
        parameters={"outcomes":asdict(PerformanceConfig()),"execution":asdict(ExecutionConfig()),
                    "high_match":asdict(MatchImpactConfig())}
        values={k:np.zeros(2) for k in ("bw","ba","bl","bs","hw","hv","hg","ew","ev")}
        values.update(bw=np.ones(2),ba=np.array([-1.,1.]),bs=np.array([.5,.5]))
        parts=allocated_index_points(values,parameters)
        self.assertLess(parts["absolute_quality_points"][0],0)
        self.assertGreater(parts["absolute_quality_points"][1],0)
        self.assertAlmostEqual(parts["allocated_index_points"].sum(),0)

    def test_paired_comparison_preserves_ties_and_exposes_overlapping_order(self):
        result=paired_score_evidence([51,52,53],[52,52,52])
        self.assertAlmostEqual(result["incumbent_higher_score_resample_share"],.5)
        self.assertEqual(result["resample_separation"],"score_order_overlaps")


if __name__=="__main__":
    unittest.main()
