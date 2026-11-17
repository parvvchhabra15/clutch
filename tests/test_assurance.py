import copy
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from clutch.execution import ExecutionConfig, combine_scores, execution_rating
from clutch.inputs import InputError
from clutch.live import PerformanceConfig
from clutch.match_impact import MatchImpactConfig, high_match_rating
from clutch.ratings import rating_components
from clutch.assurance import (
    Evidence, FIELDS, eligible, evaluate, joint_bootstrap, sensitivity, walk_forward,
)
from clutch.competitive import RefinementConfig


def fixture(twins=0):
    bcfg=PerformanceConfig(absolute_weight=.5,resilience_weight=.3,consistency_weight=.2)
    hcfg=MatchImpactConfig(gentle_adjustment=True)
    ecfg=ExecutionConfig()
    parameters={"outcomes":asdict(bcfg),"high_match":asdict(hcfg),"execution":asdict(ecfg),
                "refinements":asdict(RefinementConfig())}
    count=twins or 2
    mats={name:np.zeros((count,6)) for name in FIELDS}
    histories=[]
    for player in range(count):
        stronger_all_round=bool(player) or bool(twins)
        b,h,x=[],[],[]
        for match in range(6):
            bw=.2+.1*match
            absolute=(.2+.03*match) if stronger_all_round else (.8+.03*match)
            lift=.1+.02*match
            normal=.3+.05*match
            hw=.3+.08*match
            net=12+match if stronger_all_round else -8-match
            ew=.2+.1*match
            signal=.12+.01*match if stronger_all_round else -.12-.01*match
            gentle=1 if net>0 else -1
            b.append({"match_evidence_weight":bw,"absolute_pressure_contribution":absolute,
                      "pressure_contribution_vs_usual":lift,"usual_baseline_support":normal,"status":"delivered"})
            h.append({"high_match_weight":hw,"net_team_impact":net,"gentle_delivery_signal":gentle})
            x.append({"match_evidence_weight":ew,"mean_execution_signal":signal})
            values={"bw":bw,"ba":bw*absolute,"bl":bw*normal*lift,"bs":bw,
                    "hw":hw,"hv":hw*net,"hg":hw*gentle,"ew":ew,"ev":ew*signal,
                    "quality":4,"pballs":12,"pmatches":1,"pcap":1,"pcap2":1,
                    "hballs":12,"hmatches":1,"hcap":1,"hcap2":1,
                    "normal":8,"runs":18,"legal":12}
            for name,value in values.items():
                mats[name][player,match]=value
        histories.append((b,h,x))
    def native(player,counts):
        b,h,x=histories[player]
        expand=lambda source:[row for row,n in zip(source,counts) for _ in range(int(n))]
        return combine_scores(rating_components(expand(b),bcfg)[0],execution_rating(expand(x),ecfg)[0],
                              high_match_rating(expand(h),hcfg)[0],ecfg)
    values=[native(i,np.ones(6)) for i in range(count)]
    order=sorted(range(count),key=lambda i:(-values[i],str(i)))
    players=[{"player_id":str(i),"player":"Player "+str(i),"role":"batting","eligible":"True",
              "rank":str(order.index(i)+1),"raw_performance_index":str(values[i])} for i in range(count)]
    evidence=Evidence(players,[str(i) for i in range(6)],
        np.array(["2023-01-01","2023-03-01","2023-05-01","2023-07-01","2026-01-01","2026-02-01"]),
        np.array(["ipl"]*6),np.full((count,6),2,dtype=np.int8),
        {name:sparse.csr_matrix(value) for name,value in mats.items()},parameters,{},Path("."))
    return evidence,native


class AssuranceTests(unittest.TestCase):
    def test_full_score_matches_native_functions_for_repeated_and_omitted_matches(self):
        evidence,native=fixture()
        counts=np.array([[2,0,1,0,1,2],[0,1,0,2,0,1]]).T
        actual=evaluate(evidence.totals(counts),evidence.parameters)["score"]
        for player in range(2):
            for draw in range(2):
                self.assertAlmostEqual(actual[player,draw],native(player,counts[:,draw]),places=12)

    def test_no_evidence_returns_the_neutral_prior_and_does_not_qualify(self):
        evidence,_=fixture()
        totals=evidence.totals(np.zeros(6))
        result=evaluate(totals,evidence.parameters)
        np.testing.assert_allclose(result["score"],50)
        self.assertFalse(eligible(totals,result,evidence.parameters).any())

    def test_bootstrap_uses_all_channels_not_only_absolute_pressure_quality(self):
        evidence,_=fixture()
        summary,indices,counts,scores=joint_bootstrap(evidence,draws=80,seed=17)
        self.assertGreater(evaluate(evidence.totals(),evidence.parameters)["absolute"][0],
                           evaluate(evidence.totals(),evidence.parameters)["absolute"][1])
        self.assertTrue((scores[:,1]>scores[:,0]).all())
        self.assertEqual(summary[0]["player_id"],"1")
        np.testing.assert_array_equal(counts.sum(axis=0),6)

    def test_identical_players_share_draws_ranks_and_fractional_top_ten_places(self):
        evidence,_=fixture(twins=12)
        rows,indices,counts,scores=joint_bootstrap(evidence,draws=50,seed=19)
        np.testing.assert_allclose(scores,np.repeat(scores[:,0,None],12,axis=1),atol=1e-12)
        for row in rows:
            self.assertAlmostEqual(row["top10_resample_share"],10/12)
            self.assertEqual(row["rank_p05"],6.5)
            self.assertEqual(row["rank_p95"],6.5)
        self.assertAlmostEqual(sum(row["top10_resample_share"] for row in rows),10)

    def test_future_appearances_cannot_make_a_player_training_eligible(self):
        evidence,_=fixture()
        full=evidence.totals()
        self.assertTrue(eligible(full,evaluate(full,evidence.parameters),evidence.parameters).all())
        prior=evidence.totals((evidence.dates<"2026-01-01").astype(float))
        self.assertFalse(eligible(prior,evaluate(prior,evidence.parameters),evidence.parameters).any())
        summary,predictions=walk_forward(evidence)
        self.assertEqual(predictions,[])
        for row in summary:
            self.assertEqual(row["player_pairs"],0)
            self.assertIsNone(row["full_index_spearman"])
            self.assertEqual(row["full_index_status"],"insufficient_player_pairs")

    def test_forward_pipeline_recovers_known_persistence_using_only_prior_evidence(self):
        evidence,_=fixture(twins=10)
        evidence.matches=[str(i) for i in range(10)]
        evidence.dates=np.array([f"2023-{i:02d}-01" for i in range(1,7)]+
                               [f"2024-{i:02d}-01" for i in range(1,5)])
        evidence.families=np.array(["ipl"]*10)
        evidence.innings=np.full((10,10),2,dtype=np.int8)
        evidence.matrices={name:sparse.hstack([matrix,matrix[:,:4]],format="csr")
                           for name,matrix in evidence.matrices.items()}
        qualities=np.arange(1,11,dtype=float)/10
        evidence.matrices["ba"]=evidence.matrices["bw"].multiply(qualities[:,None]).tocsr()
        summary,predictions=walk_forward(evidence)
        row=next(r for r in summary if r["role"]=="batting" and r["test_year"]==2024
                 and r["target"]=="future_pressure_quality")
        self.assertEqual(row["player_pairs"],10)
        self.assertAlmostEqual(row["full_index_spearman"],1.0,places=12)
        self.assertEqual(row["simple_rate_status"],"constant_metric")
        self.assertIsNone(row["simple_rate_spearman"])
        self.assertEqual(len(predictions),10)
        self.assertTrue(all(r["train_ordinary_events"]==48 and r["train_pressure_balls"]==72
                            and r["test_pressure_balls"]==48 for r in predictions))

    def test_sensitivity_preserves_inputs_and_distinguishes_recency_from_subset_eligibility(self):
        evidence,_=fixture()
        parameters=copy.deepcopy(evidence.parameters)
        original=evidence.matrices["hw"].toarray()
        summary,rows=sensitivity(evidence)
        np.testing.assert_array_equal(evidence.matrices["hw"].toarray(),original)
        self.assertEqual(evidence.parameters,parameters)
        control=[r for r in rows if r["scenario"]=="control"]
        self.assertTrue(all(r["eligible_in_scenario"] for r in control))
        for row in control:
            self.assertAlmostEqual(row["scenario_score"],row["current_score"],places=12)
        recent=[r for r in rows if r["scenario"]=="recency_half_life_1y"]
        restricted=[r for r in rows if r["scenario"]=="since_2025"]
        self.assertTrue(all(r["eligible_in_scenario"] for r in recent))
        self.assertFalse(any(r["eligible_in_scenario"] for r in restricted))
        self.assertTrue(all(r["eligible_players"]==0 for r in summary if r["role"]=="bowling"))

    def test_invalid_counts_fail_explicitly(self):
        evidence,_=fixture()
        for counts in (1.0,np.ones(5),np.array([1,1,1,1,1,-1]),np.array([1,1,1,1,1,np.nan])):
            with self.assertRaises(InputError):
                evidence.totals(counts)


if __name__=="__main__":
    unittest.main()
