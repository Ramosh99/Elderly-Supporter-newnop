import unittest

from elderly_monitor.config import resolve_config
from elderly_monitor.models import Observation, State
from elderly_monitor.review.agent import ReviewAgent, identity_anchor, select_identity


class ReviewAgentTests(unittest.TestCase):
    def observation(self,t,state=State.UNKNOWN,identity=1):
        return Observation(t,state,0.9,(10,10,100,100),track_id=identity)

    def config(self,**kwargs):
        return resolve_config({'bed_region_mode':'auto',**kwargs})

    def test_stable_video_does_not_request_review(self):
        obs=[self.observation(t,State.LYING_IN_BED) for t in range(5)]
        self.assertEqual(ReviewAgent(self.config()).plan(obs,5),[])

    def test_stable_upright_over_bed_is_reviewed_with_budget(self):
        from dataclasses import replace
        obs = [replace(self.observation(t,State.STANDING), bed_relation='inside') for t in range(10)]
        windows = ReviewAgent(self.config(review_max_windows=2)).plan(obs,10)
        self.assertEqual(len(windows),2)
        self.assertEqual(windows[0]['reason'],'ambiguous_upright_over_bed')
        self.assertLessEqual(windows[0]['end_sec'],windows[1]['start_sec'])
        calibrated = [replace(o,support_evidence='feet_on_floor_region') for o in obs]
        self.assertEqual(ReviewAgent(self.config()).plan(calibrated,10),[])

    def test_windows_bounded_and_nonoverlapping(self):
        obs=[self.observation(t) for t in range(30)]
        windows=ReviewAgent(self.config(review_max_windows=2)).plan(obs,30)
        self.assertEqual(len(windows),2)
        self.assertLessEqual(windows[0]['end_sec'],windows[1]['start_sec'])
        self.assertGreaterEqual(windows[0]['start_sec'],0)

    def test_budget_and_existing_evidence_preserved(self):
        original=self.observation(0)
        agent=ReviewAgent(self.config(review_max_frames=1))
        def gather(window,budget):
            self.assertEqual(budget,1)
            return [self.observation(0,State.STANDING),self.observation(0.5,State.STANDING)],1,0
        merged,log=agent.run([original],2,gather)
        self.assertEqual(merged[0],original)
        self.assertEqual(len(merged),2)
        self.assertTrue(log['budget_exhausted'])

    def test_identity_requires_two_nearby_matching_ids(self):
        self.assertIsNotNone(identity_anchor([self.observation(0),self.observation(1)],0.5))
        self.assertIsNone(identity_anchor([self.observation(0),self.observation(1,identity=2)],0.5))
        self.assertIsNone(identity_anchor([self.observation(0),self.observation(4)],2))

    def test_ambiguous_people_are_rejected(self):
        anchor=((10,10,100,100),1)
        patient={'bbox':(10,10,100,100)}
        visitor={'bbox':(12,12,100,100)}
        self.assertIsNone(select_identity([patient,visitor],anchor))
        self.assertEqual(select_identity([patient],anchor),patient)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            self.config(review_max_frames=-1)

    def test_spatial_transition_triggers_review_without_pose_change(self):
        from dataclasses import replace
        left = replace(self.observation(0,State.STANDING),bed_relation='near')
        right = replace(self.observation(1,State.STANDING),bed_relation='away')
        windows = ReviewAgent(self.config()).plan([left,right],3)
        self.assertEqual(windows[0]['reason'],'possible_bed_transition')

    def test_zero_budget_does_not_call_evidence_tool(self):
        def gather(*args):
            self.fail('No budget available')
        original = [self.observation(0)]
        merged, log = ReviewAgent(self.config(review_max_frames=0)).run(original,2,gather)
        self.assertEqual(merged,original)
        self.assertEqual(log['attempted_frames'],0)
