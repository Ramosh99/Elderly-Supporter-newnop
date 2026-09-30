import io
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch,MagicMock
from urllib.error import HTTPError

from elderly_monitor.config import resolve_config
from elderly_monitor.models import Observation,State
from elderly_monitor.review.gemini import GeminiClient
from elderly_monitor.review.sequence import apply_sequence_assessments,guard_fragmentation
from elderly_monitor.temporal.tracker import TemporalStateTracker


class ReliabilityTests(unittest.TestCase):
    def config(self,**kwargs):
        return resolve_config({'bed_region_mode':'auto','gemini_cache_dir':'',**kwargs})

    def response(self):
        item={'index':0,'state':'STANDING','confidence':.9,'target_clear':True,'evidence':'Upright'}
        payload={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'frames':[item]})}]}}]}
        response=MagicMock();response.__enter__.return_value=response
        response.read.return_value=json.dumps(payload).encode()
        return response

    def test_503_retry_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            client=GeminiClient(self.config(gemini_cache_dir=directory))
            error=HTTPError('private',503,'private',{'Retry-After':'2'},io.BytesIO())
            with patch.dict('os.environ',{'GEMINI_API_KEY':'test'}),patch('time.sleep') as sleep,patch(
                    'urllib.request.urlopen',side_effect=[error,self.response()]) as send:
                a,_=client.assess([(0,b'image')])
                self.assertEqual(client.last_request_info['attempts'],2)
                sleep.assert_called_once_with(2.)
                b,usage=client.assess([(0,b'image')])
                self.assertEqual(a,b);self.assertEqual(usage,{})
                self.assertTrue(client.last_request_info['cache_hit'])
                self.assertEqual(send.call_count,2)

    def test_retry_budget_and_auth_failure(self):
        for code,expected in [(503,3),(403,1)]:
            def fail(*args,**kwargs):
                raise HTTPError('private',code,'private',{},io.BytesIO())
            with patch.dict('os.environ',{'GEMINI_API_KEY':'test'}),patch('time.sleep'),patch(
                    'urllib.request.urlopen',side_effect=fail) as send:
                with self.assertRaises(HTTPError):GeminiClient(self.config()).assess([(0,b'image')])
                self.assertEqual(send.call_count,expected)

    def obs(self,t,state=State.STANDING):
        return Observation(t,state,.9,(0,0,100,100),track_id=1,
            bed_polygon=[[0,0],[100,0],[100,100]],bed_relation='inside')

    def assessments(self,state):
        return [dict(index=i,state=state,confidence=.9,target_clear=True,evidence='Visible posture') for i in range(2)]

    def test_span_updates_interior_not_only_endpoints(self):
        context=[self.obs(t) for t in (0,.5,1)]
        changes,_,_=apply_sequence_assessments([context[0],context[-1]],self.assessments('SITTING_ON_BED'),.8,context)
        self.assertEqual(set(changes),{0,.5,1})

    def test_motion_and_visibility_not_overwritten(self):
        # WALKING and identity mismatch must still be blocked
        for middle in [self.obs(.5,State.WALKING), replace(self.obs(.5),track_id=2)]:
            context=[self.obs(0),middle,self.obs(1)]
            changes,_,_=apply_sequence_assessments([context[0],context[-1]],self.assessments('STANDING'),.8,context)
            self.assertFalse(changes)

    def test_no_detection_frames_eligible_for_vlm(self):
        # missing_person_detection UNKNOWN frames may now be classified by Gemini
        # (low confidence, no spatial grounding) — this is intentional for low-light/IR clips
        middle = replace(self.obs(.5), state=State.UNKNOWN, reason='missing_person_detection', bbox=None)
        context = [self.obs(0), middle, self.obs(1)]
        changes,_,_ = apply_sequence_assessments([context[0],context[-1]], self.assessments('STANDING'), .8, context)
        self.assertTrue(changes)

    def test_single_assessment_cannot_fragment_timeline(self):
        o=self.obs(0)
        changes,_,_=apply_sequence_assessments([o],[self.assessments('SITTING_ON_BED')[0]],.8,[o])
        self.assertFalse(changes)

    def test_fragmentation_guard(self):
        context=[self.obs(t) for t in (0,.5,1,1.5,2,2.5)]
        changes={1:replace(context[2],state=State.SITTING_ON_BED)}
        accepted,_=guard_fragmentation(context,changes,3,self.config())
        self.assertFalse(accepted)

    def test_supported_posture_can_leave_short_uncertain_transition(self):
        context = [self.obs(t,State.WALKING) for t in (0,.5,1,1.5,2,2.5)]
        changes = {o.timestamp_sec:replace(o,state=State.SITTING_ON_BED,
                    reason='gemini_sequence_posture') for o in context[:4]}
        accepted,info = guard_fragmentation(context,changes,3,self.config())
        self.assertTrue(accepted)
        self.assertTrue(info['accepted_supported_transition'])
        self.assertGreater(info['proposed_unknown_sec'],info['before_unknown_sec'])

    def test_short_or_unidentified_correction_cannot_bypass_guard(self):
        context = [self.obs(t,State.WALKING) for t in (0,.5,1,1.5,2,2.5)]
        for changes in (
            {1:replace(context[2],state=State.SITTING_ON_BED,reason='gemini_sequence_posture')},
            {o.timestamp_sec:replace(o,state=State.SITTING_ON_BED,track_id=None,
             reason='gemini_sequence_posture') for o in context[:4]},
        ):
            self.assertFalse(guard_fragmentation(context,changes,3,self.config())[0])

    def return_clip(self,reason='ambiguous_posture_or_hidden_legs',identity=1,gap=.4):
        tracker=TemporalStateTracker(max_sample_gap_sec=3)
        rows=[self.obs(0,State.STANDING),self.obs(2,State.SITTING_ON_BED),
              replace(self.obs(3,State.UNKNOWN),reason=reason,track_id=identity),
              self.obs(3+gap,State.LYING_IN_BED)]
        for o in rows:tracker.update(o)
        return tracker.finish(4)

    def test_short_posture_gap_retains_return_support_not_activity(self):
        result=self.return_clip()
        self.assertEqual(result['bed_return_count'],1)
        self.assertEqual(result['events'][0]['start_time_sec'],2)
        self.assertGreaterEqual(result['events'][0]['confirmed_time_sec'],3.4)
        self.assertGreaterEqual(result['total_unknown_sec'],.4)

    def test_missing_identity_or_long_gap_cannot_confirm_return(self):
        for kwargs in [{'reason':'missing_person_detection'},{'identity':2},{'gap':.6}]:
            self.assertEqual(self.return_clip(**kwargs)['bed_return_count'],0)
