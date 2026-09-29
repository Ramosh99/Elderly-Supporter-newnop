import json
import unittest
from dataclasses import replace
from unittest.mock import patch, MagicMock

from elderly_monitor.config import resolve_config
from elderly_monitor.models import Observation, State
from elderly_monitor.review.gemini import GeminiClient, validate_assessment, apply_assessments, review_with_gemini


class GeminiTests(unittest.TestCase):
    def supported_pair(self):
        a=replace(self.observation(),bbox=(0,0,100,100),state=State.WALKING)
        b=replace(a,timestamp_sec=1.5,state=State.STANDING,confidence=0.9)
        items=[{**self.assessment(),'index':i,'state':'STANDING'} for i in range(2)]
        return [a,b],items

    def test_known_correction_requires_local_and_vlm_support(self):
        obs,items=self.supported_pair()
        changes,decisions=apply_assessments(obs,items,0.8)
        self.assertEqual(changes[1].state,State.STANDING)
        self.assertEqual(decisions[0]['application'],'accepted_supported_correction')
        self.assertNotIn(1.5,changes)

    def test_identity_switch_or_unknown_gap_blocks_correction(self):
        obs,items=self.supported_pair()
        gap=replace(obs[0],timestamp_sec=1.25,state=State.UNKNOWN)
        changes,_=apply_assessments(obs,items,0.8,context=[obs[0],gap,obs[1]])
        self.assertFalse(changes)
        obs[1]=replace(obs[1],track_id=2)
        self.assertFalse(apply_assessments(obs,items,0.8)[0])

    def test_vlm_agreement_alone_is_insufficient(self):
        obs,items=self.supported_pair()
        obs[1]=replace(obs[1],state=State.WALKING)
        self.assertFalse(apply_assessments(obs,items,0.8)[0])

    def test_http_status_logged_without_provider_details(self):
        import numpy as np
        from urllib.error import HTTPError
        obs,_=self.supported_pair()
        capture=MagicMock(); capture.get.return_value=30
        capture.read.return_value=(True,np.zeros((100,100,3),dtype=np.uint8))
        client=MagicMock(); client.assess.side_effect=HTTPError('https://example.invalid/private',429,'private-message',{},None)
        with patch('cv2.VideoCapture',return_value=capture):
            result,log=review_with_gemini('unused',obs,3,self.config(),client)
        self.assertEqual(result,obs)
        self.assertEqual(log['requests'][0]['http_status'],429)
        self.assertEqual(log['requests'][0]['error_category'],'rate_limit_or_quota')
        self.assertNotIn('private',json.dumps(log))

    def config(self):
        return resolve_config({'bed_region_mode':'auto','gemini_enabled':True,'gemini_cache_dir':''})

    def assessment(self, **changes):
        return dict(index=0,state='SITTING_ON_BED',confidence=0.9,target_clear=True,
                    evidence='Hips supported by mattress, knees bent.',**changes)

    def observation(self):
        return Observation(1,State.UNKNOWN,0,None,reason='ambiguous_posture_or_hidden_legs',
                           track_id=1,bed_polygon=[[0,0],[100,0],[100,100]],bed_relation='inside',
                           )

    def test_missing_key_skips_without_network(self):
        with patch.dict('os.environ',{},clear=True), patch('urllib.request.urlopen') as request:
            obs,log=review_with_gemini('unused',[],2,self.config())
        self.assertEqual(log['status'],'skipped_missing_api_key')
        request.assert_not_called()

    def test_disabled_skips_even_with_key(self):
        config=self.config(); config['gemini_enabled']=False
        with patch('urllib.request.urlopen') as request:
            _,log=review_with_gemini('unused',[],2,config)
        self.assertFalse(log['enabled']); request.assert_not_called()

    def test_ambiguous_posture_can_be_updated(self):
        o=replace(self.observation(),bbox=(0,0,100,100))
        changes,_=apply_assessments([o],[self.assessment()],0.8)
        self.assertEqual(changes[1].state,State.SITTING_ON_BED)

    def test_missing_identity_and_known_labels_cannot_be_overwritten(self):
        for o in [self.observation(),replace(self.observation(),bbox=(0,0,100,100),track_id=None),
                  replace(self.observation(),bbox=(0,0,100,100),state=State.STANDING)]:
            changes,_=apply_assessments([o],[self.assessment()],0.8)
            self.assertFalse(changes)

    def test_conflicting_bed_evidence_rejected(self):
        o=replace(self.observation(),bbox=(0,0,100,100),bed_relation='away')
        changes,_=apply_assessments([o],[self.assessment()],0.8)
        self.assertFalse(changes)

    def test_invalid_responses_rejected(self):
        for change in [{'state':'FALL'},{'confidence':float('nan')},{'index':2},{'target_clear':'yes'}]:
            item=self.assessment(); item.update(change)
            with self.assertRaises((ValueError,TypeError)):
                validate_assessment({'frames':[item]},1)
        with self.assertRaises(ValueError):
            validate_assessment({'frames':[self.assessment(),self.assessment()]},2)

    def test_rest_request_and_token_usage(self):
        payload={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'frames':[self.assessment()]})}]}}],
                 'usageMetadata':{'promptTokenCount':500,'candidatesTokenCount':90}}
        response=MagicMock(); response.__enter__.return_value=response
        response.read.return_value=json.dumps(payload).encode()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'test-only'}), patch('urllib.request.urlopen',return_value=response) as send:
            items,usage=GeminiClient(self.config()).assess([(1,b'fake-jpeg')])
        self.assertEqual(items[0]['state'],'SITTING_ON_BED')
        self.assertEqual(usage['promptTokenCount'],500)
        body=json.loads(send.call_args.args[0].data)
        self.assertEqual(body['generationConfig']['responseMimeType'],'application/json')
        self.assertNotIn('test-only',send.call_args.args[0].full_url)

    def test_api_failure_keeps_observations(self):
        import numpy as np
        o=replace(self.observation(),bbox=(0,0,50,50))
        obs=[o,replace(o,timestamp_sec=1.5)]
        capture=MagicMock(); capture.get.return_value=30
        capture.read.return_value=(True,np.zeros((100,100,3),dtype=np.uint8))
        client=MagicMock(); client.assess.side_effect=TimeoutError('private provider details')
        with patch('cv2.VideoCapture',return_value=capture):
            result,log=review_with_gemini('unused',obs,3,self.config(),client)
        self.assertEqual(result,obs)
        self.assertEqual(log['requests'][0]['status'],'fallback')
        self.assertNotIn('private provider details',json.dumps(log))
        capture.release.assert_called_once()
