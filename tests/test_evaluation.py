import unittest
from evaluation.evaluate import evaluate, match_events


class EvaluationTests(unittest.TestCase):
    def test_ai_draft_cannot_bypass_human_review(self):
        truth = self.data()
        truth['human_reviewed'] = False
        with self.assertRaises(ValueError):
            evaluate(truth, self.data())

    def data(self):
        return {'reviewed':True,'video':'videos/demo.mp4','observation_duration_sec':10,
                'timeline':[{'start_sec':0,'end_sec':10,'state':'LYING_IN_BED'}],'events':[]}

    def test_exact_overlap_and_duration_errors(self):
        truth=self.data(); prediction=self.data()
        prediction['timeline']=[{'start_sec':0,'end_sec':7,'state':'LYING_IN_BED'},
                                {'start_sec':7,'end_sec':10,'state':'SITTING_ON_BED'}]
        r=evaluate(truth,prediction)
        self.assertEqual(r['accuracy_time_weighted'],0.7)
        self.assertEqual(r['duration_errors']['LYING_IN_BED']['signed_error_sec'],-3)
        self.assertEqual(r['confusion_matrix_seconds']['LYING_IN_BED']['SITTING_ON_BED'],3)

    def test_unreviewed_and_gapped_labels_rejected(self):
        truth=self.data(); truth['reviewed']=False
        with self.assertRaises(ValueError): evaluate(truth,self.data())
        truth['reviewed']=True; truth['timeline'][0]['start_sec']=1
        with self.assertRaises(ValueError): evaluate(truth,self.data())

    def test_wrong_video_rejected(self):
        p=self.data(); p['video']='different.mp4'
        with self.assertRaises(ValueError): evaluate(self.data(),p)

    def test_unknown_prediction_is_not_dropped(self):
        p=self.data(); p['timeline'][0]['state']='UNKNOWN'
        r=evaluate(self.data(),p)
        self.assertEqual(r['accuracy_on_known_ground_truth'],0)
        self.assertEqual(r['predicted_unknown_fraction'],1)

    def test_one_to_one_event_matching(self):
        r=match_events([2],[1.9,2.1],0.2)
        self.assertEqual(r['true_positives'],1)
        self.assertEqual(r['false_positives'],1)
        self.assertEqual(r['precision'],0.5)

    def test_matching_maximizes_matches(self):
        r=match_events([1,2],[0.5,1.5],0.6)
        self.assertEqual(r['true_positives'],2)

    def test_no_events_has_undefined_metrics(self):
        r=match_events([],[],1)
        self.assertIsNone(r['precision']); self.assertIsNone(r['recall'])
