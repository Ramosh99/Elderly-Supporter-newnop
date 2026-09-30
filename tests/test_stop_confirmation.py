import unittest
from elderly_monitor.models import Observation, State
from elderly_monitor.temporal.tracker import TemporalStateTracker


class StopConfirmationTests(unittest.TestCase):
    def run_clip(self, pause, identity=1):
        tracker = TemporalStateTracker(max_sample_gap_sec=3)
        for t, state, target in [(0,State.WALKING,1),(2,pause,identity),(2.5,State.WALKING,identity)]:
            tracker.update(Observation(t,state,.9,None,track_id=target))
        return tracker.finish(4.5)

    def test_short_stop_during_walk(self):
        r = self.run_clip(State.STANDING)
        self.assertEqual(r['activity_duration_sec']['walking'],4.5)

    def test_missing_evidence_not_bridged(self):
        self.assertEqual(self.run_clip(State.UNKNOWN)['total_unknown_sec'],.5)

    def test_identity_change_not_bridged(self):
        self.assertEqual(self.run_clip(State.STANDING,2)['total_unknown_sec'],.5)

    def test_one_second_stop_backdated(self):
        t = TemporalStateTracker(max_sample_gap_sec=3)
        t.update(Observation(0,State.WALKING,.9,None,track_id=1))
        t.update(Observation(2,State.STANDING,.9,None,track_id=1))
        r=t.finish(3)
        self.assertEqual(r['timeline'][-1]['state'],'STANDING')
        self.assertEqual(r['timeline'][-1]['start_sec'],2)
