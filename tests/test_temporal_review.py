import unittest

from elderly_monitor.models import Observation, State
from elderly_monitor.temporal.tracker import TemporalStateTracker


class TemporalReviewTests(unittest.TestCase):
    def run_clip(self, samples, end, **kwargs):
        tracker = TemporalStateTracker(**kwargs)
        for t, state, relation in samples:
            tracker.update(Observation(t, state, 0.9 if state != State.UNKNOWN else 0, None,
                                       track_id=1, bed_relation=relation))
        return tracker, tracker.finish(end)

    def test_short_sitting_is_preserved(self):
        _, result = self.run_clip([(0,State.LYING_IN_BED,'inside'),(1,State.SITTING_ON_BED,'inside'),
                                  (2,State.LYING_IN_BED,'inside')],3)
        self.assertEqual(result['activity_duration_sec']['sitting_on_bed'],1)
        self.assertEqual(result['bed_exit_count'],0)

    def test_sustained_standing_beside_bed_is_exit(self):
        samples = [(0,State.LYING_IN_BED,'inside')]+[(t,State.STANDING,'near') for t in range(1,5)]+[(5,State.SITTING_ON_BED,'inside')]
        _, result = self.run_clip(samples,6,alert_after_sec=2)
        self.assertEqual(result['bed_exit_count'],1)
        self.assertEqual(result['events'][0]['start_time_sec'],1)
        self.assertNotEqual(result['decision'],'ALERT')

    def test_brief_away_then_back_does_not_confirm_exit(self):
        _, r = self.run_clip([(0,State.LYING_IN_BED,'inside'),(1,State.STANDING,'away'),
                             (2,State.SITTING_ON_BED,'inside')],3)
        self.assertEqual(r['bed_exit_count'],0)

    def test_outside_state_changes_do_not_reset_exit_evidence(self):
        _, r = self.run_clip([(0,State.LYING_IN_BED,'inside'),(1,State.STANDING,'near'),
                             (2,State.WALKING,'away'),(3,State.STANDING,'away'),
                             (4,State.SITTING_ON_BED,'inside'),(5,State.LYING_IN_BED,'inside')],6)
        self.assertEqual(r['bed_exit_count'],1)
        self.assertEqual(r['bed_return_count'],1)
        self.assertEqual(r['events'][0]['start_time_sec'],1)
        self.assertEqual(r['events'][0]['confirmed_time_sec'],2.5)
        self.assertEqual(r['events'][1]['start_time_sec'],4)

    def test_long_occlusion_resets_event_context(self):
        _, r = self.run_clip([(0,State.LYING_IN_BED,'inside'),(1,State.UNKNOWN,'unknown'),
                             (2,State.UNKNOWN,'unknown'),(3,State.WALKING,'away'),
                             (4,State.WALKING,'away')],5)
        self.assertEqual(r['bed_exit_count'],0)
        self.assertEqual(r['total_unknown_sec'],2)

    def test_short_occlusion_preserves_baseline_but_not_away_evidence(self):
        _, r = self.run_clip([(0,State.LYING_IN_BED,'inside'),(1,State.UNKNOWN,'unknown'),
                             (2,State.WALKING,'away'),(3,State.WALKING,'away')],4)
        self.assertEqual(r['bed_exit_count'],1)
        self.assertEqual(r['events'][0]['start_time_sec'],2)
        self.assertEqual(r['total_unknown_sec'],1)

    def test_prolonged_absence_alert(self):
        _, r = self.run_clip([(t,State.WALKING,'away') for t in range(5)],5,alert_after_sec=3)
        self.assertEqual(r['decision'],'ALERT')
        self.assertIn('prolonged_confirmed_absence_from_bed',r['decision_reasons'])
        self.assertEqual(r['bed_exit_count'],0)  # video starts away

    def test_prolonged_sitting_monitor(self):
        _, r = self.run_clip([(t,State.SITTING_ON_BED,'inside') for t in range(5)],5,sitting_monitor_sec=3)
        self.assertEqual(r['decision'],'MONITOR')
        self.assertIn('prolonged_sitting_on_bed',r['decision_reasons'])

    def test_gap_and_duration_conservation(self):
        tracker, r = self.run_clip([(1,State.LYING_IN_BED,'inside'),(4,State.SITTING_ON_BED,'inside')],5)
        self.assertEqual(sum(r['activity_duration_sec'].values()),5)
        self.assertEqual(r['total_unknown_sec'],3)
        self.assertEqual(r,tracker.finish(5))

    def test_low_confidence_does_not_create_exit(self):
        tracker = TemporalStateTracker()
        tracker.update(Observation(0,State.LYING_IN_BED,0.9,None))
        for t in range(1,5):
            tracker.update(Observation(t,State.WALKING,0.1,None,bed_relation='away'))
        r = tracker.finish(5)
        self.assertEqual(r['bed_exit_count'],0)
        self.assertEqual(r['total_unknown_sec'],4)

    def test_identity_switch_cannot_create_exit(self):
        tracker = TemporalStateTracker()
        tracker.update(Observation(0,State.LYING_IN_BED,0.9,None,track_id=1))
        for t in range(1,4):
            tracker.update(Observation(t,State.WALKING,0.9,None,track_id=2,bed_relation='away'))
        self.assertEqual(tracker.finish(4)['bed_exit_count'],0)

    def test_invalid_time_rejected(self):
        tracker = TemporalStateTracker()
        tracker.update(Observation(1,State.UNKNOWN,0,None))
        with self.assertRaises(ValueError):
            tracker.update(Observation(1,State.UNKNOWN,0,None))
        with self.assertRaises(ValueError):
            tracker.finish(0)

    def test_unknown_breaks_absence_alert_clock(self):
        _, r = self.run_clip([(0,State.WALKING,'away'),(1,State.WALKING,'away'),
                             (2,State.UNKNOWN,'unknown'),(3,State.WALKING,'away'),
                             (4,State.WALKING,'away')],5,alert_after_sec=3)
        self.assertEqual(r['longest_confirmed_away_sec'],2)
        self.assertEqual(r['decision'],'MONITOR')

    def test_identity_change_breaks_absence_alert_clock(self):
        tracker = TemporalStateTracker(alert_after_sec=3)
        for t in range(4):
            tracker.update(Observation(t,State.WALKING,0.9,None,track_id=1 if t<2 else 2,bed_relation='away'))
        self.assertEqual(tracker.finish(4)['longest_confirmed_away_sec'],2)

    def test_empty_video_is_not_normal(self):
        self.assertEqual(TemporalStateTracker().finish(0)['decision'],'MONITOR')

    def test_unstable_neighbours_do_not_invent_activity(self):
        _, r = self.run_clip([(0,State.WALKING,'away'),(0.2,State.STANDING,'away'),
                             (0.4,State.WALKING,'away')],0.6)
        self.assertEqual(r['total_unknown_sec'],0.6)

    def test_standing_walking_flicker_preserves_confirmed_out_of_bed(self):
        _, r = self.run_clip([(0,State.LYING_IN_BED,'inside')]+
                            [(1+i/3,State.STANDING if i%2 else State.WALKING,'away') for i in range(6)],3)
        self.assertEqual(r['activity_duration_sec']['out_of_bed'],2)
        self.assertEqual(r['bed_exit_count'],1)
