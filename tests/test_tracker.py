import unittest

from elderly_monitor.models import Observation, State
from elderly_monitor.tracker import TemporalStateTracker


class TemporalStateTrackerTests(unittest.TestCase):
    def test_exit_and_return_are_confirmed_after_hold_period(self):
        tracker = TemporalStateTracker(hold_sec=1.0, event_hold_sec=1.0)
        samples = [
            (0, State.LYING_IN_BED),
            (1, State.LYING_IN_BED),
            (2, State.STANDING),
            (3, State.STANDING),
            (4, State.WALKING),
            (5, State.SITTING_ON_BED),
            (6, State.SITTING_ON_BED),
            (7, State.LYING_IN_BED),
        ]
        for timestamp, state in samples:
            tracker.update(Observation(timestamp, state, 0.9, None, bed_relation="away" if state in {State.STANDING, State.WALKING} else "inside"))
        result = tracker.finish(8)
        self.assertEqual(result["bed_exit_count"], 1)
        self.assertEqual(result["bed_return_count"], 1)
        self.assertEqual(result["events"][0]["start_time_sec"], 2)
        self.assertEqual(result["events"][0]["confirmed_time_sec"], 3)
        self.assertEqual(result["events"][1]["start_time_sec"], 5)

    def test_short_state_change_does_not_create_exit(self):
        tracker = TemporalStateTracker(hold_sec=2.0)
        for timestamp, state in [(0, State.LYING_IN_BED), (1, State.STANDING), (2, State.LYING_IN_BED)]:
            tracker.update(Observation(timestamp, state, 0.9, None))
        result = tracker.finish(3)
        self.assertEqual(result["bed_exit_count"], 0)


if __name__ == "__main__":
    unittest.main()
