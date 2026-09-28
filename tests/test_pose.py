import unittest

from elderly_monitor.models import Observation, State
from elderly_monitor.pose_detector import YoloPoseDetector
from elderly_monitor.pose_observer import PoseActivityObserver
from elderly_monitor.tracker import TemporalStateTracker


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.observer = PoseActivityObserver([[0,0],[300,0],[300,300],[0,300]])

    def pose(self, positions):
        points = [[0,0,0] for _ in range(17)]
        for i, xy in positions.items():
            points[i] = [*xy, 0.95]
        return points

    def test_lying(self):
        p = self.pose({5:(50,100),6:(50,120),11:(180,100),12:(180,120)})
        self.assertEqual(self.observer.observe(0,(20,70,240,80),0.9,p,1).state, State.LYING_IN_BED)

    def test_sitting(self):
        p = self.pose({5:(100,50),11:(100,150),13:(180,150),15:(180,250)})
        self.assertEqual(self.observer.observe(0,(70,20,140,260),0.9,p,1).state, State.SITTING_ON_BED)

    def test_sitting_outside_and_spatial_evidence(self):
        p = self.pose({5:(400,50),11:(400,150),13:(480,150),15:(480,250)})
        o = self.observer.observe(0,(370,20,140,260),0.9,p,1)
        self.assertEqual(o.state,State.SITTING_OUTSIDE_BED)
        self.assertEqual(o.bed_relation,'away')

    def test_horizontal_outside_is_out_of_bed(self):
        p = self.pose({5:(400,100),11:(530,100)})
        o = self.observer.observe(0,(370,70,240,80),0.9,p,1)
        self.assertEqual(o.state,State.OUT_OF_BED)
        self.assertEqual(o.bed_relation,'away')

    def test_boundary_uncertainty_is_not_forced_outside(self):
        p = self.pose({5:(305,50),11:(305,150),13:(380,150),15:(380,250)})
        o = self.observer.observe(0,(275,20,140,260),0.9,p,1)
        self.assertEqual(o.state,State.UNKNOWN)
        self.assertEqual(o.bed_relation,'near')

    def test_standing_overlapping_bed_is_not_sitting(self):
        p = self.pose({5:(100,50),11:(100,150),13:(100,200),15:(100,280)})
        self.assertEqual(self.observer.observe(0,(70,20,80,280),0.9,p,1).state, State.STANDING)

    def test_hidden_joints_are_unknown(self):
        self.assertEqual(self.observer.observe(0,(0,0,200,200),0.9,[],1).state, State.UNKNOWN)

    def test_walking_and_track_change_reset_speed(self):
        p = self.pose({5:(400,50),11:(400,150),13:(400,200),15:(400,280)})
        self.observer.observe(0,(370,20,80,280),0.9,p,1)
        moved = [[x+50,y,c] for x,y,c in p]
        self.assertEqual(self.observer.observe(1,(420,20,80,280),0.9,moved,1).state,State.WALKING)
        self.assertEqual(self.observer.observe(2,(370,20,80,280),0.9,p,2).state,State.STANDING)

    def test_no_switch_to_visitor(self):
        detector = YoloPoseDetector.__new__(YoloPoseDetector)
        detector.bed_polygon = [[0,0],[300,0],[300,300],[0,300]]
        detector.target_track_id = None
        patient = dict(id=1,bbox=(20,20,100,100))
        visitor = dict(id=2,bbox=(400,20,100,100))
        self.assertEqual(detector.select_target([patient,visitor]), patient)
        self.assertIsNone(detector.select_target([visitor]))

    def test_ambiguous_initial_identity(self):
        detector = YoloPoseDetector.__new__(YoloPoseDetector)
        detector.bed_polygon = [[0,0],[300,0],[300,300],[0,300]]
        detector.target_track_id = None
        self.assertIsNone(detector.select_target([dict(id=1,bbox=(20,20,100,100)),dict(id=2,bbox=(50,50,100,100))]))

    def test_unknown_does_not_trigger_out_of_bed_alert(self):
        tracker = TemporalStateTracker(alert_after_sec=5)
        tracker.update(Observation(0,State.UNKNOWN,0,None))
        result = tracker.finish(20)
        self.assertEqual(result['total_out_of_bed_sec'],0)
        self.assertEqual(result['total_unknown_sec'],20)
        self.assertEqual(result['longest_out_of_bed_period_sec'],0)
        self.assertEqual(result['decision'],'MONITOR')
