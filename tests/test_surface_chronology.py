import unittest
from dataclasses import replace
from elderly_monitor.config import resolve_config
from elderly_monitor.models import Observation, State
from elderly_monitor.review.chronology import classify_merged
from elderly_monitor.vision.pose_observer import PoseActivityObserver
from elderly_monitor.vision.surface import support_evidence, refresh_support
from elderly_monitor.temporal.tracker import TemporalStateTracker


SURFACE = [[0,0],[300,0],[300,200],[0,200]]
FLOOR = [[0,210],[500,210],[500,500],[0,500]]


def points(hip_y=130, foot_y=280, dx=0):
    p = [[0,0,0] for _ in range(17)]
    for i,(x,y) in {5:(80,hip_y-100),6:(100,hip_y-100),11:(80,hip_y),12:(100,hip_y),
                   13:(80,(hip_y+foot_y)/2),14:(100,(hip_y+foot_y)/2),15:(80,foot_y),16:(100,foot_y)}.items():
        p[i]=[x+dx,y,.95]
    return p


class SurfaceChronologyTests(unittest.TestCase):
    def config(self):
        return resolve_config({'bed_region_mode':'manual','bed_polygon':SURFACE})

    def test_merged_history_matches_direct_chronological_classification(self):
        c=self.config();direct=PoseActivityObserver(SURFACE);raw=[];expected=[]
        for t,dx in [(0,0),(.25,15),(.5,30),(.75,45),(1,60)]:
            p=points(dx=dx)
            raw.append(Observation(t,State.UNKNOWN,0,(0,0,300,300),keypoints=p,
                       track_id=1,bed_polygon=SURFACE,detector_confidence=.95))
            expected.append(direct.observe(t,(0,0,300,300),.95,p,1))
        actual=classify_merged(raw,c)
        self.assertEqual([o.state for o in actual],[o.state for o in expected])
        self.assertEqual([o.speed_px_sec for o in actual],[o.speed_px_sec for o in expected])
        self.assertEqual(actual[-1].state,State.WALKING)

    def test_unknown_posture_keeps_raw_detector_confidence(self):
        p=points();p[13][2]=p[14][2]=p[15][2]=p[16][2]=0
        o=PoseActivityObserver(SURFACE).observe(0,(0,0,300,300),.95,p,1)
        self.assertEqual(o.confidence,0)
        self.assertEqual(o.detector_confidence,.95)

    def test_rejected_identity_is_not_reclassified(self):
        o=Observation(.5,State.UNKNOWN,0,None,reason='review_identity_uncertain')
        self.assertEqual(classify_merged([o],self.config()),[o])

    def test_surface_projection_blocks_exit_but_not_floor_exit(self):
        for evidence,expected in [('feet_over_mattress',0),('uncertain',0),('feet_on_floor_region',1)]:
            t=TemporalStateTracker(max_sample_gap_sec=1)
            for s in range(5):
                seated=s<2
                t.update(Observation(s,State.SITTING_ON_BED if seated else State.STANDING,.9,None,
                    track_id=1,bed_relation='inside',mattress_polygon=SURFACE,
                    support_evidence='posture_over_mattress' if seated else evidence))
            self.assertEqual(t.finish(5)['bed_exit_count'],expected)

    def test_feet_require_both_visible_for_floor_support(self):
        p=points();p[16][2]=0
        self.assertEqual(support_evidence(State.STANDING,p,SURFACE,FLOOR,.5,5),'uncertain')
        self.assertEqual(support_evidence(State.STANDING,points(),SURFACE,FLOOR,.5,5),'feet_on_floor_region')

    def test_mattress_overlap_takes_priority_over_floor(self):
        self.assertEqual(support_evidence(State.WALKING,points(100,180),SURFACE,SURFACE,.5,5),
                         'feet_over_mattress')

    def test_vlm_state_change_recomputes_support(self):
        c=self.config();c['floor_polygon']=FLOOR
        o=Observation(0,State.STANDING,.8,None,keypoints=points(),mattress_polygon=SURFACE,
                      support_evidence='posture_over_mattress')
        self.assertEqual(refresh_support(o,c).support_evidence,'feet_on_floor_region')

    def test_surface_requires_calibration_and_valid_coordinates(self):
        with self.assertRaises(ValueError):
            resolve_config({'bed_region_mode':'auto','mattress_polygon':SURFACE})
        with self.assertRaises(ValueError):
            resolve_config({'bed_region_mode':'auto','floor_polygon':FLOOR})
        with self.assertRaises(ValueError):
            resolve_config({'bed_region_mode':'auto','mattress_polygon':[[0,0],[1,float('nan')],[2,0]]})

    def test_calibration_wrong_video_rejected(self):
        from pathlib import Path
        from types import SimpleNamespace
        from elderly_monitor.pipeline import _prepare_bed
        c=self.config();c.update(mattress_polygon=SURFACE,
            surface_calibration={'video':'videos/a.mp4','frame_size':[500,500]})
        with self.assertRaisesRegex(ValueError,'another video'):
            _prepare_bed(Path('videos/b.mp4'),c,SimpleNamespace(width=500,height=500))


if __name__ == '__main__':
    unittest.main()
