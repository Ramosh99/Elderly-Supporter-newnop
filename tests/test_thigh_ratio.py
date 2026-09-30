import unittest

from elderly_monitor.models import State
from elderly_monitor.vision.pose_observer import PoseActivityObserver


class ThighRatioTests(unittest.TestCase):
    def observer(self):
        return PoseActivityObserver([[0, 0], [400, 0], [400, 500], [0, 500]])

    def pose(self, straight=False):
        p = [[0, 0, 0] for _ in range(17)]
        for i, xy in {5:(100,50),6:(180,50),11:(100,150),12:(180,150),
                      13:(130,205),14:(210,205),15:(135,305),16:(215,305)}.items():
            p[i] = [*xy, .95]
        if straight:
            for h,k,a in ((11,13,15),(12,14,16)):
                p[k][0] = p[a][0] = p[h][0]
        return p

    def observe(self, observer, t, pose, identity=1):
        return observer.observe(t, (70,20,200,320), .9, pose, identity)

    def test_persistent_frontal_sitting(self):
        observer = self.observer()
        p = self.pose()
        self.assertEqual(self.observe(observer, 0, p).state, State.STANDING)
        self.assertEqual(self.observe(observer, .25, p).state, State.STANDING)
        o = self.observe(observer, .5, p)
        self.assertEqual(o.state, State.SITTING_ON_BED)
        self.assertEqual(o.reason, 'persistent_foreshortened_thighs')

    def test_straight_legs_are_not_sitting_despite_small_ratio(self):
        observer = self.observer()
        for t in (0, .25, .5, 1):
            self.assertEqual(self.observe(observer, t, self.pose(True)).state, State.STANDING)

    def test_identity_and_visibility_interrupt_confirmation(self):
        for interruption in ('identity', 'hidden', 'gap'):
            with self.subTest(interruption=interruption):
                observer = self.observer()
                p = self.pose()
                self.observe(observer, 0, p)
                self.observe(observer, .25, p)
                if interruption == 'hidden':
                    hidden = self.pose()
                    hidden[15][2] = hidden[16][2] = 0
                    self.observe(observer, .4, hidden)
                o = self.observe(observer, 2 if interruption == 'gap' else .5,
                                 p, 2 if interruption == 'identity' else 1)
                self.assertNotEqual(o.reason, 'persistent_foreshortened_thighs')

    def test_translating_pose_is_not_overridden_as_sitting(self):
        observer = self.observer()
        for t in (0, .25, .5, .75):
            p = self.pose()
            for point in p:
                point[0] += t * 70
            o = self.observe(observer, t, p)
        self.assertEqual(o.state, State.WALKING)


if __name__ == '__main__':
    unittest.main()
