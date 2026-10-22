"""Geometric behavior and missing-observation semantics of the fixed table gate."""
import copy
import unittest

from source.table_range import constrain_landings, select_fixed_table


def observation(frame, quality=.8, usable=True, corners=None):
    return {'frame_id': frame, 'proposal': {'quality': quality, 'usable_proposal': usable,
            'corners': corners if corners is not None else [[100, 100], [300, 100], [300, 300], [100, 300]]}}


def point(frame, x, y):
    return {'frame_id': frame, 'x': x, 'y': y}


class FixedTableTests(unittest.TestCase):
    def test_highest_valid_quality_applies_to_earlier_frames_and_survives_failures(self):
        rows = [observation(0, .99, False), observation(10, .8),
                observation(30, .95), observation(40, 1., False)]
        selected = select_fixed_table(rows, (640, 480))
        self.assertEqual(selected['source_frame'], 30)
        self.assertEqual(selected['candidate_count'], 2)
        kept, details = constrain_landings([point(0, 200, 200), point(41, 500, 400)], rows, (640, 480))
        self.assertEqual([p['frame_id'] for p in kept], [0])
        self.assertEqual(details['events'][1]['reason'], 'outside_fixed_table')

    def test_border_tolerance_keeps_near_edge_but_rejects_floor(self):
        points = [point(1, 200, 200), point(2, 300, 200), point(3, 305, 200), point(4, 320, 400)]
        before = copy.deepcopy(points)
        kept, details = constrain_landings(points, [observation(20)], (640, 480))
        self.assertEqual(kept, points[:3])
        self.assertEqual(points, before)
        self.assertEqual(details['events'][2]['reason'], 'edge_tolerance')
        self.assertEqual(details['rejected_count'], 1)

    def test_polygon_not_axis_aligned_bounding_box(self):
        table = observation(0, corners=[[200, 100], [250, 100], [350, 300], [100, 300]])
        kept, _ = constrain_landings([point(0, 110, 110), point(1, 225, 200)], [table], (640, 480))
        self.assertEqual([p['frame_id'] for p in kept], [1])

    def test_missing_rejected_or_malformed_table_preserves_outputs(self):
        points = [point(0, 500, 400)]
        for rows in ([], [observation(0, usable=False)],
                     [observation(0, corners=[[0, 0]] * 4)],
                     [observation(0, float('nan'))],
                     [observation(0, corners=[[0, 0], [640, 480], [0, 480], [640, 0]])]):
            with self.subTest(rows=rows):
                kept, audit = constrain_landings(points, rows, (640, 480))
                self.assertEqual(kept, points)
                self.assertFalse(audit['selected_table']['available'])
                self.assertEqual(audit['events'][0]['reason'], 'unknown_table_kept')

    def test_quality_tie_uses_earliest_frame_and_scale_preserves_classification(self):
        rows = [observation(30), observation(10)]
        self.assertEqual(select_fixed_table(rows, (640, 480))['source_frame'], 10)
        points = [point(1, 305, 200), point(2, 310, 200)]
        kept, _ = constrain_landings(points, rows, (640, 480))
        scaled = [observation(10, corners=[[2*x, 2*y] for x, y in rows[0]['proposal']['corners']])]
        kept2, _ = constrain_landings([point(p['frame_id'], p['x']*2, p['y']*2) for p in points], scaled, (1280, 960))
        self.assertEqual([p['frame_id'] for p in kept], [p['frame_id'] for p in kept2])


if __name__ == '__main__':
    unittest.main()
