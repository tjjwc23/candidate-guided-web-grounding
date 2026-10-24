import unittest


from data.candidates import normalize_candidate; from data.candidates import parse_candidate_bbox


class CandidateBBoxAlignmentTests(unittest.TestCase):
    def test_xyxy_is_converted_to_internal_xywh(self):
        for actual, expected in zip(
            parse_candidate_bbox(
                [0.25, 0.10, 0.75, 0.40],
                bbox_format="xyxy",
            ),
            [0.25, 0.10, 0.50, 0.30],
        ):
            self.assertAlmostEqual(actual, expected)
        candidate = normalize_candidate(
            {"bbox": [0.25, 0.10, 0.75, 0.40]},
            bbox_format="xyxy",
        )
        self.assertEqual(candidate["bbox_format"], "xywh")
        self.assertTrue(candidate["bbox_normalized"])


if __name__ == "__main__":
    unittest.main()
