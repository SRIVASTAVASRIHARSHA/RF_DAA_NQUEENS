import unittest

from src.feature_extraction import (
    FEATURE_COLUMNS,
    MAX_BOARD_SIZE,
    extract_candidate_features,
)


class TestFeatureExtraction(unittest.TestCase):
    def test_fixed_width_schema(self):
        features = extract_candidate_features(4, (), 0)
        self.assertEqual(tuple(features), FEATURE_COLUMNS)
        self.assertEqual(len(features), 6 + MAX_BOARD_SIZE)

    def test_existing_queens_are_encoded(self):
        features = extract_candidate_features(4, (1, 3), 0)
        self.assertEqual(features["queen_col_0"], 1)
        self.assertEqual(features["queen_col_1"], 3)
        self.assertEqual(features["queen_col_2"], -1)
        self.assertEqual(features["queen_col_11"], -1)

    def test_normalized_features(self):
        features = extract_candidate_features(5, (1,), 3)
        self.assertEqual(features["row_normalized"], 0.25)
        self.assertEqual(features["candidate_col_normalized"], 0.75)

    def test_rejects_unsafe_candidate(self):
        with self.assertRaises(ValueError):
            extract_candidate_features(4, (1,), 1)

    def test_rejects_conflicting_prefix(self):
        with self.assertRaises(ValueError):
            extract_candidate_features(4, (0, 1), 3)

    def test_rejects_board_larger_than_feature_schema(self):
        with self.assertRaises(ValueError):
            extract_candidate_features(MAX_BOARD_SIZE + 1, (), 0)


if __name__ == "__main__":
    unittest.main()
