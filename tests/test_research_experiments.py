import unittest

from src.research_experiments import ABLATIONS, _validate_sizes


class TestResearchExperimentConfiguration(unittest.TestCase):
    def test_default_train_and_unseen_sizes_are_disjoint_and_supported(self):
        _validate_sizes((4, 5, 6, 7, 8), (9, 10))

    def test_rejects_overlap_between_training_and_unseen_sizes(self):
        with self.assertRaises(ValueError):
            _validate_sizes((4, 5, 6), (6, 7))

    def test_rejects_board_size_outside_feature_schema(self):
        with self.assertRaises(ValueError):
            _validate_sizes((4, 5), (13,))

    def test_ablation_variants_are_distinct(self):
        feature_sets = list(ABLATIONS.values())
        self.assertEqual(len(feature_sets), len({tuple(features) for features in feature_sets}))
        self.assertIn("full_18_features", ABLATIONS)
        self.assertEqual(len(ABLATIONS["full_18_features"]), 18)


if __name__ == "__main__":
    unittest.main()
