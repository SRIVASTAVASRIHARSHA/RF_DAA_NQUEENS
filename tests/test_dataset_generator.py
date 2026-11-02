import unittest

from src.dataset_generator import (
    FEATURE_COLUMNS,
    STATE_COLUMN,
    TARGET_COLUMN,
    build_dataset,
    split_dataset_by_state,
)


class TestDatasetGenerator(unittest.TestCase):
    def test_four_queens_dataset_has_expected_exhaustive_counts(self):
        dataset, summary = build_dataset([4])
        self.assertEqual(len(dataset), 16)
        self.assertEqual(summary["positive_samples"], 8)
        self.assertEqual(summary["negative_samples"], 8)
        self.assertEqual(dataset[TARGET_COLUMN].sum(), 8)

    def test_dataset_contains_features_and_binary_labels(self):
        dataset, _ = build_dataset([4])
        self.assertEqual(
            list(dataset.columns),
            [STATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN],
        )
        self.assertEqual(set(dataset[TARGET_COLUMN].unique()), {0, 1})

    def test_every_candidate_has_a_state_identifier(self):
        dataset, _ = build_dataset([4])
        self.assertTrue(dataset[STATE_COLUMN].notna().all())
        self.assertGreater(dataset[STATE_COLUMN].nunique(), 1)

    def test_group_split_has_no_state_leakage(self):
        dataset, _ = build_dataset([4, 5, 6])
        train, validation = split_dataset_by_state(
            dataset, test_size=0.25, random_state=17
        )
        self.assertTrue(
            set(train[STATE_COLUMN]).isdisjoint(set(validation[STATE_COLUMN]))
        )
        self.assertEqual(len(train) + len(validation), len(dataset))
        self.assertEqual(train[TARGET_COLUMN].nunique(), 2)
        self.assertEqual(validation[TARGET_COLUMN].nunique(), 2)

    def test_invalid_split_fraction_is_rejected(self):
        dataset, _ = build_dataset([4])
        with self.assertRaises(ValueError):
            split_dataset_by_state(dataset, test_size=1.0)

    def test_duplicate_board_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            build_dataset([4, 4])


if __name__ == "__main__":
    unittest.main()
