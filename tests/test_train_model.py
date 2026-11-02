import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.dataset_generator import generate_dataset
from src.feature_extraction import FEATURE_COLUMNS
from src.train_model import calculate_ranking_metrics, train_and_evaluate


class TestTrainModel(unittest.TestCase):
    def test_ranking_metrics_compare_with_ascending_column_baseline(self):
        validation = pd.DataFrame({
            "state_id": ["state-a", "state-a", "state-b", "state-b"],
            "candidate_col": [0, 3, 0, 2],
            "can_complete": [0, 1, 1, 0],
        })
        probabilities = [0.2, 0.9, 0.8, 0.3]
        metrics = calculate_ranking_metrics(validation, probabilities)
        self.assertEqual(metrics["candidate_parent_states"], 2)
        self.assertEqual(metrics["top1_hit_rate_solvable_states"], 1.0)
        self.assertEqual(
            metrics["ascending_column_baseline_top1_hit_rate_solvable_states"],
            0.5,
        )

    def test_train_and_evaluate_writes_model_and_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            models_dir = root / "models"
            results_dir = root / "results"
            data_dir.mkdir()

            generate_dataset(
                [4, 5, 6, 7],
                csv_path=data_dir / "all.csv",
                summary_path=data_dir / "summary.json",
                train_path=data_dir / "train.csv",
                validation_path=data_dir / "validation.csv",
            )

            model_path = models_dir / "model.joblib"
            _, metrics = train_and_evaluate(
                train_csv=data_dir / "train.csv",
                validation_csv=data_dir / "validation.csv",
                model_path=model_path,
                metadata_path=models_dir / "metadata.json",
                metrics_path=results_dir / "metrics.json",
                report_path=results_dir / "report.txt",
                importance_path=results_dir / "importance.csv",
                confusion_plot_path=results_dir / "confusion.png",
                importance_plot_path=results_dir / "importance.png",
                n_estimators=15,
                max_depth=8,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue((results_dir / "metrics.json").exists())
            self.assertTrue((results_dir / "confusion.png").exists())
            loaded_model = joblib.load(model_path)
            self.assertEqual(
                list(loaded_model.feature_names_in_),
                list(FEATURE_COLUMNS),
            )
            self.assertGreaterEqual(metrics["roc_auc"], 0.0)
            self.assertLessEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
