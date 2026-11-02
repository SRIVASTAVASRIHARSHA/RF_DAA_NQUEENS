# Model artifacts

The trained Random Forest file `random_forest.joblib` is generated locally and intentionally excluded from Git because it is a relatively large, Python/scikit-learn-version-sensitive binary.

From the repository root, after generating `data/train.csv` and `data/validation.csv`, run:

```bash
python -m src.train_model
```

This creates `models/random_forest.joblib` and `models/model_metadata.json`, along with metrics and plots under `results/`. The tracked metadata and reports document the validation run used in this project; retraining under a different scikit-learn version can produce slightly different metrics.
