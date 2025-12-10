"""Helper functions and script for producing model predictions."""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import mode
import joblib

from src.config.config import PATHS
from src.model.estimators import (
    build_baseline_pipeline,
    build_svc_pipeline,
    build_lgbm_pipeline,
)


def voting_classifier(estimators, X, type):
    """Generate predictions for a set of fitted estimators.

    For soft voting, return the average of the predicted probabilities.
    For hard voting, return the most common class prediction.
    
    :param list estimators: fitted estimators
    :param pd.DataFrame X: features
    :param str type: type of prediction. must be 'soft' or 'hard'
    :return: predictions
    :rtype: np.ndarray
    """
    if type == 'soft':
        return np.mean([est.predict_proba(X)[:, 1] for est in estimators], axis=0)
    elif type == 'hard':
        # scipy.stats.mode returns (mode, count); we want just the mode array
        return mode([est.predict(X) for est in estimators], axis=0)[0]
    else:
        raise ValueError("type must be 'soft' or 'hard'")


def load_train_test_from_db():
    """Load train and test tables from the train.db SQLite database.

    Uses PATHS['train_db'] from src.config.config.

    :return: (train_df, test_df)
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    train_db_path = PATHS["train_db"]
    with sqlite3.connect(train_db_path) as conn:
        train_df = pd.read_sql("SELECT * FROM train", conn)
        test_df = pd.read_sql("SELECT * FROM test", conn)
    return train_df, test_df


def load_or_train_models(X_train, y_train):
    """Load models from disk if available, otherwise train and save them.

    Models are stored under the 'models/' directory as:
      - baseline.pkl
      - svc.pkl
      - lgbm.pkl

    :param pd.DataFrame X_train: training features
    :param pd.Series y_train: training target
    :return: list of fitted estimators
    :rtype: list
    """
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_builders = {
        "baseline": build_baseline_pipeline,
        "svc": build_svc_pipeline,
        "lgbm": build_lgbm_pipeline,
    }

    fitted_estimators = []

    for name, builder in model_builders.items():
        model_path = models_dir / f"{name}.pkl"
        if model_path.exists():
            # Load previously saved model
            model = joblib.load(model_path)
        else:
            # Build and train a new model, then save it
            model = builder()
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
        fitted_estimators.append(model)

    return fitted_estimators


def main():
    # 1. Load train + test data from SQLite
    train_df, test_df = load_train_test_from_db()

    # 2. Separate features and target
    #    We assume 'target' is the label column (as created in build.py)
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]

    # 3. Load or train the models (baseline, svc, lightgbm)
    estimators = load_or_train_models(X_train, y_train)

    # 4. Generate predictions:
    #    - soft voting → average predicted probability for class 1
    #    - hard voting → majority-vote class label
    y_prob = voting_classifier(estimators, X_test, type="soft")
    y_pred = voting_classifier(estimators, X_test, type="hard")

    # 5. Attach predictions to the test dataframe
    out_df = test_df.copy()
    # y_prob is shape (n_samples,), y_pred is shape (1, n_samples) from mode()
    out_df["pred_prob"] = np.asarray(y_prob).ravel()
    out_df["pred_label"] = np.asarray(y_pred).ravel()

    # 6. Save predictions
    out_path = Path("data/predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
