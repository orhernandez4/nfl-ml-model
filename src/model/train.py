"""Train and evaluate models."""

import datetime
import os
import joblib
import sqlite3

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from src.model.estimators import (build_baseline_pipeline,
                                  build_lgbm_pipeline,
                                  build_svc_pipeline)
from src.model.hyperoptimize import hyperoptimize
from src.model.evaluate import (custom_cv,
                                evaluate_model,
                                evaluate_features,
                                compile_scores)
from src.plot.plot import (make_and_save_plots,
                           plot_test_calibration,
                           plot_feature_importances)

from src.config.config import (PATHS,
                               CV_TRAIN_SIZE,
                               CV_TEST_SIZE,
                               CV_SHIFT_SIZE,
                               SCORING_METRIC,
                               MAX_EVALS,
                               EARLY_STOP_N)
from src.config.spaces import BASELINE_PARAMS, LIGHTGBM_SPACE, SVC_SPACE


def create_datetime_id():
    """Creates a unique identifier for the current datetime.
    
    :return: unique identifier
    :rtype: str
    """
    now = datetime.datetime.now()
    dt_id = now.strftime("%Y%m%d%H%M%S")
    return dt_id


def make_save_path(results_path):
    """Creates a directory for saving model results.
    
    :param str results_path: path to training results directory
    :return: path to directory for model results
    :rtype: str
    """
    dt_id = create_datetime_id()
    save_path = f"{results_path}/{dt_id}"
    os.makedirs(save_path)
    return save_path


def map_seasons_to_groups(X_train):
    """"""
    seasons = X_train['season'].unique()
    min_year = seasons.min()
    max_year = seasons.max() + 1
    replacements = {}
    for i, start_year in enumerate(range(min_year, max_year, 3)):
        replacements[start_year] = i + 1
        replacements[start_year + 1] = i + 1
        replacements[start_year + 2] = i + 1
    X_train['season'] = X_train['season'].replace(replacements)
    folds = X_train['season'].unique()[-1]
    return X_train, folds 


def evaluate_train_save(model_name, model, X_train, y_train, X_test, y_test,
                        cv, save_path, hyperopt=False, scoring_metric=None,
                        space=None, max_evals=None, early_stop_n=None):
    """"""
    print(f"Evaluating {model_name} on training and holdout data...")
    if hyperopt:
        best_params = hyperoptimize(model, X_train, y_train, cv,
                                    scoring=scoring_metric,
                                    space=space,
                                    max_evals=max_evals,
                                    early_stop_n=early_stop_n)
        print(f"Best params: {best_params}")
        model.set_params(**best_params)
    scores, _ = evaluate_model(model, X_train, y_train, cv)
    scores.to_csv(f"{save_path}/{model_name}_scores.csv")
    make_and_save_plots(scores, model_name, save_path)
    model.fit(X_train, y_train)
    feature_importances = evaluate_features(model, X_test, y_test,
                                            scoring_metric, n_repeats=10)
    plot_feature_importances(feature_importances, model_name, save_path)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    scores = compile_scores(y_test, y_pred, y_pred_proba)
    plot_test_calibration(scores, model_name, save_path)
    print(f"Training {name} on all data...")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    model.fit(X_full, y_full)
    joblib.dump(model, f"{save_path}/{model_name}_model.pkl")


if __name__ == "__main__":
    train_path = PATHS['train_db']
    results_path = PATHS['results']
    save_path = make_save_path(results_path)

    with sqlite3.connect(train_path) as conn:
        X_train = pd.read_sql(
            sql="SELECT * FROM train",
            con=conn,
        )
        y_train = X_train.pop('target')
        X_test = pd.read_sql(
            sql="SELECT * FROM test",
            con=conn,
        )
        y_test = X_test.pop('target')

    # X_train, folds = map_seasons_to_groups(X_train) 
    # cv = LeaveOneGroupOut()

    cv = custom_cv(CV_TRAIN_SIZE, CV_TEST_SIZE, CV_SHIFT_SIZE)

    # evaluate baseline model
    name = 'baseline'
    baseline = build_baseline_pipeline(BASELINE_PARAMS)
    evaluate_train_save(name, baseline, X_train, y_train, X_test, y_test,
                        cv, save_path)

    # evaluate svc
    name = 'svc'
    svc = build_svc_pipeline()
    evaluate_train_save(name, svc, X_train, y_train, X_test, y_test,
                        cv, save_path, hyperopt=True, scoring_metric=SCORING_METRIC,
                        space=SVC_SPACE, max_evals=MAX_EVALS,
                        early_stop_n=EARLY_STOP_N)

    # evaluate lightgbm
    name = 'lightgbm'
    lightgbm = build_lgbm_pipeline()
    evaluate_train_save(name, lightgbm, X_train, y_train, X_test, y_test,
                        cv, save_path, hyperopt=True, scoring_metric=SCORING_METRIC,
                        space=LIGHTGBM_SPACE, max_evals=MAX_EVALS,
                        early_stop_n=EARLY_STOP_N)
