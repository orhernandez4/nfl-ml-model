"""Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

The preprocess function is reserved for model-agnostic transformations that should be done before building the scikit pipelines.
"""

import numpy as np
import pandas as pd


def reduce_columns(X, columns):
    """Reduce a set of training or test data to a subset of columns.

    :param pd.DataFrame X: a set of training or test data
    :param list columns: a list of columns to keep
    :return: a set of training or test data with a subset of columns
    :rtype: pd.DataFrame
    """
    return X[columns]


def drop_columns(X, columns):
    """Drop a set of columns from a training or test set.

    :param pd.DataFrame X: a set of training or test data
    :param list columns: a list of columns to drop
    :return: a set of training or test data with columns dropped
    :rtype: pd.DataFrame
    """
    return X.drop(columns=columns,
                  errors='ignore')
