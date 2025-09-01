""""""

import sqlite3

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number, join_to_home_and_away


def get_points_for_against(scores):
    """Transform scores data into points for and against format.

    :param pl.LazyFrame scores: Polars LazyFrame containing game scores.
    :return: Transformed scores with points for and against.
    :rtype: pl.LazyFrame
    """
    obj_scores = scores.select(
        'obj_team', 'season', 'week', 'adv_team', 'obj_score', 'adv_score'
    )
    adv_scores = scores.select(
        pl.col('adv_team').alias('obj_team'),
        pl.col('season', 'week'),
        pl.col('obj_team').alias('adv_team'),
        pl.col('adv_score').alias('obj_score'),
        pl.col('obj_score').alias('adv_score'),
    )
    return (
        pl.concat([obj_scores, adv_scores], how='vertical')
        .sort('obj_team', 'season', 'week')
    )


def roll_points_for_against(team_scores):
    """Calculates rolling points for and against over a specified period.

    :param pl.LazyFrame team_scores: Polars LazyFrame containing team scores.
    :return: Rolling points for and against.
    :rtype: pl.LazyFrame
    """
    rolling_team_scores = (
        team_scores
        .sort('obj_team', 'season', 'week')
        .with_row_index('index')
        .rolling(
            index_column='index',
            group_by=['obj_team', 'season'],
            period='99i',
        )
        .agg(
            pl.col('week').last(),
            pl.col('obj_score').sum().alias('points_for'),
            pl.col('adv_score').sum().alias('points_against'),
        )
        .sort('index')
        .select('obj_team', 'season', 'week', 'points_for', 'points_against')
    )
    return rolling_team_scores


def calculate_pyexp_stats(points, period='99i', suffix=""):
    """Calculates Pythagorean expectation statistics.

    :param pl.LazyFrame points: Polars LazyFrame containing points data.
    :param str period: Rolling period for aggregation.
    :param str suffix: Optional suffix for the resulting column names.
    :return: Pythagorean expectation statistics.
    :rtype: pl.LazyFrame
    """
    return (
        points
        .select(
            pl.col('obj_team').alias('team'),
            pl.col('season', 'week'),
            (1 / (1 + (pl.col('points_against') / pl.col('points_for')).pow(2.77))).alias(f'pyexp{suffix}'),
        )
    )


def log5(games, feature_name):
    """Calculates the log5 statistic for a given feature.

    :param pl.LazyFrame games: Polars LazyFrame containing game data.
    :param str feature_name: Name of the feature to calculate log5 for.
    :return: Log5 statistic for the feature.
    :rtype: pl.LazyFrame
    """
    p_A = pl.col(f"{feature_name}_obj")
    p_B = pl.col(f"{feature_name}_adv")
    numerator = p_A - (p_A * p_B)
    denominator = p_A + p_B - (2 * p_A * p_B)
    return numerator / denominator


def convert_to_log5(games, feature_name):
    """Replaces a feature in the games DataFrame with its log5 equivalent.

    :param pl.LazyFrame games: Polars LazyFrame containing game data.
    :param str feature_name: Name of the feature to convert.
    :return: Games DataFrame with the log5 feature added.
    :rtype: pl.LazyFrame
    """
    log5_feature = f'log5_{feature_name}'
    return (
        games
        .with_columns(
            log5(games, feature_name).alias(log5_feature)
        )
        .drop(f'{feature_name}_obj', f'{feature_name}_adv')
    )


def build_pythag_features(games, scores):
    """Builds Pythagorean expectation features for NFL games.

    :param pl.LazyFrame games: Polars LazyFrame containing game data.
    :param pl.LazyFrame scores: Polars LazyFrame containing scores data.
    :param pl.LazyFrame drives: Polars LazyFrame containing drives data.
    :return: Games DataFrame with Pythagorean expectation features added.
    :rtype: pl.LazyFrame
    """
    team_points = get_points_for_against(scores)
    rolling_team_points = roll_points_for_against(team_points)
    pyexps = calculate_pyexp_stats(rolling_team_points)
    for f in [(pyexps, "pyexp")]:
        feature = (
            f[0]
            .pipe(shift_week_number)
        )
        games = join_to_home_and_away(games, feature, drop_swt=False)
        games = convert_to_log5(games, f[1])
    return games
