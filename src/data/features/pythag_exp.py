""""""

import sqlite3

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number, join_to_home_and_away


def get_team_points(games):
    """Extracts team points from game data.

    :param pl.LazyFrame games: Polars LazyFrame containing game data.
    :return: Team points data.
    :rtype: pl.LazyFrame
    """
    posteams = games.select(
        pl.col('season', 'week'),
        pl.col('obj_team').alias('posteam'),
        pl.col('adv_team').alias('defteam'),
        pl.col('obj_score').alias('points_game')
    )
    other_posteams = games.select(
        pl.col('season', 'week'),
        pl.col('adv_team').alias('posteam'),
        pl.col('obj_team').alias('defteam'),
        pl.col('adv_score').alias('points_game')
    )
    return (
        pl.concat([posteams, other_posteams], how='vertical')
        .select('posteam', 'season', 'week', 'defteam', 'points_game')
        .sort('posteam', 'season', 'week')
    )


def get_offense_points(drives):
    """Extracts offensive points from drives data.

    :param pl.LazyFrame drives: Polars LazyFrame containing drives data.
    :return: Offensive points data.
    :rtype: pl.LazyFrame
    """
    pts_mapping = {'Touchdown': 7, 'Field goal': 3}
    int_mapping = {'Interception': 1}
    stats = (
        drives
        .with_columns(
            pl.col('Result').replace_strict(pts_mapping, default=0).alias('Points'),
        )
        .group_by('posteam', 'season', 'week', 'defteam')
        .agg(
            pl.sum('Points').alias('points_game'),
        )
    )
    return stats.sort('posteam', 'season', 'week')


def calculate_points_for_against(points, side, period):
    """Calculates points for or against a team over a specified period.

    :param pl.LazyFrame points: Polars LazyFrame containing points data.
    :param str side: 'posteam' for points for, 'defteam' for points against.
    :param str period: Rolling period for aggregation.
    :return: Points for or against data.
    :rtype: pl.LazyFrame
    """
    alias = 'points_for' if side == 'posteam' else 'points_against'
    return (
        points
        .sort(side, 'season', 'week')
        .with_row_index('index')
        .rolling(
            index_column='index',
            group_by=[side, 'season'],
            period=period,
        )
        .agg(
            pl.col('week').last(),
            pl.col('points_game').sum().alias(alias),
        )
    )


def calculate_pyexp_stats(points, period='99i', suffix=""):
    """Calculates Pythagorean expectation statistics.

    :param pl.LazyFrame points: Polars LazyFrame containing points data.
    :param str period: Rolling period for aggregation.
    :param str suffix: Optional suffix for the resulting column names.
    :return: Pythagorean expectation statistics.
    :rtype: pl.LazyFrame
    """
    points_for = calculate_points_for_against(points, 'posteam', period)
    points_against = calculate_points_for_against(points, 'defteam', period)
    return (
        points_for
        .join(
            points_against,
            left_on=['posteam', 'season', 'week'],
            right_on=['defteam', 'season', 'week'],
            how='inner',
        )
        .select(
            pl.col('posteam').alias('team'),
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


def build_pythag_features(games, points, drives):
    """Builds Pythagorean expectation features for NFL games.

    :param pl.LazyFrame games: Polars LazyFrame containing game data.
    :param pl.LazyFrame points: Polars LazyFrame containing points data.
    :param pl.LazyFrame drives: Polars LazyFrame containing drives data.
    :return: Games DataFrame with Pythagorean expectation features added.
    :rtype: pl.LazyFrame
    """
    team_points = get_team_points(points)
    team_drives_points = get_offense_points(drives)
    pyexps = calculate_pyexp_stats(team_points)
    pyexps_offense = calculate_pyexp_stats(team_drives_points, suffix='_offense')
    for f in [(pyexps, "pyexp"), (pyexps_offense, "pyexp_offense")]:
        feature = (
            f[0]
            .pipe(shift_week_number)
        )
        games = join_to_home_and_away(games, feature, drop_swt=False)
        games = convert_to_log5(games, f[1])
    return games
