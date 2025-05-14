""""""

import polars as pl

from src.data.features.game_stats import extract_game_points
from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number


def calculate_points_for_against(points, side, period):
    """"""
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
    """"""
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


def calculate_adj_pyexp_stats(points, suffix=""):
    """"""
    adjusted_points = build_adjusted_features(points, aggregation='cumsum')
    return (
        adjusted_points
        .select(
            'team', 'season', 'week',
            (1 / (1 + (pl.col('points_game_scaled_posteam') / pl.col('points_game_scaled_defteam')).pow(2.77))).alias('adj_pyexp'),
        )
    )


def build_pythag_features(plays):
    """"""
    points = extract_game_points(plays)
    pyexp = calculate_pyexp_stats(points)
    return pyexp.pipe(shift_week_number)
    # adj_pyexp = calculate_adj_pyexp_stats(points)
    # return (
        # pyexp
        # .join(adj_pyexp, on=['team', 'season', 'week'], how='inner')
        # .pipe(shift_week_number)
    # )


if __name__ == '__main__':
    from src.config.config import PATHS
    from src.data.build import clean_plays

    raw_plays_path = PATHS['raw_plays']

    seasons = range(2001, 2025)
    paths = [raw_plays_path / f'play_by_play_{season}.parquet' for season in seasons]
    raw_plays = pl.concat([pl.scan_parquet(p) for p in paths],
                          how='vertical_relaxed')
    plays = clean_plays(raw_plays)

    pyexp = build_pythag_features(plays)
    print(pyexp.collect())

