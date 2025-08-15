""""""

# TODO: normal game situations
import sqlite3

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number, join_to_home_and_away


def get_drive_result():
    """"""
    return (
        pl.first('fixed_drive_result')
        .replace_strict({'Touchdown': 7, 'Field goal': 3}, default=0)
        .alias('points_drive')
    )


def extract_drive_points(plays, expected_values):
    """Extract the results for all drives.

    Example output:


    :param pl.DataFrame plays: The plays DataFrame.
    :return: The results for all drives.
    :rtype: pl.DataFrame
    """
    return (
        plays.group_by(
            ['season', 'week', 'posteam', 'fixed_drive']
        )
        .agg(
            get_drive_result(),
            pl.first('yardline_100', 'posteam_type', 'defteam')
        )
        .with_columns(
            absolute_yrdln=(100 - pl.col('yardline_100')).cast(pl.Int32),
        )
        .join(
            expected_values,
            on=['absolute_yrdln', 'posteam_type'],
            how='left'
        )
        .with_columns(
            exp_points_drive=pl.col('points_drive') - pl.col('expected_value'),
        )
        .group_by(
            ['season', 'week', 'posteam']
        )
        .agg(
            pl.col('defteam').first(),
            pl.col('exp_points_drive').sum(),
            pl.col('exp_points_drive').len().alias('count')
        )
        .sort('posteam', 'season', 'week')
    )


def build_drive_stats_features(plays, expected_values, game_id_map):
    """"""
    raw_features = (
        plays
        .pipe(extract_drive_points, expected_values=expected_values)
        .pipe(build_adjusted_features)
    )
    features = (
        raw_features
        .pipe(shift_week_number)
        .pipe(join_to_home_and_away, games=game_id_map)
        # .select(
            # 'game_id',
            # exp_points_drive_net=pl.col('exp_points_drive_posteam_home') - pl.col('exp_points_drive_defteam_away') - pl.col('exp_points_drive_posteam_away') + pl.col('exp_points_drive_defteam_home'),
        # )
    )
    # print(features.collect())
    return features
    # features.write_database(table_name='exp_points_drive',
                            # connection=f"sqlite:///{feature_store_path}",
                            # engine='sqlalchemy',
                            # if_table_exists='replace')


if __name__ == '__main__':
    from src.config.config import PATHS
    from src.data.build import clean_plays

    raw_plays_path = PATHS['raw_plays']
    expected_values = PATHS['expected_values']

    seasons = range(2001, 2025)
    paths = [raw_plays_path / f'play_by_play_{season}.parquet' for season in seasons]
    raw_plays = pl.concat([pl.scan_parquet(p) for p in paths],
                          how='vertical_relaxed')
    plays = clean_plays(raw_plays)
    expected_values = pl.scan_csv(expected_values)

    drive_points = extract_drive_points(plays, expected_values)
