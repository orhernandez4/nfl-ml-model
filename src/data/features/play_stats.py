""""""

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number, join_to_home_and_away


def build_play_stats_features(games, drives):
    """Builds play statistics features for NFL games.

    :param pl.LazyFrame games: LazyFrame containing game data.
    :param pl.LazyFrame drives: LazyFrame containing drive data.
    :return: LazyFrame with play statistics features added.
    :rtype: pl.LazyFrame
    """
    stats = (
        drives
        .group_by('posteam', 'season', 'week', 'defteam')
        .agg(
            pl.sum('Net Yds').alias('yards_play'),
            pl.sum('Plays').alias('count'),
        )
    )
    for feature_name in ['yards_play']:
        feature = (
            stats
            .select(
                'posteam', 'season', 'week', 'defteam', feature_name, 'count'
            )
            .pipe(build_adjusted_features)
            .pipe(shift_week_number)
        )
        games = join_to_home_and_away(games, feature, drop_swt=False)
    return games
