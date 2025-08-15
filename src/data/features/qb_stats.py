""""""
import polars as pl

from src.utils import join_to_home_and_away


def build_qb_stats_features(games, player_offense, starters):
    """Build out QB stats features for the games DataFrame.

    :param pl.LazyFrame games: The games DataFrame to join features to.
    :param pl.LazyFrame player_offense: The player offense DataFrame
        containing passing statistics.
    :param pl.LazyFrame starters: The starters DataFrame containing
        quarterback information.
    :return: The games DataFrame with QB stats features added.
    :rtype: pl.LazyFrame
    """
    feature = (
        player_offense
        .filter(pl.col('pass_attempts') > 0)
        .sort('player', 'season', 'week')
        .with_row_index('index')
        .rolling(
            index_column='index',
            group_by=['player'],
            period='50i'
        )
        .agg(
            pl.col('completions').sum(),
            pl.col('pass_attempts').sum(),
            pl.col('pass_yards').sum(),
            pl.col('pass_td').sum(),
            pl.col('interceptions').sum(),
            pl.col('pfr').last(),
            pl.col('season').last(),
            pl.col('week').last(),
        )
        .select( # passer rating components
            'player', 'pfr', 'season', 'week',
            a=(((pl.col('completions') / pl.col('pass_attempts')) - 0.3) * 5).clip(0, 2.375),
            b=(((pl.col('pass_yards') / pl.col('pass_attempts')) - 3) * 0.25).clip(0, 2.375),
            c=((pl.col('pass_td') / pl.col('pass_attempts')) * 20).clip(0, 2.375),
            d=((2.375 - (pl.col('interceptions') / pl.col('pass_attempts')) * 25)).clip(0, 2.375),
        )
        .select(
            'player', 'pfr', 'season', 'week',
            qb_rating=(((pl.col('a') + pl.col('b') + pl.col('c') + pl.col('d')) / 6) * 100).round(1)
        )
        .sort('player', 'season', 'week')
        .with_columns(
            pl.exclude('player', 'pfr', 'season', 'week').shift(1).over(["player"]),
        )
        .join(
            starters,
            left_on=['player', 'pfr'],
            right_on=['QB_1', 'pfr'],
            how='right',
        )
        .select(
            'posteam', 'season', 'week',
            pl.col('qb_rating').fill_null(strategy='mean')
        )
        .sort('posteam', 'season', 'week')
        .rename({'posteam': 'team'})
    )
    games = join_to_home_and_away(games, feature, drop_swt=False)
    games = (
        games
        .with_columns(
            qb_rating_net=(pl.col('qb_rating_obj') - pl.col('qb_rating_adv')).round(1),
        )
        .drop('qb_rating_obj', 'qb_rating_adv')
    )
    # print(feature.collect().glimpse())
    # print(starters.collect())
    # print(games.collect())
    return games
