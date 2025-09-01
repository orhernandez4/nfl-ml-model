""""""
import polars as pl

from src.utils import join_to_home_and_away


def make_rolling_qb_data(player_offense):
    """"""
    return (
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
    )


def make_rolling_team_data(player_offense):
    """"""
    return (
        player_offense
        .filter(pl.col('pass_attempts') > 1) # exclude gimmick pass plays
        .group_by(['posteam', 'season', 'week'])
        .agg(
            pl.col('completions').sum(),
            pl.col('pass_attempts').sum(),
            pl.col('pass_yards').sum(),
            pl.col('pass_td').sum(),
            pl.col('interceptions').sum(),
            # pl.col('season').last(),
            # pl.col('week').last(),
        )
        .sort('posteam', 'season', 'week')
        .with_row_index('index')
        .rolling(
            index_column='index',
            group_by=['posteam', 'season'],
            period='50i'
        )
        .agg(
            pl.col('completions').sum(),
            pl.col('pass_attempts').sum(),
            pl.col('pass_yards').sum(),
            pl.col('pass_td').sum(),
            pl.col('interceptions').sum(),
            # pl.col('season').last(),
            pl.col('week').last(),
        )
    )


def calculate_qbr(passer_data):
    """"""
    return (
        passer_data
        .with_columns( # passer rating components
            a=(((pl.col('completions') / pl.col('pass_attempts')) - 0.3) * 5).clip(0, 2.375),
            b=(((pl.col('pass_yards') / pl.col('pass_attempts')) - 3) * 0.25).clip(0, 2.375),
            c=((pl.col('pass_td') / pl.col('pass_attempts')) * 20).clip(0, 2.375),
            d=((2.375 - (pl.col('interceptions') / pl.col('pass_attempts')) * 25)).clip(0, 2.375),
        )
        .with_columns(
            qb_rating=(((pl.col('a') + pl.col('b') + pl.col('c') + pl.col('d')) / 6) * 100).round(1)
        )
    )


def build_starter_qbr(player_offense, starters):
    """"""
    return (
        make_rolling_qb_data(player_offense)
        .pipe(calculate_qbr)
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
        .rename({'posteam': 'team',
                 'qb_rating': 'starter_qbr'})
    )


def build_team_qbr(player_offense):
    """"""
    return (
        make_rolling_team_data(player_offense)
        .pipe(calculate_qbr)
        .sort('posteam', 'season', 'week')
        .with_columns(
            pl.exclude('posteam', 'season', 'week').shift(1).over(["posteam"]),
        )
        .rename({'posteam': 'team',
                 'qb_rating': 'team_qbr'})
        .select('team', 'season', 'week',
                pl.col('team_qbr').fill_null(strategy='mean'))
    )


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
    starter_qbr = build_starter_qbr(player_offense, starters)
    games = join_to_home_and_away(games, starter_qbr, drop_swt=False)
    games = (
        games
        .with_columns(
            (pl.col('starter_qbr_obj') - pl.col('starter_qbr_adv'))
            .round(1)
            .fill_null(0)
            .alias('qb_rating_net')
        )
        .drop('starter_qbr_obj', 'starter_qbr_adv')
    )
    # print(feature.collect().glimpse())
    # print(starters.collect())
    # print(games.collect())
    return games
