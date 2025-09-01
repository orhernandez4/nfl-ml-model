""""""

import os
import sqlite3

import polars as pl

from src.utils import shift_week_number


pl.Config.set_tbl_formatting("ASCII_MARKDOWN")


def clean_raw_games(raw_games):
    """Clean the raw games dataframe using polars.

    :param pl.LazyFrame raw_games: The raw games dataframe.
    :return: A cleaned games dataframe.
    :rtype: pl.LazyFrame
    """
    return (
        raw_games
        .filter(
            pl.col('game_type') == 'REG',
            pl.col('result').is_not_null(),
        )
    )


def reduce_games(games, min_year):
    """Reduce the games dataframe using polars.
    
    :param pl.LazyFrame games: The games dataframe.
    :param int min_year: The minimum year to keep in the games dataframe.
    :return: A reduced games dataframe.
    :rtype: pl.LazyFrame
    """
    return (
        games
        .filter(
            pl.col('week') > 4,
            pl.col('season') >= min_year,
            ~((pl.col('week') > 16) & (pl.col('season') < 2021)),
            ~((pl.col('week') > 17) & (pl.col('season') >= 2021))
        )
    )


def transform_home_away(games):
    """Transform the home and away teams in the games dataframe.

    :param pl.LazyFrame games: The games dataframe.
    :return: A transformed games dataframe with object and advantage teams.
    :rtype: pl.LazyFrame
    """
    games = games.sort('game_id')
    away_obj_games = (
        games
        .gather_every(2)
        .rename(lambda col: col.replace('away', 'obj'))
        .rename(lambda col: col.replace('home', 'adv'))
        .with_columns(
            pl.col("result") * -1,
            obj_team_is_home=0
        )
    )
    home_obj_games = (
        games
        .gather_every(2, offset=1)
        .rename(lambda col: col.replace('home', 'obj'))
        .rename(lambda col: col.replace('away', 'adv'))
        .with_columns(obj_team_is_home=1)
        .select(away_obj_games.collect_schema().names())
    )
    return (
        pl.concat([away_obj_games, home_obj_games], how='vertical_relaxed')
        .sort('game_id')
    )


def get_posteam_defteam_map(games):
    """Get the mapping of posteam and defteam from the games dataframe.

    :param pl.LazyFrame games: The games dataframe.
    :return: A dataframe with posteam and defteam mapping.
    :rtype: pl.LazyFrame
    """
    posteams = games.select(
        pl.col('pfr', 'season', 'week'),
        pl.col('obj_team').alias('posteam'),
        pl.col('adv_team').alias('defteam')
    )
    other_posteams = games.select(
        pl.col('pfr', 'season', 'week'),
        pl.col('adv_team').alias('posteam'),
        pl.col('obj_team').alias('defteam')
    )
    return (
        pl.concat([posteams, other_posteams], how='vertical')
        .select('posteam', 'season', 'week', 'defteam', 'pfr')
        .sort('posteam', 'season', 'week')
    )


def get_game_outcomes(games):
    """Get the scores data from the games dataframe.

    :param pl.LazyFrame games: The games dataframe.
    :return: A dataframe with object team, advantage team, season, week,
             object score, and advantage score.
    :rtype: pl.LazyFrame
    """
    return games.select(
        'obj_team', 'adv_team', 'season', 'week', 'obj_score', 'adv_score'
    )


def fix_pfr_team_names(df):
    """Fix the team names in the dataframe to match PFR standards.

    :param pl.LazyFrame df: The dataframe with team names to fix.
    :return: A dataframe with fixed team names.
    :rtype: pl.LazyFrame
    """
    mapping = {
        'GNB': 'GB',
        'KAN': 'KC',
        'LAR': 'LA',
        'LVR': 'LV',
        'NOR': 'NO',
        'NWE': 'NE',
        'SDG': 'SD',
        'SFO': 'SF',
        'TAM': 'TB',
    }
    return df.with_columns(
        pl.col('team').replace(mapping)
    )


if __name__ == '__main__':
    from src.data.raw.games import refresh_games_data
    from src.data.features.play_stats import build_play_stats_features
    from src.data.features.pythag_exp import build_pythag_features
    from src.data.features.qb_stats import build_qb_stats_features
    from src.config.config import (TRAINING,
                                   RAW_DATA_URLS,
                                   PATHS)

    raw_games_path = PATHS['raw_games']
    boxscore_stats_path = PATHS['boxscore_stats']
    train_db_path = PATHS['train_db']
    min_year = TRAINING['min_year']
    holdout_year_start = TRAINING['holdout_year_start']
    games_url = RAW_DATA_URLS['games']


    print('Refreshing raw games data...')
    refresh_games_data(games_url, raw_games_path)


    print('Loading and processing raw data...')
    raw_games = pl.scan_csv(raw_games_path)
    games = clean_raw_games(raw_games)
    games = transform_home_away(games)
    posteam_defteam_map = get_posteam_defteam_map(games)
    scores = get_game_outcomes(games)
    with sqlite3.connect(boxscore_stats_path) as conn:
        player_offense = (
            pl.read_database(
                query="select * from player_offense",
                connection=conn,
            )
            .lazy()
            .pipe(fix_pfr_team_names)
            .join(
                posteam_defteam_map,
                left_on=['pfr', 'team'],
                right_on=['pfr', 'posteam'],
                how='left'
            )
            .rename({'team': 'posteam'})
        )
        drives = (
            pl.read_database(
                query="select * from drives",
                connection=conn,
            )
            .lazy()
            .pipe(fix_pfr_team_names)
            .join(
                posteam_defteam_map,
                left_on=['pfr', 'team'],
                right_on=['pfr', 'posteam'],
                how='left'
            )
            .rename({'team': 'posteam'})
        )
        starters = (
            pl.read_database(
                query="select * from starters",
                connection=conn,
                infer_schema_length=None,
            )
            .lazy()
            .pipe(fix_pfr_team_names)
            .join(
                posteam_defteam_map,
                left_on=['pfr', 'team'],
                right_on=['pfr', 'posteam'],
                how='left'
            )
            .rename({'team': 'posteam'})
        )


    print('Building features...')
    features = (
        games
        .select(
            'game_id', 'season', 'week', 'obj_team', 'adv_team', 'result',
            'obj_team_is_home',
            rest_net=pl.col('obj_rest') - pl.col('adv_rest')
        )
        .pipe(build_play_stats_features, drives=drives)
        .pipe(build_pythag_features, scores=scores)
        .pipe(build_qb_stats_features, player_offense=player_offense,
              starters=starters)
        .pipe(reduce_games, min_year=min_year)
        .sort('game_id')
    )
    # print(features.collect().glimpse())


    print('Writing train and test datasets...')
    all_training_data = (
        features
        .with_columns(
            pl.when(pl.col('result') > 0).then(1).otherwise(0).alias('target')
        )
        .sort('season', 'week')
        .drop(['week', 'obj_team', 'adv_team', 'result', 'game_id'])
    )
    train = all_training_data.filter(pl.col('season') < holdout_year_start)
    test = all_training_data.filter(pl.col('season') >= holdout_year_start)
    train.collect().write_database(
        table_name='train',
        connection=f"sqlite:///{train_db_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )
    test.collect().write_database(
        table_name='test',
        connection=f"sqlite:///{train_db_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )
