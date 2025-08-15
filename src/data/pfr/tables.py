import os

from bs4 import BeautifulSoup
from bs4 import Comment
import polars as pl
import pandas as pd


def extract_row(tr):
    """"""
    return [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]


def extract_player_offense_table(soup, pfr_id):
    """"""
    table = soup.find("table", {"id": "player_offense"})
    all_rows = []
    for tr in table.find_all("tr"):
        row = extract_row(tr)
        if row[0] != "" and row[0] != "Player":
            all_rows.append(row)
    col_names = [
        "player",
        "team",
        "completions",
        "pass_attempts",
        "pass_yards",
        "pass_td",
        "interceptions",
        "sacks",
        "sack_yards",
        "pass_long",
        "qb_rating",
        "rush_attempts",
        "rush_yards",
        "rush_td",
        "rush_long",
        "targets",
        "receptions",
        "rec_yards",
        "rec_td",
        "rec_long",
        "fumbles",
        "fumbles_lost",
    ]
    return (
        pl.LazyFrame(all_rows, schema=col_names, orient="row")
        .with_columns(
            pl.col(col_names[2:10]).cast(pl.Int32, strict=False),
            pl.col('qb_rating').cast(pl.Float64, strict=False),
            pl.col(col_names[11:]).cast(pl.Int32, strict=False),
            pfr=pl.lit(pfr_id)
        )
    )


def extract_team_stats_table(soup, pfr_id):
    """"""
    table = soup.find("table", {"id": "team_stats"})
    all_rows = [extract_row(tr) for tr in table.find_all("tr")]
    away_team, home_team = all_rows.pop(0)[1:3]
    col_names = ['stat', away_team, home_team]
    team_stats = (
        # lazyframes don't have a transpose method, so we use a dataframe here
        pl.DataFrame(all_rows, schema=col_names, orient="row")
        .transpose(include_header=True, header_name="team", column_names="stat")
        .lazy()
        .with_columns(
            pl.col('First Downs', 'Net Pass Yards', 'Total Yards', 'Turnovers').cast(pl.Int32),
            pfr=pl.lit(pfr_id),
        )
    )
    return team_stats, away_team, home_team


def extract_drives_table(soup, away_team, home_team, pfr_id):
    """"""
    away_table = soup.find("table", {"id": "vis_drives"})
    away_rows = [extract_row(tr) for tr in away_table.find_all("tr")]
    col_names = away_rows.pop(0)
    col_names[0] = "num"
    away_drive = (
        pl.LazyFrame(away_rows, schema=col_names, orient="row")
        .with_columns(
            pl.col('num', 'Quarter', 'Plays', 'Net Yds').cast(pl.Int32, strict=False),
            team=pl.lit(away_team),
            pfr=pl.lit(pfr_id)
        )
    )
    home_table = soup.find("table", {"id": "home_drives"})
    home_rows = [extract_row(tr) for tr in home_table.find_all("tr")]
    col_names = home_rows.pop(0)
    col_names[0] = "num"
    home_drive = (
        pl.LazyFrame(home_rows, schema=col_names, orient="row")
        .with_columns(
            pl.col('num', 'Quarter', 'Plays', 'Net Yds').cast(pl.Int32, strict=False),
            team=pl.lit(home_team),
            pfr=pl.lit(pfr_id)
        )
    )
    return pl.concat([away_drive, home_drive], how="vertical")


def extract_starters_table(soup, away_team, home_team, pfr_id):
    """"""
    away_table = soup.find("table", {"id": "vis_starters"})
    away_rows = [extract_row(tr) for tr in away_table.find_all("tr")]
    col_names = away_rows.pop(0)
    away_starters = (
        pl.DataFrame(away_rows, schema=col_names, orient="row")
        .with_columns(
            pl.col('Pos').cum_count().over('Pos').alias('num'),
        )
        .with_columns(
            pl.col('Pos') + "_" + pl.col('num').cast(pl.Utf8),
        )
        .select('Pos', 'Player')
        .transpose(column_names="Pos")
        .lazy()
        .with_columns(
            team=pl.lit(away_team),
            pfr=pl.lit(pfr_id),
        )
    )
    home_table = soup.find("table", {"id": "home_starters"})
    home_rows = [extract_row(tr) for tr in home_table.find_all("tr")]
    col_names = home_rows.pop(0)
    home_starters = (
        pl.DataFrame(home_rows, schema=col_names, orient="row")
        .with_columns(
            pl.col('Pos').cum_count().over('Pos').alias('num'),
        )
        .with_columns(
            pl.col('Pos') + "_" + pl.col('num').cast(pl.Utf8),
        )
        .select('Pos', 'Player')
        .transpose(column_names="Pos")
        .lazy()
        .with_columns(
            team=pl.lit(home_team),
            pfr=pl.lit(pfr_id),
        )
    )
    return pl.concat([away_starters, home_starters], how="diagonal")


if __name__ == "__main__":
    from src.config.config import PATHS

    pfr_path = PATHS['pfr_data']
    boxscore_stats_path = PATHS['boxscore_stats']

    player_offense_tables = []
    team_stats_tables = []
    drives_tables = []
    starters_tables = []
    file_list = [f for f in os.listdir(pfr_path) if f.endswith('.html')]
    for file in sorted(file_list):
        print(f"Processing file: {file}")
        with open(pfr_path / file, 'r') as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        pfr_id = file[:-5]
        t = extract_player_offense_table(soup, pfr_id)
        player_offense_tables.append(t)
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        html = "\n".join([str(c) for c in comments])
        soup = BeautifulSoup(html, "html.parser")
        t, away_team, home_team = extract_team_stats_table(soup, pfr_id)
        team_stats_tables.append(t)
        t = extract_drives_table(soup, away_team, home_team, pfr_id)
        drives_tables.append(t)
        t = extract_starters_table(soup, away_team, home_team, pfr_id)
        starters_tables.append(t)



    # player offense
    player_offense = (
        pl.concat(player_offense_tables, how="vertical")
        .sort(
            ["pfr", "team", "pass_attempts", "rush_attempts", "targets"],
            descending=[False, False, True, True, True],
        )
        .fill_null(0)
    )
    print(player_offense.collect().glimpse())
    player_offense.collect().write_database(
        table_name='player_offense',
        connection=f"sqlite:///{boxscore_stats_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )

    # team stats
    team_stats = (
        pl.concat(team_stats_tables, how="vertical")
        .sort(["pfr", "team"], descending=[False, False])
    )
    print(team_stats.collect())
    team_stats.collect().write_database(
        table_name='team_stats',
        connection=f"sqlite:///{boxscore_stats_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )

    # drives
    drives = (
        pl.concat(drives_tables, how="vertical")
        .sort(["pfr", "team", "num"], descending=[False, False, False])
    )
    print(drives.collect())
    drives.collect().write_database(
        table_name='drives',
        connection=f"sqlite:///{boxscore_stats_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )

    # starters
    starters = (
        pl.concat(starters_tables, how="diagonal")
        .select(
            pl.col('team'),
            pl.all().exclude('team', 'pfr'),
            pl.col('pfr'),
        )
        .sort(["pfr", "team"], descending=[False, False])
    )
    print(starters.collect())
    starters.collect().write_database(
        table_name='starters',
        connection=f"sqlite:///{boxscore_stats_path}",
        engine='sqlalchemy',
        if_table_exists='replace'
    )
