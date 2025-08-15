import os
from time import sleep

import requests
from bs4 import BeautifulSoup
import polars as pl



def fetch_boxscore_html(pfr_game_id):
    """"""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
    }
    url = f"https://www.pro-football-reference.com/boxscores/{pfr_game_id}.htm"
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.text, "html.parser").prettify()


if __name__ == "__main__":
    from src.config.config import PATHS

    raw_games_path = PATHS['raw_games']
    pfr_data_path = PATHS['pfr_data']

    games = pl.scan_csv(raw_games_path)
    games = games.filter(
        pl.col('game_type') == 'REG',
        pl.col('result').is_not_null(),
    )

    pfr_game_ids = games.collect().get_column('pfr').to_list()
    existing_pfr_game_ids = [f[:-5] for f in os.listdir(pfr_data_path)]
    # print(existing_pfr_game_ids)

    for game_id in pfr_game_ids:
        if game_id not in existing_pfr_game_ids:
            print(f"Fetching boxscore for game ID: {game_id}")
            sleep(6.0042069)
            html = fetch_boxscore_html(game_id)
            with open(pfr_data_path / f"{game_id}.html", "w") as f:
                f.write(html)
