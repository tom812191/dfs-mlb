"""
Scrape RotoGrinders for projections data
"""

import requests
import re
import json
import pandas as pd


def scrape_batter_data(html, regex='data\s=\s(\[.*\]);'):
    """
    Scrape batter data from RotoGrinders

    :param html: projection page html text
    :param regex: the regex to extract the json data from the html

    :return: a pd.DataFrame with the columns:
        (name, id, team, opp, team_grp, position, order, order_confirmed, salary, proj_mean, proj_std)
    """
    re_compiled = re.compile(regex)
    match = re_compiled.search(html)
    if match is None:
        raise ValueError('Could not find data in rotogrinders scrape')

    player_data = json.loads(match.group(1))

    parsed_data = [{
        'name': p['player_name'],
        'id': [data['player_id'] for data in p['import_data'] if data['type'] == 'classic'][0],
        'team': p['team'],
        'opp': p['opp'],
        'team_grp': p['team'],
        'position': p['position'],
        'order': p['order'],
        'order_confirmed': p['confirmed'],
        'salary': float(p['salary']),
        'proj_mean': p['points'],
        'proj_std': p['deviation'] if p['deviation'] > 0 else p['points'] / 2,
    } for p in player_data if 'order' in p and p['import_data'] is not None]

    return pd.DataFrame(parsed_data)


def scrape_pitcher_data(html, regex='data\s=\s(\[.*\]);'):
    """
    Scrape pitcher data from RotoGrinders

    :param html: projection page html text
    :param regex: the regex to extract the json data from the html

    :return: a pd.DataFrame with the columns:
        (name, id, team, opp, team_grp, position, order, order_confirmed, salary, proj_mean, proj_std)
    """
    re_compiled = re.compile(regex)
    match = re_compiled.search(html)
    if match is None:
        raise ValueError('Could not find data in rotogrinders scrape')

    player_data = json.loads(match.group(1))

    parsed_data = [{
        'name': p['player_name'],
        'id': [data['player_id'] for data in p['import_data'] if data['type'] == 'classic'][0],
        'team': p['team'],
        'opp': p['opp'],
        'team_grp': p['opp'],
        'position': p['position'],
        'order': 10,
        'order_confirmed': 1,
        'salary': float(p['salary']),
        'proj_mean': p['points'],
        'proj_std': p['deviation'] if p['deviation'] > 0 else p['points'] / 2,
    } for p in player_data if p['import_data'] is not None]

    return pd.DataFrame(parsed_data)


def scrape_all_data(batter_url, pitcher_url):
    """
    Scrape batter and pitcher data from RotoGrinders

    :param batter_url: url to the batter data
    :param pitcher_url: url to the pitcher data

    :return: a pd.DataFrame with the columns:
        (name, id, team, opp, team_grp, position, order, order_confirmed, salary, proj_mean, proj_std)
    """
    batter_data = requests.get(batter_url).text
    pitcher_data = requests.get(pitcher_url).text

    df_batter = scrape_batter_data(batter_data)
    df_pitcher = scrape_pitcher_data(pitcher_data)

    return pd.concat([df_batter, df_pitcher], ignore_index=True)


if __name__ == '__main__':
    scrape_all_data('https://rotogrinders.com/projected-stats/mlb-hitter?site=draftkings', 'https://rotogrinders.com/projected-stats/mlb-pitcher?site=draftkings')