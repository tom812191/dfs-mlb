"""
Scrape rotogrinders for projections data
"""

import requests
import re
import json
import pandas as pd


def scrape_batter_data(html, regex='data\s=\s(\[.*\]);'):
    """
    Scrape batter data from rotogrinders
    Return a pandas dataframe with the columns:
        (name, team_grp, position, order, salary, proj_mean, proj_std)
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
    } for p in player_data if 'order' in p]

    return pd.DataFrame(parsed_data)


def scrape_pitcher_data(html, regex='data\s=\s(\[.*\]);'):
    """
    Scrape pitcher data from rotogrinders
    Return a pandas dataframe with the columns:
        (name, team_grp, position, order, salary, proj_mean, proj_std)
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
    } for p in player_data]

    return pd.DataFrame(parsed_data)


def scrape_all_data(batter_url, pitcher_url):
    """
    Scrape batter and pitcher data from rotogrinders
    Return a pandas dataframe with the columns:
        (name, team_grp, position, order, salary, proj_mean, proj_std)
    """
    batter_data = requests.get(batter_url).text
    pitcher_data = requests.get(pitcher_url).text

    df_batter = scrape_batter_data(batter_data)
    df_pitcher = scrape_pitcher_data(pitcher_data)

    return pd.concat([df_batter, df_pitcher], ignore_index=True)

if __name__ == '__main__':
    batter_url = 'https://rotogrinders.com/projected-stats/mlb-hitter?site=draftkings'
    pitcher_url = 'https://rotogrinders.com/projected-stats/mlb-pitcher?site=draftkings'

    scrape_all_data(batter_url, pitcher_url)

