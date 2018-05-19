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

    :return: parsed json player data
    """
    re_compiled = re.compile(regex)
    match = re_compiled.search(html)
    if match is None:
        raise ValueError('Could not find data in rotogrinders scrape')

    return json.loads(match.group(1))


def parse_batter_data(player_data, slate_id):
    """
    Parse the scraped data

    :param player_data: json player data
    :param slate_id: the slate to filter to

    :return: a pd.DataFrame with the columns:
        (name, id, team, opp, team_grp, position, order, order_confirmed, salary, proj_mean, proj_std)
    """
    parsed_data = []
    for p in player_data:
        if p['import_data'] is None:
            continue

        if slate_id in [data['slate_id'] for data in p['import_data']]:
            if 'order' not in p:
                continue

            parsed_data.append({
                'name': p['player_name'],
                'id': [data['player_id'] for data in p['import_data'] if data['slate_id'] == slate_id][0],
                'team': p['team'],
                'opp': p['opp'],
                'team_grp': p['team'],
                'position': p['position'],
                'order': p['order'],
                'order_confirmed': p['confirmed'],
                'salary': float(p['salary']),
                'proj_mean': p['points'],
                'proj_std': p['deviation'] if p['deviation'] > 0 else p['points'] / 2,
            })

    return pd.DataFrame(parsed_data)


def scrape_pitcher_data(html, regex='data\s=\s(\[.*\]);'):
    """
    Scrape pitcher data from RotoGrinders

    :param html: projection page html text
    :param regex: the regex to extract the json data from the html

    :return: parsed json player data
    """
    re_compiled = re.compile(regex)
    match = re_compiled.search(html)
    if match is None:
        raise ValueError('Could not find data in rotogrinders scrape')

    return json.loads(match.group(1))


def parse_pitcher_data(player_data, slate_id):
    """
    Parse the scraped data

    :param player_data: json player data
    :param slate_id: the slate to filter to

    :return: a pd.DataFrame with the columns:
        (name, id, team, opp, team_grp, position, order, order_confirmed, salary, proj_mean, proj_std)
    """
    parsed_data = []
    for p in player_data:
        if p['import_data'] is None:
            continue

        if slate_id in [data['slate_id'] for data in p['import_data']]:
            parsed_data.append({
                'name': p['player_name'],
                'id': [data['player_id'] for data in p['import_data'] if data['slate_id'] == slate_id][0],
                'team': p['team'],
                'opp': p['opp'],
                'team_grp': p['opp'],
                'position': p['position'],
                'order': 10,
                'order_confirmed': 1,
                'salary': float(p['salary']),
                'proj_mean': p['points'],
                'proj_std': p['deviation'] if p['deviation'] > 0 else p['points'] / 2,
            })

    return pd.DataFrame(parsed_data)


def choose_slate(player_data):
    """
    Get user input to choose the correct slate

    :param player_data: list of scraped rotogrinders data

    :return: The slate_id
    """
    slates = {}
    for p in player_data:
        if p['import_data'] is None:
            continue

        for slate in p['import_data']:
            if slate['slate_id'] not in slates:
                slates[slate['slate_id']] = {
                    'teams': [p['team']],
                    'type': slate['type'],
                }
            elif p['team'] not in slates[slate['slate_id']]['teams']:
                slates[slate['slate_id']]['teams'].append(p['team'])

    for slate_id, slate_info in slates.items():
        print('{}: {}'.format(slate_info['type'], slate_id))
        print(sorted(slate_info['teams']))

    while True:
        response = input('Enter the slate id: ')

        try:
            response = int(response)
        except ValueError:
            print('Must be the slate_id integer')
            continue

        if response in slates:
            return response


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

    pitcher_data = scrape_pitcher_data(pitcher_data)
    batter_data = scrape_batter_data(batter_data)

    slate_id = choose_slate(pitcher_data)

    df_batter = parse_batter_data(batter_data, slate_id)
    df_pitcher = parse_pitcher_data(pitcher_data, slate_id)

    return pd.concat([df_batter, df_pitcher], ignore_index=True)
