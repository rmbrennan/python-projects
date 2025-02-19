import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error

# FPL API Endpoints
BASE_URL = "https://fantasy.premierleague.com/api/"
BOOTSTRAP_URL = BASE_URL + "bootstrap-static/"
EVENT_LIVE_URL = BASE_URL + "event/{}/live/"

def fetch_bootstrap_data():
    response = requests.get(BOOTSTRAP_URL)
    response.raise_for_status()
    return response.json()

def fetch_gameweek_data(gameweek):
    response = requests.get(EVENT_LIVE_URL.format(gameweek))
    response.raise_for_status()
    return response.json()

def calculate_error_metrics(historical_gameweeks):
    bootstrap_data = fetch_bootstrap_data()
    players_df = pd.DataFrame(bootstrap_data['elements'])

    all_actual_points = []
    all_expected_points = []

    for gw in historical_gameweeks:
        gw_data = fetch_gameweek_data(gw)
        elements = gw_data['elements']

        for player in elements:
            player_id = player['id']
            actual_points = player['stats']['total_points']
            expected_points = players_df.loc[players_df['id'] == player_id, 'ep_next'].values
            
            if len(expected_points) > 0:
                all_actual_points.append(actual_points)
                all_expected_points.append(float(expected_points[0]))

    mae = mean_absolute_error(all_actual_points, all_expected_points)
    rmse = np.sqrt(mean_squared_error(all_actual_points, all_expected_points))
    
    return mae, rmse

# Example usage
historical_gws = list(range(1, 5))  # Analyze first 4 gameweeks as an example
mae, rmse = calculate_error_metrics(historical_gws)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")