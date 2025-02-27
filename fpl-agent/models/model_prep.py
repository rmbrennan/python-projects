import pandas as pd
import time
from tqdm import tqdm
from utils.utils import DataFetcher
from utils.constants import API_ENDPOINTS

data_fetcher = DataFetcher()
player_df = data_fetcher.load_player_data()

# Define the rolling window sizes
window_sizes = [5, 3, 1]  # 5 GWs, 3 GWs, and previous GW
max_window_size = max(window_sizes)

# TEMP - Filter players from a single team
player_df = player_df[player_df['team'] == 1]

# TEMP - Set the current gameweek to 8
current_gw = 20
# current_gw = data_fetcher.get_current_gameweek()

# Prefetch all required gameweeks at once to minimize API calls
print(f"Prefetching all required gameweeks...")
data_fetcher.load_gameweek_range(current_gw - max_window_size, current_gw)
print(f"Prefetching complete!")

# Get unique player IDs from filtered dataframe
player_ids = player_df['id'].unique().tolist()

# Initialize an empty DataFrame to store the modeling dataset
modeling_data = pd.DataFrame()

# Stats to aggregate - Add 'total_points' to the list
stats_to_track = [
    'total_points',  # Add this line to fix the error
    'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 'influence',
    'creativity', 'threat', 'ict_index', 'expected_goals', 'expected_assists',
    'expected_goal_involvements', 'expected_goals_conceded'
]

# Progress tracking
total_iterations = (current_gw - max_window_size) * len(player_df)
print(f"Processing {len(player_df)} players across {current_gw - max_window_size} gameweeks...")

# Iterate over each gameweek starting from the cutoff
for gw in tqdm(range(max_window_size + 1, current_gw + 1), desc=f"Processing gameweeks"):
    # Initialize a list to store player data for the current gameweek
    gw_player_data = []
    
    # Track time for performance monitoring
    start_time = time.time()
    
    # Aggregate data for each player over different window sizes
    for index, player in tqdm(player_df.iterrows(), total=len(player_df), desc=f"GW {gw}: Processing players", leave=False):
        player_id = player['id']
        player_name = player['web_name']

        # Initialize a dictionary to store player data
        player_data_entry = {
            'player_id': player_id,
            'player_name': player_name,
            'gameweek': gw,
        }

        # Get target gameweek data (this is now cached)
        target_gw_data = data_fetcher.get_player_data_by_id(gw, player_id)
        if not target_gw_data or 'stats' not in target_gw_data:
            # Skip players who didn't play in the target gameweek
            continue
            
        player_data_entry['actual_points'] = target_gw_data['stats'].get('total_points', 0)
        
        # For each window size, process the past gameweeks
        for window_size in window_sizes:
            # Process all stats at once for this window
            stats_sum = {stat: 0 for stat in stats_to_track}
            games_played = 0
            
            # Process the past N gameweeks
            for past_gw in range(gw - window_size, gw):
                # Get player data for this past gameweek (now cached)
                player_past_data = data_fetcher.get_player_data_by_id(past_gw, player_id)
                
                if player_past_data and 'stats' in player_past_data:
                    player_stats = player_past_data['stats']
                    
                    # Process all stats at once
                    for stat in stats_to_track:
                        if stat in player_stats:
                            stat_value = player_stats[stat]
                            # Handle type conversion
                            if isinstance(stat_value, str):
                                try:
                                    stat_value = float(stat_value)
                                except ValueError:
                                    stat_value = 0
                            stats_sum[stat] += stat_value
                    
                    # Count games played (if minutes > 0)
                    if player_stats.get('minutes', 0) > 0:
                        games_played += 1
            
            # Store the aggregated stats for this window
            player_data_entry[f'games_played_{window_size}gw'] = games_played
            
            # Calculate the sum and average for each stat
            for stat in stats_to_track:
                # Store sum
                player_data_entry[f'{stat}_sum_{window_size}gw'] = stats_sum[stat]
                
                # Calculate and store average (if games played)
                if games_played > 0:
                    player_data_entry[f'{stat}_avg_{window_size}gw'] = stats_sum[stat] / games_played
                else:
                    player_data_entry[f'{stat}_avg_{window_size}gw'] = 0
                
                # For 1gw window, also store as "previous" for clarity
                if window_size == 1:
                    player_data_entry[f'{stat}_previous'] = stats_sum[stat]
        
        # Add this player's data to the current gameweek collection
        gw_player_data.append(player_data_entry)
    
    # Calculate processing time for this gameweek
    end_time = time.time()
    print(f"GW {gw}: Processed {len(gw_player_data)} players in {end_time - start_time:.2f} seconds")
    
    # Add to modeling dataset
    if gw_player_data:
        modeling_data = pd.concat([modeling_data, pd.DataFrame(gw_player_data)], ignore_index=True)
    
    # Checkpoint every 2 gameweeks
    # if gw % 2 == 0:
    #     checkpoint_file = f"modeling_data_checkpoint_gw{gw}.csv"
    #     modeling_data.to_csv(checkpoint_file)
    #     print(f"Saved checkpoint at gameweek {gw} to {checkpoint_file}")

# Add any additional features
modeling_data['points_per_minute_5gw'] = modeling_data.apply(
    lambda row: row['total_points_sum_5gw'] / row['minutes_sum_5gw'] if row['minutes_sum_5gw'] > 0 else 0, 
    axis=1
)

# Final save
# modeling_data.to_csv("modeling_data_final.csv")
print(f"Created modeling dataset with {len(modeling_data)} rows and {len(modeling_data.columns)} features")
print(modeling_data.head())