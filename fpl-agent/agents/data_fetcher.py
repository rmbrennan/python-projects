import os
import time
import pickle
import requests
import pandas as pd
from utils.constants import API_ENDPOINTS  # Import API_ENDPOINTS from constants.py

class DataFetcher:
    def __init__(self, cache_dir='cache'):
        self.base_url = API_ENDPOINTS['base_url']
        self.gameweek_cache = {}
        self.player_data_cache = {}
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Try to load cached data if it exists
        self._load_cache()
    
    def _load_cache(self):
        """Load cached data from disk if available"""
        cache_file = os.path.join(self.cache_dir, 'gameweek_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.gameweek_cache = pickle.load(f)
                print(f"Loaded cache with {len(self.gameweek_cache)} gameweeks")
            except Exception as e:
                print(f"Error loading cache: {e}")
                
        player_cache_file = os.path.join(self.cache_dir, 'player_cache.pkl')
        if os.path.exists(player_cache_file):
            try:
                with open(player_cache_file, 'rb') as f:
                    self.player_data_cache = pickle.load(f)
                print(f"Loaded player cache with {sum(len(players) for players in self.player_data_cache.values())} player entries")
            except Exception as e:
                print(f"Error loading player cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, 'gameweek_cache.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.gameweek_cache, f)
                
            player_cache_file = os.path.join(self.cache_dir, 'player_cache.pkl')
            with open(player_cache_file, 'wb') as f:
                pickle.dump(self.player_data_cache, f)
                
            print(f"Saved cache with {len(self.gameweek_cache)} gameweeks and {sum(len(players) for players in self.player_data_cache.values())} player entries")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def load_gw_event_data(self, gw):
        """Fetch event data with caching"""
        if gw in self.gameweek_cache:
            return self.gameweek_cache[gw]
        
        # If not cached, fetch from API with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = self.base_url + API_ENDPOINTS['event'].format(event_id=gw)
                print(f"Fetching data for GW {gw} from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                events = response.json()
                
                # Cache the result
                self.gameweek_cache[gw] = events
                
                # Also pre-process and cache individual player data
                if gw not in self.player_data_cache:
                    self.player_data_cache[gw] = {}
                    
                for player in events.get('elements', []):
                    player_id = player['id']
                    self.player_data_cache[gw][player_id] = player
                
                # Save to disk after each successful fetch
                self._save_cache()
                return events
                
            except Exception as e:
                print(f"Error fetching GW {gw}, attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to fetch data for GW {gw} after {max_retries} attempts")
                    return None
    
    def get_player_data_by_id(self, gw, player_id):
        """Retrieve player data by ID for a specific gameweek using cache"""
        # Check if we already have this player's data cached
        if gw in self.player_data_cache and player_id in self.player_data_cache[gw]:
            return self.player_data_cache[gw][player_id]
        
        # If not in player cache but gameweek is cached, player doesn't exist in that gameweek
        if gw in self.gameweek_cache:
            return None
            
        # If gameweek not cached, load it
        events = self.load_gw_event_data(gw)
        if not events:
            return None
        
        # Now check if player exists in loaded data
        return self.player_data_cache.get(gw, {}).get(player_id, None)
    
    def load_gameweek_range(self, start_gw, end_gw):
        """Prefetch multiple gameweeks at once"""
        print(f"Prefetching gameweeks {start_gw} to {end_gw}...")
        for gw in tqdm(range(start_gw, end_gw + 1), desc="Loading gameweeks"):
            self.load_gw_event_data(gw)
        print(f"Prefetched {end_gw - start_gw + 1} gameweeks")
    
    def load_player_data(self):
        """Load basic player information"""
        # First try to load from cache
        player_info_file = os.path.join(self.cache_dir, 'player_info.pkl')
        if os.path.exists(player_info_file):
            try:
                with open(player_info_file, 'rb') as f:
                    player_df = pickle.load(f)
                print(f"Loaded player info for {len(player_df)} players from cache")
                return player_df
            except Exception as e:
                print(f"Error loading player info cache: {e}")
        
        # If not cached, fetch from API
        url = self.base_url + API_ENDPOINTS['bootstrap']
        print(f"Fetching player data from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Create dataframe
        player_df = pd.DataFrame(data['elements'])
        
        # Save to cache
        try:
            with open(player_info_file, 'wb') as f:
                pickle.dump(player_df, f)
            print(f"Saved player info for {len(player_df)} players to cache")
        except Exception as e:
            print(f"Error saving player info cache: {e}")
            
        return player_df

    def load_fixtures(self):
        """Fetch fixture data"""
        url = self.base_url + API_ENDPOINTS['fixtures']
        response = requests.get(url)
        response.raise_for_status()
        fixtures = response.json()
        
        return fixtures
    
    def get_current_gameweek(self):
        """Get the current gameweek"""
        url = self.base_url + API_ENDPOINTS['bootstrap']
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find the current gameweek
        for event in data['events']:
            if event['is_current']:
                return event['id']
                
        # If no current gameweek found, return the next one
        for event in data['events']:
            if event['is_next']:
                return event['id'] - 1
                
        # If neither found, return the highest finished gameweek
        max_finished = 0
        for event in data['events']:
            if event['finished'] and event['id'] > max_finished:
                max_finished = event['id']
                
        return max_finished

# Instantiate the DataFetcher
if __name__ == '__main__':
    data_fetcher = DataFetcher()
    current_gw = data_fetcher.get_current_gameweek()
    print(f"Current gameweek: {current_gw}")
    player_data = data_fetcher.load_player_data()
    print(player_data['web_name'])
    gw_data = data_fetcher.load_gw_event_data(current_gw)
    player_id = 328
    player_gw_data = data_fetcher.get_player_data_by_id(current_gw, player_id)
    print(player_gw_data.get('stats', {}))