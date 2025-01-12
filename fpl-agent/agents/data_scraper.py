# Standard library imports
import logging
from typing import Dict, Optional, Any
import json
from datetime import datetime

# Third-party imports
import pandas as pd
import requests
from requests.exceptions import RequestException
import numpy as np
import requests

# Local imports (assuming project structure from earlier)
from utils.constants import API_ENDPOINTS, POSITION_MAP
from .base_agent import FPLAgent, AgentConfig
from models.exceptions import DataFetchError

class DataScraperAgent(FPLAgent):
    def __init__(self, user_team_id: Optional[int] = None):
        super().__init__()  # Remove user_team_id from the super call
        self.user_team_id = user_team_id  # Set user_team_id as an instance variable
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self.config = AgentConfig(
            name="DataScraper",
            provides={"player_data", "fixture_data"},
            requires=set()  # No dependencies
        )

    def process(self) -> Dict:
        try:
            if not self.user_team_id:
                raise ValueError("User  team ID is required for DataScraperAgent.")
            
            data = self._fetch_data(self.user_team_id)
            return data  # Return the fetched data
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise DataFetchError(f"Failed to fetch FPL data: {e}")
        
    def validate(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}bootstrap-static/")
            return response.status_code == 200
        except RequestException as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False
        
    def _fetch_data(self, user_team_id: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from FPL API and return processed DataFrames
        """
        try:
            # Get basic data
            response = self.session.get(f"{self.base_url}bootstrap-static/")
            response.raise_for_status()
            basic_data = response.json()
            
            # Get fixtures
            fixtures_response = self.session.get(f"{self.base_url}fixtures/")
            fixtures_response.raise_for_status()
            fixtures_data = fixtures_response.json()
            
            # Process players data
            players_df = pd.DataFrame(basic_data['elements'])
            players_df['now_cost'] = players_df['now_cost'] / 10  # Convert to millions
            
            # Process fixtures data
            fixtures_df = pd.DataFrame(fixtures_data)

            # Process teams data
            teams_df = pd.DataFrame(basic_data['teams'])
            
            # Get current gameweek
            current_gw = next(
                (gw['id'] for gw in basic_data['events'] 
                 if gw['is_current'] == True),
                None
            )
            
            # Fetch user team data
            user_team_data = self.fetch_user_team(user_team_id, current_gw)
            user_team_df = self.process_user_team(user_team_data, players_df, teams_df)
            
            return {
                'all_players': players_df,
                'fixtures': fixtures_df,
                'teams': teams_df,
                'events': pd.DataFrame(basic_data['events']),
                'latest_gameweek': current_gw,
                'timestamp': pd.Timestamp.now(),
                'user_team': user_team_df,
            }
        except RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise DataFetchError(f"API request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            raise DataFetchError(f"Failed to process API response: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise DataFetchError(f"Unexpected error while fetching data: {str(e)}")
    
    def fetch_user_team(self, user_team_id: int, gameweek: int) -> Dict:
        """
        Fetch the user's current FPL team for a specific gameweek.
        """
        user_team_url = f"{self.base_url}entry/{user_team_id}/event/{gameweek}/picks/"
        try:
            response = self.session.get(user_team_url)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            self.logger.error(f"Error fetching user team data for GW {gameweek}: {e}")
            return {}
        
    def get_position_on_field(self, position: int) -> str:
        """
        Map position on field to a string representation.
        """
        position_mapping = POSITION_MAP
        return position_mapping.get(position, 'Unknown Position')

    def process_user_team(self, user_team_data: Dict, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the user's team data and cross-reference it with player data.
        """
        try:
            if not user_team_data:
                raise ValueError("No user team data found.")

            # Extract player picks from the user team data
            user_team = user_team_data.get('picks', [])
            user_team_details = []

            # Cross-reference player IDs with the players DataFrame
            for player in user_team:
                player_data = players_df[players_df['id'] == player['element']].to_dict('records')
                if player_data:
                    player_info = player_data[0]
                    team_id = player_info.get("team", 0)
                    team_name = teams_df.loc[teams_df['id'] == team_id, 'name'].iloc[0]
                    user_team_details.append({
                        'id': player['element'],
                        "name": player_info.get("web_name", "Unknown"),
                        "team_name": team_name,
                        "position": self.get_position_on_field(player_info.get("element_type", "Unknown Position")),
                        "price": player_info.get("now_cost", 0),
                        "total_points": player_info.get("total_points", 0),
                        "minutes": player_info.get("minutes", 0),
                        "goals_scored": player_info.get("goals_scored", 0),
                        "assists": player_info.get("assists", 0),
                        "clean_sheets": player_info.get("clean_sheets", 0),
                        "ownership": player_info.get("selected_by_percent", "0.0"),
                        "multiplier": player['multiplier'],
                        "is_captain": player['is_captain'],
                        "is_vice_captain": player['is_vice_captain']
                    })

            # Convert user team details into a DataFrame
            user_team_df = pd.DataFrame(user_team_details)

            # Reorder columns for better readability
            user_team_df = user_team_df[[
                "id", "name", "team_name", "position", "price", "total_points", "minutes",
                "goals_scored", "assists", "clean_sheets", "ownership", 
                "multiplier", "is_captain", "is_vice_captain"
            ]]

            return user_team_df

        except Exception as e:
            self.logger.error(f"Error processing user team data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of errors