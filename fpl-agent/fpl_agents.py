import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import logging

class FPLAgent(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = "https://fantasy.premierleague.com/api/"  # Shared base URL

    def fetch_data(self, endpoint: str) -> Dict:
        """
        Fetch data from the FPL API for the given endpoint.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data from {url}: {e}")
            return {}

    @abstractmethod
    async def process(self, data: Dict) -> Dict:
        """Process data and return results"""
        pass

class DataScraperAgent(FPLAgent):
    def __init__(self):
        super().__init__()

    def fetch_user_team(self, user_team_id: int, gameweek: int) -> Dict:
        """
        Fetch the user's current FPL team for a specific gameweek.
        """
        user_team_url = f"{self.base_url}entry/{user_team_id}/event/{gameweek}/picks/"
        try:
            response = requests.get(user_team_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching user team data for GW {gameweek}: {e}")
            return {}

    def get_latest_gameweek(self, events: List[Dict]) -> int:
        """
        Find the latest gameweek from the events data.
        """
        for event in events:
            if event.get("is_current", False):
                return event["id"]
        self.logger.error("No current gameweek found in events data.")
        return -1  # Indicate failure to find the current gameweek

    async def process(self, data: Dict) -> Dict:
        """Fetch and process FPL player and user team data."""
        try:
            # Fetch general data
            bootstrap_data = self.fetch_data("bootstrap-static/")
            elements = bootstrap_data.get("elements", [])
            teams = bootstrap_data.get("teams", [])
            events = bootstrap_data.get("events", [])

            # Get the latest gameweek
            latest_gameweek = self.get_latest_gameweek(events)
            if latest_gameweek == -1:
                raise ValueError("Failed to find the current gameweek.")

            # Fetch user team for the latest gameweek
            user_team_id = data.get("user_team_id", 1)  # Default to a valid user team ID
            user_team_data = self.fetch_user_team(user_team_id, latest_gameweek)

            # Process player data into a DataFrame
            players_df = pd.DataFrame(elements)
            players_df['team_name'] = players_df['team'].map(
                pd.DataFrame(teams).set_index('id')['name']
            )

            # Include only relevant fields for simplicity
            players_df = players_df[[
                'id', 'web_name', 'team_name', 'element_type', 'now_cost', 
                'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 
                'selected_by_percent'
            ]]

            # Rename columns for clarity
            players_df = players_df.rename(columns={
                'web_name': 'name',
                'element_type': 'position',
                'now_cost': 'price',
                'selected_by_percent': 'ownership'
            })

            # Map position IDs to position names
            position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position'] = players_df['position'].map(position_map)

            # Map user's team data to player information
            user_team = user_team_data.get('picks', [])
            user_team_details = []
            for player in user_team:
                player_data = players_df[players_df['id'] == player['element']].to_dict('records')
                if player_data:
                    player_info = player_data[0]
                    player_info.update({
                        "position": player['position'],
                        "multiplier": player['multiplier'],
                        "is_captain": player['is_captain'],
                        "is_vice_captain": player['is_vice_captain']
                    })
                    user_team_details.append(player_info)

            return {
                'players': players_df,
                'user_team': user_team_details,
                'latest_gameweek': latest_gameweek,
                'timestamp': pd.Timestamp.now(),
            }
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

class TeamOptimizationAgent(FPLAgent):
    def __init__(self):
        super().__init__()
        self.budget = 100.0
        self.position_limits = {
            'GKP': (2, 2),
            'DEF': (5, 5),
            'MID': (5, 5),
            'FWD': (3, 3)
        }

    def _calculate_expected_points(self, players_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.Series:
        try:
            # Simple expected points based on form and fixtures
            points = players_df['total_points'] / players_df['minutes'] * 90
            return points.fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating expected points: {e}")
            return pd.Series(0, index=players_df.index)

    def _select_optimal_team(self, players_df: pd.DataFrame) -> List[Dict]:
        try:
            selected_team = []
            remaining_budget = self.budget
            
            for position, (min_count, max_count) in self.position_limits.items():
                position_players = players_df[players_df['position'] == position].copy()
                position_players = position_players.nlargest(max_count, 'value')
                
                for _, player in position_players.iterrows():
                    if len(selected_team) < 15 and player['price']/10 <= remaining_budget:
                        selected_team.append(player.to_dict())
                        remaining_budget -= player['price']/10
            
            return selected_team
        except Exception as e:
            self.logger.error(f"Error selecting optimal team: {e}")
            return []

    def _calculate_remaining_budget(self, team: List[Dict]) -> float:
        try:
            spent = sum(player['price']/10 for player in team)
            return self.budget - spent
        except Exception as e:
            self.logger.error(f"Error calculating budget: {e}")
            return 0.0

    async def process(self, data: Dict) -> Dict:
            """
            Optimize and suggest a team based on player and fixture data.
            """
            try:
                self.logger.info("Starting team optimization.")

                # Validate required keys
                if 'players' not in data or 'fixtures' not in data:
                    raise KeyError("Missing required data: 'players' and/or 'fixtures'.")

                players = data['players']
                fixtures = data['fixtures']

                # Example optimization logic: pick top 15 players by total points
                recommended_team = sorted(players, key=lambda x: x.get('total_points', 0), reverse=True)[:15]

                # Construct detailed output
                processed_data = {
                    "recommended_team": recommended_team,
                    "timestamp": datetime.now().isoformat(),
                }

                self.logger.info("Team optimization complete.")
                return processed_data

            except Exception as e:
                self.logger.error(f"Error during team optimization: {e}")
                return {"error": str(e)}
        
class TransferAgent(FPLAgent):
    def __init__(self):
        super().__init__()

    def _analyze_transfers(self, current_team: List[Dict], optimal_team: List[Dict], 
                         available_transfers: int) -> List[Tuple[Dict, Dict]]:
        try:
            if not current_team or not optimal_team:
                return []

            transfers = []
            for curr_player in current_team:
                for opt_player in optimal_team:
                    if (curr_player['position'] == opt_player['position'] and 
                        curr_player['name'] != opt_player['name']):
                        transfers.append((curr_player, opt_player))
                        
            return transfers[:available_transfers]
        except Exception as e:
            self.logger.error(f"Transfer analysis failed: {e}")
            return []

    def _calculate_point_impact(self, transfers: List[Tuple[Dict, Dict]]) -> float:
        try:
            return sum(new['expected_points'] - old['expected_points'] 
                      for old, new in transfers if 'expected_points' in new and 'expected_points' in old)
        except Exception as e:
            self.logger.error(f"Point impact calculation failed: {e}")
            return 0.0

    def _calculate_transfer_cost(self, num_transfers: int, free_transfers: int) -> int:
        try:
            extra_transfers = max(0, num_transfers - free_transfers)
            return extra_transfers * 4
        except Exception as e:
            self.logger.error(f"Transfer cost calculation failed: {e}")
            return 0

    async def process(self, data: Dict) -> Dict:
        try:
            current_team = data.get('current_team', [])
            optimal_team = data.get('optimal_team', [])
            free_transfers = data.get('free_transfers', 1)
            
            transfers = self._analyze_transfers(current_team, optimal_team, free_transfers)
            
            return {
                'suggested_transfers': transfers,
                'expected_point_gain': self._calculate_point_impact(transfers),
                'transfer_cost': self._calculate_transfer_cost(len(transfers), free_transfers)
            }
        except Exception as e:
            self.logger.error(f"Transfer processing failed: {e}")
            return {'suggested_transfers': [], 'expected_point_gain': 0, 'transfer_cost': 0}

class CaptainAgent(FPLAgent):
    def __init__(self):
        super().__init__()

    def _rank_captain_choices(self, team: List[Dict], fixtures: pd.DataFrame) -> List[Dict]:
        try:
            if not team:
                return []
            
            for player in team:
                player['captain_score'] = player.get('expected_points', 0)
            
            return sorted(team, key=lambda x: x.get('captain_score', 0), reverse=True)
        except Exception as e:
            self.logger.error(f"Captain ranking failed: {e}")
            return []

    def _generate_captain_reasoning(self, player: Optional[Dict]) -> str:
        try:
            if not player:
                return "No valid captain choice available"
            return f"Selected based on expected points and fixture difficulty"
        except Exception as e:
            self.logger.error(f"Captain reasoning generation failed: {e}")
            return "Unable to generate reasoning"

    async def process(self, data: Dict) -> Dict:
        try:
            team = data.get('team', [])
            fixtures = data.get('fixtures', pd.DataFrame())
            
            choices = self._rank_captain_choices(team, fixtures)
            
            return {
                'primary_captain': choices[0] if choices else None,
                'vice_captain': choices[1] if len(choices) > 1 else None,
                'alternative_options': choices[2:],
                'reasoning': self._generate_captain_reasoning(choices[0] if choices else None)
            }
        except Exception as e:
            self.logger.error(f"Captain selection failed: {e}")
            return {
                'primary_captain': None,
                'vice_captain': None,
                'alternative_options': [],
                'reasoning': "Captain selection failed"
            }
        
# Orchestrator
class FPLOrchestrator:
    def __init__(self):
        self.data_scraper = DataScraperAgent()
        self.team_optimizer = TeamOptimizationAgent()
        self.transfer_advisor = TransferAgent()
        self.captain_advisor = CaptainAgent()
        self.logger = logging.getLogger("Orchestrator")

    async def run_workflow(self, user_team_id: int) -> Dict:
        """Run the complete FPL optimization workflow"""
        try:
            # Step 1: Gather Data
            self.logger.info("Starting data collection...")
            raw_data = await self.data_scraper.process({})
            
            # Step 2: Optimize Ideal Team
            self.logger.info("Optimizing ideal team...")
            optimization_result = await self.team_optimizer.process(raw_data)
            
            # Step 3: Generate Transfer Suggestions
            self.logger.info("Analyzing transfers...")
            transfer_suggestions = await self.transfer_advisor.process({
                'current_team': raw_data['current_team'],
                'optimal_team': optimization_result['optimal_team'],
                'free_transfers': raw_data.get('free_transfers', 1)
            })
            
            # Step 4: Captain Selection
            self.logger.info("Selecting captain...")
            captain_advice = await self.captain_advisor.process({
                'team': raw_data['current_team'],
                'fixtures': raw_data['fixtures']
            })
            
            return {
                'optimization_result': optimization_result,
                'transfer_suggestions': transfer_suggestions,
                'captain_advice': captain_advice,
                'analysis_timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow error: {e}")
            raise

# Usage Example
async def main():
    scraper = DataScraperAgent()
    user_team_id = 6256406  # My FPL team ID
    try:
        data = await scraper.process({"user_team_id": user_team_id})
        print(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())