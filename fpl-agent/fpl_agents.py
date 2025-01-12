import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

@dataclass
class Player:
    id: int
    name: str
    position: str  # 'GK', 'DEF', 'MID', 'FWD'
    team: str
    price: float
    expected_points: float

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

            # Ensure teams data is present
            if not teams:
                raise ValueError("Teams data is missing from the bootstrap-static dataset.")
            
            # Create a mapping of team IDs to team names
            team_mapping = {team['id']: team['name'] for team in teams}
            
            # Process player data into a DataFrame
            players_df = pd.DataFrame(elements)
            if players_df.empty:
                raise ValueError("No player data available in the bootstrap-static dataset.")
            
            # Add a team_name column using the team mapping
            players_df['team_name'] = players_df['team'].map(team_mapping)
            
            # Include only relevant fields
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
            
            # Fetch user team for the latest gameweek
            latest_gameweek = self.get_latest_gameweek(events)
            if latest_gameweek == -1:
                raise ValueError("Failed to find the current gameweek.")
            
            user_team_id = data.get("user_team_id", 6256406)  # Default to a valid user team ID
            user_team_data = self.fetch_user_team(user_team_id, latest_gameweek)
            
            # Map user's team data to player information
            user_team = user_team_data.get('picks', [])
            user_team_details = []
            for player in user_team:
                player_data = players_df[players_df['id'] == player['element']].to_dict('records')
                if player_data:
                    player_info = player_data[0]
                    user_team_details.append({
                        'id': player['element'],
                        "name": player_info.get("name", "Unknown"),
                        "team_name": player_info.get("team_name", "Unknown Team"),
                        "position": player_info.get("position", "Unknown Position"),
                        "price": player_info.get("price", 0) / 10,  # Convert price to decimal
                        "total_points": player_info.get("total_points", 0),
                        "minutes": player_info.get("minutes", 0),
                        "goals_scored": player_info.get("goals_scored", 0),
                        "assists": player_info.get("assists", 0),
                        "clean_sheets": player_info.get("clean_sheets", 0),
                        "ownership": player_info.get("ownership", "0.0"),
                        "position_on_field": player['position'],
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
                "position_on_field", "multiplier", "is_captain", "is_vice_captain"
            ]]
            
            return {
                'my_team': user_team_df,
                'all_players': players_df,
                'latest_gameweek': latest_gameweek,
                'timestamp': pd.Timestamp.now(),
            }
        
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

class FantasyTeamOptimizer:
    def __init__(self, all_players: List[Player], budget: float = 100.0):
        self.all_players = all_players
        self.budget = budget
        self.position_limits = {
            'GK': (2, 2),   # (min, max)
            'DEF': (5, 5),
            'MID': (5, 5),
            'FWD': (3, 3)
        }
        self.starting_position_limits = {
            'GK': (1, 1),   # (min, max)
            'DEF': (3, 5),
            'MID': (3, 5),
            'FWD': (1, 3)
        }
        
    def validate_team(self, team: List[Player]) -> bool:
        # Check total players
        if len(team) != 15:
            return False
            
        # Check budget
        if sum(p.price for p in team) > self.budget:
            return False
            
        # Check position limits
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        for player in team:
            position_counts[player.position] += 1
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
            
            # Check team limit (max 3 players per team)
            if team_counts[player.team] > 3:
                return False
        
        # Verify position counts
        for pos, (min_count, max_count) in self.position_limits.items():
            if not min_count <= position_counts[pos] <= max_count:
                return False
                
        return True
    
    def validate_starting_eleven(self, starters: List[Player]) -> bool:
        if len(starters) != 11:
            return False
            
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in starters:
            position_counts[player.position] += 1
            
        for pos, (min_count, max_count) in self.starting_position_limits.items():
            if not min_count <= position_counts[pos] <= max_count:
                return False
                
        return True
    
    def select_starting_eleven(self, squad: List[Player]) -> Tuple[List[Player], List[Player]]:
        """
        Select the optimal starting 11 from the 15-player squad.
        Returns (starting_11, bench)
        """
        # Sort players by expected points within each position
        by_position = {
            'GK': sorted([p for p in squad if p.position == 'GK'], 
                        key=lambda x: x.expected_points, reverse=True),
            'DEF': sorted([p for p in squad if p.position == 'DEF'],
                         key=lambda x: x.expected_points, reverse=True),
            'MID': sorted([p for p in squad if p.position == 'MID'],
                         key=lambda x: x.expected_points, reverse=True),
            'FWD': sorted([p for p in squad if p.position == 'FWD'],
                         key=lambda x: x.expected_points, reverse=True)
        }
        
        # Start with minimum requirements
        starting_11 = (
            by_position['GK'][:1] +  # 1 GK
            by_position['DEF'][:3] +  # 3 DEF
            by_position['MID'][:3] +  # 3 MID
            by_position['FWD'][:1]    # 1 FWD
        )
        
        # Add remaining 3 highest point scorers that maintain valid formation
        remaining_players = (
            by_position['DEF'][3:] +
            by_position['MID'][3:] +
            by_position['FWD'][1:]
        )
        remaining_players.sort(key=lambda x: x.expected_points, reverse=True)
        
        for player in remaining_players:
            temp_team = starting_11 + [player]
            if len(temp_team) <= 11 and self.validate_starting_eleven(temp_team):
                starting_11.append(player)
                
            if len(starting_11) == 11:
                break
                
        # Create bench from remaining players
        bench = [p for p in squad if p not in starting_11]
        return starting_11, bench
    
    def optimize_team(self, iterations: int = 1000) -> Tuple[List[Player], List[Player]]:
        """
        Main optimization function using a genetic algorithm approach.
        Returns (best_starting_11, best_bench)
        """
        best_squad = None
        best_score = -float('inf')
        
        def create_random_squad():
            squad = []
            for pos, (min_count, _) in self.position_limits.items():
                eligible = [p for p in self.all_players if p.position == pos]
                squad.extend(random.sample(eligible, min_count))
            return squad
        
        # Initial population
        population = []
        for _ in range(50):  # population size
            squad = create_random_squad()
            if self.validate_team(squad):
                population.append(squad)
        
        for _ in range(iterations):
            # Tournament selection
            new_population = []
            for _ in range(len(population)):
                tournament = random.sample(population, 3)
                winner = max(tournament, 
                           key=lambda squad: sum(p.expected_points for p in squad))
                new_population.append(winner.copy())
            
            # Mutation
            for squad in new_population:
                if random.random() < 0.1:  # mutation rate
                    pos = random.choice(['GK', 'DEF', 'MID', 'FWD'])
                    idx = random.randint(0, len(squad) - 1)
                    eligible = [p for p in self.all_players 
                              if p.position == pos and p not in squad]
                    if eligible:
                        squad[idx] = random.choice(eligible)
            
            # Evaluate and update best squad
            for squad in new_population:
                if self.validate_team(squad):
                    starting_11, _ = self.select_starting_eleven(squad)
                    score = sum(p.expected_points for p in starting_11)
                    if score > best_score:
                        best_score = score
                        best_squad = squad
            
            population = new_population
        
        if best_squad:
            return self.select_starting_eleven(best_squad)
        else:
            raise ValueError("Could not find a valid team configuration")
        
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
        self.optimal_team = OptimalTeamAgent()
        # self.team_optimizer = TeamOptimizationAgent()
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
    # Initialize DataScraperAgent
    data_scraper = DataScraperAgent()

    # Fetch and process data
    try:
        processed_data = await data_scraper.process(data={"user_team_id": 6256406})
        players_df = processed_data['all_players']

        # Initialize OptimalTeamAgent
        optimal_team_agent = OptimalTeamAgent(players_df)

        # Suggest the optimal team
        optimal_team_df = optimal_team_agent.suggest_optimal_team()

        # Display the optimal team
        print(optimal_team_df)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Example usage:
def select_optimal_team(player_data: List[Dict], budget: float = 100.0) -> Tuple[List[Player], List[Player]]:
    """
    Main function to select optimal team given player data.
    Returns (starting_11, bench)
    """
    # Convert raw player data to Player objects
    players = [
        Player(
            id=p['id'],
            name=p['name'],
            position=p['position'],
            team=p['team'],
            price=p['price'],
            expected_points=p['expected_points']
        )
        for p in player_data
    ]
    
    optimizer = FantasyTeamOptimizer(players, budget)
    starting_11, bench = optimizer.optimize_team(iterations=1000)
    
    # Sort bench by position for proper substitution order
    bench.sort(key=lambda x: {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}[x.position])
    
    return starting_11, bench

if __name__ == "__main__":
    asyncio.run(main())