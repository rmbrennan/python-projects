"""
Agent responsible for team optimization logic.
"""
# Standard library imports
from typing import Dict, Tuple, Any

# Third-party imports
import pandas as pd
import random
import logging

# Local imports (assuming project structure from earlier)
from .base_agent import FPLAgent, AgentConfig
from agents.data_scraper import DataScraperAgent
from utils.constants import POSITION_MAP, FORMATION_CONSTRAINTS, SQUAD_LIMITS, TEAM_CONSTRAINTS

class TeamOptimizer:
    def __init__(self, all_players, budget=100.0):
        self.all_players = all_players
        self.budget = budget
        self.formation_limits = FORMATION_CONSTRAINTS
        self.squad_limits = SQUAD_LIMITS
        self.logger = logging.getLogger(self.__class__.__name__)  # Initialize logger
        
    def validate_team(self, team):
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
        for pos, limit in self.squad_limits.items():
            if not 0 <= position_counts[pos] <= limit:
                return False
                    
        return True
        
    def validate_starting_eleven(self, starters):
        if len(starters) != 11:
            return False
                
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in starters:
            position_counts[player.position] += 1
                
        for pos, (min_count, max_count) in self.formation_limits.items():
            if not min_count <= position_counts[pos] <= max_count:
                return False
                    
        return True
    
    def select_starting_eleven(self, squad):
        """
        Select the optimal starting 11 from the 15-player squad.
        Returns (starting_11, bench)
        """
        # Sort players by expected points within each position
        by_position = {
            'GK': squad[squad['position'] == 'GK'].sort_values(by='expected_points', ascending=False),
            'DEF': squad[squad['position'] == 'DEF'].sort_values(by='expected_points', ascending=False),
            'MID': squad[squad['position'] == 'MID'].sort_values(by='expected_points', ascending=False),
            'FWD': squad[squad['position'] == 'FWD'].sort_values(by='expected_points', ascending=False)
        }
        
        # Start with minimum requirements
        starting_11 = pd.concat([
            by_position['GK'].head(1),  # 1 GK
            by_position['DEF'].head(3),  # 3 DEF
            by_position['MID'].head(3),  # 3 MID
            by_position['FWD'].head(1)   # 1 FWD
        ])
        
        # Add remaining 3 highest point scorers that maintain valid formation
        remaining_players = pd.concat([
            by_position['DEF'].iloc[3:],
            by_position['MID'].iloc[3:],
            by_position['FWD'].iloc[1:]
        ]).sort_values(by='expected_points', ascending=False)
        
        # Add the remaining 3 players
        for _, player in remaining_players.iterrows():
            # Create a temporary team by adding the current player
            temp_team = pd.concat([starting_11, player.to_frame().T])
            
            # Check if the temporary team is valid and has 11 or fewer players
            # if len(temp_team) <= 11 and self.validate_starting_eleven(temp_team):
            if len(temp_team) <= 11:
                starting_11 = temp_team
                
            # Stop once we have 11 players
            if len(starting_11) == 11:
                break
                
        # Create bench from remaining players
        bench = squad[~squad['id'].isin(starting_11['id'])]
        
        return starting_11, bench
    
    def create_random_team(self):
        """
        Creates a random valid team from the all_players dataset.
        The team adheres to FPL rules:
        - 15 players total (TOTAL_PLAYERS)
        - 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards (SQUAD_LIMITS)
        - Maximum of 3 players from any one team (MAX_PER_TEAM)
        - Total team cost within budget (BUDGET)
        """
        # Initialize an empty DataFrame for the team
        team = pd.DataFrame(columns=self.all_players.columns)
        
        # Define position limits from SQUAD_LIMITS
        position_limits = SQUAD_LIMITS
        
        # Define team constraints from TEAM_CONSTRAINTS
        max_players_per_team = TEAM_CONSTRAINTS['MAX_PER_TEAM']
        budget = TEAM_CONSTRAINTS['BUDGET']
        
        # Track team counts to ensure no more than 3 players from any one team
        team_counts = {}
        
        # Iterate through each position and select players
        for position_label, limit in position_limits.items():
            # Map the position label to the corresponding element_type code
            position_code = next(
                code for code, label in POSITION_MAP.items() if label == position_label
            )
            
            # Filter players for the current position
            eligible_players = self.all_players[self.all_players['element_type'] == position_code]
            
            # Randomly sample the required number of players for this position
            selected_players = eligible_players.sample(n=limit, replace=False)
            
            # Add selected players to the team DataFrame
            team = pd.concat([team, selected_players], ignore_index=True)
            
            # Update team counts
            for _, player in selected_players.iterrows():
                team_name = player['team_code']
                team_counts[team_name] = team_counts.get(team_name, 0) + 1
        
        # Ensure the team is within budget
        while team['now_cost'].sum() > budget:
            # Remove the most expensive player
            most_expensive_player_index = team['now_cost'].idxmax()
            team = team.drop(most_expensive_player_index).reset_index(drop=True)
            
            # Add a new random player from the same position
            position_code = team.loc[most_expensive_player_index, 'element_type']
            eligible_players = self.all_players[
                (self.all_players['element_type'] == position_code) & 
                (~self.all_players['id'].isin(team['id']))
            ]
            
            if not eligible_players.empty:
                new_player = eligible_players.sample(n=1)
                team = pd.concat([team, new_player], ignore_index=True)
        
        # Ensure no team has more than 3 players
        while any(count > max_players_per_team for count in team_counts.values()):
            # Find all teams with more than 3 players
            problematic_teams = [team for team, count in team_counts.items() if count > max_players_per_team]
            
            for problematic_team in problematic_teams:
                # Find all players from the problematic team
                problematic_players = team[team['team_code'] == problematic_team]
                
                # Calculate how many players need to be removed
                excess_players = len(problematic_players) - max_players_per_team
                
                # Remove the excess players at random
                players_to_remove = problematic_players.sample(n=excess_players)
                team = team.drop(players_to_remove.index).reset_index(drop=True)
                
                # Update team counts
                team_counts[problematic_team] -= excess_players
                
                # Replace the removed players with new players from the correct positions and teams
                for _, player in players_to_remove.iterrows():
                    position_code = player['element_type']
                    eligible_players = self .all_players[
                        (self.all_players['element_type'] == position_code) & 
                        (~self.all_players['id'].isin(team['id'])) &
                        (self.all_players['team_code'].isin([t for t in team_counts if team_counts[t] < max_players_per_team]))
                    ]
                    
                    if not eligible_players.empty:
                        new_player = eligible_players.sample(n=1)
                        team = pd.concat([team, new_player], ignore_index=True)
                        team_counts[new_player['team_code'].values[0]] = team_counts.get(new_player['team_code'].values[0], 0) + 1

        # Add the 'position' column to the team DataFrame using POSITION_MAP
        team['position'] = team['element_type'].map(POSITION_MAP)
        # Rename the 'ep_next' column to 'expected_points'
        team = team.rename(columns={'ep_next': 'expected_points'})
        # Rename the 'web_name' column to 'name'
        team = team.rename(columns={'web_name': 'name'})

        return team    

    def create_optimal_team(self, iterations=1000):
        """
        Main optimization function using a genetic algorithm approach.
        Returns (best_starting_11, best_bench)
        """
        best_squad = None
        best_score = -float('inf')
        
        for _ in range(iterations):
            # Create a random team
            squad = self.create_random_team()
            starting_11, bench = self.select_starting_eleven(squad)
            score = self.evaluate_team(starting_11)

            # Log the current iteration and score
            self.logger.info(f"Iteration: {_}, Score: {score}, Starting 11: {starting_11['name'].tolist()}")
            # Update best squad if the current score is better
            if score > best_score:
                best_score = score
                best_squad = (starting_11, bench)

        return best_squad

    def evaluate_team(self, starting_11):
        """
        Evaluate the team based on expected points.
        """
        return sum(float(player['expected_points']) for _, player in starting_11.iterrows())
    