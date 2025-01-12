"""
Agent responsible for team optimization logic.
"""
# Standard library imports
from typing import Dict, Tuple, Any

# Third-party imports
import pandas as pd
import random

# Local imports (assuming project structure from earlier)
from .base_agent import FPLAgent, AgentConfig
from utils.constants import FORMATION_CONSTRAINTS, SQUAD_LIMITS

class TeamOptimizer:
    def __init__(self, all_players, budget=100.0):
        self.all_players = all_players
        self.budget = budget
        self.formation_limits = FORMATION_CONSTRAINTS
        self.squad_limits = SQUAD_LIMITS
        
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
            if len(temp_team) <= 11:
                starting_11.append(player)
                
            if len(starting_11) == 11:
                break
                
        # Create bench from remaining players
        bench = [p for p in squad if p not in starting_11]
        return starting_11, bench
    
    def optimize_team(self, iterations=1000):
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