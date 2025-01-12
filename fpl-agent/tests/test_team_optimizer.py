import sys
import os
import unittest
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.team_optimizer import TeamOptimizer
from utils.constants import FORMATION_CONSTRAINTS, SQUAD_LIMITS

@dataclass
class Player:
    id: int
    name: str
    position: str
    team: str
    price: float
    expected_points: float

class TestTeamOptimizer(unittest.TestCase):
    def test_team_optimizer(self):
        # Create a list of players
        players = [
            Player(id=1, name='Player 1', position='GK', team='Team 1', price=6.0, expected_points=1.0),
            Player(id=2, name='Player 2', position='DEF', team='Team 1', price=6.0, expected_points=2.0),
            Player(id=3, name='Player 3', position='DEF', team='Team 2', price=6.0, expected_points=3.0),
            Player(id=4, name='Player 4', position='DEF', team='Team 2', price=6.0, expected_points=4.0),
            Player(id=5, name='Player 5', position='DEF', team='Team 3', price=6.0, expected_points=5.0),
            Player(id=6, name='Player 6', position='MID', team='Team 3', price=6.0, expected_points=6.0),
            Player(id=7, name='Player 7', position='MID', team='Team 4', price=6.0, expected_points=7.0),
            Player(id=8, name='Player 8', position='MID', team='Team 4', price=6.0, expected_points=8.0),
            Player(id=9, name='Player 9', position='MID', team='Team 4', price=6.0, expected_points=9.0),
            Player(id=10, name='Player 10', position='MID', team='Team 5', price=6.0, expected_points=10.0),
            Player(id=11, name='Player 11', position='FWD', team='Team 5', price=6.0, expected_points=11.0),
            Player(id=12, name='Player 12', position='FWD', team='Team 6', price=6.0, expected_points=12.0),
            Player(id=13, name='Player 13', position='FWD', team='Team 6', price=6.0, expected_points=13.0),
            Player(id=14, name='Player 14', position='GK', team='Team 7', price=6.0, expected_points=14.0),
            Player(id=15, name='Player 15', position='DEF', team='Team 8', price=6.0, expected_points=15.0),
        ]

        # Create a TeamOptimizer instance
        team_optimizer = TeamOptimizer(players, 100.0)

        # Test the validate_team method
        self.assertTrue(team_optimizer.validate_team(players))

        # Test the validate_starting_eleven method
        starting_eleven = players[:11]
        self.assertTrue(team_optimizer.validate_starting_eleven(starting_eleven))

        # Test the select_starting_eleven method
        starting_eleven, bench = team_optimizer.select_starting_eleven(players)
        self.assertEqual(len(starting_eleven), 11)
        self.assertEqual(len(bench), 4)

if __name__ == '__main__':
    unittest.main()