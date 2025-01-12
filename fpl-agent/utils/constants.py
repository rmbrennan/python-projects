"""
Constants used throughout the project.
"""
from typing import Dict, Tuple

POSITION_MAP: Dict[int, str] = {
    1: 'GK',
    2: 'DEF',
    3: 'MID',
    4: 'FWD'
}

FORMATION_CONSTRAINTS: Dict[str, Tuple[int, int]] = {
    'GK': (1, 1),
    'DEF': (3, 5),
    'MID': (3, 5),
    'FWD': (1, 3)
}

TEAM_CONSTRAINTS = {
    'BUDGET': 100.0,
    'MAX_PER_TEAM': 3,
    'TOTAL_PLAYERS': 15
}

API_ENDPOINTS = {
    'bootstrap': 'bootstrap-static/',
    'fixtures': 'fixtures/',
    'player_summary': 'element-summary/{player_id}/',
    'live': 'event/{event_id}/live/',
    'user_team': 'my-team/{user_id}/',
}