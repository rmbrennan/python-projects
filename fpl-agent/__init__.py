"""
Initialize the project package and expose key modules and agents.
"""
from .agents import (
    FPLAgent,
    AgentConfig,
    DataScraperAgent,
    TeamOptimizerAgent,
)

__all__ = [
    "FPLAgent",
    "AgentConfig",
    "DataScraperAgent",
    "TeamOptimizerAgent",
]