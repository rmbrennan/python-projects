"""
Data models for FPL entities.
"""
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Player:
    id: int
    name: str
    position: str
    team: str
    price: float
    expected_points: float
    
@dataclass
class Formation:
    gk: int = 1
    def_: int = 4
    mid: int = 4
    fwd: int = 2
    
    def is_valid(self) -> bool:
        """Check if formation is valid."""
        return (self.gk == 1 and 
                self.def_ >= 3 and 
                self.fwd >= 1 and 
                (self.gk + self.def_ + self.mid + self.fwd) == 11)

@dataclass
class Squad:
    starting_11: List[Player]
    bench: List[Player]
    captain_id: int
    vice_captain_id: int
