from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set, List
import logging
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    """Configuration class for agents"""
    name: str
    requires: Set[str] = field(default_factory=set)  # Required dependencies
    provides: Set[str] = field(default_factory=set)  # Data/services provided
    cache_enabled: bool = True
    timeout: int = 30

class FPLAgent(ABC):
    def __init__(self, *args, **kwargs):
        self.user_team_id = kwargs.get('user_team_id')  # Store user_team_id as an instance variable
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = "https://fantasy.premierleague.com/api/"
        
    @property
    def name(self) -> str:
        return self.config.name
        
    def add_dependency(self, agent: 'FPLAgent') -> None:
        """Add another agent as a dependency"""
        # Check if the agent provides what we need
        if not agent.config.provides.intersection(self.config.requires):
            raise ValueError(
                f"Agent {agent.name} does not provide required capabilities "
                f"for {self.name}"
            )
        
        self._dependencies[agent.name] = agent
        
    def get_dependency(self, name: str) -> Optional['FPLAgent']:
        """Get a specific dependency by name"""
        return self._dependencies.get(name)
        
    def validate_dependencies(self) -> bool:
        """Ensure all required dependencies are present and valid"""
        missing = self.config.requires - set(
            dep.config.provides for dep in self._dependencies.values()
        )
        if missing:
            self.logger.error(f"Missing required dependencies: {missing}")
            return False
        return True
        
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if available"""
        return self._cache.get(key) if self.config.cache_enabled else None
        
    def set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache"""
        if self.config.cache_enabled:
            self._cache[key] = data
            
    @abstractmethod
    def validate(self) -> bool:
        """Validate agent configuration and dependencies"""
        pass
        
    @abstractmethod
    def process(self) -> Any:
        """Main processing logic"""
        pass