"""
Headless Game Simulator for free-python-games
Extracts frames and accepts programmatic actions for ML training
"""

import numpy as np
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod


class HeadlessGameSimulator(ABC):
    """Base class for headless game simulation."""
    
    def __init__(self, width: int = 420, height: int = 420):
        self.width = width
        self.height = height
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset game to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one game step with given action"""
        pass
    
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Get current game frame as numpy array"""
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions for this game"""
        pass


