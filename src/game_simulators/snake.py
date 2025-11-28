"""
Code inspired by: https://thepythoncode.com/article/make-a-snake-game-with-pygame-in-python
"""

import numpy as np
from PIL import Image, ImageDraw
import random
from typing import Tuple, List, Dict

from .python_game_simulator import HeadlessGameSimulator


class SnakeSimulator(HeadlessGameSimulator):
    """Snake game simulator"""
    
    def __init__(self, width: int = 420, height: int = 420, cell_size: int = 10):
        self.cell_size = cell_size
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        super().__init__(width, height)
    
    def reset(self):
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = [(center_x, center_y)]
        self.direction = (0, -1)
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_frame()
    
    def _spawn_food(self) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                return (x, y)
    
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.game_over:
            return self.get_frame(), 0.0, True, {'score': self.score}
        
        action = action.lower()
        new_direction = self.direction
        
        if action == 'up' and self.direction != (0, 1):
            new_direction = (0, -1)
        elif action == 'down' and self.direction != (0, -1):
            new_direction = (0, 1)
        elif action == 'left' and self.direction != (1, 0):
            new_direction = (-1, 0)
        elif action == 'right' and self.direction != (-1, 0):
            new_direction = (1, 0)
        
        self.direction = new_direction
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        reward = 0.0
        
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.game_over = True
            reward = -10.0
            return self.get_frame(), reward, True, {'score': self.score, 'snake_length': len(self.snake)}
        
        if new_head in self.snake:
            self.game_over = True
            reward = -10.0
            return self.get_frame(), reward, True, {'score': self.score, 'snake_length': len(self.snake)}
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            reward = 0.01
        
        self.steps += 1
        
        frame = self.get_frame()
        info = {'score': self.score, 'snake_length': len(self.snake), 'steps': self.steps}
        return frame, reward, self.game_over, info
    
    def get_frame(self) -> np.ndarray:
        img = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(img)
        
        for segment in self.snake[1:]:
            x, y = segment[0] * self.cell_size, segment[1] * self.cell_size
            draw.rectangle([x, y, x + self.cell_size - 1, y + self.cell_size - 1], 
                         fill='black', outline='gray')
        
        head = self.snake[0]
        x, y = head[0] * self.cell_size, head[1] * self.cell_size
        draw.rectangle([x, y, x + self.cell_size - 1, y + self.cell_size - 1], 
                     fill='darkgreen', outline='gray')
        
        x, y = self.food[0] * self.cell_size, self.food[1] * self.cell_size
        draw.rectangle([x, y, x + self.cell_size - 1, y + self.cell_size - 1], 
                     fill='red', outline='gray')
        
        return np.array(img)
    
    def get_valid_actions(self) -> List[str]:
        return ['up', 'down', 'left', 'right', 'none']
    

def get_default_prompt(game_info: Dict, valid_actions: List[str]) -> str:
        """Generate default prompt for the game"""
        actions_str = ", ".join(valid_actions)
        
        prompt = f"""<image>
You are playing a Snake game. 
Look at the game screen and decide the best next move.

Current Score: {game_info.get('score', 0)}
Snake Length: {game_info.get('snake_length', 1)}

Valid actions: {actions_str}

Analyze the image and respond with ONLY ONE of these actions: {actions_str}
Choose the action that will help the snake reach the food while avoiding walls and itself.

Your action:"""
        
        return prompt