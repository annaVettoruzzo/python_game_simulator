import numpy as np
from PIL import Image
import torch
import random
from typing import List, Dict, Optional

from ..game_simulators.snake import get_default_prompt


class MLLMGamePlayer:
    """
    Wrapper class to connect MLLM models to the game simulator.
    """
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-Instruct"): # Small VLM for debugging
        """
        Initialize MLLM player.
        
        Args:
            model_name: HuggingFace model name for the MLLM
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the MLLM model from HuggingFace"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure to install: pip install transformers accelerate")
            raise
    
    def predict_action(
        self, 
        frame: np.ndarray, 
        game_info: Dict,
        valid_actions: List[str],
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Predict next action based on current frame.
        
        Args:
            frame: Game frame as numpy array (H, W, 3)
            game_info: Dictionary with game state info
            valid_actions: List of valid actions
            prompt_template: Optional custom prompt
            
        Returns:
            action: Predicted action string
        """
        if self.model is None:
            self.load_model()
        
        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Create prompt
        if prompt_template is None:
            prompt_template = get_default_prompt(game_info, valid_actions)
        
        # Process inputs
        inputs = self.processor(
            text=prompt_template,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action from response
        action = self._parse_action(response, valid_actions)
        
        return action
    
    def _parse_action(self, response: str, valid_actions: List[str]) -> str:
        """Parse action from model response"""
        response_lower = response.lower()
        
        # Look for valid actions in the response
        for action in valid_actions:
            if action.lower() in response_lower:
                return action
        
        # Default to 'none' if no valid action found
        return 'none'


class RandomPlayer:
    """
    Random player for testing without loading any models.
    """
    
    def predict_action(
        self,
        frame: np.ndarray,
        game_info: Dict,
        valid_actions: List[str]
    ) -> str:
        """
        Simple heuristic-based action prediction for testing.
        In reality, replace this with actual MLLM inference.
        """
        
        # Simple strategy: try to move towards red pixels (food)
        # This is just for demonstration
        actions = ['up', 'down', 'left', 'right']

        return random.choice(actions)
