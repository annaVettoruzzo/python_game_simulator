import random
from PIL import Image
from .python_game_simulator import HeadlessGameSimulator
from .snake import SnakeSimulator


# Game factory
def create_game(game_name: str, **kwargs) -> HeadlessGameSimulator:
    """Factory function to create game simulators"""
    games = {
        'snake': SnakeSimulator,
    }
    
    game_name = game_name.lower()
    if game_name not in games:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(games.keys())}")
    
    return games[game_name](**kwargs)


# Example usage
if __name__ == "__main__":
    print("Available games:", ['snake'])
    
    # Test each game
    for game_name in ['snake']:
        print(f"\n{'='*60}")
        print(f"Testing {game_name.upper()} Game")
        print('='*60)
        
        sim = create_game(game_name, width=420, height=420)
        frame = sim.reset()
        
        print(f"Frame shape: {frame.shape}")
        print(f"Valid actions: {sim.get_valid_actions()}")
        
        total_reward = 0
        for i in range(50):
            action = random.choice(sim.get_valid_actions())
            frame, reward, done, info = sim.step(action)

            img = Image.fromarray(frame)
            img.save(f'{game_name}_sample_{i}.png')

            total_reward += reward
            
            if done:
                print(f"Game ended at step {i}")
                print(f"Info: {info}")
                print(f"Total reward: {total_reward:.2f}")
                break
        
        # Save sample frame
        img = Image.fromarray(frame)
        img.save(f'{game_name}_sample.png')
        print(f"Sample frame saved to '{game_name}_sample.png'")