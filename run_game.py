from PIL import Image

from src.game_simulators.snake import SnakeSimulator
from src.game_players.game_player import MLLMGamePlayer, RandomPlayer

def run_game_with_mllm(
    simulator,
    player,
    max_steps: int = 100,
    save_frames: bool = False,
    frame_save_interval: int = 10
):
    """
    Run game with MLLM player.
    
    Args:
        simulator: Game simulator instance
        player: MLLM player instance
        max_steps: Maximum steps to run
        save_frames: Whether to save frames
        frame_save_interval: Save every N frames
    """
    print("Starting game with MLLM player...")
    print("=" * 60)
    
    frame = simulator.reset()
    total_reward = 0
    frames_saved = []
    
    for step in range(max_steps):
        # Get valid actions
        valid_actions = simulator.get_valid_actions()
        
        # Get current game info
        info = {
            'score': getattr(simulator, 'score', 0),
            'snake_length': len(getattr(simulator, 'snake', [1]))
        }
        
        # Predict action using MLLM
        action = player.predict_action(frame, info, valid_actions)
        
        # Execute action
        frame, reward, done, info = simulator.step(action)
        total_reward += reward
        
        # Log progress
        if step % 1 == 0:
            print(f"Step {step:4d} | Action: {action:5s} | "
                  f"Reward: {reward:6.2f} | Score: {info['score']:3d} | "
                  f"Length: {info['snake_length']:3d}")
        
        # Save frames if requested
        if save_frames and step % frame_save_interval == 0:
            frames_saved.append(frame.copy())
        
        if done:
            print("\n" + "=" * 60)
            print("GAME OVER!")
            print(f"Final Score: {info['score']}")
            print(f"Final Length: {info['snake_length']}")
            print(f"Total Steps: {step + 1}")
            print(f"Total Reward: {total_reward:.2f}")
            print("=" * 60)
            break
    
    return frames_saved, info


if __name__ == "__main__":
   
    print("Initializing game simulator...")
    sim = SnakeSimulator(width=420, height=420, cell_size=20)
    
    # Choose player type
    use_real_mllm = True  # Set to True to use real MLLM
    
    if use_real_mllm:
        print("\nInitializing MLLM player...")
        player = MLLMGamePlayer(model_name="llava-hf/llava-1.5-7b-hf")
        # Model will be loaded on first prediction
    else:
        print("\nUsing simple heuristic player (for testing)...")
        player = RandomPlayer()
    
    # Run game
    frames, final_info = run_game_with_mllm(
        simulator=sim,
        player=player,
        max_steps=500,
        save_frames=True,
        frame_save_interval=1
    )
    
    print(f"\nSaved {len(frames)} frames")
    
    # Optionally save frames as video or GIF
    if frames and len(frames) > 0:
        print("\nSaving frames as GIF...")
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            'game_replay.gif',
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        print("GIF saved as 'game_replay.gif'")