# Headless Game Simulator with MLLM Player

A simple Python framework for running games in headless mode and play the games through an MLLM policy action.

## Overview
This project transforms the interactive Python games into headless simulators that:
-  Extract game frames as numpy arrays for ML model input
-  Accept programmatic actions instead of keyboard/mouse input
-  Provide reward signals for reinforcement learning
-  Support data collection and training pipelines
-  Work seamlessly with vision-language models from HuggingFace.


## Supported Games
| Game   | Description            | Actions                        | Difficulty |
|--------|-------------------------|---------------------------------|------------|
| Snake  | Classic snake game      | `up`, `down`, `left`, `right`   | Easy       |


## Installation
### Prerequisites

Python 3.7+ \
PyTorch (for MLLM integration)

### Basic Installation

```bash
# Clone the repository
git clone <repo-url>
cd python-game-simulator


# Install core dependencies
pip install -r requirements.txt
```

## Quick tests
```bash
python run_game.py
```
