# GridLand

![GridLand Game in Progress](INSERT_GIF_HERE)

This repository contains the implementation of a grid-based environment game, GridLand, using PyTorch, PyQt5, and Numpy. The agent learns to navigate the grid using Proximal Policy Optimization (PPO), a reinforcement learning algorithm.

## Dependencies

The project is implemented in Python 3.9 and uses the following libraries:

- PyTorch
- PyQt5
- Numpy

## Project Structure

The project is structured as follows:

- `main.py`: The main entry point of the application. It initializes the environment, the PPO agent, and the game loop.
- `PPO/`: This directory contains the implementation of the PPO algorithm.
  - `PPO.py`: The implementation of the PPO agent. 
  - `root_network.py`: A PyTorch-based neural network model for root classification.
  - `policy_network.py`: A PyTorch-based neural network model representing a policy for reinforcement learning.
  - `value_network.py`: A PyTorch-based neural network model representing a value for reinforcement learning.
  - `dual_network.py`: The implementation of the dual network used by the PPO agent for policy and value estimation.
  - `memory.py`: The implementation of the memory buffer used by the PPO agent to store experiences.
- `environment/`: This directory contains the implementation of the GridLand environment.
  - `agent.py`: A class for creating an agent in a 2D grid. 
  - `level_manager.py`: This file manages levels for the game and translates them into in-game objects.
  - `grid_land.py`: The implementation of the GridLand environment.
- `visualize/`: This directory contains the implementation of the grid visualization using PyQt5.
  - `grid_window.py`: The implementation of the grid visualization.

## Running the Project

To run the project, execute the `main.py` script:

```bash
python main.py
