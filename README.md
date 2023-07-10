# GridLand

![GridLand Game in Progress](https://github.com/CooperJRG/GridLand/blob/main/gridland_gif.gif)

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
```
This will start the game loop, where the agent will interact with the GridLand environment. The agent's actions and the environment's responses will be visualized in a PyQt5 window.

## Project Structure

The agent is trained using the Proximal Policy Optimization (PPO) algorithm, a policy gradient method for reinforcement learning. PPO uses a dual network architecture for policy and value estimation. The policy network outputs a distribution over actions given a state, and the value network estimates the expected return given a state.

The PPO agent interacts with the environment by selecting actions based on the current policy, executing these actions in the environment, and storing the resulting state transitions in a memory buffer. The agent then uses these accumulated experiences to update the parameters of the policy and value networks.

The PPO update step involves calculating the returns from the stored rewards and value estimates using Generalized Advantage Estimation (GAE), and then optimizing the policy and value networks based on these returns and the old policy. The policy update is clipped to prevent significant policy updates, and an entropy bonus is included in the loss function to encourage exploration.

The agent is implemented from scratch, providing a detailed example of implementing the PPO algorithm in PyTorch.

## Contributing
Contributions are welcome! Please feel free to submit a pull request.
