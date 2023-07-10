# GridLand

![GridLand Game in Progress](https://github.com/CooperJRG/GridLand/blob/main/pictures/gridland_gif.gif)

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

To run the project, execute the `main.py` script. The script can be run with the following optional parameters:

1. `limiter` - (default: 300) An integer that controls the maximum number of steps per simulation.
2. `complete_count` - (default: 1) An integer that specifies the number of times an agent should complete a level.
3. `random_levels` - (default: False) A boolean flag indicating whether the levels should be randomly selected.
4. `enable_death` - (default: False) A boolean flag indicating whether to enable the death mechanism in the game.

The script is run with the following format:

```bash
python main.py [limiter] [complete_count] [random_levels] [enable_death]
```

For example, to run the script with a limiter of 500, a complete count of 2, random levels enabled, and death enabled, you would use the following command:

```bash
python main.py 500 2 true true
```
The script will run with the default values if no parameters are provided. This will start the game loop, where the agent will interact with the GridLand environment. The agent's actions and the environment's responses will be visualized in a PyQt5 window.

This will start the game loop, where the agent will interact with the GridLand environment. The agent's actions and the environment's responses will be visualized in a PyQt5 window.

## Implementation Details

![Training Progress on a Random Level](https://github.com/CooperJRG/GridLand/blob/main/pictures/training_progress.png)

The graph above represents the training progress of the agent over time. The x-axis signifies the training iterations, while the y-axis represents the average reward. 

The agent explores the environment at the beginning of training, and the rewards are relatively low and quite variable. This is the "exploration" phase of reinforcement learning, where the agent learns about the environment and tries different actions to see their effects. This phase is often characterized by high variance in reward, reflected by the noise in the graph, as the agent's decisions are initially more random.

As the agent learns from its interactions with the environment, it starts to understand the game's dynamics better, and its performance begins to improve. This is demonstrated by the increasing trend in the rewards over the training iterations.

Towards the end of the training, the reward peaks, showing that the agent has learned an effective policy to maximize its reward. However, there is still some variation in the reward, which signifies that the agent continues to explore the environment to find potentially better strategies.

This balancing act between exploration (trying out new, potentially better policies) and exploitation (sticking to the known best policy) is a fundamental challenge in reinforcement learning. The noise in the graph illustrates the agent's continual exploration of the environment, while the overall trend of increasing reward shows successful learning.

The agent is trained using the Proximal Policy Optimization (PPO) algorithm, a policy gradient method for reinforcement learning. PPO uses a dual network architecture for policy and value estimation. The policy network outputs a distribution over actions given a state, and the value network estimates the expected return given a state.

The PPO agent interacts with the environment by selecting actions based on the current policy, executing these actions in the environment, and storing the resulting state transitions in a memory buffer. The agent then uses these accumulated experiences to update the parameters of the policy and value networks.

The PPO update step involves calculating the returns from the stored rewards and value estimates using Generalized Advantage Estimation (GAE), and then optimizing the policy and value networks based on these returns and the old policy. The policy update is clipped to prevent significant policy updates, and an entropy bonus is included in the loss function to encourage exploration.

The agent is implemented from scratch, providing a detailed example of implementing the PPO algorithm in PyTorch.

## Contributing
Contributions are welcome! Please feel free to submit a pull request.
