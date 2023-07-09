# GridLand

This repository contains the implementation of a grid-based environment game, GridLand, using PyTorch, PyQt5, and Numpy. The agent learns to navigate the grid using Proximal Policy Optimization (PPO), a reinforcement learning algorithm.

## Dependencies

The project is implemented in Python 3.9 and uses the following libraries:

* PyTorch
* PyQt5
* Numpy

## Project Structure

The project is structured as follows:

* main.py: This is the main entry point of the application. It initializes the environment, the PPO agent, and the game loop.
* PPO/: This directory contains the implementation of the PPO algorithm.
  * PPO.py: This file contains the implementation of the PPO agent. The agent is initialized with a dual network for policy and value estimation. The agent selects actions, evaluates policy and value functions, and updates the network parameters based on stored experiences.
   * dual_network.py: This file contains the implementation of the dual network used by the PPO agent for policy and value estimation.
  * memory.py: This file contains the implementation of the memory buffer used by the PPO agent to store experiences.
* environment/: This directory contains the implementation of the GridLand environment.
  * grid_land.py: This file contains the implementation of the GridLand environment.
* visualize/: This directory contains the implementation of the grid visualization using PyQt5.
  * grid_window.py: This file contains the implementation of the grid visualization.

## Running the Project

To run the project, execute the main.py script:
'python main.py'

This will start the game loop, where the agent will interact with the GridLand environment. The agent's actions and the environment's responses will be visualized in a PyQt5 window.

## Implementation Details

The agent is trained using the Proximal Policy Optimization (PPO) algorithm, which is a type of policy gradient method for reinforcement learning. PPO uses a dual network architecture for policy and value estimation. The policy network outputs a distribution over actions given a state, and the value network estimates the expected return given a state.

The PPO agent interacts with the environment by selecting actions based on the current policy, executing these actions in the environment, and storing the resulting state transitions in a memory buffer. The agent then uses these stored experiences to update the parameters of the policy and value networks.

The PPO update step involves calculating the returns from the stored rewards and value estimates using Generalized Advantage Estimation (GAE), and then optimizing the policy and value networks based on these returns and the old policy. The policy update is clipped to prevent large policy updates, and an entropy bonus is included in the loss function to encourage exploration.

The agent is implemented from scratch, providing a detailed example of how to implement the PPO algorithm in PyTorch.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.
