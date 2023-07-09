import copy
import sys
import torch
from PyQt5 import QtWidgets, QtCore

from PPO.PPO import PPO
from PPO.dual_network import DualNetwork
from PPO.memory import Memory
from enviroment.grid_land import GridLand
from visualize.grid_window import GridWindow


def main():
    # Create an instance of GridLand environment
    env = GridLand()

    # Mapping of integers to actions
    int_to_action_mapping = ['Up', 'Left', 'Down', 'Right']

    # Get the current state of the environment
    state = env.generate_state()

    # Check available device for computation
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)

    # Define the network architecture parameters
    layers = [256, 128, 128, 64]
    cnn_layers = [32, 64, 64, 128]
    kernal = [(1, 1), (1, 1), (2, 2), (2, 2)]
    stride = [(1, 1), (1, 1), (1, 1), (1, 1)]

    # Initialize DualNetwork for policy and value estimation
    net = DualNetwork(input_dim=tuple(state.shape), device=device, use_cnn=True, layers=layers, cnn_layers=cnn_layers,
                      kernel_sizes=kernal, strides=stride, num_actions=len(env.possible_actions))

    # Initialize PPO for policy optimization
    ppo = PPO(net, lr=0.0003, eps_clip=0.22,
              value_coef=0.5, entropy_coef=0.05, gamma=0.99, gae_lambda=0.96)

    # Initialize memory for storing transitions
    memory = Memory(device)

    level = env.level_manager.level_names[env.level_manager.name_index]
    num_simulations = 0

    # Initialize PyQt5 Application for visualizing the grid environment
    app = QtWidgets.QApplication(sys.argv)
    window = GridWindow(env.grid)
    window.show()

    # Main loop
    while True:
        count = 1
        while not env.done and count < 300:
            # Determine the action using PPO
            action_tensor, logprob, value = ppo.select_action(state)
            action_name = int_to_action_mapping[action_tensor.item()]
            action = env.possible_actions[action_name]

            # Take a step in the environment using the selected action
            next_state, reward = env.step(action)

            # Check if the episode is terminated
            is_terminal = env.done

            # Store the transition in memory
            memory.add(state, action_tensor, logprob, torch.tensor(reward), torch.tensor(is_terminal), value)

            # Update the current state
            state = next_state
            count += 1

            # Update the grid for visualization
            grid = copy.deepcopy(env.grid)
            grid[env.agent.y][env.agent.x] = env.cells['Agent']
            window.update_grid(grid)
            QtCore.QCoreApplication.processEvents()

        env.reset_game()

        # Calculate the average reward
        mean_reward = torch.mean(torch.stack([tensor.float() for tensor in memory.rewards])).item()

        if num_simulations % 100 == 0:
            print(f'Number of Simulations: {num_simulations}')
            print(f'Average Reward: {mean_reward}')

        new_level = env.level_manager.level_names[env.level_manager.name_index]

        # If a level has been completed, print the simulation statistics
        if level != new_level:
            if num_simulations % 100 != 0:
                print(f'Number of Simulations: {num_simulations}')
                print(f'Average Reward: {mean_reward}')
            print(f'Completed {level} in {count} steps, moving on to {new_level}')
            level = new_level

        # Update the network using the collected transitions
        if len(memory.rewards) > 0:
            ppo.update(memory, 10, 32)
            num_simulations += 1

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
