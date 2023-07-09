import copy
import torch
from enviroment.agent import Agent
from enviroment.level_manager import LevelManager


class GridLand:
    """
    A class to represent a grid land in the environment.

    ...

    Attributes
    ----------
    cell_size : int
        size of each cell in the grid (default 90)
    border : int
        size of the border around the grid (default 50)
    ...

    Methods
    -------
    agent_cell():
        Returns the cell occupied by the agent
    is_goal_reached():
        Checks if the agent has reached the goal
    step(action):
        Performs the specified action and returns the new state and reward
    get_state_and_reward():
        Determines the state and reward after the step
    reset_game():
        Resets the game to its initial state
    collision_detection():
        Checks if the agent has collided with an obstacle
    distance_calculator():
        Calculates the Manhattan Distance between the agent's previous and current positions
    was_button_pressed():
        Checks if a button was pressed by the agent
    generate_state():
        Generates the current state of the grid
    """

    def __init__(self, cell_size=90, border=50):
        self.agent_was_moving = False

        # Defining the various cell types
        self.cells = {
            'Ground': 0,
            'Wall': 1,
            'Start': 2,
            'Goal': 3,
            'Electricity': 4,
            'Button': 5,
            'Car': 6,
            'Fan': 7,
            'Agent': 8
        }

        # Defining the possible actions
        self.possible_actions = {
            'Left': (-1, 0),
            'Right': (1, 0),
            'Up': (0, -1),
            'Down': (0, 1),
        }

        # Defining the weights (rewards) for each action
        self.weights = {
            'Collision': -0.5,
            'Button Press': 0.7,
            'Reach End': 1,
            'Distance': 0.2,  # Positive if it gets closer, Negative if it gets farther
            'Survival': 0.001  # Slightly positive reward for staying alive
        }

        self.cell_size = cell_size
        self.border = border
        self.level_manager = LevelManager()
        self.grid = self.level_manager.current_level
        self.start_position = self.level_manager.start_position
        self.end_position = self.level_manager.end_position
        self.agent = Agent(*self.start_position)
        self.agent_position = (
            border + self.agent.x * cell_size + cell_size / 2, border + self.agent.y * cell_size + cell_size / 2)
        self.button_down = False
        self.done = False
        self.steps_survived = 0

    def agent_cell(self):
        """Return the cell occupied by the agent."""
        return self.grid[self.agent.y][self.agent.x]

    def is_goal_reached(self):
        """Check if the agent has reached the goal and perform necessary operations."""
        if self.agent_cell() == self.cells['Goal']:
            self.level_manager.next_level()
            self.start_position = self.level_manager.start_position
            self.end_position = self.level_manager.end_position
            self.agent = Agent(*self.start_position)
            self.button_down = False
            self.done = True
            return self.weights['Reach End']
        return 0

    def step(self, action):
        """Execute the action, update the agent's position, return the new state and reward."""
        old_agent_position = (self.agent.x, self.agent.y)
        self.agent.move(action)
        self.agent_was_moving = old_agent_position != (self.agent.x, self.agent.y)
        self.level_manager.update_level()
        self.grid = self.level_manager.current_level
        return self.get_state_and_reward()

    def get_state_and_reward(self):
        """Determine the state and reward after the step."""
        reward = 0
        collision_reward = self.collision_detection()
        if collision_reward != 0:
            reward += collision_reward
            self.steps_survived = 0
        else:
            reward += self.distance_calculator()
            reward += self.was_button_pressed()
            goal_reached_reward = self.is_goal_reached()
            if goal_reached_reward != 0:
                reward += goal_reached_reward
                self.steps_survived = 0
            else:
                self.steps_survived += 1
                survival_reward = self.weights['Survival'] * 0.99 ** self.steps_survived
                reward += survival_reward
        state = self.generate_state()
        return state, reward

    def reset_game(self):
        """Reset the game to its initial state."""
        self.done = False
        self.agent.x = self.start_position[0]
        self.agent.y = self.start_position[1]
        self.agent.previous = (self.agent.x, self.agent.y)
        self.level_manager.restart_level()
        self.button_down = False
        self.steps_survived = 0

    def collision_detection(self):
        """Check if the agent has collided with an obstacle and perform necessary operations."""
        if self.agent.out_of_bounds(self.grid) or self.agent_cell() in [self.cells['Wall'],
                                                                        self.cells['Electricity'],
                                                                        self.cells['Car'], self.cells['Fan']]:
            '''
            self.done = True
            self.agent.x = self.start_position[0]
            self.agent.y = self.start_position[1]
            self.agent.previous = (self.agent.x, self.agent.y)
            self.level_manager.restart_level()
            self.button_down = False
            '''
            self.agent.x = self.agent.previous[0]
            self.agent.y = self.agent.previous[1]
            return self.weights['Collision']
        return 0

    def distance_calculator(self):
        """Calculate the Manhattan Distance between the agent's previous and current positions."""
        previous = self.agent.previous
        previous_manhattan_distance = abs(previous[0] - self.end_position[0]) + abs(
            previous[1] - self.end_position[1])
        current_manhattan_distance = abs(self.agent.x - self.end_position[0]) + abs(
            self.agent.y - self.end_position[1])
        delta = previous_manhattan_distance - current_manhattan_distance
        if delta < 0:
            delta *= self.weights['Distance']
        else:
            delta *= self.weights['Distance'] * 1.5
        return delta

    def was_button_pressed(self):
        """Check if a button was pressed by the agent."""
        if self.agent_cell() == self.cells['Button'] and not self.button_down:
            self.level_manager.button_pressed()
            self.button_down = True
            return self.weights['Button Press']
        return 0

    def generate_state(self):
        """Generate the current state of the grid."""
        state = copy.deepcopy(self.grid)
        state[self.agent.y][self.agent.x] = self.cells['Agent']

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32)

        # Normalize the values
        state /= float(self.cells['Agent'])  # Normalize between 0 and 1

        state = state.unsqueeze(0)  # Add channel dimension at index 0

        return state

