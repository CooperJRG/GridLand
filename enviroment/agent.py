class Agent:
    """
    This is a class for creating an agent in a 2D grid. The agent can move around the grid based on action tuples.

    Attributes:
        x (int): The current x-coordinate of the agent.
        y (int): The current y-coordinate of the agent.
        previous (tuple): The previous coordinates of the agent.
    """

    def __init__(self, x, y):
        """
        The constructor for the Agent class.

        Parameters:
           x (int): The initial x-coordinate of the agent.
           y (int): The initial y-coordinate of the agent.
        """
        self.x = x
        self.y = y
        self.previous = (x, y)  # Stores the initial position of the agent as the previous position

    def move(self, action_tuple):
        """
        The function to move the agent based on the action tuple.

        Parameters:
            action_tuple (tuple): A tuple containing the change in x and y coordinates.
        """
        self.previous = (self.x, self.y)  # Stores the current position as the previous position before moving
        # Update position based on action
        self.x += action_tuple[0]  # Update x-coordinate based on the first element of the action tuple
        self.y += action_tuple[1]  # Update y-coordinate based on the second element of the action tuple

    def out_of_bounds(self, grid):
        """
        The function to check if the agent is out of the grid bounds.

        Parameters:
            grid (list): A 2D list representing the grid.

        Returns:
            bool: True if the agent is out of the grid bounds, False otherwise.
        """
        # Check if the y-coordinate is within the grid bounds (0 to grid height)
        # Check if the x-coordinate is within the grid bounds (0 to grid width)
        # If the agent is out of the bounds, it returns True. Otherwise, it returns False
        return not (0 <= self.y < len(grid) and 0 <= self.x < len(grid[0]))
