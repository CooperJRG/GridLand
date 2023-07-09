class LevelManager:
    """Manages levels for the game, translates them into in-game objects.

        Attributes: translate_dict (dict): A dictionary used to translate different characters into in-game object
        representations.
        levels (dict): A dictionary containing all the game levels represented as 2D lists.
        level_names (list): A list containing names of all the levels in the game.
        name_index (int): An integer counter used to iterate over level_names.
        start_position (tuple): An tuple of two integers, (x,y), representing the start position.
        end_position (tuple): An tuple of two integers, (x,y), representing the end position.
        fan_spins (list): A list used to store fan spins.
        car_spawners (list): A list used to store car spawners.
        fan_patterns (dict): A dictionary used to store patterns of fans.
    """

    def __init__(self):
        """
        Initializes the LevelManager class.

        Sets up the dictionary to translate level elements to grid world, defines various levels with their layouts,
        sets up a list of level names and initializes certain elements used for level management.
        """

        # 0 - Ground
        # 1 - Wall
        # s - Start (represented by a 2 in grid world)
        # g - Goal (represented by a 3 in grid world)
        # ~ - Electricity (represented by a 4 in grid world)
        # b - Turns off electricity (represented by a 5 in grid world)
        # ^=n - Car Spawner, n = size of car (represented by a 6 in grid world)
        # x=n - Fan, n represents the length of fan blades (represented by a 7 in grid world)
        self.translate_dict = {0: 0, 1: 1, 's': 2, 'g': 3, '~': 4, 'b': 5}
        self.levels = {
            'Bland Land': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 'g', 0, 0, 0, 's', 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            'Blander Land': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'g', 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 's', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            'Blandest Land': [
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 'g', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 's', 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            'Flappy Bird': [
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                [0, 'g', 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 's', 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            ],
            'Frogger': [
                [0, 0, 0, 0, 'v=3', 0, 0, 'v=2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 's', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, '<=2'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 'g', 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
            'Electric Boogaloo': [
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'b', 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 'g', 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 's', 0, 0],
                [0, 0, 0, 0, '~', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            'Fan Spam': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 'x=2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 's', 0, 0, 0, 0, 0, 0, 0, 0, 0, 'x=2', 0, 0, 0, 0, 'x=2', 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 'x=2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'g', 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }
        self.level_names = list(self.levels.keys())
        self.name_index = 0
        self.start_position = (0, 0)
        self.end_position = (0, 0)
        self.fan_spins = []
        self.car_spawners = []
        self.fan_patterns = {
            0: [[7]],
            1: [
                [
                    [0, 7, 0],
                    [7, 7, 7],
                    [0, 7, 0]
                ],
                [
                    [7, 0, 7],
                    [0, 7, 0],
                    [7, 0, 7]
                ],
            ],
            2: [
                [
                    [0, 0, 7, 0, 0],
                    [0, 0, 7, 0, 0],
                    [7, 7, 7, 7, 7],
                    [0, 0, 7, 0, 0],
                    [0, 0, 7, 0, 0],
                ],
                [
                    [0, 7, 0, 0, 0],
                    [0, 0, 7, 0, 7],
                    [0, 7, 7, 7, 0],
                    [7, 0, 7, 0, 0],
                    [0, 0, 0, 7, 0],
                ],
                [
                    [7, 0, 0, 0, 7],
                    [0, 7, 0, 7, 0],
                    [0, 0, 7, 0, 0],
                    [0, 7, 0, 7, 0],
                    [7, 0, 0, 0, 7],
                ],
                [
                    [0, 0, 0, 7, 0],
                    [7, 7, 0, 7, 0],
                    [0, 0, 7, 0, 0],
                    [0, 7, 0, 7, 7],
                    [0, 7, 0, 0, 0],
                ],
            ]
        }
        self.beat_count = 0
        self.current_level = self.translate_level()
        self.draw_fans()

    def next_level(self):
        """
        This function is used to load the next level in the sequence. It resets some properties of the class,
        generates the new level from the translated template, and draws the fans for this level.
        """
        self.beat_count += 1
        if self.beat_count % 8 == 0:
            self.name_index += 1
            if self.name_index == len(self.level_names):
                self.name_index = 0
            self.fan_spins = []
            self.car_spawners = []
            self.current_level = self.translate_level()
            self.draw_fans()
            self.draw_spawners()
        else:
            self.restart_level()

    def update_level(self):
        """
        This function is used to update the game level. It redraws the fans and car spawners on the current level.
        """
        self.draw_fans()
        self.draw_spawners()

    def restart_level(self):
        """
        This function is used to reset the level. It resets some properties of the class,
        generates the new level from the translated template, and draws the fans for this level.
        """
        self.fan_spins = []
        self.car_spawners = []
        self.current_level = self.translate_level()
        self.draw_fans()

    def button_pressed(self):
        """
        This method is used to process the event of a button press in the game. It scans through each cell in
        the current game level. If the cell represents electricity (indicated by the value 4), it is set to 0,
        effectively 'pressing' the button in the game's logic.
        """
        for i, row in enumerate(self.current_level):
            for j, cell in enumerate(row):
                if cell == 4:
                    self.current_level[i][j] = 0

    def translate_level(self):
        """
        This function translates the current level template using the translation dictionary.
        It also identifies complex cells that require further handling (fans and car spawners),
        updating the fan_spins and car_spawners lists accordingly.
        Returns a 2D list of integers representing the translated level.
        """
        template = self.levels[self.level_names[self.name_index]]
        result = []
        complicated_cells = []
        for i, row in enumerate(template):
            result_row = []
            for j, cell in enumerate(row):
                if cell in self.translate_dict.keys():
                    result_row.append(self.translate_dict[cell])
                    if cell == 's':
                        self.start_position = (j, i)
                    elif cell == 'g':
                        self.end_position = (j, i)
                else:
                    complicated_cells.append((i, j))
                    result_row.append(cell)
            result.append(result_row)

        for i, j in complicated_cells:
            cell = result[i][j]
            if cell[0] == 'x':
                num_blades = int(cell[-1])
                self.fan_spins.append((i, j, num_blades, 0))
                result[i][j] = 7
            else:
                car_length = int(cell[-1])
                direction = cell[0]
                self.car_spawners.append((i, j, car_length, direction, 0))
                result[i][j] = 0
        return result

    # This function is used to draw or erase fans on the current level based on the fan spins and patterns
    def draw_fans(self):
        """
        This function is responsible for drawing or erasing fans on the current level based on the fan spins and
        patterns. It erases all the existing fans and then redraws them according to the updated fan spins.
        """
        # Get the current level template
        template = self.levels[self.level_names[self.name_index]]

        # First, erase all the existing fans on the current level
        self.update_fans(template, erase_mode=True)

        # Prepare a temporary list to hold updated fan spins
        temp_fan_spins = []

        # Then, draw all the fans on the current level
        self.update_fans(template, erase_mode=False, temp_fan_spins=temp_fan_spins)

        # Update the fan spins with the updated ones
        self.fan_spins = temp_fan_spins

    def update_fans(self, template, erase_mode, temp_fan_spins=None):
        """
        This function iterates over all fans in fan_spins. If erase_mode is True, it erases the fans.
        If not, it creates a new fan spin with an updated index and draws the fan.
        """
        for i, j, num_blades, index in self.fan_spins:
            fan_pattern = self.fan_patterns[num_blades][index]

            # Calculate the starting indices based on the number of fan blades
            start_i, start_j = i - num_blades, j - num_blades

            # If erase_mode is True, it erases the fans. If not, it draws them
            if erase_mode:
                self.modify_fan_area(start_i, start_j, fan_pattern, template, erase_mode)
            else:
                # Create a new fan spin with an updated index
                new_index = (index + 1) % len(self.fan_patterns[num_blades])
                temp_fan_spins.append((i, j, num_blades, new_index))
                self.modify_fan_area(start_i, start_j, fan_pattern, template, erase_mode)

    def modify_fan_area(self, start_i, start_j, fan_pattern, template, erase_mode):
        """
        This function modifies a specific area of the game level by either drawing or erasing a fan.
        It ensures that the modified indices are within the game level boundaries.
        """
        for i_offset in range(len(fan_pattern)):
            for j_offset in range(len(fan_pattern)):
                # Calculate the actual indices
                i_index, j_index = start_i + i_offset, start_j + j_offset

                # Ensure the indices are within the boundaries of the current level
                if self.validate_indices(i_index, j_index):
                    if erase_mode and self.current_level[i_index][j_index] == 7:
                        self.current_level[i_index][j_index] = self.get_original_value(template, i_index, j_index)
                    elif not erase_mode and fan_pattern[i_offset][j_offset] == 7:
                        self.current_level[i_index][j_index] = 7

    def validate_indices(self, i, j):
        """
        This function checks whether given indices are within the current level boundaries.
        Returns True if they are, False otherwise.
        """
        return 0 <= i < len(self.current_level) and 0 <= j < len(self.current_level[0])

    def get_original_value(self, template, i, j):
        """
        This function retrieves the original value from the level template at the given indices and
        translates it if necessary. Returns the translated value.
        """
        original_value = template[i][j]
        return self.translate_dict.get(original_value, 0)

    def draw_spawners(self):
        """
        This function updates each car spawner on the level by moving the car along its path and spawning a new car
        if necessary. It uses a helper function update_car_position to update each car's position.
        """
        temp_car_spawners = []
        for i, j, car_length, direction, counter in self.car_spawners:
            # Call helper function to update the car position
            self.update_car_position(i, j, direction)

            # If the counter is less than the car length, update the car cell at the spawner's position
            if counter < car_length:
                self.current_level[i][j] = 6

            # Update the counter or reset it if it's greater than twice the car length
            counter = counter + 1 if counter + 1 < car_length * 2 else 0
            temp_car_spawners.append((i, j, car_length, direction, counter))

        self.car_spawners = temp_car_spawners

    def update_car_position(self, i, j, direction):
        """
        This function updates a car's position based on its direction by deleting the last cell of the car.
        The updated car cells are determined based on the direction of movement.
        """
        last_cell = 0

        # Determine the iteration pattern based on the car's direction
        if direction in ['^', 'v']:
            rows = reversed(self.current_level) if direction == '^' else self.current_level
            for row in rows:
                if row[j] in [6, 0]:
                    last_cell = self.swap_cells(row, j, last_cell)
                else:
                    break

        elif direction in ['<', '>']:
            cells = reversed(self.current_level[i]) if direction == '<' else self.current_level[i]
            for x, cell in enumerate(cells):
                if direction == '<':
                    x = j - x
                if cell in [6, 0]:
                    last_cell = self.swap_cells(self.current_level[i], x, last_cell)
                else:
                    break

    @staticmethod
    def swap_cells(row, j, last_cell):
        """
        This function swaps the value of a given cell in a row with the provided last_cell value.
        Returns the original cell value before the swap.
        """
        temp_cell = row[j]
        row[j] = last_cell
        return temp_cell
