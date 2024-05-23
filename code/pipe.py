"""Pipe Mania Solver for the AI course's project."""

# Grupo 84:
# 100290 Armando Gonçalves
# 100326 João Rodrigues

from sys import stdin, stdout

import numpy as np

from search import Problem, Node, depth_first_tree_search


class PipeManiaState:
    """Describes the current state of the problem.

    This class receives an instance of the class Board
    and increases its `state_id` argument everytime it
    is called.
    """
    state_id = 0

    def __init__(self, board, current_action_piece):
        """Initializes the class.

        Args:
            board (Board): An instance of the Board class.
        """
        self.board = board
        self.current_action_piece = current_action_piece
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        """To manage ties.

        This method is used in case of tie in the management of the open 
        list in informed searches.
        """
        return self.id < other.id


class Board:
    """Describes important actions to access information about the board."""

    def __init__(self, board: np.ndarray, fixed_pieces, possible_actions):
        """Initializes the class.

        Args:
            board (np.ndarray): A multidimensional array (r*c) where 
              'r' is the number of rows and 'c' the number of columns
              containing strings representing the several pieces on
              the board and their corresponding orientation.
        """
        self.board = board
        self.fixed_pieces = fixed_pieces
        self.possible_actions = possible_actions

    def preprocessing(self):

        # List with the tuples of row and col of pieces that
        # are fixed.

        fixed_pieces = []
        possible_actions = {}
        rows, cols = self.board.shape

        # Top Left corner
        if self.board[0][0][0] == 'V':
            self.board[0][0] = 'VB'
            fixed_pieces.append((0, 0))

        if self.board[0][0][0] == 'F':
            possible_actions[(0, 0)] = ['FB', 'FD']

        # Top right corner
        if self.board[0][cols-1][0] == 'V':
            self.board[0][cols-1] = 'VE'
            fixed_pieces.append((0, cols-1))

        if self.board[0][cols-1][0] == 'F':
            possible_actions[(0, cols-1)] = ['FB', 'FE']

        # Bottom left corner
        if self.board[rows-1][0][0] == 'V':
            self.board[rows-1][0] = 'VD'
            fixed_pieces.append((rows-1, 0))

        if self.board[rows-1][0][0] == 'F':
            possible_actions[(rows-1, 0)] = ['FC', 'FD']

        # Bottom right corner
        if self.board[rows-1][cols-1][0] == 'V':
            self.board[rows-1][cols-1] = 'VC'
            fixed_pieces.append((rows-1, cols-1))

        if self.board[rows-1][cols-1][0] == 'F':
            possible_actions[(rows-1, cols-1)] = ['FC', 'FE']

        # Top edge
        for col in range(1, cols-1):
            if self.board[0][col][0] == 'B':
                self.board[0][col] = 'BB'
                fixed_pieces.append((0, col))

            if self.board[0][col][0] == 'L':
                self.board[0][col] = 'LH'
                fixed_pieces.append((0, col))

            if self.board[0][col][0] == 'F':
                possible_actions[(0, col)] = ['FB', 'FE', 'FD']

            if self.board[0][col][0] == 'V':
                possible_actions[(0, col)] = ['VB', 'VE']

        # Left edge
        for row in range(1, rows-1):
            if self.board[row][0][0] == 'B':
                self.board[row][0] = 'BD'
                fixed_pieces.append((row, 0))

            if self.board[row][0][0] == 'L':
                self.board[row][0] = 'LV'
                fixed_pieces.append((row, 0))

            if self.board[row][0][0] == 'F':
                possible_actions[(row, 0)] = ['FB', 'FC', 'FD']

            if self.board[row][0][0] == 'V':
                possible_actions[(row, 0)] = ['VB', 'VD']

        # Right edge
        for row in range(1, rows-1):
            if self.board[row][cols-1][0] == 'B':
                self.board[row][cols-1] = 'BE'
                fixed_pieces.append((row, cols-1))

            if self.board[row][cols-1][0] == 'L':
                self.board[row][cols-1] = 'LV'
                fixed_pieces.append((row, cols-1))

            if self.board[row][cols-1][0] == 'F':
                possible_actions[(row, cols-1)] = ['FB', 'FC', 'FE']

            if self.board[row][cols-1][0] == 'V':
                possible_actions[(row, cols-1)] = ['VE', 'VC']

        # Bottom edge
        for col in range(1, cols-1):
            if self.board[rows-1][col][0] == 'B':
                self.board[rows-1][col] = 'BC'
                fixed_pieces.append((rows-1, col))

            if self.board[rows-1][col][0] == 'L':
                self.board[rows-1][col] = 'LH'
                fixed_pieces.append((rows-1, col))

            if self.board[rows-1][col][0] == 'F':
                possible_actions[(rows-1, col)] = ['FD', 'FC', 'FE']

            if self.board[rows-1][col][0] == 'V':
                possible_actions[(rows-1, col)] = ['VD', 'VC']

        # Loop to infer and fix as many points on the board as possible

        dummy = [item for item in fixed_pieces]

        updated = True
        while updated:
            updated = False
            new_fixed_pieces = []   

            for (r, c) in dummy:
                
                # Check the neighboring cells and update possible actions
                if r > 0:
                    if (r-1, c) not in fixed_pieces:
                        
                        possible_pieces = check_neighbour(r, c, r-1, c, 'upper', self.board, possible_actions.get((r - 1, c), []))
                            
                        if possible_actions.get((r - 1, c)) != possible_pieces:
                            possible_actions[(r - 1, c)] = possible_pieces
                            updated = True
                        #possible_actions[(r - 1, c)] = possible_pieces

                            if len(possible_actions[(r - 1, c)]) == 1:
                                self.board[r - 1][c] = possible_actions[(r - 1, c)][0]
                                new_fixed_pieces.append((r - 1, c))

                if r < rows - 1:
                    if (r+1, c) not in fixed_pieces:
                        
                        possible_pieces = check_neighbour(r, c, r+1, c, 'lower', self.board, possible_actions.get((r + 1, c), []))
                            
                        if possible_actions.get((r + 1, c)) != possible_pieces:
                            possible_actions[(r + 1, c)] = possible_pieces
                            updated = True
                        #possible_actions[(r + 1, c)] = possible_pieces

                            if len(possible_actions[(r + 1, c)]) == 1:
                                self.board[r + 1][c] = possible_actions[(r + 1, c)][0]
                                new_fixed_pieces.append((r + 1, c))

                if c > 0:
                    if (r, c-1) not in fixed_pieces:
                    
                        possible_pieces = check_neighbour(r, c, r, c - 1, 'left', self.board, possible_actions.get((r, c - 1), []))
                            

                        if possible_actions.get((r, c - 1)) != possible_pieces:
                            possible_actions[(r, c - 1)] = possible_pieces
                            updated = True
                        #possible_actions[(r, c - 1)] = possible_pieces
                            if len(possible_actions[(r, c - 1)]) == 1:
                                self.board[r][c - 1] = possible_actions[(r, c - 1)][0]
                                new_fixed_pieces.append((r, c - 1))

                if c < cols - 1:
                    if (r, c+1) not in fixed_pieces:
                    
                        possible_pieces = check_neighbour(r, c, r, c + 1, 'right', self.board, possible_actions.get((r, c + 1), []))
                            
                        if possible_actions.get((r, c + 1)) != possible_pieces:
                            possible_actions[(r, c + 1)] = possible_pieces
                            updated = True
                        #possible_actions[(r, c + 1)] = possible_pieces
                            if len(possible_actions[(r, c + 1)]) == 1:
                                self.board[r][c + 1] = possible_actions[(r, c + 1)][0]
                                new_fixed_pieces.append((r, c + 1))

            dummy = new_fixed_pieces
            fixed_pieces.extend(new_fixed_pieces)
            fixed_pieces = list(set(fixed_pieces))

        #print(fixed_pieces)
        #print(len(fixed_pieces))
        self.fixed_pieces = list(set(fixed_pieces))
        self.possible_actions = possible_actions

    def adjacent_vertical_values(self, row: int, col: int):
        """Obtains the values immediately above and below a piece, respectively.

        When there is no piece in a certain location, it returns `None`.

        Args:
            row (int): Index of the row.
            col (int): Index of the column.

        Returns:
            upper_value, lower_value (tuple of str): Values immediately above and 
              below a piece, respectively.
        """
        upper_value = self.board[row - 1][col] if row > 0 else None
        lower_value = self.board[row +
                                 1][col] if row < len(self.board) - 1 else None
        return upper_value, lower_value

    def adjacent_horizontal_values(self, row: int, col: int):
        """Obtains the values immediately to the left and right, respectively.

        When there is no piece in a certain location, it returns `None`.

        Args:
            row (int): Index of the row.
            col (int): Index of the column.

        Returns:
            left_value, right_value (tuple of str): Values immediately left and 
              right of a piece, respectively.
        """
        left_value = self.board[row][col - 1] if col > 0 else None
        right_value = self.board[row][col +
                                      1] if col < len(self.board[0]) - 1 else None
        return left_value, right_value

    def get_value(self, row: int, col: int) -> str:
        """Returns the value at the specified position.

        Args:
            row (int): Index of the row.
            col (int): Index of the column.

        Returns:
            value (str): Description of the piece at that position.
        """
        return self.board[row][col]

    def print(self):
        """Prints the board.

        Generates an output of the board with the same specifications
        as the required input. For example:

        VB VE
        FC FC
        """
        for row in self.board:
            stdout.write("\t".join(row) + "\n")

    @staticmethod
    def parse_instance():
        """Reads input to obtain the board.

        Reads the problem instance from standard input (stdin) 
        and returns an instance of the Board class. This method
        is called everytime a standard input is used.

        Returns:
            board (class Board): Creates an instance of the board
              using the information of the respective input.
        """
        lines = stdin.readlines()
        board = np.array([line.strip().split() for line in lines])
        return Board(board, [], {})


class PipeMania(Problem):
    """Defines the search problem.

    The PipeMania class inherits from the Problem class defined in the search.py file.

    The `actions` method takes a state as an argument and returns a list of actions that
    can be executed from that state. 

    The `result` method takes a state and an action as arguments, and returns the result 
    of applying that action to that state.

    The `h` function corresponds to the heuristic and can be used to conduct a more 
    controlled and optimized search.

    """

    def __init__(self, initial_state: PipeManiaState):
        """Initializes the class.

        Args:
            initial_state (class Board): Instance of the board's initial state.
        """
        self.initial = initial_state

    def actions(self, state: PipeManiaState):
        """Actions that can be performed."""

        rows, cols = state.board.board.shape
        found_piece = False
        actions = []
        r, c = state.current_action_piece

        for row in range(r, rows):
            for col in range(cols):
                if (row == r and col <= c) or ((row, col) in state.board.fixed_pieces):
                    pass

                else:
                    if (row, col) in state.board.possible_actions:
                        actions = [(row, col, orientation)
                                   for orientation in state.board.possible_actions[(row, col)]]

                    else:
                        actions = [(row, col, orientation)
                                   for orientation in possible_orientations[state.board.board[row][col][0]]]

                    actions = check_compatibility(row, col, actions, state)
                    found_piece = True
                    break

            if found_piece:
                break

        return actions

    def result(self, state: PipeManiaState, action):
        """Result of a given action.

        Given an action, the necessary rotations are performed
        and a new state is created and then returned.

        Args:
            state (class PipeManiaState): Instance of the board's current state.
            action (3D tuple): Encode of the action to be performed.

        Returns:
            result_state (class PipeManiaState): Instance of the board's 
              resulting state.
        """
        row, col, orientation = action
        new_board = np.copy(state.board.board)
        new_board[row][col] = orientation

        return PipeManiaState(Board(new_board, state.board.fixed_pieces, state.board.possible_actions), (row, col))

    def goal_test(self, state: PipeManiaState) -> bool:

        rows, cols = state.board.board.shape
        total_pieces = rows*cols
        checked_pieces = []
        neighbours = []

        for adjacent_loc in possible_adjacent_locations[state.board.board[0][0]]:
            row, col = row_col_correspondence[adjacent_loc]
            if state.board.board[row][col] not in adjacent_pieces[adjacent_loc]:
                return False
            
            neighbours.append((row, col))

        checked_pieces = [(0,0)]

        there_are_neighbours = True
        while there_are_neighbours:
            new_neighbours = []

            for (row, col) in neighbours:
                for adjacent_loc in possible_adjacent_locations[state.board.board[row][col]]:
                    row_n = row + row_col_correspondence[adjacent_loc][0]
                    col_n = col + row_col_correspondence[adjacent_loc][1]
                    if (row_n, col_n) in checked_pieces:
                        pass
                    else:
                        if state.board.board[row_n][col_n] not in adjacent_pieces[adjacent_loc]:
                            return False
                    
                        new_neighbours.append((row_n, col_n))
                
                checked_pieces.append((row, col))

            neighbours = [neig for neig in new_neighbours]
            if not new_neighbours:
                there_are_neighbours = False

        if len(checked_pieces) != total_pieces:
            return False

        return True

    def h(self, node: Node) -> int:
        """ Heuristic function used for A* search. Number of pieces not connected."""

        rows, cols = node.state.board.board.shape
        total_pieces = rows*cols
        checked_pieces = []
        neighbours = []

        for adjacent_loc in possible_adjacent_locations[node.state.board.board[0][0]]:
            row, col = row_col_correspondence[adjacent_loc]
            if node.state.board.board[row][col] not in adjacent_pieces[adjacent_loc]:
                return total_pieces - len(checked_pieces)
            
            neighbours.append((row, col))

        checked_pieces = [(0,0)]

        there_are_neighbours = True
        while there_are_neighbours:
            new_neighbours = []

            for neighbour in neighbours:
                row, col = neighbour
                for adjacent_loc in possible_adjacent_locations[node.state.board.board[row][col]]:
                    row_n = row + row_col_correspondence[adjacent_loc][0]
                    col_n = col + row_col_correspondence[adjacent_loc][1]
                    if (row_n, col_n) in checked_pieces:
                        pass
                    else:
                        if node.state.board.board[row_n][col_n] not in adjacent_pieces[adjacent_loc]:
                            return total_pieces - len(checked_pieces)
                    
                        new_neighbours.append((row_n, col_n))
                
                checked_pieces.append(neighbour)

            neighbours = [neig for neig in new_neighbours]
            if not new_neighbours:
                there_are_neighbours = False

        return total_pieces - len(checked_pieces)

def check_neighbour(row, col, row_n, col_n, direction, board, initial_pos_actions):
    """Determine possible pieces for the neighboring cell in a given direction on the grid.

    Args:
        row (int): The row index of the current cell.
        col (int): The column index of the current cell.
        row (int): The row index of the neighbour cell.
        col (int): The column index of the neighbour cell.
        direction (str): The direction to check ('upper', 'lower', 'left', 'right').
        board (2D list): The current state of the board.

    Returns:
        list: A list of possible pieces (actions) for the neighboring cell in the given direction.
    """
    current_piece = board[row][col]
    neighbour_piece = board[row_n][col_n]
    possible_pieces = []

    #print("Current piece: ", current_piece)
    #print("Neighbour piece: ", neighbour_piece)
    #print("Initial_pos_actions: ", initial_pos_actions)

    if not initial_pos_actions:
        if direction in possible_adjacent_locations[current_piece]:
            for piece in adjacent_pieces[direction]:
                if piece[0] == neighbour_piece[0]:
                    possible_pieces.append(piece)
                
            return possible_pieces
        
        return [item for item in possible_orientations[neighbour_piece[0]] if item not in adjacent_pieces[direction]]

    if direction in possible_adjacent_locations[current_piece]:
        return [item for item in initial_pos_actions if item in adjacent_pieces[direction]]
    
    # Exemplo. A current piece é 'LV' e estamos a checkar o vizinho à esquerda que é 
    # do tipo 'B'. A peça 'LV' não pode ter ligações à sua esquerda, por isso, neste
    # caso, se a neighbour piece pertencer às peças que podem estar à esquerda (pq têm
    # ligação para a direita) podemos eliminá-las.
    return [item for item in initial_pos_actions if item not in adjacent_pieces[direction]]

def check_compatibility(row, col, initial_actions, state):
    """Checks compatibility of actions.

    Given the pieces already searched in the tree (since a piece
    by piece search is being implemented) it checks compatibility of
    the initial_actions with the pieces already checked in that state.
    If there is no way of them forming a solution, that part of the
    tree can be ignored, thus saving time.
    """
    possible_actions = [item for item in initial_actions]
    upper, lower = state.board.adjacent_vertical_values(row, col)
    left, right = state.board.adjacent_horizontal_values(row, col)

    for action in initial_actions:
        was_it_removed = False
        if not upper is None:
            if 'lower' in possible_adjacent_locations[upper]:
                if action[2] not in adjacent_pieces['lower']:
                    possible_actions.remove(action)
                    was_it_removed = True

            else:
                if action[2] in adjacent_pieces['lower']:
                    possible_actions.remove(action)
                    was_it_removed = True

        if (not left is None) and not was_it_removed:
            if 'right' in possible_adjacent_locations[left]:
                if action[2] not in adjacent_pieces['right']:
                    possible_actions.remove(action)

            else:
                if action[2] in adjacent_pieces['right']:
                    possible_actions.remove(action)
                    was_it_removed = True

        if (not lower is None) and not was_it_removed and ((row+1, col) in state.board.fixed_pieces):
            if 'upper' in possible_adjacent_locations[lower]:
                if action[2] not in adjacent_pieces['upper']:
                    possible_actions.remove(action)
                    was_it_removed = True

            else:
                if action[2] in adjacent_pieces['upper']:
                    possible_actions.remove(action)
                    was_it_removed = True

        if (not right is None) and not was_it_removed and ((row, col+1) in state.board.fixed_pieces):
            if 'left' in possible_adjacent_locations[right]:
                if action[2] not in adjacent_pieces['left']:
                    possible_actions.remove(action)

            else:
                if action[2] in adjacent_pieces['left']:
                    possible_actions.remove(action)
                    was_it_removed = True

    return possible_actions


# Dictionary for types of pieces that can be adjacent
adjacent_pieces = {
    'right': ['FE', 'BB', 'BE', 'VC', 'VE', 'LH', 'BC'],
    'left': ['FD', 'BB', 'BD', 'BC', 'VB', 'VD', 'LH'],
    'upper': ['FB', 'BB', 'BE', 'BD', 'VB', 'VE', 'LV'],
    'lower': ['FC', 'BC', 'BD', 'BE', 'VC', 'VD', 'LV']
}

possible_adjacent_locations = {
    'FC': ['upper'],
    'FB': ['lower'],
    'FE': ['left'],
    'FD': ['right'],
    'BC': ['upper', 'left', 'right'],
    'BB': ['lower', 'left', 'right'],
    'BE': ['left', 'upper', 'lower'],
    'BD': ['right', 'upper', 'lower'],
    'VC': ['upper', 'left'],
    'VB': ['right', 'lower'],
    'VE': ['left', 'lower'],
    'VD': ['right', 'upper'],
    'LH': ['left', 'right'],
    'LV': ['upper', 'lower']
}

possible_orientations = {
    'F': ['FB', 'FC', 'FD', 'FE'],
    'B': ['BB', 'BC', 'BD', 'BE'],
    'V': ['VB', 'VC', 'VD', 'VE'],
    'L': ['LH', 'LV']
}

row_col_correspondence = {
    'lower': (1, 0),
    'upper': (-1, 0),
    'right': (0, 1),
    'left': (0, -1)
}

# Example usage:
if __name__ == "__main__":
    initial_board = Board.parse_instance()
    initial_board.preprocessing()
    #initial_board.print()
    # Has to start with -1 otherwise it does not generate actions for (0,0) piece.
    s0 = PipeManiaState(initial_board, (0, -1))
    problem = PipeMania(s0)
    goal_node = depth_first_tree_search(problem)
    goal_node.state.board.print()
