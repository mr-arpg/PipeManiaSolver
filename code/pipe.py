"""Pipe Mania Solver for the AI course's project."""

# Grupo 84:
# 100290 Armando Gonçalves
# 100326 João Rodrigues

from sys import stdin, stdout

import numpy as np

from search import Problem, Node, astar_search


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

    def __init__(self, board: np.ndarray):
        """Initializes the class.

        Args:
            board (np.ndarray): A multidimensional array (r*c) where 
              'r' is the number of rows and 'c' the number of columns
              containing strings representing the several pieces on
              the board and their corresponding orientation.
        """
        self.board = board
        self.fixed_pieces, self.possible_actions = self.preprocessing()

    def preprocessing(self):

        # List with the tuples of row and col of pieces that
        # are fixed.

        fixed_pieces = []
        possible_actions = {}
        rows, cols = self.board.shape

        #Top Left corner
        if self.board[0][0][0] == 'V':
            self.board[0][0] = 'VB'
            fixed_pieces.append((0,0))
        
        if self.board[0][0][0] == 'F':
            possible_actions[(0,0)] = ['FB', 'FD']

        #Top right corner
        if self.board[0][cols-1][0] == 'V':
            self.board[0][cols-1] = 'VE'
            fixed_pieces.append((0,cols-1))

        if self.board[0][cols-1][0] == 'F':
            possible_actions[(0,cols-1)] = ['FB','FE']

        #Bottom left corner
        if self.board[rows-1][0][0] == 'V':
            self.board[rows-1][0] = 'VD'
            fixed_pieces.append((rows-1,0))

        if self.board[rows-1][0][0] == 'F':
            possible_actions[(rows-1,0)] = ['FC','FD']

        #Bottom right corner
        if self.board[rows-1][cols-1][0] == 'V':
            self.board[rows-1][cols-1] = 'VC'
            fixed_pieces.append((rows-1,cols-1))

        if self.board[rows-1][cols-1][0] == 'F':
            possible_actions[(rows-1,cols-1)] = ['FC','FE']

        #Top edge
        for col in range(1, cols-1):
            if self.board[0][col][0] == 'B':
                self.board[0][col] = 'BB'
                fixed_pieces.append((0, col))

            if self.board[0][col][0] == 'L':
                self.board[0][col] = 'LH'
                fixed_pieces.append((0, col))

            if self.board[0][col][0] == 'F':
                possible_actions[(0,col)] = ['FB','FE','FD']

            if self.board[0][col][0] == 'V':
                possible_actions[(0,col)] = ['VB','VE']

        #Left edge
        for row in range(1, rows-1):
            if self.board[row][0][0] == 'B':
                self.board[row][0] = 'BD'
                fixed_pieces.append((row, 0))

            if self.board[row][0][0] == 'L':
                self.board[row][0] = 'LV'
                fixed_pieces.append((row, 0))

            if self.board[row][0][0] == 'F':
                possible_actions[(row,0)] = ['FB','FC','FD']

            if self.board[row][0][0] == 'V':
                possible_actions[(row,0)] = ['VB','VD']

        #Right edge
        for row in range(1, rows-1):
            if self.board[row][cols-1][0] == 'B':
                self.board[row][cols-1] = 'BE'
                fixed_pieces.append((row, cols-1))

            if self.board[row][cols-1][0] == 'L':
                self.board[row][cols-1] = 'LV'
                fixed_pieces.append((row, cols-1))

            if self.board[row][cols-1][0] == 'F':
                possible_actions[(row,cols-1)] = ['FB','FC','FE']

            if self.board[row][cols-1][0] == 'V':
                possible_actions[(row,cols-1)] = ['VE','VC']

        #Bottom edge
        for col in range(1, cols-1):
            if self.board[rows-1][col][0] == 'B':
                self.board[rows-1][col] = 'BC'
                fixed_pieces.append((rows-1, col))

            if self.board[rows-1][col][0] == 'L':
                self.board[rows-1][col] = 'LH'
                fixed_pieces.append((rows-1, col))

            if self.board[rows-1][col][0] == 'F':
                possible_actions[(rows-1,col)] = ['FD','FC','FE']

            if self.board[rows-1][col][0] == 'V':
                possible_actions[(rows-1,col)] = ['VD','VC']

        return fixed_pieces, possible_actions


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
        return Board(board)


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
                if (row==r and col<=c) or ((row, col) in state.board.fixed_pieces):
                    pass

                else:
                    if (row, col) in state.board.possible_actions:
                        actions = [(row, col, orientation) for orientation in state.board.possible_actions[(row, col)]]
                    
                    else:
                        actions = [(row, col, orientation) for orientation in possible_orientations[state.board.board[row][col][0]]]
                    
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

        The actions are performed using dictionaries with the correct
        movements of the pieces given the type of rotation asked.
        This way there is no need to use multiple 'if' statements
        and a lot of time may be saved.

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

        return PipeManiaState(Board(new_board), (row, col))

    def goal_test(self, state: PipeManiaState) -> bool:

        rows, cols = state.board.board.shape
        for row in range(rows):
            for col in range(cols):
                counter = 0
                upper, lower = state.board.adjacent_vertical_values(row, col)
                left, right = state.board.adjacent_horizontal_values(row, col)

                piece = state.board.get_value(row, col)
                locations = possible_adjacent_locations[piece]
                for location in locations:
                    if (location == 'upper') and (upper in adjacent_pieces['upper']):
                        counter += 1

                    if (location == 'lower') and (lower in adjacent_pieces['lower']):
                        counter += 1

                    if (location == 'right') and (right in adjacent_pieces['right']):
                        counter += 1

                    if (location == 'left') and (left in adjacent_pieces['left']):
                        counter += 1

                if counter != len(locations):
                    return False

        return True

    def h(self, node: Node) -> int:
        """ Heuristic function used for A* search. Number of pieces not connected."""
        heu = 0
        rows, cols = node.state.board.board.shape
        for row in range(rows):
            for col in range(cols):
                counter = 0
                upper, lower = node.state.board.adjacent_vertical_values(row, col)
                left, right = node.state.board.adjacent_horizontal_values(row, col)

                piece = node.state.board.get_value(row, col)
                locations = possible_adjacent_locations[piece]
                for location in locations:
                    if (location == 'upper') and (upper in adjacent_pieces['upper']):
                        counter += 1

                    if (location == 'lower') and (lower in adjacent_pieces['lower']):
                        counter += 1

                    if (location == 'right') and (right in adjacent_pieces['right']):
                        counter += 1

                    if (location == 'left') and (left in adjacent_pieces['left']):
                        counter += 1

                #loc_len = len(locations)
                if counter != len(locations):
                    heu += 1   
        return heu


def check_compatibility(row, col, initial_actions, state):
    """Checks compatibility of actions.
    
    Given the pieces already searched in the tree (since a piece
    by piece search is being implemented) it checks compatibility of
    the initial_actions with the pieces already checked in that state.
    If there is no way of them forming a solution, that part of the
    tree can be ignored, thus saving time.
    """
    possible_actions = [item for item in initial_actions]
    upper, _ = state.board.adjacent_vertical_values(row, col)
    left, _ = state.board.adjacent_horizontal_values(row, col)

    upper_not_none = True
    left_not_none = True

    if upper is None:
        upper_not_none = False

    if left is None:
        left_not_none = False

    for action in initial_actions:
        was_it_removed = False
        if upper_not_none:
            if 'lower' in possible_adjacent_locations[upper]:
                if action[2] not in adjacent_pieces['lower']:
                    possible_actions.remove(action)
                    was_it_removed = True

        if left_not_none and not was_it_removed:
            if 'right' in possible_adjacent_locations[left]:
                if action[2] not in adjacent_pieces['right']:
                    possible_actions.remove(action)

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

# Example usage:
if __name__ == "__main__":
    initial_board = Board.parse_instance()
    #initial_board.print()
    #print("---------")
    s0 = PipeManiaState(initial_board, (0,-1)) # as to start with -1 otherwise it does not generate actions for (0,0) piece.
    problem = PipeMania(s0)
    goal_node = astar_search(problem)
    #print('Is goal?', problem.goal_test(goal_node.state))
    #print("Solution:")
    goal_node.state.board.print()
    