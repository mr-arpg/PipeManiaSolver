"""Pipe Mania Solver for the AI course's project."""
from sys import stdin
from typing import Tuple, List

import numpy as np

from search import Problem, Node

class PipeManiaState:
    """Describes the current state of the problem.
    
    This class receives an instance of the class Board
    and increases its `state_id` argument everytime it
    is called.
    """
    state_id = 0

    def __init__(self, board):
        """Initializes the class.
        
        Args:
            board (Board): An instance of the Board class.
        """
        self.board = board
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

    def adjacent_vertical_values(self, row: int, col: int) -> Tuple[str, str]:
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
        lower_value = self.board[row + 1][col] if row < len(self.board) - 1 else None
        return upper_value, lower_value

    def adjacent_horizontal_values(self, row: int, col: int) -> Tuple[str, str]:
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
        right_value = self.board[row][col + 1] if col < len(self.board[0]) - 1 else None
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
            print(" ".join(row))

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
    def __init__(self, initial_state: PipeManiaState): #, goal_state: PipeManiaState):
        """Initializes the class.
        
        Args:
            initial_state (class PipeManiaState): Instance of the board's initial state.
        """
        self.initial = initial_state
        #self.goal = goal_state

    def actions(self, state: PipeManiaState) -> List[Tuple[int, int, bool]]:
        """Actions that can be performed.
        
        Returns a list of actions that can be executed from the given state.
        The actions that can be performed are represented by the piece's
        location and and integer between 1 and 3:
            1 ---> Clockwise rotation of the piece (if not type 'L')
            2 ---> Counter-clockwise rotation of the piece (if not type 'L')
            3 ---> Rotation of the 'L' type piece.
        
        Args:
            state (class PipeManiaState): Instance of the board's current state.

        Returns:
            actions (list of tuples): List of actions that can be performed.
        """
        actions = []
        rows, cols = state.board.shape
        for row in range(rows):
            for col in range(cols):
                if state.board.board[row][col] != 'L':
                    actions.append((row, col, 1))  # Rotate clockwise
                    actions.append((row, col, 2))  # Rotate counterclockwise

                if state.board.board[row][col] == 'L':
                    actions.append((row, col, 3)) # Rotate 'L' piece
        return actions

    def result(self, state: PipeManiaState, action: Tuple[int, int, int]) -> PipeManiaState:
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
        row, col, action_type = action
        new_board = np.copy(state.board.board) #Maybe usar o deepcopy?
        if action_type == 1:
            # Rotate the pipe piece clockwise
            new_board[row][col] = rotate_clockwise(new_board[row][col])
        if action_type == 2:
            # Rotate the pipe piece counterclockwise
            new_board[row][col] = rotate_counterclockwise(new_board[row][col])
        if action_type == 3:
            new_board[row][col] = rotate_Lpiece(new_board[row][col])

        return PipeManiaState(Board(new_board))

    def h(self, node: Node) -> int:
        """ Heuristic function used for A* search. """
        # For now, return 0 (trivial heuristic)
        return 0

# Dictionary for clockwise rotation
clockwise_rotation = {
    'B': 'E',
    'E': 'C',
    'C': 'D',
    'D': 'B'
}

# Dictionary for counterclockwise rotation
counterclockwise_rotation = {
    'C': 'E',
    'E': 'B',
    'B': 'D',
    'D': 'C'
}

# Dictionary for 'L' piece rotation
L_piece_rotation = {
    'H': 'V',
    'V': 'H'
}

def rotate_clockwise(pipe_piece: str) -> str:
    """Rotate the given piece clockwise.
    
    Args: 
        pipe_piece (str): String that represents the type of piece
          and its orientation.
    """
    return pipe_piece[0] + clockwise_rotation[pipe_piece[1]]

def rotate_counterclockwise(pipe_piece: str) -> str:
    """Rotate the given piece counterclockwise.
    
    Args: 
        pipe_piece (str): String that represents the type of piece
          and its orientation.
    """
    return pipe_piece[0] + counterclockwise_rotation[pipe_piece[1]]

def rotate_Lpiece(pipe_piece: str) -> str:
    """Rotates the 'L' piece of the board.
    
    Args: 
        pipe_piece (str): String that represents the type of piece
          and its orientation.
    """
    return pipe_piece[0] + L_piece_rotation[pipe_piece[1]]

# Example usage:
if __name__ == "__main__":
    board = Board.parse_instance()
    board.print()
    problem = PipeMania(board)
    s0 = PipeManiaState(board)
    s1 = problem.result(s0, (0, 1, False))
    s1.board.print()
