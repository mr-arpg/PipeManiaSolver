import numpy as np
from sys import stdin
from search import Problem, Node
from typing import Tuple, List

class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        """ This method is used in case of tie in the management of the open list in informed searches."""
        return self.id < other.id

class Board:
    def __init__(self, board: np.ndarray):
        self.board = board

    def adjacent_vertical_values(self, row: int, col: int) -> Tuple[str, str]:
        """ Returns the values immediately above and below, respectively. """
        upper_value = self.board[row - 1][col] if row > 0 else None
        lower_value = self.board[row + 1][col] if row < len(self.board) - 1 else None
        return upper_value, lower_value

    def adjacent_horizontal_values(self, row: int, col: int) -> Tuple[str, str]:
        """ Returns the values immediately to the left and right, respectively. """
        left_value = self.board[row][col - 1] if col > 0 else None
        right_value = self.board[row][col + 1] if col < len(self.board[0]) - 1 else None
        return left_value, right_value

    def get_value(self, row: int, col: int) -> str:
        """ Returns the value at the specified position. """
        return self.board[row][col]

    def print(self):
        """ Prints the board. """
        for row in self.board:
            print(" ".join(row))

    @staticmethod
    def parse_instance():
        """ Reads the problem instance from standard input (stdin) and returns an instance of the Board class. """
        lines = stdin.readlines()
        board = np.array([line.strip().split() for line in lines])
        return Board(board)

class PipeMania(Problem):
    def __init__(self, initial_state: PipeManiaState, goal_state: PipeManiaState):
        self.initial = initial_state
        self.goal = goal_state

    def actions(self, state: PipeManiaState) -> List[Tuple[int, int, bool]]:
        """ Returns a list of actions that can be executed from the given state. """
        actions = []
        rows, cols = state.board.shape
        for row in range(rows):
            for col in range(cols):
                # For each pipe piece, add actions to rotate it clockwise and counterclockwise
                if state.board[row][col] != 'F':
                    actions.append((row, col, True))  # Rotate clockwise
                    actions.append((row, col, False))  # Rotate counterclockwise
        return actions

    def result(self, state: PipeManiaState, action: Tuple[int, int, bool]) -> PipeManiaState:
        """ Returns the result of applying the action to the given state. """
        row, col, clockwise = action
        new_board = np.copy(state.board)
        if clockwise:
            # Rotate the pipe piece clockwise
            new_board[row][col] = rotate_clockwise(new_board[row][col])
        else:
            # Rotate the pipe piece counterclockwise
            new_board[row][col] = rotate_counterclockwise(new_board[row][col])
        return PipeManiaState(new_board)

    def h(self, node: Node) -> int:
        """ Heuristic function used for A* search. """
        # For now, return 0 (trivial heuristic)
        return 0

def rotate_clockwise(pipe_piece: str) -> str:
    """ Rotate the given pipe piece clockwise. """
    if pipe_piece[1] in ['C', 'B']:
        # Rotate from C to D or from B to A
        return pipe_piece[0] + chr(ord(pipe_piece[1]) + 1)
    elif pipe_piece[1] in ['E', 'D']:
        # Rotate from E to C or from D to B
        return pipe_piece[0] + chr(ord(pipe_piece[1]) - 1)
    else:
        return pipe_piece

def rotate_counterclockwise(pipe_piece: str) -> str:
    """ Rotate the given pipe piece counterclockwise. """
    if pipe_piece[1] in ['C', 'D']:
        # Rotate from C to E or from D to A
        return pipe_piece[0] + chr(ord(pipe_piece[1]) + 2)
    elif pipe_piece[1] in ['B', 'E']:
        # Rotate from B to D or from E to C
        return pipe_piece[0] + chr(ord(pipe_piece[1]) - 2)
    else:
        return pipe_piece

# Example usage:
if __name__ == "__main__":
    board = Board.parse_instance()
    problem = PipeMania(None, None)  # Replace None with initial and goal states
    # Perform actions and search for the solution
    # Print the solution
