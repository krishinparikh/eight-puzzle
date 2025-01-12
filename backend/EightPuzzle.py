import random
import sys
from collections import deque
import queue
from scipy.optimize import fsolve
from tabulate import tabulate
import numpy as np

class EightPuzzle:

    # Initializes an eight-puzzle and sets the initial state to a solved position
    def __init__(self, initial_state=None):
        # Default to the solved position if no state is provided
        self.state = initial_state if initial_state else [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # Helper method
    # Returns a 2D array given a string of numbers
    def stringToState(self, stringState):
        matrix = []
        for i in range(0, 9, 3):
            row = [int(stringState[i]), int(stringState[i+1]), int(stringState[i+2])]
            matrix.append(row)
        
        return matrix
    
    # Returns a string of integers 0-8 given a 2D state array
    def stateToString(self):
        s = ""
        for i in self.state:
            for j in i:
                s += str(j)
        return s
    
    # Helper method
    # Returns indeces i, j that correspond with the location of the zero in self.state
    def findZeroIndeces(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j

    # Sets self.state given a string of numbers
    def setState(self, stringState):
        self.state = self.stringToState(stringState)
    
    # Prints self.state in the terminal as a 2D array
    def printState(self):
        for row in self.state:
            print(row)
        print()
    
    # Changes self.state based on a given string direction
    def move(self, direction):
        i, j = self.findZeroIndeces()
        
        if direction == 'up' and i > 0:
            self.state[i][j], self.state[i-1][j] = self.state[i-1][j], self.state[i][j]
            return True
        elif direction == 'down' and i < 2:
            self.state[i][j], self.state[i+1][j] = self.state[i+1][j], self.state[i][j]
            return True
        elif direction == 'left' and j > 0:
            self.state[i][j], self.state[i][j-1] = self.state[i][j-1], self.state[i][j]
            return True
        elif direction == 'right' and j < 2:
            self.state[i][j], self.state[i][j+1] = self.state[i][j+1], self.state[i][j]
            return True
        else:
            # print("Error: Invalid move")
            return False

    # Scrambles self n times
    def scrambleState(self, n, seed=10):
        self.setState("012345678")

        seen_stringStates = set()
        seen_stringStates.add(self.stateToString())

        if seed is not None:
            random.seed(seed)

        for x in range(int(n)):
            i, j = self.findZeroIndeces()
            possible_moves = []

            if i > 0:  # Can move up
                possible_moves.append("up")
            if i < 2:  # Can move down
                possible_moves.append("down")
            if j > 0:  # Can move left
                possible_moves.append("left")
            if j < 2:  # Can move right
                possible_moves.append("right")

            current_stateString = self.stateToString()

            while self.move(possible_moves[random.randint(0, len(possible_moves) - 1)]):
                if (self.stateToString() not in seen_stringStates):
                    seen_stringStates.add(self.stateToString())
                    break
                else:
                    self.setState(current_stateString)

    
    # Checks if the setState value is valid
    @staticmethod
    def isValidString(stringState):
        valid_chars = set("012345678")
        
        if len(stringState) != 9 or set(stringState) != valid_chars:
            return False
        
        return True


# A node in the BFS tree
class Node:

    def __init__(self, parent, stringState, prev_move, path_cost=0):
        self.parent = parent
        self.stringState = stringState
        self.prev_move = prev_move
        self.path_cost = path_cost
        self.total_cost = 0

    def __lt__(self, other):
        return self.total_cost < other.total_cost

# The Agent that solves the eight-puzzle
class Agent:

    goal_stringState = "012345678"
    goal_arrayState = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # Solves the 8-puzzle using a DFS approach
    @staticmethod
    def solveDFS(puzzle, maxnodes=1000, gettingNodesCreated=False):
        # Initialize the start node as the current state of the puzzle
        start_node = Node(None, puzzle.stateToString(), None)
        nodes_created = 1

        # Base case
        if (start_node.stringState == Agent.goal_stringState):
            return Agent.printSearchSolution(start_node, nodes_created)
        
        # Initialize a stack (LIFO data structure) for storing successor nodes
        frontier = deque([start_node])

        # Initialize a set for storing all previously visited states
        visited = set()
        visited.add(start_node.stringState)

        while frontier and nodes_created < maxnodes:
            current_node = frontier.pop()

            # Explores all possible moves that can be made from the current state
            for move in ["left", "right", "up", "down"]:
                if nodes_created < maxnodes:
                    puzzle.setState(current_node.stringState)

                    if puzzle.move(move):
                        child_stringState = puzzle.stateToString()

                        if child_stringState not in visited:
                            child_node = Node(current_node, child_stringState, move)
                            nodes_created += 1

                            # Check if the child is the goal
                            if child_stringState == Agent.goal_stringState:
                                # Originally, this line used Agent.printSearchSolution(), but was changed to support Flask API call
                                return nodes_created if gettingNodesCreated else Agent.stringSearchSolution(child_node, nodes_created)

                            frontier.append(child_node)
                            visited.add(child_stringState)
                else:
                    return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")
        
        return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")


    # Solves the 8-puzzle using a BFS approach
    @staticmethod
    def solveBFS(puzzle, maxnodes=1000, gettingNodesCreated=False):
        # Initialize the start node as the current state of the puzzle
        start_node = Node(None, puzzle.stateToString(), None)
        nodes_created = 1

        # Base case
        if (start_node.stringState == Agent.goal_stringState):
            return Agent.printSearchSolution(start_node, nodes_created)
        
        # Initialize a queue (FIFO data structure) for storing successor nodes
        frontier = deque([start_node])

        # Initialize a set for storing all previously visited states
        visited = set()
        visited.add(start_node.stringState)

        while frontier and nodes_created < maxnodes:
            current_node = frontier.popleft()

            # Explores all possible moves that can be made from the current state
            for move in ["left", "right", "up", "down"]:
                if nodes_created < maxnodes:
                    puzzle.setState(current_node.stringState)

                    if puzzle.move(move):
                        child_stringState = puzzle.stateToString()

                        if child_stringState not in visited:
                            child_node = Node(current_node, child_stringState, move)
                            nodes_created += 1
                            # print("Child: " + child_stringState)
                            # Check if the child is the goal
                            if child_stringState == Agent.goal_stringState:
                                # Originally, this line used Agent.printSearchSolution(), but was changed to support Flask API call
                                return nodes_created if gettingNodesCreated else Agent.stringSearchSolution(child_node, nodes_created)

                            frontier.append(child_node)
                            visited.add(child_stringState)
                        # else:
                            # print(f"{child_stringState} already visited")
                else:
                    # print(visited)
                    return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")
        
        return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")


    # Deconstructs the solution for either search algorithm by traversing up the tree and prints the solution
    @staticmethod
    def printSearchSolution(goal_node, nodes_created):
        print("Nodes created during search: " + str(nodes_created))
        
        current_node = goal_node
        solution = []
        
        # Reconstructing moves to solve puzzle
        while current_node.parent != None:
            # Appends the moves to the solution list in reverse order
            solution = [current_node.prev_move] + solution
            # Makes the current node the parent
            current_node = current_node.parent

        print("Solution length: " + str(len(solution)))
        print("Move sequence: ")

        # Prints the moves in the solution list
        if len(solution) > 0:
            for move in solution:
                print("move " + move)
        print("")

    # Deconstructs the solution for either search algorithm by traversing up the tree and returns a string of the solution
    @staticmethod
    def stringSearchSolution(goal_node, nodes_created):
        # Initialize the result string
        result = ""
        result += "Nodes created during search: " + str(nodes_created) + "\n\n"
        
        current_node = goal_node
        solution = []
        
        # Reconstructing moves to solve puzzle
        while current_node.parent is not None:
            # Appends the moves to the solution list in reverse order
            solution = [current_node.prev_move] + solution
            # Makes the current node the parent
            current_node = current_node.parent
        
        result += "Solution length: " + str(len(solution)) + "\n\n"
        result += "Move sequence:\n\n"
        
        # Appends the moves in the solution list to the result string
        if len(solution) > 0:
            for move in solution:
                result += "move " + move + "\n\n"
        
        return result


    
    # Returns the number of misplaced tiles
    @staticmethod
    def h1(puzzle):
        count = 0
        for i in range(len(Agent.goal_stringState)):
            if Agent.goal_stringState[i] != puzzle.stateToString()[i]:
                count += 1
        return count

    # Returns the sum of the distances of the tiles from their goal positions
    @staticmethod
    def h2(puzzle):
        distances_sum = 0
        for x in range(len(puzzle.state)):
            for y in range(len(puzzle.state[x])):
                if (puzzle.state[x][y] != 0):
                    i, j = Agent.findGoalPositionIndices(puzzle.state[x][y])
                    distances_sum += (abs(x - i) + abs(y - j))
        return distances_sum

    # Returns the indices of the goal position of some given tile
    @staticmethod
    def findGoalPositionIndices(tile):
        for i in range(len(Agent.goal_arrayState)):
            for j in range(len(Agent.goal_arrayState[i])):
                if Agent.goal_arrayState[i][j] == tile:
                    return i, j

    # Solves the 8-puzzle using A* search using either heuristic h1 or h2
    @staticmethod
    def solveAstar(puzzle, heuristic, maxnodes=1000, gettingNodesCreated=False):
        # Initialize the start node as the current state of the puzzle
        start_node = Node(None, puzzle.stateToString(), None, 0)
        
        # Compute the initial heuristic cost
        if heuristic == "h1":
            start_node.total_cost = start_node.path_cost + Agent.h1(puzzle)
        elif heuristic == "h2":
            start_node.total_cost = start_node.path_cost + Agent.h2(puzzle)

        nodes_created = 1

        # Base case: check if the start state is already the goal
        if start_node.stringState == Agent.goal_stringState:
            return Agent.printSearchSolution(start_node, nodes_created)
        
        # Initialize a priority queue for storing successor nodes
        frontier = queue.PriorityQueue()
        frontier.put(start_node)

        # Initialize a dictionary to track the lowest path cost for each state
        visited = {start_node.stringState: start_node.path_cost}

        while not frontier.empty() and nodes_created < maxnodes:
            current_node = frontier.get()

            # Explore all possible moves from the current state
            for move in ["left", "right", "up", "down"]:
                if nodes_created < maxnodes:
                    puzzle.setState(current_node.stringState)

                    if puzzle.move(move):
                        child_stringState = puzzle.stateToString()
                        child_path_cost = current_node.path_cost + 1

                        # If this state has not been visited or found with a lower cost
                        if child_stringState not in visited or visited[child_stringState] > child_path_cost:
                            # Create the child node
                            child_node = Node(current_node, child_stringState, move, child_path_cost)

                            # Compute the heuristic cost
                            if heuristic == "h1":
                                child_node.total_cost = child_node.path_cost + Agent.h1(puzzle)
                            elif heuristic == "h2":
                                child_node.total_cost = child_node.path_cost + Agent.h2(puzzle)

                            nodes_created += 1

                            # Check if the child is the goal state
                            if child_stringState == Agent.goal_stringState:
                                # Originally, this line used Agent.printSearchSolution(), but was changed to support Flask API call
                                return nodes_created if gettingNodesCreated else Agent.stringSearchSolution(child_node, nodes_created)

                            # Add child node to the priority queue and update the visited map
                            frontier.put(child_node)
                            visited[child_stringState] = child_path_cost
                else:
                    return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")

        return print(f"Error: maxnodes limit ({str(maxnodes)}) reached\n")
    

    def computeAvgNodesCreated(depth, num_trials, algorithm):
        puzzle = EightPuzzle()
        sum = 0
        
        for i in range(num_trials):
            puzzle.scrambleState(depth, seed=i)
            if algorithm == "BFS":
                sum += Agent.solveBFS(puzzle, maxnodes=100000, gettingNodesCreated=True)
            elif algorithm == "A* h1":
                sum += Agent.solveAstar(puzzle, "h1", maxnodes=100000, gettingNodesCreated=True)
            elif algorithm == "A* h2":
                sum += Agent.solveAstar(puzzle, "h2", maxnodes=100000, gettingNodesCreated=True)
        
        return round(sum / num_trials)


    def effectiveBranchingFactor(nodes_created, depth):
        def equation(b):
            return (b**(depth + 1) - 1) / (b - 1) - nodes_created
        
        return round(round(fsolve(equation, 2.0)[0], 2) + 0.01, 2)



# Defines all of the testcmds.txt commands
def cmd(command, puzzle, line_num):
    words = command.split()

    if command == "" or words[0] == "#" or words[0] == "//":
        pass

    elif words[0] == "setState":
        if (puzzle.isValidString(words[1])):
            puzzle.setState(words[1])
        else:
            printError(line_num)
    
    elif words[0] == "printState":
        puzzle.printState()
    
    elif words[0] == "move":
        if (words[1] == "up" or words[1] == "down" or words[1] == "left" or words[1] == "right"):
            puzzle.move(words[1])
        else:
            printError(line_num)
    
    elif words[0] == "scrambleState":
        if words[1].isdigit():
            puzzle.scrambleState(words[1])
        else:
            printError(line_num)

    elif words[0] == "heuristic":
        if words[1] == "h1":
            print(Agent.h1(puzzle))
        elif words[1] == "h2":
            print(Agent.h2(puzzle))
        else:
            printError(line_num)

    elif words[0] == "solve":
        if words[1] == "DFS":
            if len(words) == 2:
                Agent.solveDFS(puzzle)
            else:
                if words[2][0:9] == "maxnodes=":
                    maxnodes = int(words[2][9:])
                    Agent.solveDFS(puzzle, maxnodes=maxnodes)
                else:
                    printError(line_num)
        elif words[1] == "BFS":
            if len(words) == 2:
                Agent.solveBFS(puzzle)
            else:
                if words[2][0:9] == "maxnodes=":
                    maxnodes = int(words[2][9:])
                    Agent.solveBFS(puzzle, maxnodes=maxnodes)
                else:
                    printError(line_num)
        elif words[1] == "A*":
            # if no maxnodes specified
            if len(words) == 3:
                if words[2] == "h1":
                    Agent.solveAstar(puzzle, "h1")
                elif words[2] == "h2":
                    Agent.solveAstar(puzzle, "h2")
                else:
                    printError(line_num)
            # if maxnodes is specified
            else:
                if words[3][0:9] == "maxnodes=":
                    maxnodes = int(words[3][9:])
                    if words[2] == "h1":
                        Agent.solveAstar(puzzle, "h1", maxnodes=maxnodes)
                    elif words[2] == "h2":
                        Agent.solveAstar(puzzle, "h2", maxnodes=maxnodes)
                    else:
                        printError(line_num)
                else:
                    printError(line_num)
        else:
            printError(line_num)
    else:
        printError(line_num)


# Prints error message at specified line
def printError(line_num):
    print("Error: invalid command: " + str(line_num))

# Generates comparison table
def genComparisonTable():
    arr_header = ["d",  "DFS (N)", "BFS (N)", "A* h1 (N)", "A* h2 (N)", "DFS (b*)", "BFS (b*)", "A* h1 (b*)", "A* h2 (b*)"]
    arr_data = []

    depth = 6
    while depth <= 22:
        row = []

        # Calculates average nodes and effective branching factor for each algorithm
        row.append(depth)
        row.append("N/A")
        row.append(Agent.computeAvgNodesCreated(depth, 10, "BFS"))
        row.append(Agent.computeAvgNodesCreated(depth, 10, "A* h1"))
        row.append(Agent.computeAvgNodesCreated(depth, 10, "A* h2"))
        row.append("N/A")
        row.append(Agent.effectiveBranchingFactor(row[2], depth))
        row.append(Agent.effectiveBranchingFactor(row[3], depth))
        row.append(Agent.effectiveBranchingFactor(row[4], depth))

        arr_data.append(row)

        depth += 2

    print(tabulate(arr_data, arr_header, tablefmt="pretty"))



# Initializing an eight-puzzle
puzzle = EightPuzzle()

# Reading the testcmds.txt file from the terminal script
if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]

        with open(filename, 'r') as file:
            line_num = 1
            for line in file:
                print(line.strip())
                cmd(line.strip(), puzzle, line_num)
                line_num += 1
    else:
        genComparisonTable()