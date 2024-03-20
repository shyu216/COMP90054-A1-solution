# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from copy import deepcopy

import util
from game import Actions
from game import Directions
from game import Grid
from util import foodGridtoDic


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    This heuristic is trivial.
    """
    return 0


def goalCountingHeuristic(state, problem):
    return state[1].count()


def minFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    if foodGrid.count() == 0:
        return 0

    min_dis = 999999
    for food_pos in foodGrid.asList():
        min_dis = min(util.manhattanDistance(food_pos, pos), min_dis)
    return max(min_dis, len(state[1].asList()))


def maxFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    max_dis = 0
    for food_pos in foodGrid.asList():
        max_dis = max(util.manhattanDistance(food_pos, pos), max_dis)
    return max(max_dis, len(state[1].asList()))


def maxMazeFoodHeuristic(state, problem):
    foodGrid = state[1]
    pos = state[0]
    max_dis = 0
    for food_pos in foodGrid.asList():
        max_dis = max(mazeDistance(food_pos, pos, problem), max_dis)
    return max(max_dis, len(state[1].asList()))


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"
    return maxMazeFoodHeuristic(state, problem)

    # comment the below line after you implement the algorithm
    # util.raiseNotDefined()


def manhattanHeuristic(state, problem):
    pos = state[0]
    assert state[1].count() <= 1
    if state[1].count() == 0:
        return 0

    food_pos = state[1].asList()[0]
    return util.manhattanDistance(food_pos, pos)


def mazeDistance(point1, point2, problem):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    if (x1, y1, x2, y2) in problem.heuristicInfo:
        return problem.heuristicInfo[(x1, y1, x2, y2)]

    walls = problem.walls
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    food_grid = problem.start[1]
    new_food_grid = Grid(food_grid.width, food_grid.height)
    new_food_grid[x2][y2] = True
    prob = SingleFoodSearchProblem(pos=point1, food=new_food_grid, walls=walls)
    dis = len(aStarSearch(prob, heuristic=manhattanHeuristic))
    problem.heuristicInfo[(x1, y1, x2, y2)] = dis
    return dis


from itertools import product


class MAPFProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPositions, foodGrid ) where
      pacmanPositions: a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
      foodGrid: a Grid (see game.py) of either pacman_name or False, specifying each pacman's target food. For example, if foodGrid[x][y] == 'A', it means pacman A want to eat food at position (x, y)
    """

    def __init__(self, startingGameState):
        "Initial function"
        "*** WARNING: DO NOT CHANGE!!! ***"
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()

    def getStartState(self):
        "Get start state"
        "*** WARNING: DO NOT CHANGE!!! ***"
        return self.start

    def isGoalState(self, state):
        "Return if the state is the goal state"
        "*** YOUR CODE HERE for task2 ***"
        return state[1].count(False) == state[1].width * state[1].height

        # comment the below line after you implement the function
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        "*** YOUR CODE HERE for task3 ***"
        all_succs = []
        for pacman in state[0].keys():
            p_succs = []
            x, y = state[0][pacman]


            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:

                dx, dy = Actions.directionToVector(direction)
                next_x, next_y = int(x + dx), int(y + dy)
                if not self.walls[next_x][next_y]:
                    p_succs.append((pacman, (next_x, next_y), direction))

            all_succs.append(p_succs)

        successors = []
        for succ in product(*all_succs):
            poss = set()
            edges = set()
            pos_dic = {}
            new_food_grid = state[1].copy()
            action_dic = {}
            for atom in succ:
                pacman, pos, direction = atom
                prev_pos = state[0][pacman]

                # Avoid vertex conflict
                if not pos in poss:
                    poss.add(pos)
                else:
                    break

                # Avoid edge conflict
                if not (pos, prev_pos) in edges:
                    edges.add((prev_pos, pos))
                else:
                    break

                pos_dic[pacman] = pos
                if new_food_grid[pos[0]][pos[1]] == pacman:
                    new_food_grid[pos[0]][pos[1]] = False

                action_dic[pacman] = direction
            else:
                successors.append(((pos_dic, new_food_grid), action_dic, 1))  # uniform cost

        return successors

        # comment the below line after you implement the function
        # util.raiseNotDefined()


def aStarSearchConstraints(problem, vertex_constraints=[], edge_constraints=[]):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [startState], [])
    myPQ.push(startNode, cbsHeuristic(startState, problem))
    closed = set([c for c in vertex_constraints])
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, trajs, path = node
        if not (state[0], cost) in closed:
            closed.add((state[0], cost))
            if problem.isGoalState(state):
                for (pos,
                     t) in vertex_constraints:  # if there is a constraint that the pacman can not be in the current position at time t,
                    if state[0] == pos and cost <= t:  # then it is not a goal state
                        break
                else:
                    return trajs, path

            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                if not (state[0], succState[0], cost) in edge_constraints:
                    new_cost = cost + succCost
                    newNode = (succState, new_cost, trajs + [succState], path + [succAction])
                    myPQ.push(newNode, cbsHeuristic(succState, problem) + new_cost)

    return None, None  # Goal not found


def cbsHeuristic(state, problem):
    """
    Heuristic function used by constrained astar search.
    """
    return manhattanHeuristic(state, problem)


def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores path for each pacman as a list {pacman_name: [a1, a2, ...]}.

        A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
          pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
          foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    "*** YOUR CODE HERE for task3 ***"
    pacman_positions, food_grid = problem.getStartState()

    v_constraints_dic = {p: [] for p in pacman_positions.keys()}
    e_constraints_dic = {p: [] for p in pacman_positions.keys()}
    trajs = {} # each item is a tuple of (trajs, path)
    walls = problem.walls
    probs = {}
    for pacman in pacman_positions.keys():
        x1, y1 = pacman_positions[pacman]
        x2, y2 = food_grid.asList(pacman)[0]
        new_food_grid = Grid(walls.width, walls.height)
        new_food_grid[x2][y2] = True
        probs[pacman] = SingleFoodSearchProblem(pos=(x1, y1), food=new_food_grid, walls=walls)
        trajs[pacman] = aStarSearchConstraints(probs[pacman])

    start_node = (trajs, v_constraints_dic, e_constraints_dic)
    sum_cost = sum([len(trajs[p]) - 1 for p in pacman_positions.keys()])
    pq = util.PriorityQueue()
    pq.push(start_node, sum_cost)

    while not pq.isEmpty():
        s = pq.getMinimumPriority()
        ts, vcs, ecs = pq.pop()
        c = validate({p:ts[p][0] for p in ts})  # c = (p_i, p_j, pos, t) or (p_i, p_j, pos1, pos2, t)
        if not c:
            return {p:ts[p][1] for p in ts} # only return dictionary of path

        if len(c) == 4:
            p_i, p_j, pos, t = c

            new_constraints = deepcopy(vcs)
            new_constraints[p_i].append((pos, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_i] = aStarSearchConstraints(probs[p_i], new_constraints[p_i], ecs[p_i])
            if new_trajs[p_i] != (None, None):
                pq.push((new_trajs, new_constraints, ecs), s - len(ts[p_i][0]) + len(new_trajs[p_i][0]))

            new_constraints = deepcopy(vcs)
            new_constraints[p_j].append((pos, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_j] = aStarSearchConstraints(probs[p_j], new_constraints[p_j], ecs[p_j])
            if new_trajs[p_j] != (None, None):
                pq.push((new_trajs, new_constraints, ecs), s - len(ts[p_j][0]) + len(new_trajs[p_j][0]))
        else:  # len(c) == 5, edge conflict
            p_i, p_j, pos1, pos2, t = c
            new_constraints = deepcopy(ecs)
            new_constraints[p_i].append((pos1, pos2, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_i] = aStarSearchConstraints(probs[p_i], vcs[p_i], new_constraints[p_i])
            if new_trajs[p_i] != (None, None):
                pq.push((new_trajs, vcs, new_constraints), s - len(ts[p_i][0]) + len(new_trajs[p_i][0]))

            new_constraints = deepcopy(ecs)
            new_constraints[p_j].append((pos2, pos1, t))
            new_trajs = deepcopy(ts)
            new_trajs[p_j] = aStarSearchConstraints(probs[p_j], vcs[p_j], new_constraints[p_j])
            if new_trajs[p_j] != (None, None):
                pq.push((new_trajs, vcs, new_constraints), s - len(ts[p_j][0]) + len(new_trajs[p_j][0]))

    return None

    # comment the below line after you implement the function
    # util.raiseNotDefined()


def validate(trajs):
    # vertex conflict
    t = 1
    flag = True

    while flag:
        poss = {}
        flag = False
        for p in trajs:
            if len(trajs[p]) > t:
                pos = trajs[p][t][0]
                flag = True
            else:
                pos = trajs[p][-1][0]

            if pos in poss:
                return p, poss[pos], pos, t
            poss[pos] = p

        t += 1

    # edge conflict
    t = 1
    flag = True

    while flag:
        edges = {}
        flag = False
        for p in trajs:
            if len(trajs[p]) > t:
                pre_pos = trajs[p][t - 1][0]
                pos = trajs[p][t][0]
                if (pos, pre_pos) in edges:
                    return p, edges[(pos, pre_pos)], pre_pos, pos, t - 1
                edges[(pre_pos, pos)] = p
                flag = True

        t += 1

    return None


"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"

class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                nextFood = state[1].copy()
                nextFood[next_x][next_y] = False
                successors.append((((next_x, next_y), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class SingleFoodSearchProblem(FoodSearchProblem):
    """
    A special food search problem with only one food and can be generated by parsing pacman position, food grid and wall grid
    """
    def __init__(self, pos, food, walls):
        self.start = (pos, food)
        self.walls = walls
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch
