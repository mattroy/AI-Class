# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostDistances = map(lambda x: util.manhattanDistance(newPos, x), newGhostPositions)
        ghostDis = sum(ghostDistances)
        foodCount = len(newFood.asList())
        foodDis = map(lambda x: util.manhattanDistance(newPos, x), newFood.asList())
        if len(foodDis):
          closestFood = min(foodDis)
          farthestFood = max(foodDis)

        if foodCount == 0:
          return ghostDis

        return ghostDis - foodCount
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          """
        succesors = []
        actions = gameState.getLegalActions(0)

        for action in actions:
          succesors.append((gameState.generateSuccessor(0, action), action))
        
        values = []
        for succesor in succesors:
          values.append((self.minState(succesor[0], 1, 1), succesor[1]))

        maxState = max(values, key = lambda x: x[0])
        
        return maxState[1]


    def minState(self, gameState, agentIndex, depth):
      succesors = []
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)

      for action in actions:
        succesors.append(gameState.generateSuccessor(agentIndex, action))
      
      values = []
      for succesor in succesors:
        if agentIndex == gameState.getNumAgents() - 1:
          values.append(self.maxState(succesor, 0, depth + 1))
        else:
          values.append(self.minState(succesor, agentIndex + 1, depth ))

      return min(values)

    def maxState(self, gameState, agentIndex, depth):
      if depth > self.depth:
        return self.evaluationFunction(gameState)
        
      succesors = []
      actions = gameState.getLegalActions(agentIndex)

      if len(actions) == 0:
        return self.evaluationFunction(gameState)

      for action in actions:
        succesors.append(gameState.generateSuccessor(agentIndex, action))
      
      values = []
      for succesor in succesors:
          values.append(self.minState(succesor, agentIndex + 1, depth ))

      return max(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        succesors = []
        actions = gameState.getLegalActions(0)

        for action in actions:
          succesors.append((gameState.generateSuccessor(0, action), action))
        
        self.alpha = float("-inf")
        self.beta = float("inf")

        values = []
        for succesor in succesors:
          values.append((self.minState(succesor[0], 1, 1, self.alpha, self.beta), succesor[1]))

        maxState = max(values, key = lambda x: x[0])
        
        return maxState[1]


    def minState(self, gameState, agentIndex, depth, alpha, beta):
      succesors = []
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)

      for action in actions:
        succesors.append(gameState.generateSuccessor(agentIndex, action))
      print "Starting beta is ", beta
      self.beta = beta
      value = float("inf")
      for succesor in succesors:
        if agentIndex == gameState.getNumAgents() - 1:
          value = min(value, self.maxState(succesor, 0, depth + 1, self.alpha, self.beta))
          if value < self.alpha:
            return value
          self.beta = min(self.beta, value)
          print "New beta in min is ", self.beta
        else:
          value = min(value, self.minState(succesor, agentIndex + 1, depth, self.alpha, self.beta))
          if value > self.beta:
            return value
          self.alpha = min(self.alpha, value)
          print "new alpha in min is ", self.alpha

      return value

    def maxState(self, gameState, agentIndex, depth, alpha, beta):
      if depth > self.depth:
        return self.evaluationFunction(gameState)
        
      succesors = []
      actions = gameState.getLegalActions(agentIndex)

      if len(actions) == 0:
        return self.evaluationFunction(gameState)

      for action in actions:
        succesors.append(gameState.generateSuccessor(agentIndex, action))
      
      value = float("-inf")
      for succesor in succesors:
          value = min(value, self.minState(succesor, agentIndex + 1, depth, self.alpha, self.beta))
          if value > self.beta:
            return value
          self.alpha = min(self.alpha, value)
          return "new alpha in min is ", self.alpha

      return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

