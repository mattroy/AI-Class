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
        newScore = successorGameState.getScore()

        ghostDistances = map(lambda x: util.manhattanDistance(newPos, x), newGhostPositions)
        ghostDis = min(ghostDistances)
        if(ghostDis > 0):
          ghostDis = 1/ ghostDis
        else: 
          ghostDis = 0

        foodCount = len(newFood.asList())
        foodDis = map(lambda x: util.manhattanDistance(newPos, x), newFood.asList())
        if len(foodDis):
          closestFood = min(foodDis)
        else:
          closestFood = 0

        return ghostDis * -5 + closestFood * -1 + newScore * 2 + foodCount * -1
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
        alpha = float("-inf")
        beta = float("inf")
        value = float("-inf")

        actions = gameState.getLegalActions(0)

        for action in actions:
          succesor = gameState.generateSuccessor(0, action)
          newValue = self.minState(succesor, 1, 1, alpha, beta)
          if newValue > value:
            value = newValue
            newAction = action
          if value > beta:
            return newAction
          alpha = max(alpha, value)
          
        return newAction


    def minState(self, gameState, agentIndex, depth, alpha, beta):
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)
      
      value = float("inf")

      for action in actions:
        succesor = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == gameState.getNumAgents() - 1:
          value = min(value, self.maxState(succesor, 0, depth + 1, alpha, beta))
          if value < alpha:
            return value
          beta = min(beta, value)
        else:
          value = min(value, self.minState(succesor, agentIndex + 1, depth, alpha, beta))
          if value < alpha:
            return value
          beta = min(beta, value)

      return value

    def maxState(self, gameState, agentIndex, depth, alpha, beta):
      if depth > self.depth:
        return self.evaluationFunction(gameState)
        
      actions = gameState.getLegalActions(agentIndex)

      if len(actions) == 0:
        return self.evaluationFunction(gameState)

      value = float("-inf")
      for action in actions:
        succesor = gameState.generateSuccessor(agentIndex, action)
        value = max(value, self.minState(succesor, agentIndex + 1, depth, alpha, beta))
        if value > beta:
          return value
        alpha = max(alpha, value)

      return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
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
      count = 0
      for succesor in succesors:
        count += 1
        if agentIndex == gameState.getNumAgents() - 1:
          values.append(self.maxState(succesor, 0, depth + 1))
        else:
          values.append(self.minState(succesor, agentIndex + 1, depth ))

      return float(sum(values)) / float(count)

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

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Linear combination of the following weighted features
      -score: score value of current state
      --weight: 1
      
      -Number of Capsules
      --weight: -5
      
      -Distance to closest food
      --weight: -2
      
      -Number of Food left
      --weight: -4
      
      -Distance to closest ghost
      --weight: reciprical
      When little food was left, pacman would only care about distance to 
      ghosts, take the reciprical so he only worries about close ghosts

    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPositions = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    score = currentGameState.getScore()
    scoreWeight = 1

    capsules = len(currentGameState.getCapsules())
    capsuleWeight = -5

    foodDisWeight = -2
    disFood = map(lambda x: util.manhattanDistance(newPos, x), newFood.asList())
    if len(disFood):
      minDisFood = min(disFood)
    else:
      minDisFood = 0

    foodCountWeight = -4
    foodCount = len(newFood.asList())

    ghostDistanceWeight = 1
    ghostMinDistance =  min(map(lambda x: util.manhattanDistance(newPos, x), newGhostPositions))
    if ghostMinDistance != 0:
      ghostMinDistance = 1 / ghostMinDistance

    return foodDisWeight * minDisFood + foodCountWeight * foodCount + ghostDistanceWeight * ghostMinDistance + \
      score * scoreWeight + capsuleWeight * capsules 

# Abbreviation
better = betterEvaluationFunction

