# multiAgents.py
# --------------
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        mdistances = []
        mdscore = 0
        mdglist = [manhattanDistance(newPos, i.getPosition()) for i in newGhostStates]
        mdgav = reduce(lambda x, y: x+y, mdglist)/len(mdglist)

        if not newFood.asList():
            mdscore = 0
        else:
            for x2 in newFood.asList():
                mdistances.append(manhattanDistance(newPos,x2))
                mdscore = min(mdistances)

        return successorGameState.getScore() + min(newScaredTimes) + 1/(mdscore + 0.1) - (1/(mdgav + 0.1))

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

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #The only maximizer is pacman Index 0, minimizer gets called here
        def maximizer(gameState, agentIndex, currentDepth):
            succScore = []

            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, OMG agentIndex needs to start with 1 when maximzer is called
                succScore.append(minimizer(succState, 1, currentDepth))

            return max(succScore)

        #returns the min value for a set of actions - Only Ghosts are minimizer
        def minimizer(gameState, agentIndex, currentDepth):
            succScore = []
            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, but check if next state is max or min
                if agentIndex % (gameState.getNumAgents() - 1) == 0:
                    #next maxmizer need to increment due to ply depth
                    succScore.append(maximizer(succState, 0, currentDepth + 1))
                else:
                    succScore.append(minimizer(succState, agentIndex + 1, currentDepth))

            return min(succScore)

        #get maximizer value of root node and pick action
        actions = gameState.getLegalActions(0)
        #dictionary of actions {value: action}
        actionDict = {}
        scores = []

        for a in actions:
            succState = gameState.generateSuccessor(0, a)
            #Successor states are minimizers starts with agentIndex 1 and depth 0
            curScore = minimizer(succState, 1, 0)

            #check if score is greater than the max of scores
            scores.append(curScore)
            actionDict[curScore] = a

        return actionDict[max(scores)]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        beta = float('inf')
        alpha = float('-inf')
        #print("num of agents: ", gameState.getNumAgents())

        #The only maximizer is pacman Index 0, minimizer gets called here
        def maximizer(gameState, alpha, beta, agentIndex, currentDepth):
            #print("max agent: ",agentIndex)

            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            #maximizer initializes -inf
            nodeAlpha = float('-inf')

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, OMG agentIndex needs to start with 1 when maximzer is called
                succScore = minimizer(succState, alpha, beta, 1, currentDepth)

                if succScore > nodeAlpha:
                    nodeAlpha = succScore

                #prune condition for max just return current alpha
                if nodeAlpha > beta:
                    return nodeAlpha
                if nodeAlpha > alpha:
                    alpha = nodeAlpha

            return nodeAlpha

        #returns the min value for a set of actions - Only Ghosts are minimizer
        def minimizer(gameState, alpha, beta, agentIndex, currentDepth):
            #print("min agent: ",agentIndex)

            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            #minimizer stores the beta
            nodeBeta = float('inf')

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, but check if next state is max or min
                if agentIndex % (gameState.getNumAgents() - 1) == 0:
                    #next maxmizer need to increment due to ply depth
                    succScore = maximizer(succState, alpha, beta, 0, currentDepth + 1)
                else:
                    succScore = minimizer(succState, alpha, beta, agentIndex + 1, currentDepth)

                if succScore < nodeBeta:
                    nodeBeta = succScore
                if nodeBeta < alpha:
                    return nodeBeta
                if nodeBeta < beta:
                    beta = nodeBeta

            return nodeBeta

        #get maximizer value of root node and pick action
        actions = gameState.getLegalActions(0)
        #dictionary of actions {value: action}
        actionDict = {}
        scores = [alpha]

        for a in actions:
            succState = gameState.generateSuccessor(0, a)
            #Successor states are minimizers starts with agentIndex 1 and depth 0
            curScore = minimizer(succState, alpha, beta, 1, 0)

            #check if score is greater than the max of scores
            if curScore > max(scores):
                scores.append(curScore)
                actionDict[curScore] = a
            #prune condition
            if curScore > beta:
                return actionDict[max(scores)]

            alpha = max(scores)
        return actionDict[alpha]


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
        #The only maximizer is pacman Index 0, minimizer gets called here
        def maximizer(gameState, agentIndex, currentDepth):
            succScore = []

            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, OMG agentIndex needs to start with 1 when maximzer is called
                succScore.append(minimizer(succState, 1, currentDepth))

            return max(succScore)

        #returns the min value for a set of actions - Only Ghosts are minimizer
        def minimizer(gameState, agentIndex, currentDepth):
            succScore = []
            #base case, leaf node
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            #iterate through states and check for pruning conditions
            for a in actions:
                succState = gameState.generateSuccessor(agentIndex, a)

                #get score of successor State, but check if next state is max or min
                if agentIndex % (gameState.getNumAgents() - 1) == 0:
                    #next maxmizer need to increment due to ply depth
                    succScore.append(maximizer(succState, 0, currentDepth + 1))
                else:
                    succScore.append(minimizer(succState, agentIndex + 1, currentDepth))

            return float(sum(succScore))/float(len(succScore))

        #get maximizer value of root node and pick action
        actions = gameState.getLegalActions(0)
        #dictionary of actions {value: action}
        actionDict = {}
        scores = []

        for a in actions:
            succState = gameState.generateSuccessor(0, a)
            #Successor states are minimizers starts with agentIndex 1 and depth 0
            curScore = minimizer(succState, 1, 0)

            #check if score is greater than the max of scores
            scores.append(curScore)
            actionDict[curScore] = a

        return actionDict[max(scores)]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      Important Factors that raise score (+):
      DP - distance to nearest pellet, DGS - distance to nearest ghost (when scared),
      DPP - distance to nearest power pellet

      Important Factors that reduce score (-):
      DNG - distance to nearest ghost (not scared, increase the shorter the distance, reciprocal),

      Some linear combo of: a(DP) + b(DGS) + c(DPP) - d(DNG)
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)



    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    foodDistances = []
    mdscore = 0
    ghostPositions = [manhattanDistance(newPos, i.getPosition()) for i in newGhostStates]
    mdgav = reduce(lambda x, y: x+y, ghostPositions)/len(ghostPositions)

    if not newFood.asList():
        mdscore = 0
    else:
        foodDistances = [manhattanDistance(newPos,i) for i in newFood.asList()]
        mdscore = min(foodDistances)

    return currentGameState.getScore() + min(newScaredTimes) + 1/(mdscore + 0.1) - (1/(mdgav + 0.11))
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
