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
    """

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(agent_index, depth, game_state):
            if agent_index >= game_state.getNumAgents():
                agent_index = 0
                depth += 1
            if depth == self.depth or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state), None

            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state), None

            if agent_index == 0:
                best_score, best_action = float('-inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = minimax(agent_index + 1, depth, successor)
                    if score > best_score:
                        best_score, best_action = score, action
                return best_score, best_action
            else:
                best_score, best_action = float('inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = minimax(agent_index + 1, depth, successor)
                    if score < best_score:
                        best_score, best_action = score, action
                return best_score, best_action

        _, action = minimax(0, 0, gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction with alpha-beta pruning.
        """
        def alpha_beta(curr_depth, agent_index, gameState, alpha, beta):
            """
            Performs the alpha-beta pruning for the minimax tree.
            """
            if agent_index >= gameState.getNumAgents():
                agent_index = 0
                curr_depth += 1

            # Check if we reached the depth limit or there are no legal actions left
            if curr_depth == self.depth or not gameState.getLegalActions(agent_index):
                return self.evaluationFunction(gameState)

            if agent_index == 0:  # Pacman (MAX player)
                value = float('-inf')
                for action in gameState.getLegalActions(agent_index):
                    successor = gameState.generateSuccessor(agent_index, action)
                    value = max(value, alpha_beta(curr_depth, agent_index + 1, successor, alpha, beta))
                    if value > beta:  # Prune
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts (MIN players)
                value = float('inf')
                for action in gameState.getLegalActions(agent_index):
                    successor = gameState.generateSuccessor(agent_index, action)
                    value = min(value, alpha_beta(curr_depth, agent_index + 1, successor, alpha, beta))
                    if value < alpha:  # Prune
                        return value
                    beta = min(beta, value)
                return value

        # Root call for Pacman
        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alpha_beta(0, 1, successor, alpha, beta)
            if value > alpha:
                alpha = value
                best_action = action

        return best_action



     


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(agent_index, depth, game_state):
            if agent_index >= game_state.getNumAgents():
                agent_index = 0
                depth += 1
            if depth == self.depth or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state), None

            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state), None

            if agent_index == 0:
                best_score, best_action = float('-inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = expectimax(agent_index + 1, depth, successor)
                    if score > best_score:
                        best_score, best_action = score, action
                return best_score, best_action
            else:
                expected_score = 0
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = expectimax(agent_index + 1, depth, successor)
                    expected_score += score / len(legal_actions)
                return expected_score, None

        _, action = expectimax(0, 0, gameState)
        return action

def betterEvaluationFunction(currentGameState):
    return currentGameState.getScore()

better = betterEvaluationFunction
