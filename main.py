# qlearningAgents.py
# ------------------
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
from copy import copy, deepcopy
from typing import Any, Iterable
import dataclasses as dto
import collections
import random, math


class Game:
    grid: list[list[int]]
    workers: list[list[tuple[int, int]]]
    actual_player: int
    actual_worker: int
    agent: 'Any'

    def _set_grid(self, initi, initj, player):
        k = 0
        for i in range(initi, initi + 3):
            for j in range(initj, initj + 3):
                if i - initi == 1 and j - initj == 1:
                    continue
                self.grid[i][j] = player + 4
                self.workers[player][k] = (i, j)
                k += 1

    def __init__(self):
        self.grid = [[-1 for i in range(40)] for _ in range(40)]
        self.workers = [[(-1, -1) for i in range(8)] for _ in range(4)]
        k = 0
        for i in [1, 40 - 1 - 3]:
            for j in [1, 40 - 1 - 3]:
                self._set_grid(i, j, k)
                k += 1
        self.actual_player = 0
        self.actual_worker = 0
        self.agent = ApproximateQAgent()

    def get_pos_worker(self):
        return self.workers[self.actual_player][self.actual_worker]

    def possibles_moves(self):
        res = []
        x, y = self.get_pos_worker()
        if y > 0 and self.grid[x][y - 1] < 4:
            res.append('Down')
        if y < 40 - 1 and self.grid[x][y + 1] < 4:
            res.append('Up')
        if x > 0 and self.grid[x - 1][y] < 4:
            res.append('Left')
        if x < 40 - 1 and self.grid[x + 1][y] < 4:
            res.append('Right')
        res.append('None')
        return res

    def move(self, action):
        assert action in self.possibles_moves()
        x, y = self.get_pos_worker()
        x0, y0 = x, y
        match action:
            case "Down":
                y -= 1
            case "Up":
                y += 1
            case "Right":
                x += 1
            case "Left":
                x -= 1
            case "None":
                pass
        self.grid[x0][y0] = self.actual_player
        self.workers[self.actual_player][self.actual_worker] = (x, y)
        self.grid[x][y] = self.actual_player + 4
        self.actual_worker += 1
        if self.actual_worker >= 8:
            self.actual_worker = 0
            player_ = self.actual_player + 1
            self.actual_player = player_ if player_ < 4 else 0

    def scores(self):
        is_visited = collections.defaultdict(lambda: False)
        stack = [(0, 0)]
        res = [0] * 4
        while stack:
            x, y = stack.pop()
            if is_visited[(x, y)]:
                continue
            is_visited[(x, y)] = True
            y_ = self.grid[x][y]
            if y_ != -1:
                y_ = y_ if y_ < 4 else y_ - 4
                stack_player = []
                self.add_to_stack(is_visited, stack_player, x, y)
                k = self.connected_cells(is_visited, stack, stack_player, y_)
                res[y_] = max(res[y_], k)
            else:
                self.add_to_stack(is_visited, stack, x, y)
        return [-x for x in res]

    def get_scores_feature(self):
        res = self.scores()
        return (res + res)[self.actual_player: self.actual_player + 4]

    def connected_cells(self, is_visited, stack, stack_player, player):
        k = 1
        while stack_player:
            x, y = stack_player.pop()
            y_ = self.grid[x][y]
            if is_visited[(x, y)]:
                continue
            if y_ != player and y_ != player + 4:
                stack.append((x, y))
                continue
            is_visited[(x, y)] = True
            k += 1
            self.add_to_stack(is_visited, stack_player, x, y)
        return k

    def add_to_stack(self, is_visited, stack_player, x, y):
        for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if not is_visited[(i + x, j + y)] and 0 <= i + x < 40 and 0 <= j + y < 40:
                stack_player.append((i + x, j + y))

    def __str__(self):
        return "\n".join(("".join(str(x if x >= 0 else ' ') for x in xs) for xs in self.grid))

    def __repr__(self):
        return str(self)



def max_list(iterable: Iterable[Any], key=lambda x: x) -> list[Any]:
    res = []
    vl = None
    for i in iterable:
        new_value = key(i)
        if vl is None or vl < new_value:
            res = [i]
            vl = new_value
        if vl is not None and vl == new_value:
            res.append(i)
    return res


class QLearningAgent:
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def getLegalActions(self, state: Game):
        return state.possibles_moves()

    def __init__(self, **args):
        "You can initialize Q-values here..."
        self.q_values = collections.Counter()
        self.epsilon = 0.7
        self.alpha = 0.7
        self.discount = 0.3

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        try:  # How do I know how to get the terminal state?
            return max(self.getQValue(state, action) for action in self.getLegalActions(state))
        except ValueError:  # max() arg is an empty sequence
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        res = max_list(self.getLegalActions(state), key=lambda action: self.getQValue(state, action))
        if res:
            return random.choice(res)  # If some actions are optimal it returns randomly
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use til.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if random.random() < self.epsilon:
            return random.choice(legalActions)
        action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward) -> None:
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        old_q = self.getQValue(state, action)
        next_v = self.computeValueFromQValues(nextState)
        learned_q = reward + self.discount * next_v
        new_q = old_q + self.alpha * (learned_q - old_q)

        self.q_values[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor=lambda st, a: st, weights = None, **args):
        super().__init__(**args)
        self.featExtractor = extractor
        if weights is not None:
            self.weights = weights
        else:
            self.weights = collections.defaultdict(int)
        self.epsilon = 0.05
        self.gamma = 0.01
        self.alpha = 0.7
        self.discount = 0.01

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * feature_vector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor(state, action)
        q = 0
        for key in feature_vector:
            q += feature_vector[key] * self.getWeights()[key]
        return q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        self.alpha = self.alpha * 0.8
        old_q = self.getQValue(state, action)
        next_v = self.computeValueFromQValues(nextState)
        learned_q = reward + self.discount * next_v
        difference = learned_q - old_q

        feature_vector = self.featExtractor(state, action)
        for key in feature_vector:
            self.weights[key] = self.weights[key] + feature_vector[key] * self.alpha * difference


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        print(self.weights)

def euclidean_distance(x,y, a, b):
    return abs(x + y - a  - b)


def feat_extractor(g: Game, action: str):
    x, y = g.get_pos_worker()
    xs = [abs(x + y - a - b) for i, xs in enumerate(g.workers) if i != g.actual_player for a, b in xs]
    ys = [abs(x + y - a - b) for a, b in g.workers[g.actual_player]]
    min_dist = 40 + 40
    max_dist = 0
    a = deepcopy(g)
    a.move(action)
    scores = a.get_scores_feature()
    score = scores[0]
    adv_score = max(scores[1:])
    for i in range(40):
        for j in range(40):
            j_ = g.grid[i][j]
            if -1 <= j_ < 4 and j_ != g.actual_player:
                min_dist = min(min_dist, euclidean_distance(x, y, i, j))
                max_dist = max(max_dist, euclidean_distance(x, y, i, j))

    return {
        "min_adv": min(xs),
        # "max_adv": max(xs),
        # "min_com": min(ys),
        # "max_com": max(ys),
        "free_cell_min": min_dist,
        # "free_cell_max": max_dist,
        "score": score,
        "adv_score": adv_score,
    }


if __name__ == '__main__':
    g = Game()
    agent = [ApproximateQAgent(feat_extractor) for _ in range(4)]
    try:
        for i in range(600 * 8 * 4):
            if i % 100 == 0:
                print(f'Epoch {i}, {[a.weights for a in agent]}')
            action = agent[g.actual_player].getAction(g)
            before = deepcopy(g)
            g.move(action)
            reward = g.scores()[before.actual_player]
            agent[g.actual_player].update(before, action, g, reward)
    finally:
        print([a.weights for a in agent])
    print([a.weights for a in agent])




