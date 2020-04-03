
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding

import random
import sys

class DeepSeaTreasureDanger(gym.Env):
    

    def __init__(self):
        
        self.debug_file = open('debug' ,'w')
    
        self.depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
        self.treasure = [1, 2, 3, 5, 8, 16, 24, 50, 74, 124]
        self.numRows = 11
        self.numCols = 10
        self.TREASURE = 0
        self.TIME = 1
        
        

        self.action_space = spaces.Discrete(4)              # 2 actions, go fish and go wood
        #self.observation_space = spaces.Discrete(2)         # 2 states, fishing and in the woods
        self.num_rewards = 2                                # 2 objectives, amount of fish and amount of wood

        #self._seed()
        self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        # Pick an initial state at random
        #self._state = self.FISH
        self._timestep = 0
        self.agentRow = 0
        self.agentCol = 0
        self.agentRow = 0
        self.agentCol = 0
        self.state = 0

        return self.agentRow, self.agentCol, self.state

    def isValid(self, row, col):

        valid = True

        if row > self.depths[col]:
            valid = False

        return valid

    def updatePosition(self, action, row, col):

        row = row
        col = col

        if action == 0:
            col += 1
            if col > (self.numCols - 1):
                col -= 1

        if action == 1:
            col -= 1

            if col < 0:
                col = 0
            elif self.isValid(row, col) == False:
                col += 1

        if action == 2:
            row += 1

        if action == 3:
            row -= 1

            if row < 0:
                row = 0

        self.agentRow = row
        self.agentCol = col

        return self.agentRow, self.agentCol

    def getRewards(self, row, col):
        rewards = np.zeros((2,))

        rewards[self.TIME] = -1
        rewards[self.TREASURE] = 0

        #print("Row ", row, file = self.debug_file)
        #print("Col ", col, file = self.debug_file)
        #print("Depths : ", self.depths[col], file = self.debug_file)

        if row == self.depths[col]:
            rewards[self.TREASURE] = self.treasure[col]

        return rewards

    def isTerminal(self, row, col):

        if row == self.depths[col]:
            return True
        else:
            return False

    def getState(self, row, col):

        return col * self.numRows + row



    def step(self, state, action, timestep, row, col):

        rewards = np.zeros((2,))
        done = False

        _row_, _col_ = self.updatePosition(action, row, col)
        rewards = self.getRewards(_row_, _col_)
        timestep += 1
        state = self.getState(_row_, _col_)
        #print("Timestep", timestep, file = self.debug_file)
        _done_ = self.isTerminal(_row_, _col_)

        if timestep == 1000:
            done = True

        if _done_ == True:
            done = True

        return state, rewards, timestep, done, _row_, _col_, {}
