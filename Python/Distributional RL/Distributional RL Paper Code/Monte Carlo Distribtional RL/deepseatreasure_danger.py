
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding

import random
import sys

class DeepSeaTreasureDanger(gym.Env):
    

    def __init__(self):
        
        #self.debug_file = open('debug' ,'w')
    
        self.depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
        self.treasure = [1, 2, 3, 5, 8, 16, 24, 50, 74, 124]
        self.numRows = 11
        self.numCols = 10
        self.TREASURE = 0
        self.HEALTH = 1
        self.TIME = 2
        
        

        self.action_space = spaces.Discrete(4)              # 2 actions, go fish and go wood
        #self.observation_space = spaces.Discrete(2)         # 2 states, fishing and in the woods
        self.num_rewards = 3                               # 2 objectives, amount of fish and amount of wood

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
        self._health_ = 0

        return self.state

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

        #self.agentRow = row
        #self.agentCol = col

        return row, col
    def getRewards(self, row, col):
        rewards = np.zeros((3,))

        rewards[self.TIME] = -1
        rewards[self.TREASURE] = 0
        rewards[self.HEALTH] = 0

        #print("Row ", row, file = self.debug_file)
        #print("Col ", col, file = self.debug_file)
        #print("Depths : ", self.depths[col], file = self.debug_file)

        if row == self.depths[col]:
            rewards[self.TREASURE] = self.treasure[col]



        if row == 1 and col == 4:
            if random.random() < 0.5:

                rewards[self.HEALTH] = -75

        elif row == 1 and col == 5:

            if random.random() < 1:
                
                rewards[self.HEALTH] = -100

        elif row == 1 and col == 6:

            if random.random() < 0.5:
                
                rewards[self.HEALTH] = -75

        elif row == 2 and col == 4:
            if random.random() < 0.5:

                rewards[self.HEALTH] = -75

        elif row == 2 and col == 5:

            if random.random() < 1:
                
                rewards[self.HEALTH] = -100

        elif row == 2 and col == 6:

            if random.random() < 0.5:
                
                rewards[self.HEALTH] = -75

        elif row == 4 and col == 6:
            if random.random() < 0.5:

                rewards[self.HEALTH] = -75

        elif row == 4 and col == 7:

            if random.random() < 1:
                
                rewards[self.HEALTH] = -100

        elif row == 4 and col == 8:

            if random.random() < 0.5:
                
                rewards[self.HEALTH] = -75

        elif row == 5 and col == 6:
            if random.random() < 0.5:

                rewards[self.HEALTH] = -75

        elif row == 5 and col == 7:

            if random.random() < 1:
                
                rewards[self.HEALTH] = -100

        elif row == 5 and col == 8:

            if random.random() < 0.5:
                
                rewards[self.HEALTH] = -75

        elif row == 6 and col == 6:
            if random.random() < 0.5:

                rewards[self.HEALTH] = -75

        elif row == 6 and col == 7:

            if random.random() < 1:
                
                rewards[self.HEALTH] = -100

        elif row == 6 and col == 8:

            if random.random() < 0.5:
                
                rewards[self.HEALTH] = -75


        return rewards

    def isTerminal(self, row, col):

        if  row == 1 and col == 5:
            return True
        elif  row == 2 and col == 5:
            return True
        elif  row == 4 and col == 7:
            return True
        elif  row == 5 and col == 7:
            return True
        elif  row == 6 and col == 7:
            return True
        elif row == self.depths[col]:
            return True
        else:
            return False    

    def getState(self, col, row):
        state = [col, row]

        numState = 11 * 10
        basesForStateNo = [10, 10]
        stateNo = 0

        for i in range(2):
            stateNo = int(stateNo * basesForStateNo[i] + state[i])

        return stateNo


    def getXYfromState(self,state):
        
        basesForStateNo = [10, 10]
        stateNo = [0,0]
        # stateNo = 0
        inputstateNo = state
        j = 1
        for i in range(2):

            check = int(inputstateNo % basesForStateNo[i])
            stateNo[j] = check
            inputstateNo = int(inputstateNo / basesForStateNo[i])
            j -= 1

        return stateNo



    def step(self, state, action, health):

        rewards = np.zeros((3,))
        done = False

        state_ = self.getXYfromState(state)
        col = state_[0]
        row = state_[1]

        #print("Row", row, file = self.debug_file)
        #print("Col",col, file = self.debug_file)

        _row_, _col_ = self.updatePosition(action, row, col)

        #print("Row A", _row_, file = self.debug_file)
        #print("Col A",_col_, file = self.debug_file)


        rewards = self.getRewards(_row_, _col_)
        state = self.getState(_col_, _row_)

        #print("State ", state, file = self.debug_file)

        _done_ = self.isTerminal(_row_, _col_)

        health = rewards[self.HEALTH] + health

        if health <= -100:
            health = -100
            rewards[1] = -100
            _done_ = True


        if _done_ == True:
            done = True



        return state, rewards, done, health, {}
