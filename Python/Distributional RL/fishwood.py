
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding

import random
import sys

class FishWood(gym.Env):
    

    def __init__(self, fishproba, woodproba):
        self._fishproba = fishproba
        self._woodproba = woodproba
        self.debug_file = open('debug' ,'w')
        self.FISH = 0
        self.WOOD = 1

        self.action_space = spaces.Discrete(2)              # 2 actions, go fish and go wood
        self.observation_space = spaces.Discrete(2)         # 2 states, fishing and in the woods
        self.num_rewards = 2                                # 2 objectives, amount of fish and amount of wood

        #self._seed()
        self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        # Pick an initial state at random
        self._state = self.FISH
        self._timestep = 0

        return self._state

    def step(self, state, action, timestep):
        # Obtain a resource from the current state
        rewards = np.zeros((2,))
        self._state = state
        self._timestep = timestep

        if state == self.WOOD and random.random() < self._woodproba:
            rewards[self.WOOD] = 1.0
        elif state == self.FISH and random.random() < self._fishproba:
            rewards[self.FISH] = 1.0

        # Execute the action
        #print("Environment State :: ", self._state, file = self.debug_file)
        self._state = action
        timestep += 1
        #print("Environment Reward :: ", rewards, file = self.debug_file)
        #print("Environment After State :: ", self._state, file = self.debug_file)
        
        return action, rewards, timestep, timestep == 200, {}
