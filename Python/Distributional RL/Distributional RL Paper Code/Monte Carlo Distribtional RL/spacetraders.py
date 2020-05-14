import gym
import numpy as np

from gym import spaces
from gym.utils import seeding

import random
import sys


class SpaceTraders(gym.Env):

    def __init__(self):

        self.action_space = spaces.Discrete(3)
        self.num_rewards = 2

        self.reset()

    def reset(self):

        self.state = 0

        return self.state

    def step(self, state, action):

        rewards = np.zeros((2,))
        _random_ = random.random()
        done = False

        if action > 2:
            return -1000000, [-1000000, -1000000], True, {}

        # Timestep 1
        # Planet A -> Planet B        
        if state == 0 and action == 0:
            next_state = 1
            rewards = [0, -12]
            done = False
    
            return next_state, np.array(rewards), done, {}
    
        if state == 0 and action == 1 and _random_ <= 0.9:
    
            next_state = 1
            rewards = [0, -6]
            done = False
    
            return next_state, np.array(rewards), done, {}
    
        elif state == 0 and action == 1 and _random_ > 0.9:
    
            next_state = 1
            rewards = [0, -1]
            done = True
    
            return next_state, np.array(rewards), done, {}
    
        if state == 0 and action == 2 and _random_ <= 0.85:
            next_state = 1
            rewards = [0, 0]
            done = False
    
            return next_state, np.array(rewards), done, {}
    
        elif state == 0 and action == 2 and _random_ > 0.85:
            next_state = 1
            rewards = [0, 0]
            done = True
    
            return next_state, np.array(rewards), done, {}
    
        # Timestep 2
        # Planet B -> Planet A
        if state == 1 and action == 0:
            next_state = 2
            rewards = [1, -10]
            done = True
            return next_state, np.array(rewards), done, {}
    
        if state == 1 and action == 1 and _random_ <= 0.9:
    
            next_state = 2
            rewards = [1, -8]
            done = True
    
            return next_state, np.array(rewards), done, {}
    
        elif state == 1 and action == 1 and _random_ > 0.9:
            next_state = 2
            rewards = [0, -7]
            done = True
    
            return next_state, np.array(rewards), done, {}
    
        if state == 1 and action == 2 and _random_ <= 0.85:
            next_state = 2
            rewards = [1, 0]
            state = 0
            done = True
    
            return next_state, np.array(rewards), done, {}
    
        elif state == 1 and action == 2 and _random_ > 0.85:
            next_state = 2
            rewards = [0, 0]
            done = True
    
            return next_state, np.array(rewards), done, {}

      
