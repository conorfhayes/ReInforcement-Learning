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
        done = False

        if action > 2:
        	return -1000000, [-1000000, -1000000], True

    	# Timestep 1
        # Planet A -> Planet B
        if state == 0 and action == 0:
        	state = 1
        	rewards = [0, -12]

        	return state, rewards, done, {}

    	if state == 0 and action == 1 and random.random() <= 0.9:
    		state = 1
    		rewards = [0, -6]

    		return state, rewards, done, {}

		elif state == 0 and action == 1 and random.random() > 0.9:
			rewards = [0, -1]
			done == True

			return state, rewards, done, {}

		if state == 0 and action == 2 and random.random() <= 0.85:
			rewards = [0,0]
			state = 1

			return state, rewards, done, {}

		elif state == 0 and action == 2 and random.random() > 0.85:
			rewards = [0,0]
			done == True

			return state, rewards, done, {}



		#Timestep 2
		#Planet B -> Planet A
		if state == 1 and action == 0:
			state = 0
        	rewards = [1, -10]
        	done == True

        	return state, rewards, done, {}   

    	if state == 1 and action == 1 and random.random() <= 0.9:
    		state = 0
    		rewards = [1, -8]
    		done == True

    		return state, rewards, done, {}

		elif state == 1 and action == 1 and random.random() > 0.9:
			rewards = [0, -7]
			done == True

			return state, rewards, done, {}

		if state == 1 and action == 2 and random.random() <= 0.85:
			rewards = [1,0]
			state = 0
			done = True

			return state, rewards, done, {}

		elif state == 0 and action == 2 and random.random() > 0.85:
			rewards = [0,0]
			done == True

			return state, rewards, done, {}   
        
       

        return state, rewards, done, {}
