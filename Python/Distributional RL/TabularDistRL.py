from __future__ import print_function
import os
import sys
import argparse

import keras
import math
import random
import keras.backend as K
import gym
import numpy as np
from random import randint
import datetime
import tensorflow as tf
from collections import deque
from keras.layers import Dense, Input, Activation
from keras.models import Sequential, load_model, Model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from array import *
import collections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from gym.envs.registration import register



register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 100, 'nobjectives': 4, 'nactions': 8, 'nsuccessor': 12, 'seed': 1}
)
register(
    id='FishWood-v0',
    entry_point='fishwood:FishWood',
    reward_threshold=0.0,
    kwargs={'fishproba': 0.1, 'woodproba': 0.9}
)


class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        #self.dist_table = [100][8]
        s = 2
        a = 2
        self.num_actions = 2

        self.dist_table = [[[0 for x in range(1)] for y in range(a)] for z in range(s)]

        #for i in range(100):
        #    for j in range(8):
        #        self.dist_table[i][j].append([])

        self.prob_table = [[[0 for x in range(1)] for y in range(a)] for z in range(s)]
        self.action_taken = [[0 for x in range(2)] for y in range(2)]
        #self.action_taken_ = [2]

        for s in range(2):
            for a in range(2):
                self.dist_table[s][a][:] = np.trim_zeros(self.dist_table[s][a][:])
                self.prob_table[s][a][:] = np.trim_zeros(self.prob_table[s][a][:])



        # Make environment
        self.debug_file = open('debug' ,'w')
        self.action_true = [0,0,0,0,0,0,0,0] 
        #self.action_taken = [100][8]

        self.epsilon = 0.1

        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state
        self.learning_rate = 0.1
        self.gamma = 1.0

        self.reward_counter = []
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])
        self.reward_counter.append([])    
        

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        self._num_rewards = getattr(self._env, 'num_rewards', 1)
        self._num_actions = np.prod([a.n for a in aspace])
        self._aspace = aspace

        # Make an utility function
        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

        # Build network
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_vars = self._env.observation_space.n                    # Prepare for one-hot encoding
        else:
            self._state_vars = np.product(self._env.observation_space.shape)

        if self._extra_state == 'none':
            self._actual_state_vars = self._state_vars
        elif self._extra_state == 'timestep':
            self._actual_state_vars = self._state_vars + 1                      # Add the timestep to the state
        elif self._extra_state == 'accrued':
            self._actual_state_vars = self._state_vars + self._num_rewards      # Accrued vector reward
        elif self._extra_state == 'both':
            self._actual_state_vars = self._state_vars + self._num_rewards + 1  # Both addition        

        #print('Number of primitive actions:', self._num_actions)
        #print('Number of state variables', self._actual_state_vars)
        #print('Number of objectives', self._num_rewards)

        # Lists for policy gradient
        self._experiences = []
    


    def encode_state(self, state, timestep, accrued, reward, previous_reward):
        """ Encode a raw state from Gym to a Numpy vector
        """
        index = 0
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            #print("State" , state)
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        # Add the extra state information
        extratimestep = [(50 - timestep) * 0.1]
        extraaccrued = accrued * 0.1

        index = np.where(rs == 1)
        index = index[0][0]

        return index
            
    
    def encode_reward(self, reward):
        """ Encode a scalar or vector reward as an array
        """
        if ifinstance(reward, float):
            return np.array([reward])
        else:
            return np.array(reward)

    def addTo(line):

        multi_dim_list = []
        
        return multi_dim_list   

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
            return eval(self._utility, {}, {'r'+str(i+1): rewards[i] for i in range(self._num_rewards)})

    

    def compare(self, lista, listb):

        for list_ in listb:

            if np.array_equal(np.array(lista), np.array(list_)):
                
                return 0

        return 1

    def getRewardIndex(self, action, rewards, rewardsMatrix):

        index = 0

        for i in rewardsMatrix[action]:
            if np.array_equal(np.array(i), np.array(rewards)):
                return index

            index += 1


        return index

    

    def run(self):
        """ Execute an option on the environment
        """
        env_state = self._env.reset()

        done = False

        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        rewards = np.zeros(shape=(self._num_rewards,))
        previous_culmulative_rewards = np.zeros(shape=(self._num_rewards,))

        timestep = 0
        scalar_reward = 0        

        action = 0
        action_ = 0
        time_since_utility = 0

        action_taken = np.zeros(shape=(self._num_rewards,))
        times_action_taken = np.zeros(shape=(self._num_rewards,))

        reward_wood = np.zeros(shape=(self._num_rewards,))
        reward_fish = np.zeros(shape=(self._num_rewards,))
        previous_utility = 0
        

        previous_reward = [0, 0]
        previous_action = 1

        

        rewards_ = []
        #rewards_recieved = [[0 for x in range(2)] for y in range(1)]
        count_actions_taken = [8]
        reward_prob = [8]
        potential_reward_action = []

        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        rewards_.append([])
        
        #potential_reward_action.append([])
        #potential_reward_action.append([])

        while not done:
            timestep += 1
            #print(" *** timestep *** : " , timestep , file = self.debug_file)
            # Select an action or option based on the current state            
            old_env_state = env_state

            state = self.encode_state(env_state, timestep, cumulative_rewards, cumulative_rewards, previous_culmulative_rewards)
            #print("State :: " , state,  file = self.debug_file)
            #print(state)            
            check = random.uniform(0, 1)        

            increase_utility = 0

            potential_reward_action = []
            potential_reward_action.append([])
            potential_reward_action.append([])
            prob_reward = 0

            if check < self.epsilon:
                # select random action
                action = random.randint(0, 1)
            #    print("Random Action : " , action , file = self.debug_file)
            else:

                if timestep < 4:
                    action = random.randint(0, 1)

                else:

                    best_prob = -2147483648                    
                    actionTaken = 0
                    rewardTaken = 0
                    times_action_taken = 0
                    reward_prob = 0
                    
                    # Tabular Dist RL Algorithm
                    # for each action
                    for i in range(self.num_actions):
                        #print(" **** Stats **** :: " , file = self.debug_file)

                        # for each reward learned for an action in a state
                        
                        a = 0
                        for j in self.dist_table[state][i]:    
                                                       
                            reward = j

                            #calculate the probability of getting that reward                            
                            actionTaken = self.prob_table[state][i][a]
                            reward_prob = actionTaken / self.action_taken[state][i]                                                    

                            # Calculate the potential increase/decrease in utility by taking an action that recieves reward
                            # Multiply the expected increase/decrease in utility by the probaility of getting that reward
                            potential_reward = (self.scalarize_reward(reward + cumulative_rewards) - self.scalarize_reward(cumulative_rewards))  * reward_prob                             

                            # take the action that returns the maximimum increase in utility
                            if potential_reward > best_prob:
                                action = i
                                best_prob = potential_reward
                                if best_prob <= 0:
                                    action = random.randint(0, 1) 

                            a += 1   


                            #print("Times Action Taken :: " , self.action_taken[state][i],  file = self.debug_file)                           
                            #print("Times Reward Recieved :: " , actionTaken,  file = self.debug_file)   
                            #print("Times action taken vector :: " , self.prob_table[state][i],  file = self.debug_file)    
                            #print("potential_reward :: " , potential_reward,  file = self.debug_file)      

                            #print("Action :: " , i, file = self.debug_file)
                            #print("Times Action Taken", self.action_taken[state][i], file = self.debug_file)
                            #print("actionTaken :: " , self.prob_table[state][i][:], file = self.debug_file)
                            #print("Reward  :: " , reward,  file = self.debug_file) 
                            #print("Reward Prob :: " , reward_prob,  file = self.debug_file)  
                            #print("Rewards :: " , self.dist_table[state][i],  file = self.debug_file)            
                     
            
            self.action_taken[state][previous_action] += 1
            


            previous_utility = self.scalarize_reward(cumulative_rewards)


            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                env_state, rewards, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                env_state, rewards, done, __ = self._env.step(action)

            if self._render:
                self._env.render()           
            
            cumulative_rewards += rewards     
            

            if self.compare(rewards, self.dist_table[state][previous_action][:]) == 1:
                self.dist_table[state][previous_action].append(rewards)
                self.prob_table[state][previous_action].append(0)            

            reward_index = self.getRewardIndex(previous_action, rewards, self.dist_table[state])
            
            self.prob_table[state][previous_action][reward_index] += 1              
            previous_action = action          

        # Mark episode boundaries
        

        #previous_culmulative_rewards = cumulative_rewards
        return cumulative_rewards



def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", action="store_true", default=False, help="Enable a graphical rendering of the environment")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--avg", type=int, default=1, help="Episodes run between gradient updates")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--ret", type=str, choices=['forward', 'both'], default='both', help='Type of return used for training, only forward-looking or also using accumulated rewards')
    parser.add_argument("--utility", type=str, help="Utility function, a function of r1 to rN")
    parser.add_argument("--extra-state", type=str, choices=['none', 'timestep', 'accrued', 'both'], default='none', help='Additional information given to the agent, like the accrued reward')
    parser.add_argument("--hidden", default=50, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")

    # Next and Sub from arguments
    args = parser.parse_args()

    # Instantiate learner
    learner = Learner(args)
    #learner.model = self.make_network()

    # Learn
    f = open('out-' + args.name, 'w')
    loss_file = open('loss_file', 'w')

    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)

    try:
        old_dt = datetime.datetime.now()
        avg = np.zeros(shape=(learner._num_rewards,))

        for i in range(args.episodes):
            rewards = learner.run()
            #print("Episode ::", i,  file=f)
            if i == 0:
                avg = rewards
            else:
                avg = 0.99 * avg + 0.01 * rewards            
           

            # decay epsilon
            learner.epsilon = learner.epsilon * 0.99
            if learner.epsilon < 0.1:
                learner.epsilon = 0.1

            scalarized_avg = learner.scalarize_reward(avg)

            #print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
            #print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, )
            print(scalarized_avg, file = f)
            f.flush()

    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

if __name__ == '__main__':
    main()
