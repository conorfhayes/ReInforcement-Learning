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
from collections import deque
from array import *
import collections


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


class Node():     

    def __init__(self, env, action, state, id, cumulative_rewards, timestep, layer, file, _dict_, probability):
        self.simulations = 5   
        
        #self.debug_file = open('Node ' + str(id) ,'w')
        self.debug_file = file
        self.children = []
        self.layer = layer + 1
        self.env = env
        self.unique_id = layer + 1
        self.num_actions = 2
        self.data = []
        self.name = state
        self.times_visited = 0
        self.reward = 0
        self.timestep = timestep
        self.cumulative_rewards = cumulative_rewards
        self.id = id
        self.done = False
        self.action = action
        self.probability = probability

        if self.id == 0:
            self.parent = None
        else:
            self.parent = self.id - 1

        #self.state = self.getState(state, action)
        self.state = state
        self.rewards = [[[0,0], [1,0]], [[0,0], [0,1]]]
        self.rewards_counter = [[[0], [0]], [[0], [0]]]
        self.run(self.state, _dict_, probability)

    def getDetails(self):
        
        return self.env, self.action, self.state, self.id, self.cumulative_rewards, self.timestep

    def create_children(self, state, action, cumulative_rewards, _dict_, probability):
        child_id = self.id + 1
       # env_timestep = child_id + self.timestep
        #for i in range(self.num_actions):.
        
        self.children.append(Node(self.env, action, state, child_id, cumulative_rewards, self.timestep + 1, self.layer, self.debug_file,_dict_, probability))
        self.unique_id += 1      
    

    def getState(self, state, action):

        if state == 0 and action == 1:
            state = 1
        elif state == 1 and action == 1:
            state = 1
        elif state == 0 and action == 0:
            state = 0
        elif state == 0 and action == 0:
            state = 0

        return state

    def simultation(self, state, action, cumulative_rewards, timestep, _dict_):
        next_state , reward, timestep, self.done, __ = self.env.step(state, action, timestep)
        #prob = dict_[0][0][0]['times_here'] / dict_[0][0][0][str([0,1])]['times_here'] 
        print("Prob : ", _dict_[state][action][next_state][str(reward)]['count'], file = self.debug_file)
        print("Prob2 : ", _dict_[state][action][next_state]['count'], file = self.debug_file)
        probability = _dict_[state][action][next_state][str(reward)]['count'] / _dict_[state][action][next_state]['count']
        print("Probability : ", probability, file = self.debug_file)
        self.reward = reward
        #self.state = state

        return probability, next_state, reward + cumulative_rewards

    def run(self, state, _dict_, probability):
        timestep = self.timestep
        #state = 0
        for action in range(2):
            if self.layer < self.simulations + 1:
                probability, next_state, cumulative_rewards =  self.simultation(state, action, self.cumulative_rewards, timestep, _dict_)
                #self.create_children()  
                if self.probability == 0:
                    self.probability = 0
                else:    
                    self.probability = probability * self.probability
                self.state = next_state
                #print("Simulation Ran : ",file = self.debug_file)          

            if self.layer < self.simulations and self.done == False:
                #self.cumulative_rewards =  self.simultation(self.action, self.cumulative_rewards, self.timestep)
                self.create_children(next_state, action, cumulative_rewards,_dict_, self.probability)

        self.timestep = self.timestep + 1





class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        #self.dist_table = [100][8]
        s = 2
        a = 2
        self.num_actions = 2
        self.num_timesteps = 200

        self.dict = {}

        self.dist_table = [[[0 for x in range(1)] for y in range(a)] for z in range(s)]
        self.dist_time_table = [[[[0 for x in range(1)] for y in range(s)] for z in range(self.num_timesteps)] for j in range(a)] 

        self.state_table = [[[0 for x in range(s)] for y in range(a)] for z in range(s)]
        self.state_time_table = [[[[0 for x in range(s)] for y in range(a)] for z in range(s)] for j in range(self.num_timesteps)] 

        self.prob_table = [[[0 for x in range(1)] for y in range(a)] for z in range(s)]
        self.prob_time_table = [[[[0 for x in range(1)] for y in range(a)] for z in range(self.num_timesteps)] for j in range(a)]

               
        self.action_taken = [[0 for x in range(2)] for y in range(2)]
        self.action_time_taken = [[[0 for x in range(2)] for y in range(self.num_timesteps + 1)] for j in range(2)]
        #self.action_taken_ = [2]
        self.state_time_counter = [[0 for x in range(200)] for y in range(2)]
        
        self.all_reward = [[0 for x in range(1)] for y in range(2)]
        self.RtProb = 0

        for s in range(2):
            self.all_reward[s][:] = np.trim_zeros(self.all_reward[s][:])
            for a in range(2):
                self.dist_table[s][a][:] = np.trim_zeros(self.dist_table[s][a][:])
                self.prob_table[s][a][:] = np.trim_zeros(self.prob_table[s][a][:])

        for s in range(2):
            for t in range(200):
                for a in range(2):
                    self.dist_time_table[s][t][a][:] = np.trim_zeros(self.dist_time_table[s][t][a][:])
                    self.prob_time_table[s][t][a][:] = np.trim_zeros(self.prob_time_table[s][t][a][:])




        # Make environment
        self.debug_file = open('debug' ,'w')
        self.action_true = [0,0,0,0,0,0,0,0] 
        #self.action_taken = [100][8]

        self.epsilon = 1.0

        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state
        self.learning_rate = 0.1
        self.gamma = 1.0

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


    def getResults(self, tree):

        dist_rewards = []
        dist_prob = []

        a = 0

        #for node in tree:
        #for node in range(2):
        def _getResults(tree):
            if len(tree.children) == 0:
                dist_rewards.append(tree.cumulative_rewards)
                dist_prob.append(tree.probability)
                #print("Dist Rewards : " ,dist_rewards, file = self.debug_file)
            
            for n in range(len(tree.children)):
                _getResults(tree.children[n])

            
        _getResults(tree)

        #if len(tree.children) == 0:
        #    dist_reward.append(tree.children.cumulative_rewards)

        
        #print("Dist Rewards : " ,dist_rewards, file = self.debug_file)
        return dist_rewards, dist_prob
    


    def rollOut(self, env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_):

        #self.rollOut(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0)

        tree = Node(env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_, 0)
        action_rewards, action_prob = self.getResults(tree)
        """
        print("CR", cumulative_rewards, file = file)
        print("Layer A0 1 State ", tree.state, file = file)
        print("Layer A0 1 R ", tree.reward, file = file)
        print("Layer A0 2 State ", tree.children[0].state, file = file)
        print("Layer A0 2 R ", tree.children[0].reward, file = file)
        print("Layer A0 3 State ", tree.children[0].children[0].state, file = file)
        print("Layer A0 3 R ", tree.children[0].children[0].reward, file = file)
        print("Layer A0 4 State ", tree.children[0].children[0].children[0].state, file = file)
        print("Layer A0 4 R ", tree.children[0].children[0].children[0].reward, file = file)
        print("Layer A0 5 State", tree.children[0].children[0].children[0].children[0].state, file = file)
        print("Layer A0 5 R", tree.children[0].children[0].children[0].children[0].reward, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A0 1 CR ", tree.cumulative_rewards, file = file)
        print("Layer A0 2 CR ", tree.children[0].cumulative_rewards, file = file)
        print("Layer A0 3 CR ", tree.children[0].children[0].cumulative_rewards, file = file)
        print("Layer A0 4 CR ", tree.children[0].children[0].children[0].cumulative_rewards, file = file)
        print("Layer A0 5 CR", tree.children[0].children[0].children[0].children[0].cumulative_rewards, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A1 1 R ", tree.reward, file = file)
        print("Layer A1 2 R ", tree.children[1].reward, file = file)
        print("Layer A1 3 R ", tree.children[1].children[1].reward, file = file)
        print("Layer A1 4 R ", tree.children[1].children[1].children[1].reward, file = file)
        print("Layer A1 5 R", tree.children[1].children[1].children[1].children[1].reward, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A1 1 CR ", tree.cumulative_rewards, file = file)
        print("Layer A1 2 CR ", tree.children[1].cumulative_rewards, file = file)
        print("Layer A1 3 CR ", tree.children[1].children[1].cumulative_rewards, file = file)
        print("Layer A1 4 CR ", tree.children[1].children[1].children[1].cumulative_rewards, file = file)
        print("Layer A1 5 CR", tree.children[1].children[1].children[1].children[1].cumulative_rewards, file = file)
        """

        return action_rewards, action_prob


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

        #print("List A :: ", lista, file = self.debug_file)
        #print("List B :: ", listb, file = self.debug_file)
        for list_ in listb:

            if np.array_equal(np.array(lista), np.array(list_)):
                
                return 0

        return 1

    def getRewardIndex(self, action, rewards, rewardsMatrix):

        index = -10
        a = 0
        for i in rewardsMatrix[action]:
            comparison = np.array(i) == np.array(rewards)
            
            if comparison.all() == True:
                index = a                
                return index

            a += 1
            

        if index >= 2:
            index = -10


        return index

    def getTimeRewardIndex(self, action, rewards, rewardsMatrix):

        index = 0

        for i in rewardsMatrix:
            #print("i", i, file = self.debug_file)
            #print("rewards matrix", rewardsMatrix[action], file= self.debug_file)
            if np.array_equal(np.array(i), np.array(rewards)):
                return index

            index += 1


        return index

    def getNextState(self, state, action):

        if action == 0:
            return 0

        if action == 1:
            return 1

    #def Generate(rootNode, cumulative_rewards, computational_timesteps):
    #    timestep = 0

    #    self.rootNode.ex





    def getRtplus1(self, cumulative_rewards):

        probReward = 0
        maxAction = -1
        maxUtility = -1
        prob = 0
        #bestReward = []

        prob_reward = 0
        prob_reward1 = 0
        maxAction1 = -1
        maxUtility1 = -1
        bestaction = -1
        bestaction1 = -1 
        comparison1 = 0
        prob1 = 0
        #comparison = 0
        #bestReward1 = np.array(shape = (2))
        totalProb = 0
        #print(self.all_reward, file = self.debug_file)
        for action in range(2):
            for reward in self.all_reward[action]:
                #reward = np.array(reward)
                #comparison = np.array(reward) == np.array([0,0])
                #if comparison.all() == 1 and action == 1:
                #    prob_reward = 0.1

                comparison = np.array(reward) == np.array([0,1])
                if comparison.all() == 1:                
                    prob_reward = 0.9

                #comparison = np.array(reward) == np.array([0,0])
                #if comparison.all() == 1 and action == 0:                
                #    prob_reward = 0.9
                
                comparison = np.array(reward) == np.array([1,0])   
                if comparison.all() == 1:
                    prob_reward = 0.1

                maxUtilityCalc = self.scalarize_reward(cumulative_rewards + np.array(reward).all()) - self.scalarize_reward(cumulative_rewards)

                if maxUtilityCalc > maxUtility:
                    maxUtility = maxUtilityCalc
                    maxAction = action
                    prob = prob_reward
                    bestReward = np.array(reward)
                    bestaction = action

                


        for action1 in range(2):
            for reward1 in self.all_reward[action1]:
                #reward1 = np.array(reward1)
                #comparison = np.array(reward1) == np.array([0,0])
                #if comparison.all() == 1 and action1 == 1:
                #    prob_reward1 = 0.1

                comparison = np.array(reward1) == np.array([0,1])
                if comparison.all() == 1:                
                    prob_reward1 = 0.9

                #comparison = np.array(reward1) == np.array([0,0])
                #if comparison.all() == 1 and action1 == 0:                
                #    prob_reward1 = 0.9
                
                comparison = np.array(reward1) == np.array([1,0])   
                if comparison.all() == 1:
                    prob_reward1 = 0.1


                maxUtilityCalc1 = self.scalarize_reward(cumulative_rewards + bestReward.all() + reward1.all()) - self.scalarize_reward(cumulative_rewards + bestReward.all())

                if maxUtilityCalc1 > maxUtility1:
                    maxUtility1 = maxUtilityCalc1
                    maxAction1 = action1
                    prob1 = prob_reward1
                    bestReward1 = reward1
                    #print("Best Reward Rt+1", bestReward1, file = self.debug_file)

                

        if maxUtility == 0:
            a = round(randint(0,1))
            b = round(randint(0,1))
            bestReward = self.all_reward[a][b]

            #comparison = np.array(bestReward) == np.array([0,0])
            #if comparison.all() == 1 and bestaction == 1:
            #    prob = 0.1

            comparison = np.array(bestReward) == np.array([0,1])
            if comparison.all() == 1:                
                prob = 0.9

            #comparison = np.array(bestReward) == np.array([0,0])
            #if comparison.all() == 1 and bestaction == 0:                
            #    prob = 0.9
                
            comparison = np.array(bestReward) == np.array([1,0])   
            if comparison.all() == 1:
                prob = 0.1

        if maxUtility1 == 0:
           #print("In the money", bestReward1, file = self.debug_file)
            a = round(randint(0,1))
            b = round(randint(0,1))
            bestReward1 = self.all_reward[a][b]

            
            comparison = np.array(bestReward1) == np.array([0,0])
            comparison1 = np.array(bestReward) == np.array([0,0])

            if comparison.all() == 1 and comparison1.all() == 1:
                prob1 = 0
                prob = 0

            #comparison = np.array(bestReward1) == np.array([0,0])
            #if comparison.all() == 1 and bestaction1 == 1:
            #    prob1 = 0.1

            comparison = np.array(bestReward1) == np.array([0,1])
            if comparison.all() == 1:                
                prob1 = 0.9

            #comparison = np.array(bestReward1) == np.array([0,0])
            #if comparison.all() == 1 and bestaction1 == 0:                
            #    prob1 = 0.9
                
            comparison = np.array(bestReward1) == np.array([1,0])   
            if comparison.all() == 1:
                prob1 = 0.1


        #print("Best Reward Rt", bestReward, file = self.debug_file)
        #print("Prob", prob, file = self.debug_file)
        #print("Best Reward Rt+1", bestReward1, file = self.debug_file)
        #print("Prob1", prob1, file = self.debug_file)

        totalProb = prob1 * prob

        #print("Best Reward Rt+1 Prob", totalProb, file = self.debug_file)

        return np.array(bestReward1), totalProb

       

    def reward_estimator(self, state, cumulative_rewards):
        reward_estimator = 0
        potential_action = 0
        reward_prob = 0
        state_prob = 0
        rewards = []
        rewardHolder = []
        tester = 0
        value = 0

        for i in range(self.num_actions):                    
            a = 0
            rewards.append([])
            for j in self.all_reward[i]: 
            #for j in self.dist_table[state][i] 

                                      
                reward = self.all_reward[i][a]
                reward = self.dist_table[state][i][a]
                next_state = self.getNextState(state, i)
                                             
                potential_action = self.prob_table[state][i][a]
                if self.action_taken[state][i] == 0:
                    reward_prob = 0
                else:
                    reward_prob = potential_action / self.action_taken[state][i]

                if self.action_taken[state][i]  == 0:
                    state_prob = 0
                else:
                    state_prob = self.state_table[state][i][next_state] / self.action_taken[state][i]    
                #tester =    self.prob_table[next_state][i][a] /self.state_table[state][i][next_state]  
                Rt1, Rt1Prob = self.getRtplus1(cumulative_rewards)      
                value = (reward * reward_prob)

                reward_estimator = (value + (self.gamma * 1 )) + (Rt1 + Rt1Prob)
                rewards[i].append([])
                #print("Value ::", value, file =self.debug_file)
                #print("Reward ::", reward, file =self.debug_file)
                #print("state_prob ::", state_prob, file =self.debug_file)
                rewards[i][a] = reward_estimator
                #rewardHolder.append(j)

                a +=1

        return rewards

    def max_utility(self, rewards, cumulative_rewards, current_rewards):

        maxAction = -1
        maxUtility = -2147483648
        nextReward = []


        for i in range(self.num_actions):
            a = 0
            for estreward in current_rewards[i]:

                if self.scalarize_reward(rewards + cumulative_rewards + estreward) - self.scalarize_reward(rewards + cumulative_rewards) > maxUtility:

                    maxAction = i
                    maxUtility = self.scalarize_reward(rewards + cumulative_rewards + estreward) - self.scalarize_reward(rewards + cumulative_rewards)
                    nextReward = rewards

                a += 1

        return maxAction, nextReward

    

        

    def reward_estimator_prob(self, estRewards, state, timestep, cumulative_rewards):

        reward_estimator = 0
        potential_action = 0
        potential_TimeAction = 0
        reward_prob = 0
        state_prob = 0
        rewardHolder = []
        action_nxtState = 0
        rewardIndex = -1
        nextTimestep = 0
        rewardVector  = []
        Rt1 = 0
        Rt1Prob = 0
        
        #print("Beginning of Est :: ", file = self.debug_file)
        rewards = []

        for i in range(self.num_actions):                    
            a = 0

            action = i
            rewardVector.append([])
            #rewards.append([])
            ##print("Dist time table ::", self.dist_time_table, file = self.debug_file)
            #for j in self.all_reward[i]:
            for j in self.dist_table[state][i]:

            #for j in self.dist_time_table[state][timestep-1][i]: 
                #print("estRewards :: " , estRewards, file = self.debug_file)
                rewardVector[i].append([])
                reward = j
                reward__ = self.all_reward[i][a]
                next_state = self.getNextState(state, i)

                potential_action = self.prob_table[state][i][a]

                if self.action_taken[state][i] == 0:
                    reward_prob = 0
                else:
                    reward_prob = potential_action / self.action_taken[state][i]

                if self.action_taken[state][i]  == 0:
                    state_prob = 0
                else:
                    state_prob = self.state_table[state][i][next_state] / self.action_taken[state][i]
                
                #reward_prob = potential_action / self.action_taken[state][i]  
                #state_prob = potential_action / self.state_table[state][i][next_state]

                rewards = self.reward_estimator(next_state, cumulative_rewards)
                Rt1, Rt1Prob = self.getRtplus1(cumulative_rewards) 

                #print("State ::" , state, file = self.debug_file)
                #print("Cumulative Rewards ::", cumulative_rewards, file = self.debug_file)
                #print("Cumulative Rewards Time = t ::", cumulative_rewards + j, file = self.debug_file)
                #print("Dist Table State ::", self.dist_table[state], file = self.debug_file)
                action_nxtState, reward = self.max_utility( j, cumulative_rewards, self.all_reward)
                nextTimestep = timestep + 1

                #print("Max Action::", action_nxtState, file = self.debug_file)

                if timestep == 200:
                    nextTimestep = 1

                #print("reward", reward, file = self.debug_file)
                #print("action next state", action_nxtState, file = self.debug_file)
                #print("Timestep", timestep, file = self.debug_file)
                #print("Next State", next_state, file = self.debug_file)
                #print("self.dist_time_table[next_state]",self.dist_time_table[next_state] , file = self.debug_file)

                rewardIndex = self.getRewardIndex(action, reward, self.dist_time_table[next_state][nextTimestep-1])

                #print("rewardex", reward, file = self.debug_file)
                #print("Dist time table", self.dist_time_table[next_state][nextTimestep-1], file = self.debug_file)
                #print("Dist time table2", self.dist_time_table[next_state][nextTimestep-1][action_nxtState], file = self.debug_file)
                #print("Check", self.prob_time_table[next_state][nextTimestep-1][action_nxtState][:], file = self.debug_file)
                #print("Check Length ", len(self.prob_time_table[next_state][nextTimestep-1][action_nxtState][:]), file = self.debug_file)
                #print("Reward :: ", reward__, file = self.debug_file)
                #print("Reward Index :: ", rewardIndex, file = self.debug_file)
                #print("Reward Table :: ", self.dist_time_table[next_state][nextTimestep-1], file = self.debug_file)
                #print("Reward Table :: ", self.dist_time_table[next_state][nextTimestep-1][action_nxtState], file = self.debug_file)
                #print("Prob Table :: ", self.prob_time_table[next_state][nextTimestep-1], file = self.debug_file)

                if self.prob_time_table[next_state][nextTimestep-1][action] == []:
                    potential_TimeAction = 0

                elif rewardIndex == -10:
                    potential_TimeAction = 0

                else:
                    potential_TimeAction = self.prob_time_table[next_state][nextTimestep-1][action][rewardIndex]
                    #print("Made It :: ", potential_TimeAction, file = self.debug_file)
                    #print("Made it 2 :: ", self.action_time_taken[action_nxtState][nextTimestep-1][action_nxtState], file = self.debug_file)

                if self.action_time_taken[next_state][nextTimestep-1][action] == 0:
                    rewardTime_prob = 0

                elif potential_TimeAction == 0:
                    rewardTime_prob = 0

                else:
                    rewardTime_prob = potential_TimeAction / self.action_time_taken[next_state][nextTimestep-1][action] 

                rewardVector[i][a] = reward_prob + Rt1Prob#(self.gamma * 1 * rewardTime_prob) * self.RtProb

                #print("RT+1 ", Rt1, file=self.debug_file)
                #print("RT+1 Prob ", Rt1Prob, file=self.debug_file)
                
                #print("Probability :: ", reward_prob + (self.gamma * 1 * rewardTime_prob), file = self.debug_file)
                #print("Reward Prob ::", reward_prob, file=self.debug_file)
                #print("state_prob :: ", state_prob, file = self.debug_file)
                #print("rewardTime_prob :: ", rewardTime_prob, file = self.debug_file)
                #print("rewardVector[i][a] :: ", rewardVector[i][a], file = self.debug_file)
                a += 1

        return rewardVector

    def updateDictionary(self, env_state, action, new_env_state, rewards, _dict_):

        if env_state in _dict_:
            if action in _dict_[env_state]:
                if new_env_state in _dict_[env_state][action]:
                    if str(rewards) in _dict_[env_state][action][new_env_state]:
                        pass
                    else:
                        _dict_[env_state][action][new_env_state].update({str(rewards): {'count' : 0}})
                else:
                    _dict_[env_state][action].update({new_env_state : {str(rewards): {'count' : 0}}, 'count' : 0})
            else:
                _dict_[env_state].update({action : {new_env_state : {str(rewards): {'count' : 0}, 'count' : 0}}})                
        else:
            _dict_.update({env_state : {action : {new_env_state : {str(rewards): {'count' : 0}, 'count' : 0}}}})


        if str(rewards) in _dict_[env_state][action][new_env_state]:
            pass
        else:
            _dict_[env_state][action][new_env_state][str(rewards)]['count'] = 0


        return _dict_



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

        action = -10
        #action_ = 0
        time_since_utility = 0

        
        #self.dict = lambda : defaultdict(self.dict)

        action_taken = np.zeros(shape=(self._num_rewards,))
        times_action_taken = np.zeros(shape=(self._num_rewards,))

        reward_wood = np.zeros(shape=(self._num_rewards,))
        reward_fish = np.zeros(shape=(self._num_rewards,))
        previous_utility = 0
        self.timestep = 0
        

        previous_reward = [0, 0]
        previous_action = 0
        new_env_state = -1

        

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
            self.timestep += 1
            #print(" *** timestep *** : " , timestep , file = self.debug_file)
            # Select an action or option based on the current state            
            

            state = self.encode_state(env_state, timestep, cumulative_rewards, cumulative_rewards, previous_culmulative_rewards)
            state = env_state
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

                best_prob = -2147483648 
                best_prob = 0                   
                actionTaken = 0
                rewardTaken = 0
                times_action_taken = 0
                reward_prob = 0
                next_state = 0
                rewards = []
                valueCheck = 0
                actionPicked = False


                #rewards_ = [[0 for x in range(self.num_actions)] for y in range(self.dist_table[state][])]

                rewards_ = self.reward_estimator(env_state, cumulative_rewards)
                #rewards_ = []
                #for j in self.dist_table[state][i]:
                #    rewards_.append(j)

                test = self.reward_estimator_prob(rewards_, env_state, timestep, cumulative_rewards)
                #print("State :: ", state, file = self.debug_file)
                #print("Rewards ::", rewards_, file = self.debug_file) 
                #print("Rewards ::", self.dist_table[state][:], file = self.debug_file) 

                #print("cumulative_rewards ::", cumulative_rewards, file = self.debug_file)
                #print("Rewards ::", rewards_, file = self.debug_file)
                #print("All Rewards ::", self.all_reward, file = self.debug_file)
                #print("Test", test, file = self.debug_file)
                
                #for state in range(2):
                #    print(" Begin action selection :: State", state, file = self.debug_file)
                    
                for action_ in range(self.num_actions): 
                    a = 0
                    for reward in self.dist_table[env_state][action_][:]:
                        rewards = rewards_[env_state][a]
                        rewards = self.all_reward[action_][a]
                        #reward = self.dist_table[state][action_][a]
                        #print("Reward :: ", self.dist_table[state][action_], file = self.debug_file)                  
                        #valueCheck = (self.scalarize_reward(cumulative_rewards + rewards_[state][action_])) * test[state][action_]

                        #valueCheck = (self.scalarize_reward(cumulative_rewards + rewards)) * test[action_][a]
                        #print("Reward :::" , rewards, file = self.debug_file)
                        #print("Test State ", test[action_], file = self.debug_file)
                        #print("Test State Action", test[action_][a], file = self.debug_file)
                        valueCheck = (self.scalarize_reward(cumulative_rewards + rewards) - self.scalarize_reward(cumulative_rewards)) * test[state][action_]
                        #valueCheck = test[state][action_]
                        #print("State", state, file = self.debug_file)
                        #print("action", action_, file = self.debug_file)
                        #print("reward", reward, file = self.debug_file)
                        #print("valueCheck", valueCheck, file = self.debug_file)
                        #print("cumulative_rewards", cumulative_rewards, file = self.debug_file)
                        #print("Probability :: " , test[state][action_], file = self.debug_file)
                        
                         
                        if valueCheck >= best_prob:
                            action = action_
                            best_prob = valueCheck
                            actionPicked = True


                        #elif actionPicked == False:
                        #    action = random.randint(0, 1) 

                        a += 1
                action_rewards = []
                action_prob = []
                for action_ in range(self.num_actions):

                    # env, action, state, id, cumulative_rewards, timestep)

                    action_rewards.append(self.rollOut(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0, self.debug_file, self.dict)[0])
                    action_prob.append(self.rollOut(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0, self.debug_file, self.dict)[1]);

                    #tree = Node(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0)

                #list1 = [0, 1]
                #string1 = str(list1)
                #dict = {string1 : 2}
                #print("Dictionary :: ", dict, file = self.debug_file) 
                print("Action Rewards :: ", action_rewards, file = self.debug_file)  
                print("Action Probabilities :: ", action_prob, file = self.debug_file) 
            
            self.action_taken[env_state][previous_action] += 1
            self.state_table[env_state][previous_action][self.getNextState(state, previous_action)] += 1
            
            self.action_time_taken[env_state][timestep - 1][previous_action] += 1


            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                new_env_state, rewards, self.timestep, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                new_env_state, rewards, self.timestep, done, __ = self._env.step(env_state, action, self.timestep)

            if self._render:
                self._env.render()           
            
            cumulative_rewards += rewards                 
            
            for all_states in range(2):
                if self.compare(rewards, self.dist_table[all_states][previous_action][:]) == 1:
                    self.dist_table[all_states][previous_action].append(rewards)
                    self.prob_table[all_states][previous_action].append(0)            

                #print("State :: ", state, file=self.debug_file)
                #print("Timestep :: ", timestep, file=self.debug_file)
                #print("Action :: ", action, file=self.debug_file)
                #print("Dist Time Table :: ", self.dist_time_table, file=self.debug_file)

                if self.compare(rewards, self.dist_time_table[all_states][timestep - 1][previous_action][:]) == 1:
                    self.dist_time_table[all_states][timestep - 1][previous_action].append(rewards)
                    self.prob_time_table[all_states][timestep - 1][previous_action].append(0)      

            if self.compare(rewards, self.all_reward[previous_action][:]) == 1:
                self.all_reward[previous_action].append(rewards)         

            
            #print("dist time table" , self.dist_time_table[state][timestep - 1], file = self.debug_file)
            #print("prob_time_table ::", self.prob_time_table, file = self.debug_file) 
            #print("rewards::", rewards, file = self.debug_file)
           

            reward_index = self.getRewardIndex(previous_action, rewards, self.dist_table[env_state])
            reward_time_index = self.getRewardIndex(previous_action, rewards, self.dist_time_table[env_state][timestep - 1])
            dict_ = {0:{0:{0:{str([0,1]):{'times_here' : 110}, 'times_here':100}}}}
            dict_[0][0][0]['times_here'] += 1
            prob = dict_[0][0][0]['times_here'] / dict_[0][0][0][str([0,1])]['times_here'] 

            #print("Dictionary Test ::", dict_ , file = self.debug_file) 
            
            self.prob_table[env_state][previous_action][reward_index] += 1    
            self.prob_time_table[env_state][timestep - 1][previous_action][reward_time_index] += 1           
            previous_action = action   
            #print("State - Reward Test ::", self.dict , file = self.debug_file)
            self.dict = self.updateDictionary(env_state, action, new_env_state, rewards, self.dict)
            """
            if env_state in self.dict:
                if action in self.dict[env_state]:
                    if new_env_state in self.dict[env_state][action]:
                        if str(rewards) in self.dict[env_state][action][new_env_state]:
                            pass
                        else:
                            self.dict[env_state][action][new_env_state].update({str(rewards): {'count' : 0}})
                    else:
                        self.dict[env_state][action].update({new_env_state : {str(rewards): {'count' : 0}}, 'count' : 0})
                else:
                    self.dict[env_state].update({action : {new_env_state : {str(rewards): {'count' : 0}, 'count' : 0}}})                
            else:
                self.dict.update({env_state : {action : {new_env_state : {str(rewards): {'count' : 0}, 'count' : 0}}}})

            if str(rewards) in self.dict[env_state][action][new_env_state]:
                pass
            else:
                self.dict[env_state][action][new_env_state][str(rewards)]['count'] = 0

            """


            

            self.dict[env_state][action][new_env_state]['count'] += 1
            self.dict[env_state][action][new_env_state][str(rewards)]['count'] += 1

            #print("State ::", env_state , file = self.debug_file) 
            #print("Action ::", action , file = self.debug_file) 
            #print("State - Reward Test ::", self.dict[env_state][action][new_env_state] , file = self.debug_file) 
            env_state = new_env_state 


        # Mark episode boundaries
        
        #print("Prob", self.prob_time_table[env_state][timestep][:], file = self.debug_file)
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
            learner.epsilon = learner.epsilon * 0.9
            if learner.epsilon < 0.001:
                learner.epsilon = 0.001

            scalarized_avg = learner.scalarize_reward(avg)

            print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
            #print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, file = f )
            #print(scalarized_avg, file = f)
            f.flush()

    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

if __name__ == '__main__':
    main()
