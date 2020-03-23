from __future__ import print_function
import os
import sys
import argparse

import math
import random
import gym
import numpy as np
from random import randint
import datetime
from collections import deque
from array import *
import collections

import pandas as pd
import time
from datetime import datetime, date


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

    def __init__(self, env, action, state, id, cumulative_rewards, timestep, layer, file, _dict_, probability, parent,_all_):

        self.simulations = 0
        self.num_actions = 2

        self.action = action
        self.probability = probability
        self.cumulative_rewards = cumulative_rewards
        self.state = state
        
        self.debug_file = file
        self.children = []
        self.parent = parent
        self.layer = layer + 1
        self.env = env
        self.unique_id = self.layer + 1
        self.num_actions = 2
        
        self.action = action
        self.times_visited = 0
        self.timestep = timestep
        
        self.cumulative_rewards = cumulative_rewards
        self.id = id
        self.done = False
        self.action = action
        self.probability = probability
        self._all_ = _all_
        
        
        #self.rewardprobs = {0: {str([0, 0]) : 0.45, str([1, 0]) : 0.05}, 1: {str([0, 0]) : 0.05, str([0, 1]) : 0.45}}
        #self.rewards = {0: {str([0, 0]) : [0, 0], str([1, 0]) : [1, 0]}, 1: {str([0, 0]) : [0, 0], str([0, 1]) : [0, 1]}}
        
        if self.parent == False:
            self._run_(self.state, _dict_, self.probability, self._all_)



    def create_children(self, state, action, cumulative_rewards, _dict_, probability, _all_):
        child_id = self.id + 1
       
        node = Node(self.env, action, state, child_id, cumulative_rewards, self.timestep + 1, self.layer, self.debug_file,
                 _dict_, probability, False, _all_)

        self.children.append(node)
        self.unique_id += 1

        return node
    


    def simultation(self, state, action, cumulative_rewards, timestep, _dict_):
        next_state, reward, timestep, self.done, __ = self.env.step(state, action, timestep)

        probability = (_dict_[state][action][next_state][str(reward)]['count'] / _dict_[state][action][next_state][
            'count']) / self.num_actions


        self.reward = reward

        return probability, next_state, reward + cumulative_rewards
    

        # Create Tree for each state, add to tree if needed and reuse! Hopefully this will save computation!

    def createTree(self, state, _dict_, probability, _all_):

        tree = self._run_(state, _dict_, probability, _all_)
        #print("Tree " , tree[0].children, file = self.debug_file)

        return tree

    def push(self,tree, cumulative_rewards, timestep, _dict_, probability, action):

        self.cumulative_rewards = cumulative_rewards
        self.timestep = timestep  
        self.probability = probability
        self.action = action
        rewards = []
        probabilities = []

        def _push(node, cumulative_rewards, timestep, _dict_, probability): 

            self.cumulative_rewards = cumulative_rewards
            self.probability = probability

            if node.parent == True:
                _probability_, next_state, _cumulative_rewards_ = node.simultation(self.action, self.action, self.cumulative_rewards,timestep,_dict_) 

                self.probability = _probability_
                cumulative_rewards = _cumulative_rewards_ 

                node.probability = self.probability
                node.cumulative_rewards = cumulative_rewards                        

            if len(node.children) == 0:              
                
                rewards.append(node.cumulative_rewards)
                probabilities.append(node.probability)
            
            else:  

                a = 0  
                for child in node.children:

                    _probability_, next_state, _cumulative_rewards_ = child.simultation(a, a, cumulative_rewards,timestep,_dict_) 
                    
                    child.probability = _probability_ * self.probability
                    child.cumulative_rewards = _cumulative_rewards_
                    
                    _push(child, child.cumulative_rewards, timestep +1, _dict_, child.probability)

                    a += 1

        _push(tree, self.cumulative_rewards, timestep, _dict_, self.probability)

        return rewards, probabilities

    def _run_(self, state, _dict_, probability, _all_):
        timestep = self.timestep
        
       
        if _all_ == True:

            if self.parent is True:                
                for reward in self.rewardprobs[self.action]: 

                    self.probability = self.rewardprobs[self.action].get(str(reward))
                    cumulative_rewards = self.cumulative_rewards + self.rewards[self.action].get(str(reward))                   

                    for action in range(self.num_actions):
                        for reward in self.rewardprobs[action]:
                            if self.layer <= self.simulations:
                                cumulative_rewards1 = cumulative_rewards + self.rewards[action].get(str(reward))
                                probability = self.rewardprobs[action].get(reward)

                                if self.probability == 0:
                                    self.probability = probability
                                else:
                                    probability = probability * self.probability

                            if self.layer <= self.simulations and self.timestep <= 200:
                                self.create_children(action, action, cumulative_rewards1, _dict_, probability, _all_)

            else:
                for action in range(self.num_actions):
                    for reward in self.rewardprobs[action]:
                        if self.layer <= self.simulations:
                            cumulative_rewards = self.cumulative_rewards + self.rewards[action].get(str(reward))
                            probability = self.rewardprobs[action].get(reward)

                            if self.probability == 0:
                                self.probability = probability
                            else:
                                self.probability = probability * self.probability

                        if self.layer <= self.simulations and self.done == False:
                            self.create_children(action, action, cumulative_rewards, _dict_, self.probability, _all_)

        # Beginning of generation of the tree search algorithm

        else:
            if self.parent == True:
                _probability_, next_state, cumulative_rewards = self.simultation(self.action, self.action,
                                                                                      self.cumulative_rewards,
                                                                                      timestep, _dict_)
                self.state = next_state
                self.probability = _probability_
                #print("Action", self.action, file = self.debug_file)
                #print("Cumulative Rewards ::", self.cumulative_rewards, file = self.debug_file)
                self.cumulative_rewards = cumulative_rewards

                #self.create_children(self.action, self.action, self.cumulative_rewards, _dict_, self.probability, self._all_)

            # timestep = self.timestep + 1
            if self.simulations > 0:

                for action in range(self.num_actions):
                    if self.layer <= self.simulations:
                        probability, next_state, cumulative_rewards = self.simultation(action, action,
                                                                                       self.cumulative_rewards, timestep,
                                                                                       _dict_)
                                             
                        self.probability = probability * self.probability


                    if self.layer <= self.simulations and self.done == False:
                        # self.cumulative_rewards =  self.simultation(self.action, self.cumulative_rewards, self.timestep)
                        # print("Cumulative Reward : ", self.cumulative_rewards, file = self.debug_file)
                        self.create_children(next_state, action, cumulative_rewards, _dict_, self.probability, self._all_)

        self.timestep = self.timestep + 1

        return self


class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # self.dist_table = [100][8]
        s = 2
        a = 2
        self.num_actions = 2
        self.num_timesteps = 200
        self.random_action = 0
        self._treedict_ = {}
        self.dict = {}           

        for i in range(s):
            self.dict.update({i : {}}) 
            self._treedict_.update({i : {}})             
            for j in range(a):
                self.dict[i].update({j : {}})                            
                for k in range(s):
                    self.dict[i][j].update({k : {'count' : 0}})

           

        # Make environment
        self.debug_file = open('debug', 'w')
        self.action_true = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.action_taken = [100][8]

        #print("Test Self Instantiate ::", self.dict, file= self.debug_file)   

        self.epsilon = 1.0

        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state
        

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]  # Ensure that the action space is a list for all the environments

        self._num_rewards = getattr(self._env, 'num_rewards', 1)
        self._num_actions = np.prod([a.n for a in aspace])
        self._aspace = aspace

        # Make an utility function
        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

        # Build network
        

        # print('Number of primitive actions:', self._num_actions)
        # print('Number of state variables', self._actual_state_vars)
        # print('Number of objectives', self._num_rewards)

        # Lists for policy gradient
        self._experiences = []
    

    def rollOut(self, env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_, parent, _all_):

        # self.rollOut(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0)

        _rewards_ = []
        _probs_ = []
        
        if action in self._treedict_[state]:
            _tree_ = self._treedict_[state].get(action)

        else:  

            tree = Node(env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_, 0, parent, _all_)
            #(state, _dict_, probability, _all_
            _tree_ = tree.createTree(state, _dict_, 0, _all_ )
            self._treedict_[state].update({action : _tree_})     
    
                    
        retNode = _tree_.push(_tree_, cumulative_rewards, timestep, _dict_, 0, action)

        _rewards_ = retNode[0] + _rewards_
        _probs_ = retNode[1] + _probs_      

        #print("Rewards :", _rewards_, file = self.debug_file)
        #print("Probs :", _probs_, file = self.debug_file) 

        return _rewards_, _probs_

    def encode_state(self, state, timestep, accrued, reward):
        """ Encode a raw state from Gym to a Numpy vector
        """
        index = 0
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            # print("State" , state)
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

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(self._num_rewards)})   
    

    def updateDictionary(self, env_state, action, new_env_state, rewards, _dict_):
        
        if str(rewards) not in _dict_[env_state][action][new_env_state]:
            _dict_[env_state][action][new_env_state].update({str(rewards): {'count': 0}})
        
       
        return _dict_

    def run(self):
        

        """ Execute an option on the environment
        """
        env_state = self._env.reset()
        done = False

        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        rewards = np.zeros(shape=(self._num_rewards,))
        
        timestep = 0
        scalar_reward = 0

        action = -10        
        self.timestep = 0       
        new_env_state = -1
        

        while not done:
            
            state = env_state            
            check = random.uniform(0, 1)


            if check < self.epsilon:
                # select random action
                action = random.randint(0, 1)
            #    print("Random Action : " , action , file = self.debug_file)
            else:

                
                _all_ = False
                action_selection = 0               
                action_rewards = []
                action_prob = []

                for _action_ in range(self.num_actions): 
                                   
                    a = self.rollOut(self._env, _action_, env_state, 0, cumulative_rewards, self.timestep, 0,
                                     self.debug_file, self.dict, True, _all_)                                     
                    action_rewards.append(a[0])                        
                    action_prob.append(a[1])                    

                chance = 0 
                prob_chance = 0                  

                for _action_ in range(2):
                    #print("Here::", file = self.debug_file)
                    #for reward in dictResults[_action_]:
                    a = 0
                    for reward in action_rewards[_action_]:
                    
                        scalarize_reward = self.scalarize_reward(cumulative_rewards)                        
                        potential_reward = self.scalarize_reward(reward)
                        utility = potential_reward - scalarize_reward  
                        prob = action_prob[_action_][a]      
                        utility_chance = utility * prob                 

                        """
                        #Debug File Output ::
                        print('Cumulative Rewards :: ', cumulative_rewards, file = self.debug_file)
                        print('Action Examined :: ', _action_, file = self.debug_file)
                        print('Scalarize Rewards', scalarize_reward, file=self.debug_file)
                        
                        print('Potential Rewards', potential_reward, file=self.debug_file)
                        print('Utility ', utility, file=self.debug_file)
                        print('Utility Chance ', utility_chance, file=self.debug_file)
                        print('Reward ', reward, file=self.debug_file)
                        print('Prob', prob, file=self.debug_file)
                        """
                        

                        

                        # Standard Action Selection Technique : 0
                        if action_selection == 0:
                            if utility_chance > chance:
                                #if prob > prob_chance:
                                chance = utility_chance
                                prob_chance = prob                                
                                action = _action_

                        if action_selection == 1:
                            if utility >= chance and prob > prob_chance:
                                #if prob > prob_chance:
                                chance = utility
                                prob_chance = prob
                                #print('Chance ', chance, file=self.debug_file)
                                #print('Prob Chance ', prob_chance, file=self.debug_file)
                                #print('Action ', action, file=self.debug_file)
                                action = _action_  

                        if action_selection == 2:
                            if utility_chance >= chance:
                                #if prob > prob_chance:
                                chance = utility_chance
                                prob_chance = prob                                
                                action = _action_

                        if action_selection == 3:
                            if utility == 0:
                                if prob > prob_chance:
                                    prob_chance = prob
                                    action = _action_
                            if utility > chance and prob > prob_chance:
                                chance = utility
                                prob_chance = prob 
                                action = _action_

                        if action_selection == 4:
                            val = potential_reward * prob
                            if val > chance:
                                chance = potential_reward * prob
                                action = _action_

                        a += 1



                if action == -10:
                    #print('Random Action', file=self.debug_file)
                    self.debug_file.flush()
                    action = random.randint(0, 1)



                #print('Cumulative Rewards', cumulative_rewards, file=self.debug_file)
                #print("Dictionary Action 0 :: ", dictResults[0], file=self.debug_file)
                #print("Dictionary Action 1 :: ", dictResults[1], file=self.debug_file)
                #print("Action Selected :: ", action, file=self.debug_file)
                #print("Action Reward :: ", action_rewards, file=self.debug_file)
                #print("Action Prob :: ", action_prob, file=self.debug_file)

                #self.graph(dictResults[0])
            
            #action = random.randint(0, 1)
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
            
            self.dict = self.updateDictionary(env_state, action, new_env_state, rewards, self.dict)
            

            self.dict[env_state][action][new_env_state]['count'] += 1
            self.dict[env_state][action][new_env_state][str(rewards)]['count'] += 1

            
            env_state = new_env_state

            # Mark episode boundaries

        # print("Prob", self.prob_time_table[env_state][timestep][:], file = self.debug_file)
        # previous_culmulative_rewards = cumulative_rewards
        return cumulative_rewards

def main():
    # Parse parameters
    num_runs = 1
    episodes = 10000
    
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
    import time
    
    # Instantiate learner
    #learner = Learner(args)
    df = pd.DataFrame()
    # Learn
    f = open('Experiment_Output', 'w')
    start_time = datetime.time(datetime.now())
    
    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)
    for run in range(num_runs): 
        runData = []    
        learner = Learner(args)
        try:
            #old_dt = datetime.datetime.now()
            avg = np.zeros(shape=(learner._num_rewards,))

            for i in range(episodes):
                rewards = learner.run()

                if i == 0:
                    avg = rewards
                else:
                    avg = 0.99 * avg + 0.01 * rewards

                #print("Percentage Completed....", i%100, "% ", "Run : ", num_runs, " Episode : ", i,  file = f)
                scalarized_avg = learner.scalarize_reward(avg)

                learner.epsilon = learner.epsilon * 0.8
                if learner.epsilon < 0.1:
                    learner.epsilon = 0.1



                if i % 100 == 0 and i >= 0:
                    r = (i/episodes) * 100
                    time = datetime.time(datetime.now())
                    time_elapsed = datetime.combine(date.today(), time) - datetime.combine(date.today(), start_time)

                    print("Percentage Completed...", r, "% ", "Run : ", run, "Time Elapsed : ", time_elapsed, "Average Reward : ", scalarized_avg, file = f)
                    f.flush()

                #print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
                #print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg)
                runData.append(scalarized_avg)
                #print("Run Data:", runData, file = f)
                f.flush()

            #data = pd.DataFrame({"Run " + str(run) : runData})
            #df = df.append(data)
            df['Run ' + str(run)] = runData
            #print("DataFrame:", df, file = f)
            #f.flush()


        except KeyboardInterrupt:
            pass

        if args.monitor:
            learner._env.monitor.close()
    t = datetime.now()
    #timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    df.to_csv(r'Experiments/Exp_MODistRL_TreeSearch_' + str(t) + '.csv')     
        #f.close()

if __name__ == '__main__':
    main()
