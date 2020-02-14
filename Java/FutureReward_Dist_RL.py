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
        self.num_timesteps = 200

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

        for s in range(2):
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
        self.gamma = 0.1

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

        #print("List A :: ", lista, file = self.debug_file)
        #print("List B :: ", listb, file = self.debug_file)
        for list_ in listb:

            if np.array_equal(np.array(lista), np.array(list_)):
                
                return 0

        return 1

    def getRewardIndex(self, action, rewards, rewardsMatrix):

        index = 0

        for i in rewardsMatrix[action]:
            #print("i", i, file = self.debug_file)
            #print("rewards matrix", rewardsMatrix[action], file= self.debug_file)
            if np.array_equal(np.array(i), np.array(rewards)):
                return index

            index += 1


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

        if state == 0 and action == 0:
            return 0

        if state == 0 and action == 1:
            return 1

        if state == 1 and action == 0:
            return 0

        if state == 1 and action == 1:
            return 1


    def reward_estimator(self, state):
        reward_estimator = 0
        potential_action = 0
        reward_prob = 0
        state_prob = 0
        rewards = []
        rewardHolder = []
        tester = 0

        for i in range(self.num_actions):                    
            a = 0
            rewards.append([])
            for j in self.dist_table[state][i]:    
                                      
                reward = j
                next_state = self.getNextState(state, i)
                                             
                potential_action = self.prob_table[state][i][a]
                reward_prob = potential_action / self.action_taken[state][i]
                state_prob = self.state_table[state][i][next_state] / self.action_taken[state][i]    
                #tester =    self.prob_table[next_state][i][a] /self.state_table[state][i][next_state]        

                reward_estimator = (reward * reward_prob) + (self.gamma * state_prob )
                rewards[i].append([])
                #print("Rewards ::", rewards, file =self.debug_file)
                rewards[i][a] = reward_estimator
                #rewardHolder.append(j)

                a +=1

        return rewards

    def max_utility(self, rewards, cumulative_rewards, current_reward):

        maxAction = -1
        maxUtility = -2147483648
        nextReward = 0


        for i in range(self.num_actions):
            a = 0
            for reward in rewards[i][:]:

                if self.scalarize_reward(reward + cumulative_rewards + current_reward) > maxUtility:

                    maxAction = i
                    maxUtility = self.scalarize_reward(reward + cumulative_rewards + current_reward)# - self.scalarize_reward(reward + cumulative_rewards)
                    nextReward = reward

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
        rewardIndex = 0
        nextTimestep = 0
        rewardVector  = []
        #rewardVector = []
        #for s in range(2):
        #    for a in range(2):
        #        rewardVector[s][a][:] = np.trim_zeros(rewardVector[s][a][:])

        rewards = []

        for i in range(self.num_actions):                    
            a = 0
            rewardVector.append([])
            #rewards.append([])
            #print("Dist time table ::", self.dist_time_table, file = self.debug_file)
            #for j in estRewards[i]:
            for j in self.dist_table[state][i]:
            #for j in self.dist_time_table[state][timestep-1][i]: 
                #print("estRewards :: " , estRewards, file = self.debug_file)
                rewardVector[i].append([])
                reward = j
                next_state = self.getNextState(state, i)

                potential_action = self.prob_table[state][i][a]
                reward_prob = potential_action / self.action_taken[state][i]  
                state_prob = potential_action / self.state_table[state][i][next_state]

                rewards = self.reward_estimator(next_state)

                action_nxtState, reward = self.max_utility(rewards, cumulative_rewards, j)
                nextTimestep = timestep + 1

                if timestep == 200:
                    nextTimestep = 1

                #print("reward", reward, file = self.debug_file)
                #print("action next state", action_nxtState, file = self.debug_file)
                #print("Timestep", timestep, file = self.debug_file)
                #print("Next State", next_state, file = self.debug_file)
                #print("self.dist_time_table[next_state]",self.dist_time_table[next_state] , file = self.debug_file)

                rewardIndex = self.getRewardIndex(action_nxtState, reward, self.dist_time_table[next_state][nextTimestep-1])

                #print("rewardIndex", rewardIndex, file = self.debug_file)
                #print("Dist time table", self.dist_time_table[next_state][nextTimestep-1], file = self.debug_file)
                #print("Dist time table2", self.dist_time_table[next_state][nextTimestep-1][action_nxtState], file = self.debug_file)
                #print("Prob", self.prob_time_table[next_state][nextTimestep-1][action_nxtState][:], file = self.debug_file)

                if self.prob_time_table[next_state][nextTimestep-1][action_nxtState] == []:
                    potential_TimeAction = 0
                else:
                    potential_TimeAction = self.prob_time_table[next_state][nextTimestep-1][action_nxtState][rewardIndex - 1]

                if self.action_time_taken[next_state][nextTimestep-1][action_nxtState] == 0:
                    rewardTime_prob = 0

                elif potential_TimeAction == 0:
                    rewardTime_prob = 0

                else:
                    rewardTime_prob = potential_TimeAction / self.action_time_taken[next_state][nextTimestep-1][action_nxtState] 

                rewardVector[i][a] = reward_prob + (self.gamma * state_prob * rewardTime_prob)
                a += 1
                #print("Probability :: ", reward_prob + (self.gamma * state_prob *rewardTime_prob), file = self.debug_file)
                #print("estRewards :: ", estRewards, file = self.debug_file)

        return rewardVector



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

        action_taken = np.zeros(shape=(self._num_rewards,))
        times_action_taken = np.zeros(shape=(self._num_rewards,))

        reward_wood = np.zeros(shape=(self._num_rewards,))
        reward_fish = np.zeros(shape=(self._num_rewards,))
        previous_utility = 0
        

        previous_reward = [0, 0]
        previous_action = 1
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
            timestep += 1
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

                rewards_ = self.reward_estimator(state)
                #rewards_ = []
                #for j in self.dist_table[state][i]:
                #    rewards_.append(j)

                test = self.reward_estimator_prob(rewards_, state, timestep, cumulative_rewards)
                #print("State :: ", state, file = self.debug_file)
                #print("Rewards ::", rewards_, file = self.debug_file) 
                print("Rewards ::", self.dist_table[state][:], file = self.debug_file) 
                print("Test", test, file = self.debug_file)
                
                for state in range(2):
                    print(" Begin action selection :: State", state, file = self.debug_file)
                    
                    for action_ in range(self.num_actions): 
                        a = 0
                        for reward in self.dist_table[state][action_][:]:
                            reward = rewards_[state][a]
                            reward = self.dist_table[state][action_][a]
                            #print("Reward :: ", self.dist_table[state][action_], file = self.debug_file)                  
                            #valueCheck = (self.scalarize_reward(cumulative_rewards + rewards_[state][action_])) * test[state][action_]
                            valueCheck = (self.scalarize_reward(cumulative_rewards + reward) - self.scalarize_reward(cumulative_rewards)) * test[state][action_]
                            print("State", state, file = self.debug_file)
                            print("action", action_, file = self.debug_file)
                            print("reward", reward, file = self.debug_file)
                            print("valueCheck", valueCheck, file = self.debug_file)
                            print("cumulative_rewards", cumulative_rewards, file = self.debug_file)
                            print("Probability :: " , test[state][action_], file = self.debug_file)
                            
                             
                            if valueCheck > best_prob:
                                action = state
                                best_prob = valueCheck
                                actionPicked = True

                            elif actionPicked == False:
                                action = random.randint(0, 1) 

                            a += 1


            
            print("Selected Action", action, file = self.debug_file)  
            
            self.action_taken[env_state][action] += 1
            self.state_table[env_state][action][self.getNextState(state, action)] += 1
            
            self.action_time_taken[env_state][timestep - 1][action] += 1


            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                new_env_state, rewards, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                new_env_state, rewards, done, __ = self._env.step(action)

            if self._render:
                self._env.render()           
            
            cumulative_rewards += rewards                 
            
            
            if self.compare(rewards, self.dist_table[env_state][action][:]) == 1:
                self.dist_table[env_state][action].append(rewards)
                self.prob_table[env_state][action].append(0)            

            #print("State :: ", state, file=self.debug_file)
            #print("Timestep :: ", timestep, file=self.debug_file)
            #print("Action :: ", action, file=self.debug_file)
            #print("Dist Time Table :: ", self.dist_time_table, file=self.debug_file)

            if self.compare(rewards, self.dist_time_table[env_state][timestep - 1][action][:]) == 1:
                self.dist_time_table[env_state][timestep - 1][action].append(rewards)
                self.prob_time_table[env_state][timestep - 1][action].append(0)             

            
            #print("dist time table" , self.dist_time_table[state][timestep - 1], file = self.debug_file)
            #print("prob_time_table ::", self.prob_time_table, file = self.debug_file) 
            #print("rewards::", rewards, file = self.debug_file)
           

            reward_index = self.getRewardIndex(action, rewards, self.dist_table[env_state])
            reward_time_index = self.getRewardIndex(action, rewards, self.dist_time_table[env_state][timestep - 1])

            #print("reward_time_index ::", reward_time_index, file = self.debug_file) 
            
            self.prob_table[env_state][action][reward_index] += 1    
            self.prob_time_table[env_state][timestep - 1][action][reward_time_index - 1] += 1           
            previous_action = action      
            env_state = new_env_state    

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
