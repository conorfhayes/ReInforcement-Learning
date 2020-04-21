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
    kwargs={'nstates': 100, 'nobjectives': 4, 'nactions': 8, 'nsuccessor': 12, 'seed': 1, 'nrewards': 2}
)
register(
    id='FishWood-v0',
    entry_point='fishwood:FishWood',
    reward_threshold=0.0,
    kwargs={'fishproba': 0.1, 'woodproba': 0.9}
)
register(
    id='DeepSeaTreasure-v0',
    entry_point='deepseatreasure:DeepSeaTreasure',
    reward_threshold=0.0,
    kwargs={}
)
register(
    id='DeepSeaTreasureDanger-v0',
    entry_point='deepseatreasure_danger:DeepSeaTreasureDanger',
    reward_threshold=0.0,
    kwargs={}
)

class Tree:

    def __init__(self, state, env, row, col, file):
        self.state = state
        self.children = []
        self.env = env
        self.row = row
        self.col = col
        self.num_actions = 4
        self.rewards = []
        self.probabilities = []
        self.file = file
        self.parent = "Null"
        self.root = "Null"
        self.CR = [0,0]
        self.hasChanceNodes = False

        for action in range(self.num_actions): 
            state, reward, done, row, col, __ = self.env.step(self.state, action, self.row, self.col)
            node = Node("Null", state, action, self.env, row, col, reward, done, False)
            self.children.append(node)
            self.simulate(node)

    def step(self):
        start = 0

        def select(node):
            import random
            node.timesVisted += 1

            #print("Is Leaf Node ??", node.isleaf, file = self.file)

            if node.isleaf == True and node.done == False:
                self.expand(node)
                return
            
            elif len(node.children) > 0 and node.done == False:                    
                v = sys.maxsize
                for nodes in node.children:                    
                    if nodes.timesVisted < v:
                        v = nodes.timesVisted
                        _node_ = nodes                                
                select(_node_)

            if node.done == True:
                #print("Node is done and we went here", file = self.file)
                #self.backPropogation(node, node.reward, node.probability)
                return  
            

        while start < 4:
            chance = random.randint(0, 1)
            node = self.root

            if self.root == "Null":
                pick = random.randint(0, len(self.children) - 1)
                _node_ = self.children[pick]
                select(_node_)

            else:
                v = sys.maxsize
                for nodes in node.children:
                    
                    if nodes.timesVisted < v:
                        v = nodes.timesVisted
                        _node_ = nodes
                                
                select(_node_)

            start += 1

        return 


    def expand(self, node):
        node.isleaf = False

        for action in range(self.num_actions):

            state, reward, done, row, col, __ = self.env.step(node.state, action, node.row, node.col)

            child = node.createChild(node, state, action, row, col, reward, done, False)

        child = random.choice(node.children)
   
        self.simulate(child)

        return


    def simulate(self, node):

        cumulative_reward = [0,0]
        estProb = 0
        numActions = 4
        done = False
        time = 0
        _timer_ = 35
        state = node.state
        action = node.action
        row = node.row
        col = node.col

        if node.done == True:
           
            cumulative_reward += node.reward
            probability = node.probability
            self.backPropogation(node, cumulative_reward, probability)
            pass

        else:

            a = 0  

            while time < _timer_ and done == False:
                action = random.randint(0, 3)
                next_state, reward, done, next_row, next_col, __ = self.env.step(state, action, row, col)
                cumulative_reward += reward
                if a == 0:
                    prob = node.probability
                else:
                    prob = prob * node.probability

                #print("Prob :", prob, file = self.file)
                state = next_state
                row = next_row
                col = next_col
                #action = random.randint(0, self.num_actions - 1)
                #done = _done_

                time += 1
                a += 1
                                
            cumulative_reward += node.reward
            probability = prob * node.probability

            self.backPropogation(node, cumulative_reward, probability)


        return

    def backPropogation(self, node, cumulative_reward, probability):
        
        a = 0      
        #print("backPropogation probability ::", probability, file = self.file)

        if node.parent == "Null":

            node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)

        else:

            #node.rewards.append(cumulative_reward)
            #node.probabilities.append(probability)
            #node = node.parent

            while a == 0:
                cumulative_reward = cumulative_reward + node.reward
                probability = probability * node.probability

                node.rewards.append(cumulative_reward)
                node.probabilities.append(probability)
                
                if node.parent == "Null":
                    
                    cumulative_reward = cumulative_reward + node.reward
                    probability = probability * node.probability

                    node.rewards.append(cumulative_reward)  
                    node.probabilities.append(probability)

                    a = 1

                else:
                    node = node.parent
            
        return

    def run(self):

        if self.root == "Null":
            self.rewards = [[] for x in range(self.num_actions)]
            self.probabilities = [[] for x in range(self.num_actions)]

            for node in self.children:
                self.rewards[node.action].append(node.rewards)
                self.probabilities[node.action].append(node.probabilities)           


            return self, self.rewards, self.probabilities

        else:

            node = self.root
            childrenRewards = [[] for x in range(4)]
            childrenProbabilities = [[] for x in range(4)]          

            #if node.done == True:
            #    childrenRewards[node.action].append(node.reward)
            #    childrenProbabilities[node.action].append(node.probability)

            #else:
                #print("Number of Children :", len(node.children), file = self.file)
            for node in node.children:

                if node.done == True:
                    childrenRewards[node.action].append(node.reward)
                    childrenProbabilities[node.action].append(node.probability)


                childrenRewards[node.action].extend(node.rewards)
                childrenProbabilities[node.action].append(node.probabilities) 

            return self, childrenRewards, childrenProbabilities
        

    def reset(self):

        self.root = 'Null'
        self.CR = [0,0]

        return


    def takeAction(self, action):

        if self.root == "Null":
            #node = self.children[action]
            #print("Action attempt", action, file = self.file)
            next_state, reward, done, next_row, next_col, __ = self.env.step(self.state, action, self.row, self.col)

            flag = 0
            for node in self.children:
                if node.action == action:
                    if reward.all() == node.reward.all():
                        self.root = node
                        flag = 1

            if flag == 0:        
                #print("Here 1", file = self.file)                    
                node = self.root.createChild("Null", next_state, action, next_rol, next_col, reward, done, True)

            self.CR += [0, -1]

            if node.done == True:
                #self.backPropogation(node, reward, node.probability)
                self.reset()

        else:
            node = self.root
            if node.done == False:

                next_state, reward, done, next_row, next_col, __ = self.env.step(node.state, action, node.row, node.col)
                
                flag = 0
                #print("Action:", action, file = self.file)
                #print("Node Reward:", node.reward, file = self.file)
                for _node_ in node.children:
                    if _node_.action == action:
                        if reward.all() == _node_.reward.all():
                            self.root = _node_
                            flag = 1

                if flag == 0:     
                    #print("Here 2", file = self.file)                       
                    _node_ = node.createChild(node, next_state, action, next_rol, next_col, reward, done, True)
                    #self.backPropogation(node, reward, _node_.probability)
                    self.root = _node_

                if done == True:
                    #print("Terminal Reward ::", node.reward, file = self.file)
                    #self.backPropogation(node, reward, node.probability)
                    self.reset()

                

                #print("Action ::", action, file = self.file)

            else:

                self.reset()

        return reward




class Node:

    def __init__(self, node, state, action, env, row, col, reward, done, _type_):
        self.numActions = 4
        self.parent = node
        self.state = state
        self.numActions = 4
        self.action = action
        self.children = []
        self.reward = reward
        self.childrenNotCreated = [0,1,2,3]
        self.isleaf = True
        self.env = env
        self.timesVisted = 1
        self.isExpanded = False
        self.rewards = []
        self.probabilities = []
        self.childrenRewards = [[] for x in range(self.numActions)]
        self.childrenProbabilities = [[] for x in range(self.numActions)]
        self.row = row
        self.done = done
        self.hasChanceNodes = _type_
        self.col = col
        self.numSimulations = 1
        self.nodeData = {str(reward) : {"count" : 1, str(reward) : reward}}
        self.probability = self.getRewardProbability(self.reward)
        

    def getRewardProbability(self, reward):

        rewardCount = self.nodeData[str(reward)]["count"]
        _prob_ = rewardCount / self.numSimulations
        probability = _prob_ / self.numActions

        return probability

   
    def createChild(self, node, state, action, rol, col, reward, done, _type_):
        #node, node.state, action, node.row, node.col
        child = Node(node, state, action, self.env, rol, col, reward, done, _type_)

        self.children.append(child)

        if len(self.childrenNotCreated) > 0:
            self.childrenNotCreated.remove(action)
        return child



class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # self.dist_table = [100][8]
        s = 121
        a = 4
        self._env = gym.make(args.env)
        self.debug_file = open('debug', 'w') 
        self.tree = Tree(0, self._env, 0, 0, self.debug_file)
        self.num_actions = 4
        self.num_timesteps = 200
        #self.random_action = 0
        self._treedict_ = {}
        self.dict = {}     
        self.TS = {}      

           

        # Make environment 

        self.epsilon = 1.0

        
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

    def compare(self, a, b, thresholds):
        #print(a, file = self.debug_file)

        for i in range(len(thresholds)):

            thresholdA = min(a[i], thresholds[i])
            thresholdB = min(b[i], thresholds[i])

            if thresholdA > thresholdB:
                return 1

            elif thresholdA < thresholdB:
                return -1

        if a[len(thresholds) - 1] > b[len(thresholds) - 1]:
            return 1

        elif a[len(thresholds) - 1] < b[len(thresholds) - 1]:
            return -1

        for i in range(len(thresholds)):
            if a[i] > b[i]:
                return 1
            elif a[i] < b[i]:
                return -1

        return 0

    def selectTLOAction(self, actionValues, actionProb, thresholds):
        bestActions = []
        bestProbs = []

        bestActions.append(0)

        
        for a in range(len(actionValues)):
            #print(bestActions, file = self.debug_file)
            compareResult = self.compare(actionValues[a], actionValues[bestActions[0]], thresholds)

            if compareResult > 0:
                bestActions.clear()
                bestActions.append(a)
                bestProbs.clear()
                bestProbs.append(actionProb[a])

            elif compareResult == 0:
                bestActions.append(a)
                bestProbs.append(a)

        if len(bestActions) > 1:

            index = bestProbs.index(max(bestProbs))
            return bestActions[index]

        else:

            return bestActions[0]


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
    


    def run(self):
        

        """ Execute an option on the environment
        """
        self.row, self.col, env_state = self._env.reset()
        done = False
        self._num_rewards = 2
        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        rewards = np.zeros(shape=(self._num_rewards,))
        
        timestep = 0
        scalar_reward = 0

        action = -10        
        self.timestep = 0       
        new_env_state = -1

        _all_ = False
        action_selection = 0
        self.num_actions = 4
        #self.row = 0
        #self.col = 0
        self.done = False
        
        self.tree.root = "Null"
        while not self.done:
            
            state = env_state            
            check = random.uniform(0, 1)

            #print(self.epsilon, file = self.debug_file)
            if check < self.epsilon:
                # select random action
                action = random.randint(0, self.num_actions - 1)
            #    print("Random Action : " , action , file = self.debug_file)
            else:

                action_rewards = []
                action_prob = []

                

            self.tree.step()
            tree, testReward , testProbs = self.tree.run()

            action = random.randint(0, 3)

            #print(len(testReward), file = self.debug_file)
            #print(len(testProbs), file = self.debug_file)

            

            #self.TS.update({env_state : tree})
            
            # if self.row == 8 and self.col == 8:     
             
            print("Cumulative Reward : ", cumulative_rewards, file = self.debug_file)         
            print("State ", env_state, file = self.debug_file)
            print("Row :", self.row, " Col :", self.col, file = self.debug_file)
            print("Rewards Action 0:: ", testReward[0], file = self.debug_file)
            print("Rewards Prob 0:: ", testProbs[0][0], file = self.debug_file)

            if len(testProbs[0][0]) > 1:
                print("Rewards Prob 0:: ", sum(testProbs[0][0]), file = self.debug_file)
            else:
                print("Rewards Prob 0:: ", testProbs[0][0], file = self.debug_file)
            """
            print("Rewards Action 1:: ", testReward[1], file = self.debug_file)
            if len(testProbs[1][0]) > 1:
                print("Rewards Prob 1 : ", sum(testProbs[1][0]), file = self.debug_file)
            else:
                print("Rewards Prob 1 : ", testProbs[1][0], file = self.debug_file)
            
            """
            print("Rewards Action 2:: ", testReward[2], file = self.debug_file)
           
            print("Rewards Prob 2:: ", testProbs[2][0], file = self.debug_file)
            """

            print("Rewards Action 3:: ", testReward[3], file = self.debug_file)
            if len(testProbs[3][0]) > 1:
                print("Rewards Prob 3:: ", sum(testProbs[3][0]), file = self.debug_file)
            else:
                print("Rewards Prob 3:: ", testProbs[3][0], file = self.debug_file)
            """
                        
            

                
                #print("Rewards Action 1 :: ", testReward[1], file = self.debug_file)
                #print("Rewards Action 2 :: ", testReward[2], file = self.debug_file)
                #print("Rewards Action 3 :: ", testReward[3], file = self.debug_file)
                #print(" ", file = self.debug_file)
            
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                new_env_state, rewards, self.timestep, self.done, self.row, self.col, __ = self._env.step(actions)
            else:
                # Simple scalar action
                new_env_state, rewards, self.done, self.row, self.col, __ = self._env.step(env_state, action, self.row, self.col)

            
            rewards = self.tree.takeAction(action)

            

                #print("Is Row : ", self.row, file = self.debug_file)
                #print("Is Col : ", self.col, file = self.debug_file)

            #print("Done : ", done, file = self.debug_file)
            if self._render:
                self._env.render()

            #rewards[1] = rewards[1] / abs(rewards[2])
            #print(cumulative_rewards, file = self.debug_file)
            cumulative_rewards += rewards
            #print(rewards, file = self.debug_file)
            #print("Cumulative Rewards :", cumulative_rewards, file = self.debug_file)
            #print(" ", file = self.debug_file)
            #cumulative_rewards[1] = cumulative_rewards[1] / abs(cumulative_rewards[2])
            #tree = self.TS[env_state]
            
                #print("Tester 2 :: ", test2, file = self.debug_file)
            
            #self.dict = self.updateDictionary(env_state, action, new_env_state, rewards, self.dict)
            
            #self.dict[env_state][action][new_env_state]['count'] += 1
            #self.dict[env_state][action][new_env_state][str(rewards)]['count'] += 1

            #print("Action 1 Reward :", self.dict[1][1][1], file = self.debug_file)
            #print("Action 0 Reward :", self.dict[0][0][0], file = self.debug_file)
            #self.dict[env_state][action]['count'] += 1
            #self.dict[env_state][action][str(rewards)]['count'] += 1

            
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
    f = open('DeepSeaTreasure-Experiment-Output', 'w')
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
                    #avg = 0.99 * avg + 0.01 * rewards
                    avg = avg + rewards

                #print("Percentage Completed....", i%100, "% ", "Run : ", num_runs, " Episode : ", i,  file = f)
                scalarized_avg = learner.scalarize_reward(avg)

                if i < 0:
                    learner.epsilon = 1.0
                else:
                    learner.epsilon = learner.epsilon * 0.99
                    if learner.epsilon < 0.001:
                        learner.epsilon = 0.001



                if i % 100 == 0 and i >= 0:
                    r = (i/episodes) * 100
                    time = datetime.time(datetime.now())
                    time_elapsed = datetime.combine(date.today(), time) - datetime.combine(date.today(), start_time)

                    print("Percentage Completed...", r, "% ", "Run : ", run, "Time Elapsed : ", time_elapsed, "Average Reward : ", scalarized_avg, file = f)
                    f.flush()

                print("Cumulative reward:", rewards, file=f)
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
