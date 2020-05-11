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
from itertools import chain

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

    def __init__(self, state, args, file):
        self.state = state
        self.children = []
        self.args = args
        self.env = gym.make(self.args.env)
        self.numRewards = 2
        self.num_actions = 4
        self.childChanceRewards = {}
        self.rewards = []
        self.probabilities = []
        self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        self.file = file
        self.node = Node("Null", 0, -10, self.args, [0,0], False, 0, False)
        self.root = self.node
        self.CR = [0,0]
        self.health = 0
        self.state = 0
        self.cr = 0
        

    def step(self, cumulative_reward):
        import random
        self.num_expansions = 0

        def findExpansionSpot(node):  

            if node.done == True:
                return node

            if node.isleaf == True:                
                node = self.expand(node)
                return node            
            
            node = self.UCT(node)
            return findExpansionSpot(node)           

        def rollOut(node):
            self.simulate(node)        

        while self.num_expansions < 50:

            node = self.root    
            node = findExpansionSpot(node)
            rollOut(node)

            self.num_expansions += 1      

        return  


    def expand(self, node):

        action = random.choice(node.childrenRemaining)

        if self.numRewards == 3:

            state, reward, done, health,  __ = self.env.step(node.state, action, node.health)
            child = node.createChild(node, state, action, reward, done, health, False)
        else:
            state, reward, done, __ = self.env.step(node.state, action)
            child = node.createChild(node, state, action, reward, done,0, False)

        node.childChanceRewards[action].append(reward)
        node.childrenRemaining.remove(action)

        if len(node.children) == self.num_actions:
            node.isleaf = False
        else:
            node.isleaf = True

        return child

    def stepsToRoot(self,node):
        step = 0

        if node.done == True:
            node = node.parent

        while node != "Null":
            
            step += 1
            node = node.parent

        return step


    def UCT(self, node):

        c = 50

        
        x = -sys.maxsize
        uct = [0] * len(node.children)
        sumValue = 0
        a = 0

        for child in node.children:

            if child.timesVisited == 0:
                uct[a] = sys.maxsize
            else:
                uct[a] = (child.rollUCT / child.timesVisited +  c * np.sqrt (2 * np.log(node.timesVisited) / child.timesVisited ))

            a += 1
        
        bestActions = []

        for i in range(len(uct)):
            if uct[i] > x:
                bestActions.clear()
                bestActions.append(i)
                x = uct[i]
            elif x == uct[i]:
                bestActions.append(i)

        if len(bestActions) > 1:
            index = random.choice(bestActions)
        else:
            index = bestActions[0]

        #index = uct.index(max(uct))

        move = node.children[index]

        return move




    def simulate(self, node):

        cumulative_reward = [0] * self.numRewards
        estProb = 0
        numActions = 4
        done = False
        state = node.state
        prob = 1
        health = 0
        _done_ = False

        if node.done == True:
           
            cumulative_reward += node.reward
            probability = node.getProbability()

        else:
            a = 0  

            while _done_ == False:

                action = random.randint(0, 3)
                
                if self.numRewards == 3:
                    next_state, reward, done, health, __ = self.env.step(state, action, health)
                else:
                    next_state, reward, done, __ = self.env.step(state, action)

                cumulative_reward += reward

                if len(cumulative_reward) > 2:
                    if cumulative_reward[1] <= -100:
                        cumulative_reward[1] = -100
                        done == True

                if a == 0:
                    probability = node.getProbability()
                    a += 1
                else:
                    probability = probability * node.getProbability()

                #print("Prob :", prob, file = self.file)
                state = next_state
                _done_ = done

            if len(cumulative_reward) > 2:

                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

        steps = self.stepsToRoot(node)
        cumulative_reward += [0, -steps]

        self.backPropogation(node, cumulative_reward, probability)


        return

    def backPropogation(self, node, cumulative_reward, probability):
        
        a = 0     

        """

        if node.done == True:
            
            node.rewards.append(cumulative_reward)
            node.probabilities.append(node.getProbability())
            node.rollUCT += node.scalarize_reward(cumulative_reward) * 1

            node.timesVisited += 1
            node = node.parent 
        """
                

        while node != self.root:            
           
            #cumulative_reward = cumulative_reward + node.reward
            probability = probability * node.getProbability()
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * 1

            node.timesVisited += 1
            node = node.parent
        


        if node == self.root:

            #cumulative_reward = cumulative_reward + node.reward
            probability = probability * node.getProbability()
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * 1

            node.timesVisited += 1     
        
        
            
        return
    

    def run(self):

        node = self.root
        childrenRewards = [[] for x in range(4)]
        childrenProbabilities = [[] for x in range(4)]    
        expectedUtility = [None] * len(node.children)
        nodes = [None] * len(node.children)

        a = 0
        for child in node.children:            

            childrenRewards[child.action].append(child.rewards)
            childrenProbabilities[child.action].append(child.probabilities)             

            nodes[a] = child 
            expectedUtility[a] = child.rollUCT / child.timesVisited
            a += 1


        return self, childrenRewards, childrenProbabilities, expectedUtility, nodes
    

    def reset(self):

        self.root = self.node
        self.CR = [0,0]

        return

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


    def takeAction(self, action):
        
        node = self.root

        if self.numRewards == 3:
            next_state, reward, done, health, __ = self.env.step(node.state, action, node.health)
        else:
            next_state, reward, done, __ = self.env.step(node.state, action)        
        flag = 0
        
        a = 0       
        
        for _node_ in node.children:
            if _node_.action == action:
                _node_.timesActionTaken += 1                        
                if any((reward == q).all() for q in self.root.childChanceRewards[action]):
                    comparison = reward == _node_.reward
                    #print("Node State", node.state, file = self.file)
                    if comparison.all():
                        self.root = _node_
                        a += 1                      

                        #_node_.health = health
                        _node_.timeRewardReceived += 1
                        flag = 1

    
        if flag == 0:     
            #print("Here 2", file = self.file)                       
            _node_ = node.createChild(node, next_state, action, reward, done, reward, True)
            
            node.childChanceRewards[action].append(reward)
            self.root = _node_
            _node_.health = health
            _node_.timesActionTaken += 1
            _node_.timeRewardReceived += 1  

        return next_state, reward, node, done




class Node:

    def __init__(self, node, state, action, args, reward, done, health, _type_):
        self.numActions = 4
        self.health = health
        self.parent = node
        self.state = state
        self.numActions = 4
        self.action = action
        self.children = []
        self.reward = reward
        self.isleaf = True
        self.args = args
        self.env = gym.make(self.args.env)
        self.timesVisited = 0
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.chanceRewards.append(reward)
        self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        
        self.done = done
        self.hasChanceNodes = _type_
        self.numSimulations = 1
        self.timesActionTaken = 0
        self.timeRewardReceived = 0
        self.probability = self.getProbability()
        self._num_rewards = 2
        self.rollUCT = 0
        self.expectedUtility = 0
        self.childrenRemaining = [0,1,2,3]

        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

    def getProbability(self):
        if self.timesActionTaken > 0 and self.timeRewardReceived > 0:
            probability = self.timeRewardReceived / self.timesActionTaken
            probability = probability# / self.numActions
        else:
            _prob_ = 1
            probability = _prob_ #/ self.numActions

        return probability

   
    def createChild(self, node, state, action, reward, done, health, _type_):
        #node, node.state, action, node.row, node.col
        child = Node(node, state, action, self.args,reward, done, health, _type_)

        self.children.append(child)

        return child

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function#

            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(self._num_rewards)})



class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # self.dist_table = [100][8]
        s = 121
        a = 4
        self.args = args
        self._env = gym.make(args.env)
        self.debug_file = open('debug', 'w') 
        self.tree = Tree(0, self.args,self.debug_file)
        self.num_actions = 4
        self.num_timesteps = 200
        #self.random_action = 0
        self._treedict_ = {}
        self.dict = {}     
        self.TS = {}      
        self.args = args           

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

            #print(a[i], file = self.debug_file)

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

        #print("A ", actionValues, file = self.debug_file)
        for a in range(len(actionValues)):
            compareResult = self.compare(actionValues[a], actionValues[bestActions[0]], thresholds)

            if compareResult > 0:
                bestActions.clear()
                bestActions.append(a)
                #bestProbs.clear()
                #bestProbs.append(actionProb[a])

            elif compareResult == 0:
                bestActions.append(a)
                #bestProbs.append(a)

        if len(bestActions) > 1:

            #index = bestProbs.index(max(bestProbs))
            action = random.choice(bestActions)
            return action

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

    def selectAction(self, method, rewards, probabilities, cumulative_rewards):

        #print("Rewards :", rewards, file = self.debug_file)
        #print("Probabilities :", probabilities, file = self.debug_file)

        if method == "Expected Utility":
            sumof = [0,0,0,0]
            
            for i in range(len(rewards)):

                for j in range(len(rewards[i])):                    
                    sumof[i] += sum([self.scalarize_reward((rewards[i][j][r]) * probabilities[i][j][r]) for r in range(len(rewards[i][j]))])

            for i in range(len(sumof)):
                sumof[i] +=  self.scalarize_reward(cumulative_rewards)
                    
            action = sumof.index(max(sumof))
            #print("SumOf :", sumof, file = self.debug_file)

        if method == "TLO":
            thresholds = [124, -19]
            actions = [0] * 4
            for a in range(len(rewards)): 
                for i in range(len(rewards[a])):
                    actions[a] = rewards[a][i][self.selectTLOAction(rewards[a][i], probabilities, thresholds)]

            #print("Actions ::", actions, file = self.debug_file)

            action = self.selectTLOAction(actions, probabilities, thresholds)

        if method == "Policy":
            sumof = [0,0,0,0]
            for i in range(len(rewards)):
                val = -sys.maxsize
                for j in range(len(rewards[i])):
                    for k in range(len(rewards[i][j])):
                        if self.scalarize_reward(rewards[i][j][k]) > val:
                            val =  self.scalarize_reward(cumulative_rewards + rewards[i][j][k])
                sumof[i] = val

            action = sumof.index(max(sumof))

        return action

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


    def run(self):
        

        """ Execute an option on the environment
        """
        env_state = self._env.reset()
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
        health = 0
        
        #self.tree.root = "Null"
        self.tree.reset()
        a = 0
        check = random.random()
        #self.epsilon = 1

        while not self.done:
            
            state = env_state            

            action_rewards = []
            action_prob = []                

            self.tree.step(cumulative_rewards)
            tree, testReward , testProbs, expectedUtility, nodes = self.tree.run()

            x, y = self.getXYfromState(state)

            print("X :", x, "Y :", y, file = self.debug_file) 
            #print("Rewards Action:: ", testReward, file = self.debug_file)
            #print(" ", file = self.debug_file)
            #print("probabilities ::", testProbs, file = self.debug_file)
            print("Expected Utility on each Node :", expectedUtility, file = self.debug_file)
             
            
            if random.random() < 0.0:
                action = random.randint(0,3)
            else:
                index = expectedUtility.index(max(expectedUtility))
                node = nodes[index]
                action = node.action

            env_state, rewards, node, done = self.tree.takeAction(action)

            cumulative_rewards += rewards
            self.done = done


            if len(cumulative_rewards) > 2:

                if cumulative_rewards[1] <= -100 or health <= -100:
                    cumulative_rewards[1] = -100
                    self.done = True
                    tree.reset()

            a += 1
            #if a > 1000:
            #    self.done = True
            #    tree.reset()
        #print("Rewards Action:: ", testReward, file = self.debug_file)

        tree.reset()
        return cumulative_rewards

def main():
    # Parse parameters
    num_runs = 1
    episodes = 20000
    
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

                if i < 2:
                    learner.epsilon = 1.0
                else:
                    learner.epsilon = learner.epsilon * 0.9999
                    if learner.epsilon < 0.1:
                        learner.epsilon = 0.1



                if i % 1 == 0 and i >= 0:
                    r = (i/episodes) * 100
                    time = datetime.time(datetime.now())
                    time_elapsed = datetime.combine(date.today(), time) - datetime.combine(date.today(), start_time)

                    print("Episode", i, "Time Elapsed : ", time_elapsed, "Cumulative reward:", rewards, file = f)
                    f.flush()

                #print("Cumulative reward:", rewards, file=f)
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
