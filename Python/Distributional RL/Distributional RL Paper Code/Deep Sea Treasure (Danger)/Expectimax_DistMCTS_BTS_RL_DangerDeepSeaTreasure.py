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
import matplotlib.pyplot as plt



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
register(
    id='SpaceTraders-v0',
    entry_point='spacetraders:SpaceTraders',
    reward_threshold=0.0,
    kwargs={}
)


class Tree:

    def __init__(self, state, args, file):
        self.state = state
        self.count = 0
        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None
        self.epsilon = 1
        self.children = []
        self.args = args
        self.env = gym.make(self.args.env)
        self.numRewards = 3
        self.num_actions = 4
        self.childChanceRewards = {}
        self.rewards = []
        self.probabilities = []
        self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        self.file = file
        self.node = decisionNode("Null", 0, 0, self.args, [0,0,0], False, 0, False)
        self.root = self.node
        self.CR = [0,0]
        self.health = 0
        self.state = 0
        self.cr = 0
        self.distTable = {}
        self.artificialSamples = []

        for i in range(2000):
            choice = [0, 10]
            r1 = random.randint(0, 1000)
            r2 = random.randint(0, 10)
            r3 = random.randint(1, 1000)
            self.artificialSamples.append([r1, 0, -r3])

        for x in range(200):
            self.distTable.update({x : {'count' : 0}})
            for y in range(4):
                self.distTable[x].update({y : {'count' : 0}}) 
                for z in range(200):
                    self.distTable[x][y].update({z : {'count' : 0}}) 


    def step(self, cumulative_reward):
        import random
        self.num_expansions = 0

        def findExpansionSpot(node, cumulative_reward):  
            if node.parent == "Null":
                pass
            
            elif node.type == "chance":
                node, cumulative_reward = self.runSimulate(node, cumulative_reward)


            if node.type == "decision" and node.done == True:
                return node

            if node.isleaf == True and node.type == "decision":                
                node, cumulative_reward = self.expand(node, cumulative_reward)
                #for i in range(10000):
                    #choice = [0, 10]
                    #r1 = random.randint(0, 1000)
                    #r2 = random.randint(0, 10)
                    #r3 = random.randint(1, 1000)
                node.rewards = self.artificialSamples
                #print(node.rewards, file = self.file)
                
                return node            
            
            
            #node = self.UCT(node)
            node = self.thompsonSampling(node, cumulative_reward)
            #node = random.choice(node.children)
            return findExpansionSpot(node, cumulative_reward)           

        def rollOut(node):
            self.simulate(node)        

        while self.num_expansions < 50:

            node = self.root 
            node = findExpansionSpot(node, cumulative_reward)
            rollOut(node)
            self.count += 1

            self.num_expansions += 1      

        return 

    def TESTscalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        #print("Scalarise R", rewards, file = self.debug_file)
        if rewards[0] >= 0.88:
            return rewards[1]
        else:
            return -1000000000

    def TESTscalarize_reward1(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        #print("Scalarise R", rewards, file = self.debug_file)
        if rewards[0] >= 1:
            return rewards[1]
        else:
            return -1000

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
        """
        return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(len(rewards))})

    def runSimulate(self, node, cumulative_reward):
        flag = 0

        if self.numRewards == 3:
            state, reward, done, __ = self.env.step(node.parent.state, node.action, cumulative_reward[1])
        else:
            state, reward, done, __ = self.env.step(node.state, node.action)

        if str(reward) in node.distributionTable:
            flag = 1
            for _node_ in node.children:
                comparison = reward == _node_.reward
                if comparison.all():
                    n = _node_

        self.updateDistributionTable(node.parent.state, node.action, reward, state, node)
        
       


    
        if flag == 0:                      
            _node_ = node.createChild(node, state, node.action, reward, done, reward, True)            
            #node.parent.childChanceRewards[node.action].append(reward)
            _node_.timeRewardReceived += 1
            self.updateDistributionTable(node.parent.state, node.action, reward, state, _node_)
            n = _node_

        #if random.random() < 0.1:
            #self.simulate(np.random.choice(node.children))



        return n, cumulative_reward + reward

    def integrals(self, utility, probs, limits):

        increment = 1
        #print("Utility", utility, file = self.file)
        #print("Probabilites", probs, file = self.file)

        area = [[] for x in range(3)]

        for i in range(len(utility)):
            l1 = limits[0]
            l2 = limits[1]

            #area = []
            _area_ = 0
        
            a = 0
                
            while l1 < limits[1]:
                if a < len(utility[i]) - 1 :
                    if a == 0 and l1 < utility[i][a]:
                        p = 0
                    if l1 >= utility[i][a] and l1 < utility[i][a + 1]:
                        p = probs[i][a]
                    else:
                        a += 1
                        p = probs[i][a]

                elif a == 0 and l2 < utility[i][a]:
                    p = 0
                elif a == 0 and l2 > utility[i][a]:
                    p = probs[i][a]

                calc = (l1 * p) - (l2 * p)
                #print("Calc", p, file = self.file)
                _area_ += calc

                #print("Area Val", _area_, file = self.file)
                area[i].append(abs(round(_area_, 2)))

                l1 += increment
                l2 += increment

        dominantDist = -10
        count = []
        #print("Area", len(area), file = self.file)
        for x in range(len(area)):
            val = 0
            #print("X", x, file = self.file)
            for y in range(len(area)):
                #print("Y", y, file = self.file)
                

                dist = np.array(area[x]) - np.array(area[y])
                #print("X :", x, "-", "Y :", y, "=", dist, file = self.file)
                if min(dist) >= 0:
                    #dominantDist = x
                    val += 1
            
            count.append(val)

        if dominantDist == -10:
            dominantDist = count.index(max(count))
       

        return dominantDist


    def stochasticDominance(self, rewards, probabilities):


        maxUtility = 0
        minUtility = -1000

        #print("Rewards", rewards, file = self.file)
        #print("Probs", probabilities, file = self.file)

        compare = len(rewards)
        limits = [minUtility, maxUtility]

        dist_sorted = []
        prob_sorted = []
        cumulative_probability = []

        

        for i in range(len(rewards)):
            if len(rewards[i]) > 1:
                
                _dist_sorted_, _prob_sorted_ = zip(*sorted(zip(rewards[i], probabilities[i])))
                
                dist_sorted.append(_dist_sorted_)
                prob_sorted.append(_prob_sorted_)
            
            else:
                dist_sorted.append(rewards[i])
                prob_sorted.append(probabilities[i])


            #print("Probabilites", prob_sorted, file = self.file)

            cp = 0
            p = []
            for j in range(len(dist_sorted[i])):
                #print("Probabilites hello", prob_sorted[i][j], file = self.file)
                cp = cp + prob_sorted[i][j]
                p.append(cp)
                #print("Probabilites hello", prob_sorted[i][j], file = self.file)
            cumulative_probability.append(p)

        index = self.integrals(dist_sorted, cumulative_probability, limits)



        return index

    def thompsonSampling(self, node, cumulative_reward):
        
        
        randomNum = random.random()
        bestNode = -10
        #action = 0
        minVal = -sys.maxsize
        
        
        for child in node.children:

            
            sample = np.random.choice(len(child.samples_mean))
            #randomSample = self.scalarize_reward(np.random.choice(child.samples_mean[sample]) + cumulative_reward)
            randomSample = self.scalarize_reward(child.samples_mean[sample])# + self.scalarize_reward(cumulative_reward)
            
            if randomSample > minVal:
                minVal = randomSample
                bestNode = child  

       
        if randomNum < 0.05:
            #self.simulate(np.random.choice())
            bestNode = np.random.choice(node.children)

        self.epsilon = self.epsilon * 0.999
       

        return bestNode


    def expand(self, node, cumulative_reward):

        action = random.choice(node.childrenRemaining)
        #self, node, state, action, args, _type_
        #chance = chanceNode(node, node.state, node.action, self.args, 0)
        #print("Expanding", file = self.file)
        chance = node.createChild(node, node.state, action, [0,0,0], False, 0, False)
        #print("Check type", chance.type, file = self.file)

        if self.numRewards == 3:
            #print("CR", cumulative_reward, file = self.file)
            state, reward, done,  __ = self.env.step(node.state, action, cumulative_reward[1])
            #print("Reward", reward, file = self.file)
            child = chance.createChild(chance, state, action, reward, done, 0, False)
        else:
            #print("Node State ", node.state, file = self.file)
            #print("Node Action", action, file = self.file)
            #print("CR", cumulative_reward, file = self.file)
            state, reward, done, __ = self.env.step(node.state, action)
            child = chance.createChild(chance, state, action, reward, done, 0, False)

        #chance.reward = reward
        node.childChanceRewards[action].append(reward)
        node.childrenRemaining.remove(action)

     


        self.updateDistributionTable(node.state, action, reward, state, chance)
        self.updateDistributionTable(node.state, action, reward, state, child)
      
        

        if len(node.children) == self.num_actions:
            node.isleaf = False
        else:
            node.isleaf = True

        return child, cumulative_reward

    def stepsToRoot(self,node):
        step = np.array([0,0])


        if node.done == True:
            node = node.parent

        while node != "Null":
            
            step += node.reward
            node = node.parent

        #print("Step", step, file = self.file)

        return step

    def updateDistributionTable(self, state, action, reward, next_state, node):

        if str(reward) not in node.distributionTable:
            self.distTable[state][action][next_state].update({str(reward) : {'count' : 0}})
            node.distributionTable.update({str(reward) : {'count' : 0}})        

        node.distributionTable[str(reward)]['count'] += 1
        node.distributionTable['count'] += 1

        

        return


    def UCT(self, node):
        #print("Node UCT Type", node.type, file = self.file)

        c = 30
        
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
        #print("Best Actions", bestActions, file = self.file)
        if len(bestActions) > 1:
            index = random.choice(bestActions)
        else:
            index = bestActions[0]

        #index = uct.index(max(uct))

        move = node.children[index]

        return move


    def simulate(self, node):

        cumulative_reward = np.array([0,0,0])
        estProb = 0
        numActions = 4
        done = False
        state = node.state
        prob = 1
        health = 0
        _done_ = False
        #print(node.reward, file = self.file)
        #prob = 1


        if node.done == True:
           
            cumulative_reward = node.reward + cumulative_reward
            #probability = (self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']) #* (1/4)



        else:
            a = 0  

            _node_ = node

            while _done_ == False and a < 50:

                action = random.randint(0, 3)
                
                if self.numRewards == 3:
                    next_state, reward, done, __ = self.env.step(state, action, cumulative_reward[1])
                else:
                    next_state, reward, done, __ = self.env.step(state, action)


                #self.updateDistributionTable(state, action, reward, next_state)

                cumulative_reward = reward + cumulative_reward

                if len(cumulative_reward) > 2:
                    if cumulative_reward[1] <= -100:
                        cumulative_reward[1] = -100
                        done == True

                #prob = (self.distTable[state][action][next_state][str(reward)]['count'] / self.distTable[state][action]['count']) #* (1/4)
                """"
                if a == 0:
                    probability = prob
                    a += 1
                else:
                    probability = probability * prob 
                """
                state = next_state
                _done_ = done

                a += 1

            
        self.distributionBackPropogation(node, cumulative_reward)
        #self.backPropogation(node, cumulative_reward)


        return

    

    def distributionBackPropogation(self, node, cumulative_reward):
        
        a = 0
        n = 10
        probability = 1
        #print(self.distTable[node.parent.state][node.action][node.state], file = self.file)
        #node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
        node.probability = node.distributionTable[str(node.reward)]['count'] / node.distributionTable['count']
        #print("CR Check", cumulative_reward, file = self.file)
        distRewards = [cumulative_reward]
        #print("Dist Rewards", distRewards, file = self.file)
        distProbability = [node.probability]
        startNode = node

        if node.done == True:

            node.timesVisited += 1
            node.parent.timesActionTaken[node.action] += 1

            strCumulativeReward = str(cumulative_reward)

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) #* probability
            #node.rewards.append(self.scalarize_reward(cumulative_reward))
            #node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
            node.probability = node.distributionTable[str(node.reward)]['count'] / node.distributionTable['count']
            probability = node.probability
            distRewards = [cumulative_reward]

            distProbability = [1]

            node.data[str(cumulative_reward)] = {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability }
            if len(node.distribution.keys()) == 0:
                node.distribution.update({'reward' : distRewards})
                node.distribution.update({'probability' : distProbability})
            else:
                sum1 = 0
                sum2 = 0
                for val in range(len(node.distribution['reward'])):
                    #print(reward, file = self.file)
                    sum1 += self.scalarize_reward(node.distribution['reward'][val]) * node.distribution['probability'][val]
                    
                for val in range(len(distRewards)):
                    sum2 += self.scalarize_reward(distRewards[val]) * distProbability[val]

                if sum2 > sum1:
                    node.distribution.update({'reward' : distRewards})
                    node.distribution.update({'probability' : distProbability})

                    distRewards = node.distribution['reward']
                    distProbability = node.distribution['probability']
                #else:
                    #distRewards = node.distribution['reward']
                    #distProbability = node.distribution['probability']

            #distRewards = node.distribution['reward']
            #distProbability = node.distribution['probability']
            #print("Dist Checker", distRewards[0], file = self.file)

            #_rewards_ = node.rewards
            holder = [i for i in range(len(distRewards))]
            #print(self.distTable[node.parent.state][node.action][node.state], file = self.file)
            #print(distProbability, file = self.file)
            _rewards_ = np.random.choice(holder, 10, p=distProbability)
            _rewards_ = [distRewards[i] for i in _rewards_]
            node.rewards.append(_rewards_)
            #print("Node rewards",node.rewards, file = self.file)
            #print(distRewards, file = self.file)
            #print(distProbability, file = self.file)
            sample_indices = np.random.choice(len(node.rewards), n )
            sample = [node.rewards[j] for j in sample_indices]
            #print(sample, file = self.file)

            
            _sample_ = [self.scalarize_reward(j) for j in sample]
            #_sample_ = [j for j in sample]
            meanSample = sum(_sample_) / n
            node.samples_mean.append(meanSample)
            oldNode = node
            node = node.parent
            

        while node != self.root.parent:

            cumulative_reward = cumulative_reward + node.reward            
            distRewards = [node.reward + i for i in distRewards]

            if node.parent == "Null":
                pass
            else:

                node.parent.timesActionTaken[node.action] += 1

                if node.type == "chance":

                    xx = 0
                    array = []
                    childProbDistribution = []
                    for child in node.children:
                        
                        if child == oldNode:
                            prob = node.distributionTable[str(child.reward)]['count'] / node.distributionTable['count']
                            distProbability = [x * prob for x in distProbability]
                            

                        else:
                            childRewardDistribution = child.distribution['reward']# + node.reward
                            #print("b4 Test 3", distRewards, file = self.file)
                            for i in childRewardDistribution:
                                distRewards.append(i)
                            #distRewards = distRewards.append(np.array(childRewardDistribution))
                            #print("after Test 3", distRewards, file = self.file)
                            childProbDistribution = child.distribution['probability']
                            prob = node.distributionTable[str(child.reward)]['count'] / node.distributionTable['count']
                            childProbDistribution = [x * prob for x in childProbDistribution]
                
                    if len(childProbDistribution) > 0:
                        #distProbability = np.append(distProbability, childProbDistribution)
                        for i in childProbDistribution:
                            distProbability.append(i)
                        
                    #print("End", file = self.file)

                    #print("Len???", distRewards, file = self.file)

                    if len(node.distribution.keys()) == 0:
                        #print(bool(node.distribution), file = self.file)
                        node.distribution.update({'reward' : distRewards})
                        node.distribution.update({'probability' : distProbability})
                    
                    elif len(node.distribution['reward']) != len(distRewards) and node == startNode.parent:
                        node.distribution.update({'reward' : distRewards})
                        node.distribution.update({'probability' : distProbability})

                        distRewards = distRewards
                        distProbability = distProbability

                    else:
                        sum1 = 0
                        sum2 = 0
                        for val in range(len(node.distribution['reward'])):
                            #print(reward, file = self.file)
                            sum1 += self.scalarize_reward(node.distribution['reward'][val]) * node.distribution['probability'][val]
                            
                        for val in range(len(distRewards)):
                            sum2 += self.scalarize_reward(distRewards[val]) * distProbability[val]

                        if sum2 > sum1:
                            node.distribution.update({'reward' : distRewards})
                            node.distribution.update({'probability' : distProbability})

                            distRewards = node.distribution['reward']
                            distProbability = node.distribution['probability']

                        #else:
                            #distRewards = node.distribution['reward']
                            #distProbability = node.distribution['probability']

                    

                    #print("Check Rewards", distRewards, file = self.file)
                    #print("Check Probabilities", distProbability, file = self.file)
                    #node.distribution.update({'reward' : distRewards})
                    #print("After", node.distribution, file = self.file)
                    #node.distribution.update({'probability' : distProbability})


                else:                    
                    #prob = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count'] #* (1 / node.numActions)
                    prob = 1
                    #node.probability = node.distributionTable[str(node.reward)]['count'] / node.distributionTable['count']
                    #print(self.distTable[node.parent.state][node.action][node.state], file = self.file)
                    distRewards = distRewards# + node.reward
                    distProbability = distProbability


                    if len(node.distribution.keys()) == 0:
                        #print(bool(node.distribution), file = self.file)
                        node.distribution.update({'reward' : distRewards})
                        node.distribution.update({'probability' : distProbability})
                    else:
                        sum1 = 0
                        sum2 = 0
                        for val in range(len(node.distribution['reward'])):
                            #print(reward, file = self.file)
                            sum1 += self.scalarize_reward(node.distribution['reward'][val]) * node.distribution['probability'][val]
                            
                        for val in range(len(distRewards)):
                            sum2 += self.scalarize_reward(distRewards[val]) * distProbability[val]

                        if sum2 > sum1:
                            node.distribution.update({'reward' : distRewards})
                            node.distribution.update({'probability' : distProbability})

                            distRewards = node.distribution['reward']
                            distProbability = node.distribution['probability']
                        #else:
                            #distRewards = node.distribution['reward']
                            #distProbability = node.distribution['probability']

                    #print("distProbability", distProbability, file = self.file)

                #probability = prob * probability #*  (1 / node.numActions)
                #probability = probability / 3
                node.parent.timesActionTaken[node.action] += 1
                #print(distRewards, file = self.file)
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.timesVisited += 1           

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) #* probability

            node.data[str(cumulative_reward)] = {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability }

            
            holder = [i for i in range(len(distRewards))]
            _rewards_ = np.random.choice(holder, 10, p=distProbability)
            _rewards_ = [distRewards[i] for i in _rewards_]
            node.rewards.append(_rewards_)
            sample_indices = np.random.choice(len(node.rewards), n )
            sample = [node.rewards[j] for j in sample_indices]

            
            _sample_ = [self.scalarize_reward(j) for j in sample]
            #_sample_ = [j for j in sample]
            meanSample = sum(_sample_) / n
            node.samples_mean.append(meanSample)
            oldNode = node
            #cumulative_reward = cumulative_reward + node.reward
            #distRewards = [node.reward + i for i in distRewards]
            node = node.parent

        
            
        return
    

    def backPropogation(self, node, cumulative_reward):
        
        a = 0
        n = 10
        #print("Node Data", node._data_, file = self.file)
        probability = 1
        if node.done == True:
            node.timesVisited += 1
            node.parent.timesActionTaken[node.action] += 1
            #cumulative_reward = cumulative_reward
            #node.rewards.append(cumulative_reward)
            strCumulativeReward = str(cumulative_reward)
            #node.samples_mean.append(cumulative_reward)

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) #* probability
            node.rewards.append(self.scalarize_reward(cumulative_reward))
            node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
            probability = node.probability

            node.data[str(cumulative_reward)] = {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability }
            """
            if str(cumulative_reward) not in node._data_:

                #if node.parent.parent == self.root:
                node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability} })
                #else:
                    #* (1 / node.numActions)
                    #node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability } })
            else:

                node._data_[str(cumulative_reward)]['probability'] = probability
                
                if node.parent.parent == self.root:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability
                else:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability #* (1 / node.numActions)
                
                """

            #n = 200
            _rewards_ = node.rewards
            #print("Node rewards",node.rewards, file = self.file)
            sample_indices = np.random.choice(len(_rewards_), n )
            sample = [_rewards_[j] for j in sample_indices]

            
            #_sample_ = [self.scalarize_reward(j) for j in sample]
            _sample_ = [j for j in sample]
            meanSample = sum(_sample_) / n
            node.samples_mean.append(meanSample)
            node = node.parent
        

        while node != self.root.parent:

            cumulative_reward = cumulative_reward + node.reward
            #print("Cumulative Reward", cumulative_reward, file = self.file)            

            if node.parent == "Null":
                pass
            else:

                node.parent.timesActionTaken[node.action] += 1
                if node.type == "chance":
                    prob = 1
                else:                    
                    prob = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count'] #* (1 / node.numActions)
                    #node.setProbability(prob)
                    #print("Prob Update", node.probability, file = self.file)

                probability = prob * probability #*  (1 / node.numActions)
                #probability = probability / 3
                node.parent.timesActionTaken[node.action] += 1
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.timesVisited += 1           

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) #* probability

            node.data[str(cumulative_reward)] = {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability }

            """
            if str(cumulative_reward) not in node._data_:
                
                    # (1 / node.numActions)
                node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability } })
            
                #node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
            else:
                node._data_[str(cumulative_reward)]['probability'] = probability

            
                if node.parent == "Null":
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability #* (1 / node.numActions)
                elif node.parent.parent == self.root:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability
                else:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability #* (1 / node.numActions)
        
            
            #print("Node Data", node._data_, file = self.file)
            #print(" " , file = self.file)
        """ 
            node.rewards.append(self.scalarize_reward(cumulative_reward))
            #n = 200
            _rewards_ = node.rewards
            sample_indices = np.random.choice(len(_rewards_), n) 
            sample = [_rewards_[j] for j in sample_indices]

            
            #_sample_ = [self.scalarize_reward(j) for j in sample]
            _sample_ = [j for j in sample]
            meanSample = sum(_sample_) / n
            node.samples_mean.append(meanSample)

            node = node.parent
        
            
        return
    

    def run(self):

        numActions = 4

        node = self.root
        childrenRewards = [[] for x in range(numActions)]
        childrenProbabilities = [[] for x in range(numActions)]    
        expectedUtility = [0] * numActions
        nodes = [None] * len(node.children)
        allData = [[] for x in range(numActions)] 
        rewards = [[] for x in range(numActions)]
        probabilities = [[] for x in range(numActions)]
        _probs_ = [[] for x in range(numActions)]
        scaled_probabilities = [[] for x in range(numActions)]

        

        a = 0
        #print("Number of Children", len(node.children), file = self.file)
        for child in node.children:            

            childrenRewards[child.action].append(child.rewards)
            childrenProbabilities[child.action].append(child.probabilities)             

            nodes[a] = child
            #print("Times Visited", child.timesVisited, file = self.file)
            #print("Node Type", child.type, file = self.file)
            for c in child.children:
                pass#print("Child UCT", c.rollUCT, file = self.file)
            #print("Node Children", child.children, file = self.file)
            #print("Node UCT", child.rollUCT, file = self.file)
            #print(" ", file = self.file)
            expectedUtility[child.action] += child.rollUCT / child.timesVisited
            allData[child.action].append(child.data)
        
            #for action in child._data_:
            for each in child._data_:
                #print("Test", each, file = self.file)
                rewards[child.action].append(child._data_[str(each)]['reward'])
                probabilities[child.action].append(child._data_[str(each)]['probability'])
                scaled_probabilities[child.action].append(child._data_[str(each)]['scaled probability'])

                #rewards[child.action].append(child._data_[each]['reward'])
                #probabilities[child.action].append(child._data_[each]['probability'])
                #scaled_probabilities[child.action].append(child._data_[each]['scaled probability'])

            a += 1


        return self, childrenRewards, childrenProbabilities, expectedUtility, nodes, node.data, allData, rewards, probabilities, scaled_probabilities
    

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


    def takeAction(self, action, cumulative_reward):
        
        node = self.root

        if self.numRewards == 3:
            next_state, reward, done, __ = self.env.step(node.state, action, cumulative_reward[1])
        else:
            next_state, reward, done, __ = self.env.step(node.state, action)        
        flag = 0

        #node.timesActionTaken[node.action] += 1
        #self.updateDistributionTable(node.state, action, reward, next_state)

        a = 0  
        
        for nodes in node.children:
            if nodes.action == action:
                node = nodes    
        for child in node.children:
            comparison = reward == child.reward
            if comparison.all():
                if child.done == done:
                    self.root = child 
                    flag = 1
        
        if flag == 0:     
            #print("Here 2", file = self.file)                       
            _node_ = node.createChild(node, next_state, action, reward, done, reward, True)
            
            node.childChanceRewards[action].append(reward)
            self.root = _node_
            #_node_.health = health
            #_node_.timesActionTaken += 1
            _node_.timeRewardReceived += 1 
        
        #probability = self.distTable[node.state][action][next_state][str(reward)]['count'] / self.distTable[node.state][action]['count']
        probability = 1
        """     
        
        for _node_ in node.children:
            if _node_.action == action:
                #_node_.timesActionTaken += 1                        
                if any((reward == q).all() for q in node.childChanceRewards[action]):
                    comparison = reward == _node_.reward
                    #print("Node State", node.state, file = self.file)
                    if comparison.all():
                        if _node_.done == done:
                            self.root = _node_
                            a += 1                      

                            #_node_.health = health
                            #_node_.timeRewardReceived += 1
                            flag = 1

    
        if flag == 0:     
            #print("Here 2", file = self.file)                       
            #_node_ = node.createChild(node, next_state, action, reward, done, reward, True)
            
            node.childChanceRewards[action].append(reward)
            self.root = _node_
            #_node_.health = health
            #_node_.timesActionTaken += 1
            _node_.timeRewardReceived += 1  
        """

        return next_state, reward, node, done, probability


class chanceNode:

    def __init__(self, node, state, action, args, _type_):
        self.numActions = 4
        #self.health = health
        self.type = "chance"
        self.parent = node
        self.samples_mean = []
        self.state = state
        self.distback_rewards = []
        self.distback_props = []
        self.action = action
        self.children = []
        self.reward = [0,0,0]
        self.isleaf = False
        self.args = args
        self.env = gym.make(self.args.env)
        self.timesVisited = 0
        self.timesActionTaken = {}
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.distributionalBackpropDict = {}
        self.distributionTable = {'count' : 0}
        #self.chanceRewards.append(reward)
        self.childChanceRewards = {}
        for i in range(self.numActions):
            self.timesActionTaken.update({i : 0})
            self.childChanceRewards.update({i : []})
        self._data_ = {}
        #for x in range(3):
        #    self._data_.update({x : {}})

        self.distribution = {}
        
        for x in range(self.numActions):
            self.distribution.update({x : {}})

        self.distributionData = {}
            
        self.chanceNode = False
        self.expectedReturn = -sys.maxsize
        #self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        
        #self.done = done
        self.hasChanceNodes = _type_
        self.numSimulations = 1
        #self.timesActionTaken = 0
        self.timeRewardReceived = 0
        self.probability = 1
        self._num_rewards = 3
        self.rollUCT = 0
        self.expectedUtility = 0
        #self.childrenRemaining = [0,1,2]
        self.data = {}
        self.distribution = {}
        self.childrenRemaining = [0,1,2,3]


        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

    def getProbability(self):
        if self.timeRewardReceived > 0:
            probability = self.timeRewardReceived
            probability = probability# / self.numActions
        else:
            _prob_ = 1
            probability = _prob_ #/ self.numActions

        return probability

    

    def createChild(self, node, state, action, reward, done, health, _type_):
        #node, node.state, action, node.row, node.col
        child = decisionNode(node, state, action, self.args, reward, done, health, _type_)


        self.children.append(child)

        return child

    def getExpectedReturn(self, node):
        var = 0
        rewards =[]
        probabilities = []
        scales = []
        scalr = [0,0]
        reward = node.reward
        probability = node.probability        

        for child in self.children:
            rewards.append(child.reward)
            probabilities.append(child.probability)
            scalr += (reward + child.reward) * (child.probability * probability)
            scales.append(scalr)


        if scalr[0] >= 0.88:
            var = scalr[1]
        else:
            var = -1000
        return var, rewards, probabilities, scales



    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function#

            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(self._num_rewards)})

class decisionNode:

    def __init__(self, node, state, action, args, reward, done, health, _type_):
        self.numActions = 4
        self.health = health
        self.parent = node
        self.type = "decision"
        self.state = state
        self.distback_rewards = []
        self.distback_props = []
        self.action = action
        self.children = []
        self.reward = reward
        self.isleaf = True
        self.samples_mean = []
        self.args = args
        self.env = gym.make(self.args.env)
        self.timesVisited = 0
        self.timesActionTaken = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.distributionTable = {'count': 0}
        self.chanceRewards.append(reward)
        self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3: []}
        self._data_ = {}

        if node == "Null":
            self.distributionTable = {'count' : 1, str(reward) : {'count' : 1}}


        #for x in range(3):
        #    self._data_.update({x : {}})

        self.chanceNode = False
        self.probability = 1
        #self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        
        self.done = done
        self.hasChanceNodes = _type_
        self.numSimulations = 1
        #self.timesActionTaken = 0
        self.timeRewardReceived = 0
        self.probability = self.getProbability()
        self._num_rewards = 2
        self.rollUCT = 0
        self.expectedUtility = 0
        self.distribution = {}
        #self.childrenRemaining = [0,1,2]
        self.data = {}
        self.childrenRemaining = [0,1,2,3]

        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

    def getProbability(self):
        if self.timeRewardReceived > 0:
            probability = self.timeRewardReceived
            probability = probability# / self.numActions
        else:
            _prob_ = 1
            probability = _prob_ #/ self.numActions

        return probability

    def setProbability(self, probability):
        self.probability = probability

    def createChild(self, node, state, action, reward, done, health, _type_):
        #node, node.state, action, node.row, node.col
        #self, node, state, action, args, _type_
        child = chanceNode(node, state, action, self.args,  _type_)

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

            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(3)})


class Node:

    def __init__(self, node, state, action, args, reward, done, health, _type_):
        self.numActions = 3
        self.health = health
        self.parent = node
        self.state = state
        self.numActions = 3
        self.action = action
        self.children = []
        self.reward = reward
        self.isleaf = True
        self.args = args
        self.env = gym.make(self.args.env)
        self.timesVisited = 0
        self.timesActionTaken = {0 : 0, 1 : 0, 2 : 0}
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.chanceRewards.append(reward)
        self.childChanceRewards = {0 : [], 1 : [], 2 : []}
        self._data_ = {}
        self.chanceNode = False
        #self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        
        self.done = done
        self.hasChanceNodes = _type_
        self.numSimulations = 1
        #self.timesActionTaken = 0
        self.timeRewardReceived = 0
        self.probability = self.getProbability()
        self._num_rewards = 2
        self.rollUCT = 0
        self.expectedUtility = 0
        self.childrenRemaining = [0,1,2]
        self.data = {}
        self.distribution = {}
        #self.childrenRemaining = [0,1,2,3]

        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

    def getProbability(self):
        if self.timeRewardReceived > 0:
            probability = self.timeRewardReceived
            probability = probability# / self.numActions
        else:
            _prob_ = 1
            probability = _prob_ #/ self.numActions

        return probability

    def isChanceNode(self):
        if self.isChanceNode == True:
            return True
        else:
            return False

    def isParentChanceNode(self):
        if self.parent.isChanceNode() == True:
            return True 
        else:
            return False

    def editNodeType(self):
        chanceNodeChildren = []
        for child in node.parent.children:
            if child.action == node.action:
                node.parent.children.remove(child)
                chanceNodeChildren.append(child)

        chanceNode = Node(node.parent, node.state, node.action, self.args, [0,0], False, 0, False)
        chanceNode.isChanceNode = True

        for nodes in chanceNodeChildren:

            chanceNode.chanceRewards.append(nodes.reward)
            chanceNode.children.append(nodes)
            nodes.parent = chanceNode

        node.parent.children.append(chanceNode)


   
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
        a = 3
        self.args = args
        self._env = gym.make(args.env)
        self.debug_file = open('debug', 'w') 
        self.tree = Tree(0, self.args,self.debug_file)
        self.num_actions = 3
        self.num_timesteps = 200
        #self.random_action = 0
        self._treedict_ = {}
        self.dict = {}     
        self.TS = {}      
        self.args = args   
        self.act = 0  

        self.sampleDict = {}
        for x in range(2):
            self.sampleDict.update({x : {}})
            for y in range(3):
                self.sampleDict[x].update({y : {}})      

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

    def TESTscalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        #print("Scalarise R", rewards, file = self.debug_file)
        if rewards[0] >= 0.88:
            return rewards[1]
        else:
            return -1000000000

    def TESTscalarize_reward1(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        #print("Scalarise R", rewards, file = self.debug_file)
        if rewards[0] == 1:
            return rewards[1]
        else:
            return -1000

    def optimalRoute(self, node, tree):
        self.node = node
        self.rewards = []
        self.probabilities = []
        self.stareward = [0,0]
        self.staprob = 1
        self.bestChild = 0
        self.reward = []
        self.probabilities = []

        self.cumulativeReward = [0,0]
        self.probability = 1

        def backup(node):
            _reward_ = [0,0]
            _prob_ = 1

            while node != self.node.parent:

                _reward_ += node.reward
                if node.type == "chance":
                    _prob_ = _prob_ * 1
                
                else:
                    _prob_ = _prob_ * tree.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / tree.distTable[node.parent.state][node.action]['count']
                node = node.parent

            self.rewards.append(_reward_)
            self.probabilities.append(_prob_)


        def find(node):            
            
            for outcome in node.children:

                if len(outcome.children) == 0:  
                    backup(outcome) 
                
                else:
                    _val_ = -sys.maxsize
                    
                    for action in outcome.children:
                        val, testA, testB, scales = action.getExpectedReturn(outcome)
                        #print("Child Rewards", testA, file = self.debug_file)
                        #print("Child Probabilities", testB, file = self.debug_file)
                        #print("Child Scales", scales, file = self.debug_file)
                        #print(" ", file = self.debug_file)
                        if val > _val_:
                            _val_  = val
                            _node_ = action
                    
                    find(_node_)                    

                    

        def finder(node, action):            
            
            for outcome in node.children:

                if len(outcome.children) == 0:  
                    backup(outcome) 
                
                else:
                    #_val_ = -sys.maxsize
                    
                    for _node in outcome.children:
                        if _node.action == action:
                            _node_ = _node
                    
                    finder(_node_, action)
                        

                        #finder(outcome)  


        #for action in range(3):
            #finder(node,action)
        find(node)


        return self.rewards, self.probabilities

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

    def selectAction(self, method, tree, rewards, probabilities, cumulative_rewards, cumulative_probability, state):

        #print("Rewards :", rewards, file = self.debug_file)
        #print("Probabilities :", probabilities, file = self.debug_file)

        if method == "UCT":
            val = -sys.maxsize
            cr = self.scalarize_reward(cumulative_rewards)
            for node in self.tree.root.children:
                if (node.rollUCT / node.timesVisited) + cr > val:
                    val = (node.rollUCT / node.timesVisited) + cr
                    action = node.action
                    #nodes[a] = child 
            #expectedUtility[a] = child.rollUCT / child.timesVisited


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
            actions = [0] * 3
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

        if method == "Optimal Policy":
            node = tree.root
            rewards = []
            probabilities = []
            actions = []

            for child in node.children:
                rewards.append(self.optimalRoute(child, tree)[0])
                probabilities.append(self.optimalRoute(child, tree)[1])
                actions.append(child.action)
            
            #print("Rewards Checker", rewards, file = self.debug_file)
            #print("Probabilities Checker", probabilities, file = self.debug_file)
            #utility = 0
            val = -sys.maxsize
            utilities = []


            
            bestAction = -10
            avg_reward = []
            for action in range(3):
                a = 0
                utility = 0
                scalar_reward = [0,0]
                for reward in rewards[action]:
                    #print("Reward", reward, file = self.debug_file)
                    #print("Probability", probabilities[action][a], file = self.debug_file)
                    #print("Scaled", reward * probabilities[action][a], file = self.debug_file)

                    scalar_reward += (cumulative_rewards + reward) * (cumulative_probability * probabilities[action][a])
                    #print("Scalar Reward", scalar_reward, file = self.debug_file)

                    #utility += self.TESTscalarize_reward((reward) * probabilities[action][a])
                    #print("Utility", utility, file = self.debug_file)
                    a += 1
                utility = self.TESTscalarize_reward1(scalar_reward)

                #if utility > val:
                    #val = utility
                    #bestAction = actions[action]
                utilities.append(utility)
                avg_reward.append(scalar_reward)

            index = utilities.index(max(utilities))
            action = actions[index]

        if method == "Bootstrap Thompson":
            node = tree.root
            minVal = -sys.maxsize

            means = [[] for x in range(4)]
            #print("Node type...", node.type, file = self.debug_file)
            for child in node.children:
                #print("Child type...", child.type, file = self.debug_file)

                meanSample_indices = np.random.choice(len(child.samples_mean))            
                randomSample = child.samples_mean[meanSample_indices]
                #self.sampleDict[node.state][child.action].update({'distribution' : child.samples_mean})

                #mean = sum(child.samples_mean) / len(child.samples_mean)
                #means[child.action].append(mean)           
                """
                if randomSample > minVal:
                    minVal = randomSample 
                    action = child.action
                """
               #print(child.action, file = self.debug_file)
               #print("Hello", file = self.debug_file)
                #if self.timestep == 0 and child.action == 0:
                print("State", state, file = self.debug_file)
                xy = self.getXYfromState(state)
                print("Timestep", self.timestep, "X", xy[0], "Y", xy[1], "Action", child.action, file = self.debug_file)
                #print("Action", child.action, file = self.debug_file)
                print("Random Sample", randomSample, file = self.debug_file)
                print("Cumulative Reward", cumulative_rewards, file = self.debug_file)
                print("Sample + Reward", randomSample + cumulative_rewards, file = self.debug_file)
                print("Distribition", child.distribution, file = self.debug_file)
                print(" ", file = self.debug_file)
                means[child.action] = self.scalarize_reward(randomSample + cumulative_rewards) 
                #if self.scalarize_reward(randomSample + cumulative_rewards) > minVal:
                    #minVal = self.scalarize_reward(randomSample + cumulative_rewards) 
                    #action = child.action
                action = means.index(max(means))
                #print("Sample", randomSample, file = self.debug_file)
                """
                if randomSample > minVal:
                    minVal = randomSample
                    action = child.action
                """
            
            #actions = [0, 1, 2, 3]
            #action = random.choice(actions)

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
        self._num_rewards = 3
        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        cumulative_probability = 1
        rewards = np.zeros(shape=(self._num_rewards,))
        
        timestep = 0
        scalar_reward = 0

        action = -10        
        self.timestep = 0       
        new_env_state = -1



        _all_ = False
        action_selection = 0
        self.num_actions = 3
        #self.row = 0
        #self.col = 0
        self.done = False
        health = 0
        
        #self.tree.root = "Null"
        self.tree.reset()
        x = 0
        check = random.random()
        #self.epsilon = 1

        while not self.done:
            
            state = env_state            

            action_rewards = []
            action_prob = []                



            self.tree.step(cumulative_rewards)
            #tree, testReward , testProbs, expectedUtility, nodes, nodeDist, allData, a, b, scaled_probs = self.tree.run()
            testReward = []
            testProbs = []
            expectedUtility = 0
            nodes = []
            nodeDist = []
            allData = {}
            a = []
            b = []
            scaled_probs = []
            #x, y = self.getXYfromState(state)
            #print("Cumulative Rewards", cumulative_rewards, file = self.debug_file)
            action = self.selectAction("Bootstrap Thompson", self.tree, testReward, testProbs, cumulative_rewards, cumulative_probability, env_state)
            #actions = [0, 1, 2, 3]
            #action = random.choice(actions)
            #print("Action", action, file = self.debug_file)

            #print("X :", x, "Y :", y, file = self.debug_file) 
            #print("State", state, file = self.debug_file)
            #print("Rewards Action:: ", testReward, file = self.debug_file)
            #print("Probabilities ::", testProbs, file = self.debug_file)
            #print("All Data", allData, file = self.debug_file)
            #print("Data 0", allData[0], file = self.debug_file)
            #print("Data 1", allData[1], file = self.debug_file)
            #print("Data 2", allData[2], file = self.debug_file)
            #print("Expected Utility on each Node :", expectedUtility, file = self.debug_file)
            #print("Distribution ", nodeDist, file = self.debug_file)
            #print("Check", testReward, file = self.debug_file)
            #print("*** Start ***", file = self.debug_file)
            #print("Rewards 0 ", a[0], file = self.debug_file)
            #print("Probabilities 0", b[0], file = self.debug_file)
            #print("Scaled Probabilities", scaled_probs[0], file = self.debug_file)
            #print(" ", file = self.debug_file)

            #print("Rewards 1 ", a[1], file = self.debug_file)
            #print("Probabilities 1", b[1], file = self.debug_file)
            #print("Scaled Probabilities 1", scaled_probs[1], file = self.debug_file)
            #print("Sum SP1", sum(scaled_probs[1]), file = self.debug_file)
            #print(" ", file = self.debug_file)

            #print("Rewards 2 ", a[2], file = self.debug_file)
            #print("Probabilities 2", b[2], file = self.debug_file)
            #print("Scaled Probabilities 2", scaled_probs[2], file = self.debug_file)
            #print("Sum SP2", sum(scaled_probs[2]), file = self.debug_file)
            #print(" ", file = self.debug_file)

            #print("End", file = self.debug_file)
            
            #print("Action", action, file = self.debug_file)
            #print("Probabilities ALl", b, file = self.debug_file)
            #print("Sum Prob 0", sum(b[0]), file = self.debug_file)
            #print("Sum Prob 1", sum(b[1]), file = self.debug_file)
            #print("Sum Prob 2", sum(b[2]), file = self.debug_file)
            #print(" ", file = self.debug_file)
             
            """
            if random.random() < 0.0:
                if self.act < 0:
                    action = 2
                else:
                    action = random.randint(0,2)
                self.act += 1
            else:
                index = expectedUtility.index(max(expectedUtility))
                node = nodes[index]
                #action = node.action

                #print("Action", action, file = self.debug_file)
            """
            env_state, rewards, node, done, probability = self.tree.takeAction(action, cumulative_rewards)

            cumulative_rewards += rewards
            cumulative_probability *= probability
            print("Action Taken:", action, "Reward Total:", cumulative_rewards, file = self.debug_file)
            self.done = done


            if len(cumulative_rewards) > 2:

                if cumulative_rewards[1] <= -100 or health <= -100:
                    cumulative_rewards[1] = -100
                    self.done = True
                    self.tree.reset()

            if x > 50:
                self.done = True
                self.tree.reset()

            x += 1
            self.timestep += 1
        self.tree.reset()
        return cumulative_rewards, testReward, testProbs, self.sampleDict

def main():
    # Parse parameters
    num_runs = 1
    episodes = 1000
    
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
    dfDist = pd.DataFrame()
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
            utility = 0
            avgUtility = 0

            for i in range(episodes):
                rewards, allRewards, allProbabilities, sampleDict = learner.run()

                if i == 0:
                    avg = rewards
                else:
                    #avg = 0.99 * avg + 0.01 * rewards
                    avg = avg + rewards

                #print("Percentage Completed....", i%100, "% ", "Run : ", num_runs, " Episode : ", i,  file = f)
                scalarized_avg = learner.scalarize_reward(avg)
                if rewards[0] >= 0.88:
                    utility += rewards[1]
                else:
                    utility += -1000

                if i < 2:
                    learner.epsilon = 1.0
                else:
                    learner.epsilon = learner.epsilon * 0.9999
                    if learner.epsilon < 0.1:
                        learner.epsilon = 0.1

                avgUtility = utility / (i + 1)

                if i % 1 == 0 and i >= 0:
                    r = (i/episodes) * 100
                    time = datetime.time(datetime.now())
                    time_elapsed = datetime.combine(date.today(), time) - datetime.combine(date.today(), start_time)

                    print("Episode", i, "Time Elapsed : ", time_elapsed, "Cumulative reward:", rewards, file = f)
                    f.flush()

                """

                if i > 0 and i % 100 == 0:
                    graphDict = {}
                    dfDist = pd.DataFrame()
                    for state in sampleDict:
                        graphDict.update({state : {}})
                        for action in sampleDict[state]:
                            distribution = sampleDict[state][action]['distribution']
                            unique, frequency = np.unique(distribution, return_counts=True)
                            graphDict[state].update({action : {'unique' : unique, 'frequency' : frequency}})
                            #plt.bar(unique, frequency, alpha = 0.25)
                            #print(unique, file = f)
                            dfDist['State ' + str(state) + ' Action ' + str(action) + " Unique "] = pd.Series(unique)
                            dfDist['State ' + str(state) + ' Action ' + str(action) + " Frequency "] = pd.Series(frequency) 
                    #print(dfDist, file = f)
                    #dfDist.to_csv(r'ExperimentGraphs/DistributionGraphs_Episode' + str(i) + '.csv')

                        #plt.show()
                """



                #if i % 1000 == 0 and i >= 0:
                    #print("Episode", i, file = learner.debug_file)
                    #print("Test Rewards :", allRewards, file = learner.debug_file)
                    #print("Test Probabilities :", allProbabilities, file = learner.debug_file)
                    #print(" ", file = learner.debug_file)

                #print("Cumulative reward:", rewards, file=f)
                #print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
                #print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg)
                runData.append(avgUtility)
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
