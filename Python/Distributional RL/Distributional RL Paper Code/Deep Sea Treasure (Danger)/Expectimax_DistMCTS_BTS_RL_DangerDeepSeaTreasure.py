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

        self.coverageSet = []
        self.hypervolume = []

        self.coverage0 = False
        self.coverage1 = False
        self.coverage2 = False
        self.coverage3 = False
        self.coverage4 = False
        self.coverage5 = False
        self.coverage6 = False
        self.coverage7 = False
        self.coverage8 = False
        self.coverage9 = False

        

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
            #print(node, file = self.file)
            if node.parent == "Null":
                pass
            
            elif node.type == "chance":
                node, cumulative_reward = self.runSimulate(node, cumulative_reward)


            if node.type == "decision" and node.done == True:
                return node

            if node.isleaf == True and node.type == "decision":                
                node, cumulative_reward = self.expand(node, cumulative_reward)
                node.rewards = self.artificialSamples
                #node.samples_mean = node.rewards
                #print(node.rewards, file = self.file)
                
                return node            
            
            
            #node = self.UCT(node)

            node = self.thompsonSampling(node, cumulative_reward)
            #node = random.choice(node.children)
            return findExpansionSpot(node, cumulative_reward)           

        def rollOut(node):
            
            self.simulate(node)

        #start = time.time()
        while self.num_expansions < 10:

            node = self.root 
            node = findExpansionSpot(node, cumulative_reward)
            rollOut(node)
            self.count += 1

            self.num_expansions += 1      
        end = time.time()
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
            state, reward, done, __ = self.env.step(node.state, node.action, cumulative_reward[1])
        else:
            state, reward, done, __ = self.env.step(node.state, node.action)

        if str(reward) in node.distributionTable:
            flag = 1
            for _node_ in node.children:
                comparison = reward == _node_.reward
                if comparison.all():
                    n = _node_

        self.updateDistributionTable(node.state, node.action, reward, state, node)
    
        if flag == 0:                      
            _node_ = node.createChild(node, state, node.action, reward, done, reward, True)            
            #node.parent.childChanceRewards[node.action].append(reward)
            _node_.timeRewardReceived += 1
            self.updateDistributionTable(node.state, node.action, reward, state, _node_)
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
        c = 30
        #n = 1500
        n = 250
        #n = 1000
        #action = 0
        minVal = -sys.maxsize
        arr = np.array([124, 0, -1])
        checkReward = []
        checkCount = []
        sample = np.random.choice(len(node.onlineRewards) + n, len(node.children))
        #print(sample, file = self.file)
        #print(len(node.children), file = self.file)
        for child in node.children:

            #print(child.rewards, file = self.file)
            #sample = np.random.choice(len(child.onlineRewards))
            #print(sample, file = self.file)
            #randomSample = self.scalarize_reward(np.random.choice(child.samples_mean[sample]) + cumulative_reward)
            #print()
            #print(child.onlineRewards[sample] / (child.count[sample] ), file = self.file)
            #print(child.onlineRewards[sample], file = self.file)
            #print(child.artificialRewards, file = self.file)
            if child.timesVisited == 0:
                randomSample = sys.maxsize
            else:
                #print(child.artificialRewards, file = self.file)
                checkReward = np.vstack((child.onlineRewards, child.artificialRewards))

                checkCount = np.concatenate((child.count, child.artificialCount))
                #print(child.onlineRewards, file = self.file)
                #checkCount = child.count
                #for i in range(n):
                    #checkReward = np.vstack((checkReward, arr))
                    #checkReward.append(arr)
                    #checkCount = np.append(checkCount, 1)
                    
                #randomSample = self.scalarize_reward(child.onlineRewards[sample[child.action]] / child.count[sample[child.action]])# + self.scalarize_reward(cumulative_reward))
                #print(checkReward, file = self.file)
                randomSample = self.scalarize_reward(checkReward[sample[child.action]] / checkCount[sample[child.action]])
                randomSample = randomSample #+ 60 * np.sqrt (2 * np.log(node.timesVisited) / child.timesVisited)
                #print(randomSample, file = self.file)
                
                if randomSample > minVal:
                    minVal = randomSample
                    bestNode = child  

       
        if randomNum < 0:
            #self.simulate(np.random.choice())
            bestNode = np.random.choice(node.children)

        #distRewards = bestNode.distribution['reward']
        #distProbability = bestNode.distribution['probability']
        #self.onlineBootstrap(bestNode, distRewards, distProbability, 1)

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
        #print(node.distributionTable, file = self.file)

        

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

            while _done_ == False and a <= 30:

                action = random.randint(0, 3)
                
                if self.numRewards == 3:
                    next_state, reward, done, __ = self.env.step(state, action, cumulative_reward[1])
                else:
                    next_state, reward, done, __ = self.env.step(state, action)


                #self.updateDistributionTable(state, action, reward, next_state)

                cumulative_reward = reward + cumulative_reward

                if len(cumulative_reward) > 2:
                    if cumulative_reward[1] <= -10:
                        cumulative_reward[1] = -10
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

    def hypervolumeCalc(self, cumulative_reward):

        #if node.parent == "Null":
        cumulative_reward = np.array(cumulative_reward)
        #print(cumulative_reward, file = self.file)
        comparison0 = cumulative_reward == np.array([1, 0, -1])
        #print(comparison0, file = self.file)
        comparison1 = cumulative_reward == np.array([2, 0, -3])
        comparison2 = cumulative_reward == np.array([3, 0, -5])
        comparison3 = cumulative_reward == np.array([5, 0, -7])
        comparison4 = cumulative_reward == np.array([8, 0, -8])
        comparison5 = cumulative_reward == np.array([16, 0, -9])
        comparison6 = cumulative_reward == np.array([24, 0, -13])
        comparison7 = cumulative_reward == np.array([50, 0, -14])
        comparison8 = cumulative_reward == np.array([74, 0, -17])
        comparison9 = cumulative_reward == np.array([124, 0, -19])

        #comparison0 = cumulative_reward == np.array([1, 0, -1])
        comparison21 = cumulative_reward == np.array([34, 0, -3])
        comparison22 = cumulative_reward == np.array([58, 0, -5])
        comparison23 = cumulative_reward == np.array([78, 0, -7])
        comparison24 = cumulative_reward == np.array([86, 0, -8])
        comparison25 = cumulative_reward == np.array([92, 0, -9])
        comparison26 = cumulative_reward == np.array([112, 0, -13])
        comparison27 = cumulative_reward == np.array([116, 0, -14])
        comparison28 = cumulative_reward == np.array([122, 0, -17])
        comparison29 = cumulative_reward == np.array([124, 0, -19])
            


        if comparison0.all() == True and self.coverage0 == False:
            self.hypervolume.append(2)
            self.coverageSet.append([1,-1])
            self.coverage0 = True

        if comparison1.all() == True and self.coverage1 == False:
            self.hypervolume.append(68)
            self.coverageSet.append([2, -3])
            self.coverage1 = True

        if comparison2.all() == True and self.coverage2 == False:
            self.hypervolume.append(116)
            self.coverageSet.append([3, -5])
            self.coverage2 = True

        if comparison3.all() == True and self.coverage3 == False:
            self.hypervolume.append(78)
            self.coverageSet.append([5, -7])
            self.coverage3 = True

        if comparison4.all() == True and self.coverage4 == False:
            self.hypervolume.append(86)
            self.coverageSet.append([8, -8])
            self.coverage4 = True

        if comparison5.all() == True and self.coverage5 == False:
            self.hypervolume.append(368)
            self.coverageSet.append([16, -9])
            self.coverage5 = True

        if comparison6.all() == True and self.coverage6 == False:
            self.hypervolume.append(112)
            self.coverageSet.append([24, -13])
            self.coverage6 = True

        if comparison7.all() == True and self.coverage7 == False:
            self.hypervolume.append(348)
            self.coverageSet.append([50, -14])
            self.coverage7 = True

        if comparison8.all() == True and self.coverage8 == False:
            self.hypervolume.append(244)
            self.coverageSet.append([72, -17])
            self.coverage8 = True

        if comparison9.all() == True and self.coverage9 == False:
            self.hypervolume.append(744)
            self.coverageSet.append([124, -19])
            self.coverage9 = True




        if comparison21.all() == True and self.coverage1 == False:
            self.hypervolume.append(68)
            self.coverageSet.append([34, -3])
            self.coverage1 = True

        if comparison22.all() == True and self.coverage2 == False:
            self.hypervolume.append(116)
            self.coverageSet.append([58, -5])
            self.coverage2 = True

        if comparison23.all() == True and self.coverage3 == False:
            self.hypervolume.append(78)
            self.coverageSet.append([78, -7])
            self.coverage3 = True

        if comparison24.all() == True and self.coverage4 == False:
            self.hypervolume.append(86)
            self.coverageSet.append([86, -8])
            self.coverage4 = True

        if comparison25.all() == True and self.coverage5 == False:
            self.hypervolume.append(368)
            self.coverageSet.append([92, -9])
            self.coverage5 = True

        if comparison26.all() == True and self.coverage6 == False:
            self.hypervolume.append(112)
            self.coverageSet.append([112, -13])
            self.coverage6 = True

        if comparison27.all() == True and self.coverage7 == False:
            self.hypervolume.append(348)
            self.coverageSet.append([116, -14])
            self.coverage7 = True

        if comparison28.all() == True and self.coverage8 == False:
            self.hypervolume.append(244)
            self.coverageSet.append([122, -17])
            self.coverage8 = True

        if comparison29.all() == True and self.coverage9 == False:
            self.hypervolume.append(744)
            self.coverageSet.append([124, -19])
            self.coverage9 = True

        return

    def onlineBootstrap(self, node, distRewards, distProbability, n, cumulative_reward):
        a = 1
        n = 10
        holder = []
        _rewards_ = []
        sample_indices = []
        sample = []
        #distRewards = node.distribution['reward']
        #distProbability = node.distribution['probability']
        holder = [i for i in range(len(distRewards))]
        _rewards_ = np.random.choice(holder, 1, p=distProbability)
        _rewards_ = [distRewards[i] for i in _rewards_]
        #_rewards_ = sum(_rewards_) / n
        _sample_ = sample 
        #r = 2
        r = 4
        ar = np.random.randint(r, size = (len(node.onlineRewards)))
        node.onlineRewards[ar == 1] = np.add(node.onlineRewards[ar == 1] , _rewards_[0])
        node.count[ar == 1] += 1

        #node.sample = np.vstack((node.onlineRewards, node.artificialRewards))
        #node.sampleCount = np.concatenate((node.count, node.artificialCount))
        

        node.samples_mean = node.onlineRewards

        return


    def distributionSample(self, node, distRewards, distProbability, n):
        a = 100
        n = 100
        holder = []
        _rewards_ = []
        sample_indices = []
        sample = []
        
        holder = [i for i in range(len(distRewards))]
        _rewards_ = np.random.choice(holder, a, p=distProbability)
        #_rewards_ = [distRewards[i] for i in _rewards_]
        _rewards_ = list(map(lambda x: distRewards[x], _rewards_)) 

        for i in range(len(_rewards_)):
            node.rewards.append(_rewards_[i])

        sample_indices = np.random.choice(len(node.rewards), n )
        #sample = [node.rewards[j] for j in sample_indices]
        sample = list(map(lambda x: node.rewards[x], sample_indices)) 
        #print("Sample", sample, file = self.file)
        #print(sample, file = self.file)

        
        #print("B4 Sample Scalarized", sample, file = self.file)
        #_sample_ = [self.scalarize_reward(j) for j in sample]
        _sample_ = sample        
        
        meanSample = sum(_sample_) / n
        #print(meanSample, file = self.file)
        node.samples_mean.append(meanSample)
        #print("Samples Mean", node.samples_mean, file = self.file)


        return

    def distributionCompare(self, node, oldNode, distRewards, distProbability, cumulative_reward, startNode):
       
        if node.type == "chance":
            #print("Check 0ff", distRewards, file = self.file)
            #print("Check 0ff", distProbability, file = self.file)
            xx = 0
            array = []
            childProbDistribution = []
            #print(len(node.children), file = self.file)
            holderRewards = []
            holderProbabilities = []

            if len(node.distribution.keys()) == 0:
                #print(bool(node.distribution), file = self.file)
                node.distribution.update({'reward' : distRewards})
                node.distribution.update({'probability' : distProbability})

            prob = node.distributionTable[str(oldNode.reward)]['count'] / node.distributionTable['count']
            for reward in distRewards:
                    holderRewards.append(reward)
            for probs in distProbability:
                holderProbabilities.append(prob * probs)

            if len(node.children) > 1:
                for child in node.children:
                    
                    if child == oldNode:
                        pass
                        #print()                    
                        #print("Check 1", distRewards, file = self.file)
                        #print("Check 1", distProbability, file = self.file)
                        

                    else:
                        childRewardDistribution = child.distribution['reward']# + node.reward
                        childProbDistribution = child.distribution['probability']
                        #print("b4 Test 3", distRewards, file = self.file)
                        #print("childProbDistribution", childProbDistribution, file = self.file)
                        #print("DistRewards", distRewards, file = self.file)
                        #print("childRewardDistribution", childRewardDistribution, file = self.file)
                        #print("childProbDistribution", childProbDistribution, file = self.file)
                        
                        prob = node.distributionTable[str(child.reward)]['count'] / node.distributionTable['count']
                        #print("Prob", prob, file = self.file)
                        childProbDistribution = [x * prob for x in childProbDistribution]
                        for i in childRewardDistribution:
                            holderRewards.append(i)
                        for i in childProbDistribution:
                            holderProbabilities.append(i)
                            #print("Check 2 Append", i, file = self.file)
                            #print("Check 2 Append", distRewards, file = self.file)
                        #print("Check 2", distRewards, file = self.file)
                        #print("Check 2", distProbability, file = self.file)
                    xx += 1
                #if len(childProbDistribution) > 0:
                    #distProbability = np.append(distProbability, childProbDistribution)
                    
            
            distRewards = holderRewards
            distProbability = holderProbabilities  
            #print("XX", xx ,file = self.file)

            #print("Len???", distProbability, file = self.file)

            if len(node.distribution.keys()) == 0:
                #print(bool(node.distribution), file = self.file)
                node.distribution.update({'reward' : distRewards})
                node.distribution.update({'probability' : distProbability})

                #distRewards = node.distribution['reward']
                #distProbability = node.distribution['probability']

                #print("Check 3", distRewards, file = self.file)
                #print("Check 3", distProbability, file = self.file)
            
            elif len(node.distribution['reward']) != len(distRewards) and node == startNode.parent:
                #print("Check", node.distribution['reward'], file = self.file)
                node.distribution.update({'reward' : distRewards})
                node.distribution.update({'probability' : distProbability})

                #distRewards = node.distribution['reward']
                #distProbability = node.distribution['probability']

                #print("Check 4", distRewards, file = self.file)
                #print("Check 4", distProbability, file = self.file)

            else:
                sum1 = 0
                sum2 = 0
                #print("Check 5", distRewards, file = self.file)
                #print("Check 5", distProbability, file = self.file)
                #print("Check 5", node.distribution['reward'], file = self.file)
                #print("Check 5", node.distribution['probability'], file = self.file)
                for val in range(len(node.distribution['reward'])):
                    #print("Index out of range", node.distribution['reward'], file = self.file)
                    #print("Index out of range",node.distribution['probability'], file = self.file)

                    sum1 += self.scalarize_reward(node.distribution['reward'][val]) * node.distribution['probability'][val]
                    
                for val in range(len(distRewards)):
                    #print("Here", distRewards, file = self.file)
                    #print("Scala", self.scalarize_reward(distRewards[val])  * distProbability[val], file = self.file)
                    sum2 += self.scalarize_reward(distRewards[val]) * distProbability[val]

                #print(sum1, sum2, file = self.file)
                if sum2 > sum1:
                    node.distribution.update({'reward' : distRewards})
                    node.distribution.update({'probability' : distProbability})

                    distRewards = node.distribution['reward']
                    distProbability = node.distribution['probability']

                    #print("Check 6", distRewards, file = self.file)
                    #print("Check 6", distProbability, file = self.file)

                #else:

                    #distRewards = node.distribution['reward']
                    #distProbability = node.distribution['probability']

        else:
            if len(node.distribution.keys()) == 0:
                #print(bool(node.distribution), file = self.file)
                node.distribution.update({'reward' : distRewards})
                node.distribution.update({'probability' : distProbability})
                
                distRewards = node.distribution['reward']
                distProbability = node.distribution['probability']

                #print("Check 7", distRewards, file = self.file)
                #print("Check 7", distProbability, file = self.file)
            else:
                sum1 = 0
                sum2 = 0
                #print("Check 8", distRewards, file = self.file)
                #print("Check 8", distProbability, file = self.file)
                #print("Check 8", node.distribution['reward'], file = self.file)
                #print("Check 8", node.distribution['probability'], file = self.file)
                for val in range(len(node.distribution['reward'])):
                    
                    sum1 += self.scalarize_reward(node.distribution['reward'][val]) * node.distribution['probability'][val]
                    
                for val in range(len(distRewards)):

                    sum2 += self.scalarize_reward(distRewards[val]) * distProbability[val]

                if sum2 > sum1:

                    node.distribution.update({'reward' : distRewards})
                    node.distribution.update({'probability' : distProbability})

                    distRewards = node.distribution['reward']
                    distProbability = node.distribution['probability']
                    #print("Here...", file = self.file)
                    #print("Check 9", distRewards, file = self.file)
                    #print("Check 9", distProbability, file = self.file)
                #else:
                    #distRewards = node.distribution['reward']
                    #distProbability = node.distribution['probability']

        #print("Here...", node.distribution['reward'], file = self.file)
        #print("Here...", node.distribution['probability'], file = self.file)
        #print("Here dist", distRewards, file = self.file)
        #print("Here Prob", distProbability, file = self.file)
        return distRewards, distProbability
    

    def distributionBackPropogation(self, node, cumulative_reward):
        
        a = 0
        n = 10
        probability = 1
        #print(self.distTable[node.parent.state][node.action][node.state], file = self.file)
        #node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
        node.probability = node.distributionTable[str(node.reward)]['count'] / node.distributionTable['count']
        #print("CR Check", cumulative_reward, file = self.file)
        
        #print("Dist Rewards", distRewards, file = self.file)
        distProbability = [1]
        startNode  = node
        oldNode = node

        if len(cumulative_reward) > 2:
            if cumulative_reward[1] <= -10:
                cumulative_reward[1] = -10
                #done == True

        distRewards = [cumulative_reward]

        while node != self.root:
            distRewards, distProbability = self.distributionCompare(node, oldNode, distRewards, distProbability, cumulative_reward, startNode)
            #print("In while", distRewards, file = self.file)
            self.onlineBootstrap(node, distRewards, distProbability, 1, cumulative_reward)

            oldNode = node
            if node.type == "decision" and node.done == False:                
                cumulative_reward = cumulative_reward + node.reward
                distRewards = [node.reward + i for i in distRewards]

            #print(distRewards, file = self.file)
            #print(distProbability, file = self.file)
            node.timesVisited += 1
            oldNode = node
            node = node.parent


        if node == self.root:
            distRewards, distProbability = self.distributionCompare(node, oldNode, distRewards, distProbability, cumulative_reward, startNode)
            #print("Out while", distRewards, file = self.file)
            self.onlineBootstrap(node, distRewards, distProbability, 1, cumulative_reward)

            oldNode = node
            if node.type == "decision" and node.done == False:                
                cumulative_reward = cumulative_reward + node.reward
                distRewards = [node.reward + i for i in distRewards]

            #distRewards = [node.reward + i for i in distRewards]
            node.timesVisited += 1
            oldNode = node
            node = node.parent

       
            
        
        if node == "Null":
            self.hypervolumeCalc(cumulative_reward)  
            #print("Hypervolume", self.hypervolume, "Coverage Set", self.coverageSet, file = self.file)

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
                if cumulative_reward[1] <= -10:
                    cumulative_reward[1] = -10

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
        #self.updateDistributionTable(node, node.state, action, reward, next_state)

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

        n = 10000
        an = 1500
        #an = 250

        #an = 1000
        self.onlineRewards = []
        artificialData = []
        

        self.onlineRewards = np.tile(np.array([1, 0, 0]),(n, 1))
        self.count = np.empty(n)
        self.count.fill(1)

        self.artificialRewards = []
        #artificialData =  np.array([artificialData])
        r1 = np.random.randint(0, 124, size=(an, 1))
        r2 = np.zeros(shape=(an, 1))
        r3 = np.random.randint(-30,0, size=(an, 1))
        #self.artificialRewards = np.tile(np.array(np.array([random,randint(0, 124), 0, -random.randint(0, 30)]),(an, 1))
        self.artificialRewards = np.hstack((r1,r2,r3))

        self.artificialCount = np.empty(an)
        self.artificialCount.fill(1)

        self.sample = np.vstack((self.onlineRewards, self.artificialRewards))
        self.sampleCount = np.concatenate((self.count, self.artificialCount))
        

        #self.health = health
        self.type = "chance"
        self.parent = node
        self.samples_mean = self.onlineRewards
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

        n = 10000
        an = 1500
        #an = 250

        #an = 1000
        self.onlineRewards = []
        artificialData = []
        

        self.onlineRewards = np.tile(np.array([1, 0, 0]),(n, 1))
        self.count = np.empty(n)
        self.count.fill(1)

        self.artificialRewards = []
        #artificialData =  np.array([artificialData])
        r1 = np.random.randint(0,124, size=(an, 1))
        r2 = np.zeros(shape=(an, 1))
        r3 = np.random.randint(-30,0, size=(an, 1))
        #self.artificialRewards = np.tile(np.array(np.array([random,randint(0, 124), 0, -random.randint(0, 30)]),(an, 1))
        self.artificialRewards = np.hstack((r1,r2,r3))

        self.artificialCount = np.empty(an)
        self.artificialCount.fill(1)

        self.sample = np.vstack((self.onlineRewards, self.artificialRewards))
        self.sampleCount = np.concatenate((self.count, self.artificialCount))
        

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
        self.samples_mean = self.onlineRewards
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
            sample = np.random.choice(len(node.onlineRewards), len(node.children))
            for child in node.children:
                #print("Child type...", child.type, file = self.debug_file)
                randomSample = child.onlineRewards[sample[child.action]] / child.count[sample[child.action]]
                
                xy = self.getXYfromState(state)
                #print("Timestep", self.timestep, "X", xy[0], "Y", xy[1], "Action", child.action, file = self.debug_file)
                #print("Action", child.action, file = self.debug_file)
                #print("Random Sample", randomSample, file = self.debug_file)
                #print("Cumulative Reward", cumulative_rewards, file = self.debug_file)
                #print("Sample + Reward", randomSample + cumulative_rewards, file = self.debug_file)
                #print("Distribution", child.distribution, file = self.debug_file)
                #print(" ", file = self.debug_file)

                means[child.action] = self.scalarize_reward((randomSample) + cumulative_rewards) 

            action = means.index(max(means))

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
            
            env_state, rewards, node, done, probability = self.tree.takeAction(action, cumulative_rewards)

            cumulative_rewards += rewards
            cumulative_probability *= probability
            print("Action Taken:", action, "Reward Total:", cumulative_rewards, file = self.debug_file)
            self.done = done


            if len(cumulative_rewards) > 2:

                if cumulative_rewards[1] <= -10 or health <= -10:
                    cumulative_rewards[1] = -10
                    self.done = True
                    self.tree.reset()

            if x > 50:
                self.done = True
                self.tree.reset()

            x += 1
            self.timestep += 1
        self.tree.reset()
        return cumulative_rewards, testReward, testProbs, self.sampleDict, self.tree.hypervolume, self.tree.coverageSet

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
                rewards, allRewards, allProbabilities, sampleDict, hypervolume, coverageSet = learner.run()

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

                    print("Episode", i, "Time Elapsed : ", time_elapsed, "Cumulative reward:", rewards, "Hypervolume", hypervolume, "Coverage Set", coverageSet, file = f)
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
