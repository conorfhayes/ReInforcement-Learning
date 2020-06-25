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
register(
    id='SpaceTraders-v0',
    entry_point='spacetraders:SpaceTraders',
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
        self.num_actions = 3
        self.childChanceRewards = {}
        self.rewards = []
        self.probabilities = []
        self.childChanceRewards = {0 : [], 1 : [], 2 : [], 3 : []}
        self.file = file
        self.node = decisionNode("Null", 0, 0, self.args, [0,0], False, 0, False)
        self.root = self.node
        self.CR = [0,0]
        self.health = 0
        self.state = 0
        self.cr = 0
        self.distTable = {}
        for x in range(3):
            self.distTable.update({x : {'count' : 0}})
            for y in range(3):
                self.distTable[x].update({y : {'count' : 0}}) 
                for z in range(3):
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

            if node.isleaf == True:                
                node, cumulative_reward = self.expand(node, cumulative_reward)
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

    def runSimulate(self, node, cumulative_reward):

        state, reward, done, __ = self.env.step(node.state, node.action)

        self.updateDistributionTable(node.state, node.action, reward, state)
        flag = 0

        for _node_ in node.children:
            comparison = reward == _node_.reward
            if comparison.all():
                if _node_.done == done:
                    n = _node_
                    flag = 1


    
        """
        for _node_ in node.parent.children:
            if _node_.action == node.action:                      
                if any((reward == q).all() for q in node.parent.childChanceRewards[node.action]):
                    comparison = reward == _node_.reward
                    if comparison.all():
                        if _node_.done == done:
                            #a += 1    
                            flag = 1

        """
    
        if flag == 0:                      
            _node_ = node.createChild(node, state, node.action, reward, done, reward, True)            
            #node.parent.childChanceRewards[node.action].append(reward)
            _node_.timeRewardReceived += 1
            n = _node_



        return n, cumulative_reward + reward

    def thompsonSampling(self, node, cumulative_reward):
        n = 1000
        samples = [0] * 3
        _samples_ = -sys.maxsize
        bestNode = 0
        rewards = [[] for x in range(3)]
        probabilities = [[] for x in range(3)]
        #action = 0
        for child in node.children:
            _l_ = [0] * 2
            if len(child.rewards) == 0:
                return child            

            for children in child.children:
                #print("Action", child.action, file = self.file)
                #print("Child Data", children._data_, file = self.file)
                for a in children._data_:
                    #print("Child Data", children._data_, file = self.file)

                    rewards[child.action].append(children._data_[a]['reward'] + cumulative_reward)
                    
                    if len(children.children) > 0:
                        probabilities[child.action].append(children._data_[a]['probability'] * (1 / children.numActions))
                    else:
                        probabilities[child.action].append(children._data_[a]['probability'] * (1))

             
        #print("Rewards 0", rewards[0], file = self.file)
        #print("Probabilities 0", probabilities[0], file = self.file)
        #print("Sum 0", sum(probabilities[0]), file = self.file)
        #print(" ", file = self.file)

        #print("Rewards 1", rewards[1], file = self.file)
        #print("Probabilities 1", probabilities[1], file = self.file)
        #print("Sum 1", sum(probabilities[1]), file = self.file)
        #print(" ", file = self.file)

        #print("Rewards 2", rewards[2], file = self.file)
        #print("Probabilities 2", probabilities[2], file = self.file)
        #print("Sum 2", sum(probabilities[2]), file = self.file)
        #print(" ", file = self.file)
        
        samples = [[] for x in range(3)] 
        minVal = -sys.maxsize

        for child in node.children:
            #_sample_ = []
            #print("Round", round(sum(probabilities[child.action]), 1), file = self.file)
            check = round(sum(probabilities[child.action]), 2)
            if check < 1.0 or check > 1.0:
                #print("Hello", file = self.file)
                sample_indices = np.random.choice(len(child.rewards), n) 
                sample = [child.rewards[j] for j in sample_indices]


            else:
                _probabilities_ = [round(num,5) for num in probabilities[child.action]]
                #print("Here",rewards[i], file = self.file)
                #print(1/3, file = self.file)
                sample = random.choices(population = rewards[child.action],weights= _probabilities_, k =n)
            
            _sample_ = [self.TESTscalarize_reward1(j) for j in sample]
            #print("Before", sample[0:10], file = self.file)
            #print("After", _sample_[0:10], file = self.file)
            meanSample = sum(_sample_) / n
            #print("Mean Sample", meanSample, file = self.file)


            child.samples_mean.append(meanSample)

            meanSample_indices = np.random.choice(len(child.samples_mean))
            #print(meanSample_indices, file = self.file)
            randomSample = child.samples_mean[meanSample_indices]
            #print(meanSample, file = self.file)
            #print("Sample Mean :", child.samples_mean, file = self.file)

            if randomSample > minVal:
                minVal = randomSample
                bestNode = child        
        

        return bestNode


    def expand(self, node, cumulative_reward):

        action = random.choice(node.childrenRemaining)
        #self, node, state, action, args, _type_
        #chanceNode = chanceNode(node, node.state, node.action,self.args, 0)
        #print("Expanding", file = self.file)
        chance = node.createChild(node, node.state, action, [0,0], False, 0, False)

        if self.numRewards == 3:

            state, reward, done, health,  __ = self.env.step(node.state, action, node.health)
            child = node.createChild(node, state, action, reward, done, health, False)
        else:
            #print("Node State ", node.state, file = self.file)
            #print("Node Action", action, file = self.file)
            state, reward, done, __ = self.env.step(node.state, action)
            child = chance.createChild(chance, state, action, reward, done, 0, False)

        node.childChanceRewards[action].append(reward)
        node.childrenRemaining.remove(action)

        self.updateDistributionTable(node.state, action, reward, state)

        

        if len(node.children) == self.num_actions:
            node.isleaf = False
        else:
            node.isleaf = True

        return child, cumulative_reward + reward

    def stepsToRoot(self,node):
        step = np.array([0,0])


        if node.done == True:
            node = node.parent

        while node != "Null":
            
            step += node.reward
            node = node.parent

        #print("Step", step, file = self.file)

        return step

    def updateDistributionTable(self, state, action, reward, next_state):
        #print("Dist Table", self.distTable, file = self.file)
        if str(reward) not in self.distTable[state][action][next_state]:
            self.distTable[state][action][next_state].update({str(reward) : {'count' : 0}})

        self.distTable[state]['count'] += 1
        self.distTable[state][action]['count'] += 1
        self.distTable[state][action][next_state]['count'] += 1
        self.distTable[state][action][next_state][str(reward)]['count'] += 1


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

        cumulative_reward = np.array([0,0])
        estProb = 0
        numActions = 3
        done = False
        state = node.state
        prob = 1
        health = 0
        _done_ = False

        if node.done == True:
           
            cumulative_reward += node.reward
            probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']



        else:
            a = 0  

            _node_ = node
            while _done_ == False:

                action = random.randint(0, 2)
                
                if self.numRewards == 3:
                    next_state, reward, done, health, __ = self.env.step(state, action, health)
                else:
                    next_state, reward, done, __ = self.env.step(state, action)


                self.updateDistributionTable(state, action, reward, next_state)

                cumulative_reward += reward

                if len(cumulative_reward) > 2:
                    if cumulative_reward[1] <= -100:
                        cumulative_reward[1] = -100
                        done == True

                prob = self.distTable[state][action][next_state][str(reward)]['count'] / self.distTable[state][action]['count']

                if a == 0:
                    probability = prob
                    a += 1
                else:
                    probability = probability * prob 

                state = next_state
                _done_ = done

                a += 1

            if len(cumulative_reward) > 2:

                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

        #self.distributionBackPropogation(node, cumulative_reward, probability)
        self.backPropogation(node, cumulative_reward, probability)


        return

    def distributionBackPropogation(self, node, cumulative_reward, probability):
        
        a = 0
        move = 0
        oldNode = "Null"
        #print("Node Data", node._data_, file = self.file)
        flag = False
        distRewards = []
        distProps = []
        if node.done == True:
            node.timesVisited += 1
            node.parent.timesActionTaken[node.action] += 1
            #cumulative_reward = cumulative_reward
            node.rewards.append(cumulative_reward)
            strCumulativeReward = str(cumulative_reward)
            #node.samples_mean.append(cumulative_reward)

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * probability
            node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
            #print("Data", node._data_, file = self.file)

            if str(cumulative_reward) not in node._data_:
                node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
            else:
                node._data_[str(cumulative_reward)]['probability'] = probability
                node._data_[str(cumulative_reward)]['scaled probability'] = probability * (1 / node.numActions)

            #print("Hello **************", file = self.file)

            oldNode = node
            node = node.parent
            move += 1


            if node.type == "chance" and flag == False:
                #print("Data", node._data_, file = self.file)
                node.distback_props = []
                node.distback_rewards = []
                for reward in node._data_:
                    #print("Parent", node.parent.parent, file = self.file)
                    print("Cumulative Reward", cumulative_reward, file = self.file)
                    print("Check ", node._data_[str(reward)]['reward'], file = self.file)
                    print("Check Data ", node._data_, file = self.file)
                    node.distback_rewards.append(node._data_[str(reward)]['reward'])
                    node.distback_props.append(node._data_[str(reward)]['probability'])

                distRewards = node.distback_rewards
                distProps = node.distback_props
                flag = True

            
                
                #print("Chance Check ", node._data_, file = self.file)
                #print("Cumulative Reward", cumulative_reward, file = self.file)

        

        while node != self.root.parent:

            cumulative_reward = cumulative_reward + node.reward

            if node.parent == self.root and node.type == "chance":
                #print("Hello!", file = self.file)
                for child in node.children:
                    #print("Number of Children", len(node.children), file = self.file)
                    if child != oldNode:
                        #if oldNode.done == True and child.done == True:
                            #print("Rewards ", distRewards, file = self.file)
                            #print("Type", child.type, file = self.file)
                            #print("Data", child._data_, file = self.file)
                            #print(" ", file = self.file)
                        #else:
                        #print("Node State", node.state, file = self.file)
                        #print("Rewards ", distRewards, file = self.file)
                        #for reward in distRewards:

                        node.distributionalBackpropDict.update({str(cumulative_reward) : {'rewards' : distRewards.copy()}})
                        if move > 1:
                            #print("Data", child._data_, file = self.file)
                            for el in child._data_:
                                res = not child._data_[el]
                                #print(res, file = self.file)
                                if res == True:
                                    #print("Why??????", child._data_[el], file = self.file)

                                    pass
                                else:
                                    for reward in child._data_[el]: 
                                        #print("Checker",child._data_[el][reward], file = self.file)
                                        reward_update = node.distributionalBackpropDict[str(cumulative_reward)]['rewards']
                                        reward_update.append(child._data_[el][reward]['reward'])
                                        #print("RU", reward_update, file = self.file)
                                        node.distributionalBackpropDict.update({str(cumulative_reward) : {'rewards' : reward_update}})
                                #print("Checker", child._data_[el], file = self.file)
                        
                        #print("Backprop Distribution ", node.distributionalBackpropDict, file = self.file)
                        print(" ", file = self.file)


            #print("Before", distRewards, file= self.file)
            
            a = 0      

            

            if node.parent == "Null":
                pass
            else:

                node.parent.timesActionTaken[node.action] += 1
                if node.type == "chance":
                    prob = 1
                else:
                    prob = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
                    node.setProbability(prob)

                for each in distRewards:
                    distRewards[a] = distRewards[a] + node.reward
                    distProps[a] *= prob
                    a += 1

                probability = prob * probability
                #probability = probability / 3
                node.parent.timesActionTaken[node.action] += 1
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.timesVisited += 1           

            node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * probability

            if str(cumulative_reward) not in node._data_:
                #print("Before", node._data_, file = self.file)
                node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
                #print("After", node._data_, file = self.file)
            else:
                node._data_[str(cumulative_reward)]['probability'] = probability #/ node.numActions
                node._data_[str(cumulative_reward)]['scaled probability'] = probability * (1 / node.numActions)
            
            oldNode = node
            node = node.parent
            if node != "Null":
                node.distback_props = distProps
                node.distback_rewards = distRewards
            move += 1
        
            
        return
    

    def backPropogation(self, node, cumulative_reward, probability):
        
        a = 0
        #print("Node Data", node._data_, file = self.file)
        if node.done == True:
            node.timesVisited += 1
            node.parent.timesActionTaken[node.action] += 1
            #cumulative_reward = cumulative_reward
            node.rewards.append(cumulative_reward)
            strCumulativeReward = str(cumulative_reward)
            #node.samples_mean.append(cumulative_reward)

            #node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * probability
            node.probability = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']


            if str(cumulative_reward) not in node._data_:

                if node.parent.parent == self.root:
                    node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability} })
                else:
                    node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
            else:

                node._data_[str(cumulative_reward)]['probability'] = probability

                if node.parent.parent == self.root:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability
                else:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability * (1 / node.numActions)
            
            node = node.parent
        

        while node != self.root.parent:

            cumulative_reward = cumulative_reward + node.reward
            #

            if node.parent == "Null":
                pass
            else:

                node.parent.timesActionTaken[node.action] += 1
                if node.type == "chance":
                    prob = 1
                else:
                    prob = self.distTable[node.parent.state][node.action][node.state][str(node.reward)]['count'] / self.distTable[node.parent.state][node.action]['count']
                    node.setProbability(prob)
                    #print("Prob Update", node.probability, file = self.file)

                probability = prob * probability
                #probability = probability / 3
                node.parent.timesActionTaken[node.action] += 1
            
            if len(cumulative_reward) > 2:
                if cumulative_reward[1] <= -100:
                    cumulative_reward[1] = -100

            node.timesVisited += 1           

            node.rewards.append(cumulative_reward)
            node.probabilities.append(probability)
            node.rollUCT += node.scalarize_reward(cumulative_reward) * probability

            if str(cumulative_reward) not in node._data_:
                if node.parent == "Null":
                    node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
                elif node.parent.parent == self.root:
                    node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability} })
                else:
                    node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
            
                #node._data_.update({str(cumulative_reward) : {'probability' : probability, 'reward' : cumulative_reward, 'scaled probability' : probability * (1 / node.numActions)} })
            else:
                node._data_[str(cumulative_reward)]['probability'] = probability

                if node.parent == "Null":
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability * (1 / node.numActions)
                elif node.parent.parent == self.root:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability
                else:
                    node._data_[str(cumulative_reward)]['scaled probability'] = probability * (1 / node.numActions)
            
            #print("Node Data", node._data_, file = self.file)
            #print(" " , file = self.file)
            node = node.parent
        
            
        return
    

    def run(self):

        numActions = 3

        node = self.root
        childrenRewards = [[] for x in range(numActions)]
        childrenProbabilities = [[] for x in range(numActions)]    
        expectedUtility = [0] * 3
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


    def takeAction(self, action):
        
        node = self.root

        if self.numRewards == 3:
            next_state, reward, done, health, __ = self.env.step(node.state, action, node.health)
        else:
            next_state, reward, done, __ = self.env.step(node.state, action)        
        flag = 0

        #node.timesActionTaken[node.action] += 1
        self.updateDistributionTable(node.state, action, reward, next_state)

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

        probability = self.distTable[node.state][action][next_state][str(reward)]['count'] / self.distTable[node.state][action]['count']

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
        self.numActions = 3
        #self.health = health
        self.type = "chance"
        self.parent = node
        self.samples_mean = []
        self.state = state
        self.numActions = 3
        self.distback_rewards = []
        self.distback_props = []
        self.action = action
        self.children = []
        self.reward = [0,0]
        self.isleaf = False
        self.args = args
        self.env = gym.make(self.args.env)
        self.timesVisited = 0
        self.timesActionTaken = {0 : 0, 1 : 0, 2 : 0}
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.distributionalBackpropDict = {}
        #self.chanceRewards.append(reward)
        self.childChanceRewards = {0 : [], 1 : [], 2 : []}
        self._data_ = {}
        #for x in range(3):
        #    self._data_.update({x : {}})

        self.distribution = {}
        
        for x in range(3):
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
        self._num_rewards = 2
        self.rollUCT = 0
        self.expectedUtility = 0
        self.childrenRemaining = [0,1,2]
        self.data = {}
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

    

    def createChild(self, node, state, action, reward, done, health, _type_):
        #node, node.state, action, node.row, node.col
        child = decisionNode(node, state, action, self.args,reward, done, health, _type_)

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
        self.numActions = 3
        self.health = health
        self.parent = node
        self.type = "decision"
        self.state = state
        self.numActions = 3
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
        self.timesActionTaken = {0 : 0, 1 : 0, 2 : 0}
        self.rewards = []
        self.chanceRewards = []
        self.probabilities = []
        self.chanceRewards.append(reward)
        self.childChanceRewards = {0 : [], 1 : [], 2 : []}
        self._data_ = {}


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
        self.childrenRemaining = [0,1,2]
        self.data = {}
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

            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(self._num_rewards)})


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

    def selectAction(self, method, tree, rewards, probabilities, cumulative_rewards, cumulative_probability):

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

            means = [[] for x in range(3)]

            for child in node.children:

                meanSample_indices = np.random.choice(len(child.samples_mean))            
                randomSample = child.samples_mean[meanSample_indices]

                print("Random Sample", randomSample, file = self.debug_file) 

                mean = sum(child.samples_mean) / len(child.samples_mean)
                means[child.action].append(mean)           

                if randomSample > minVal:
                    minVal = randomSample
                    action = child.action

            print("Means", means, file = self.debug_file)
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
        a = 0
        check = random.random()
        #self.epsilon = 1

        while not self.done:
            
            state = env_state            

            action_rewards = []
            action_prob = []                

            self.tree.step(cumulative_rewards)
            tree, testReward , testProbs, expectedUtility, nodes, nodeDist, allData, a, b, scaled_probs = self.tree.run()

            #x, y = self.getXYfromState(state)
            #print("Cumulative Rewards", cumulative_rewards, file = self.debug_file)
            action = self.selectAction("Bootstrap Thompson", tree, testReward, testProbs, cumulative_rewards, cumulative_probability)

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

            env_state, rewards, node, done, probability = self.tree.takeAction(action)

            cumulative_rewards += rewards
            cumulative_probability *= probability
            #print("Cumulative Probability", cumulative_probability, file = self.debug_file)
            self.done = done


            if len(cumulative_rewards) > 2:

                if cumulative_rewards[1] <= -100 or health <= -100:
                    cumulative_rewards[1] = -100
                    self.done = True
                    tree.reset()

            #a += 1

        tree.reset()
        return cumulative_rewards, testReward, testProbs

def main():
    # Parse parameters
    num_runs = 5
    episodes = 2000
    
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
            utility = 0
            avgUtility = 0

            for i in range(episodes):
                rewards, allRewards, allProbabilities = learner.run()

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
