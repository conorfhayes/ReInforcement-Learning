# Dynamic Economic Emissions Dispatch Problem implemenation with Q-learning and Deep Q-Learning (DQN).
# Code to be updated



from __future__ import print_function
import numpy as np
import math
from sklearn import preprocessing
from keras.layers import (Dense)
from keras.models import Model, Sequential
from random import randint
import time, os, fnmatch, shutil
from collections import Counter, deque
from random import seed
import random
import pandas as pd
from keras.optimizers import Adam


import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras import initializers
sys.stderr = stderr
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import tensorflow as tf
from keras import regularizers
from keras.backend.tensorflow_backend import tf
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)




class Agent:

    qTable = []
    actions = []

    UHolder = [[1] for y in range(14)]

    UHolder[0] = [150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    UHolder[1] = [135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    UHolder[2] = [73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.038, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80]
    UHolder[3] = [60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50]
    UHolder[4] = [73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    UHolder[5] = [57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    UHolder[6] = [20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    UHolder[7] = [47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    UHolder[8] = [20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30]
    UHolder[9] = [10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30]

    PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

    def agent(self, numActions, alpha, gamma, epsilon, id):
        self.UHolder
        self.id = int(id)
        self.numActions = 101
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.P1M_array = []
        self.selectedActions = []
        self.previousActions = []
        self.powerArray = []
        self.returnState = 0
        self.qValues = []
        self.previousStates = []
        self.currentState = 0
        self.action = 0
        self.maxQ = 0
        self.stateVector = 0
        self.action_holder = []
        self.action_ = []
        self.dReward = 0
        self.P1M_array_D = []
        self.Pnm = []
        self.previousAgentCost = 0
        self.previousAgentEmissions = 0
        self.previousAgentPower = 0
        self.P1M_Minus_D = 0
        self.P1M_Minus = 0
        self.replay_memory = deque(maxlen=2000)
        self.counter = 0
        self.learningRate = 0.0001
        self.create_model(self.learningRate)
        self.power = 0
        self.deltaDemands = []
        self.minDeltaDemand = sys.float_info.max
        self.maxDeltaDemand = sys.float_info.min
        self.demandRange = 0
        self.demandOffset = 0
        self.genRanges = []
        self.genOffsets = []
        self.calculateNumStatesActionsMARL()
        self.qTable = self.initialiseQvalue((self.demandRange*101) + 1000, 101)
        self.numStates = self.demandRange*100

        return self

    def decayEpsilon(self):
        self.epsilon = self.epsilon * 0.995
        return self.epsilon

    def decayAlpha(self):
        self.alpha = self.alpha * 0.99
        return self.alpha

    def getAgentPower(self):

        return self.power

    def setAgentPower(self, power):

        self.power = power

    def create_model(self, learningRate):

        loss = tf.keras.losses.Huber(delta=100.0)

        self.model = Sequential()
        self.model.add(Dense(24,activation="relu",kernel_initializer=keras.initializers.VarianceScaling(scale=2), input_shape=(1,)))
        self.model.add(Dense(128,kernel_initializer=keras.initializers.VarianceScaling(scale=2), activation="relu"))
        self.model.add(Dense(101,kernel_initializer=keras.initializers.VarianceScaling(scale=2), activation="linear"))
        self.model.compile(loss = loss, optimizer= SGD(lr=learningRate), metrics=['accuracy'])

        self.target_model = Sequential()
        self.target_model.add(Dense(24,kernel_initializer=keras.initializers.VarianceScaling(scale=2), activation="relu", input_shape=(1,)))
        self.target_model.add(Dense(128,kernel_initializer=keras.initializers.VarianceScaling(scale=2), activation="relu"))
        self.target_model.add(Dense(101,kernel_initializer=keras.initializers.VarianceScaling(scale=2), activation="linear"))
        self.target_model.compile(loss = loss, optimizer= SGD(lr=learningRate), metrics=['accuracy'])


        self.target_model.set_weights(self.model.get_weights())


    def train(self):
        miniBatch = random.sample(self.replay_memory, 64)

        for index, (currentState, action, reward, nextState, done) in enumerate(miniBatch):

            target = self.model.predict([currentState])

            future_qs_list = self.target_model.predict([nextState])

            max_future_q = np.max(future_qs_list)
            target_f = reward + (self.gamma * max_future_q)
            target[0][action] = target_f

            self.model.fit([currentState], target, epochs= 1, verbose=0)

    def setDifferenceReward(self, reward):
        self.dReward = reward

    def calculateNumStatesActionsMARL(self):
        self.deltaDemands.append(0)

        for i in range(1, len(self.PDM_hold)):
            self.deltaDemands.append(self.PDM_hold[i] - self.PDM_hold[i - 1])

        for i in range(1, len(self.deltaDemands)):
            if self.deltaDemands[i] < self.minDeltaDemand:
                self.minDeltaDemand = self.deltaDemands[i]
            if self.deltaDemands[i] > self.maxDeltaDemand:
                self.maxDeltaDemand = self.deltaDemands[i]

        self.demandRange = 1 + self.maxDeltaDemand + abs(self.minDeltaDemand)
        self.demandOffset = abs(self.minDeltaDemand)

        self.genRanges.append((self.UHolder[0][1] - self.UHolder[0][0]) / 1 + 1)
        self.genOffsets.append(int(self.UHolder[0][0]))

        for j in range(1, 10):
            a = ((self.UHolder[j][1] - self.UHolder[j][0]) / 1 + 1)
            self.genRanges.append(a)
            self.genOffsets.append(int(self.UHolder[j][0]))



    def getDifferenceReward(self):

        return self.dReward

    def setP1M_MinusD(self, P1M_Minus_D):
        self.P1M_Minus_D = P1M_Minus_D

    def getP1M_MinusD(self):
        return self.P1M_Minus_D

    def setP1M_Minus(self, P1M_Minus):
        self.P1M_Minus = P1M_Minus

    def getP1M_Minus(self):
        return self.P1M_Minus

    def initialiseQvalue(self, numStates, numActions):
        self.qTable = np.zeros((numStates, numActions))
        return self.qTable

    def getAgentID(self):

        return self.id

    def updateQTable(self,previousState, selectedAction, currentState, reward, agent):
        oldQ = self.qTable[previousState][selectedAction]
        maxQ = self.getMaxQValue(agent, currentState)
        #maxQ = 0
        newQ = oldQ + self.alpha * (reward + self.gamma * maxQ - oldQ)
        self.qTable[previousState][selectedAction] = newQ
        return self

    def saveAction(self, action):
        self.action = action

        return self.action

    def saveMaxQ(self, maxQ):
        self.maxQ = maxQ

        return self.action

    def getMaxQ(self):

        return self.action

    def getAction(self):

        return self.action

    def getPreviousAgentCost(self):
        return self.previousAgentCost

    def setPreviousAgentCost(self, cost):
        self.previousAgentCost = cost

        return self.previousAgentCost

    def getPreviousAgentPower(self):
        return self.previousAgentPower

    def setPreviousAgentPower(self, power):
        self.previousAgentPower = power

    def getPreviousAgentEmissions(self):
        return self.previousAgentEmissions

    def setPreviousAgentEmissions(self, emissions):
        self.previousAgentEmissions = emissions

    def create_bins(self, lower_bound, width, quantity):
        bins = []
        for low in range(lower_bound,
                         lower_bound + quantity * width + 1, width):
            bins.append((low, low + width))
        return bins



    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)



    def selectAction(self,state,  agent, minAllowedAction, maxAllowedAction):
        #check = random.uniform(0,100)
        check = random.uniform(0,1)
        #print(check)
        if check < 0:
            #print(check)
            selectedAction = self.selectrandomAction(minAllowedAction, maxAllowedAction)
        else:
            selectedAction = self.getSelectedAction(state, agent, minAllowedAction, maxAllowedAction)

        return selectedAction

    def selectrandomAction(self, minAllowedAction, maxAllowedAction):
        self.randomAction = round(randint(minAllowedAction,maxAllowedAction))

        return self.randomAction

    def saveCurrentState(self, currentState):

        self.currentState = currentState
        return self.currentState

    def getState(self):

        return self.currentState

    def getStateFromXY(self, state, basesForStateNo):

        numStates = basesForStateNo[0] * basesForStateNo[1]

        stateNo = 0
        for i in range(len(state)):
            stateNo = stateNo * basesForStateNo[i] + state[i]

        return stateNo

    def getStateMARL(self, hour, agent, power_):
        state = 0

        if hour > 1 and hour <= 24:
            deltaDemand = self.deltaDemands[hour - 1] + self.demandOffset
            power = power_ - self.genOffsets[agent.getAgentID() - 1] * 101 / (1 * (self.genRanges[agent.getAgentID() - 1]- 1))
            state = self.getStateFromXY([deltaDemand, power], [self.demandRange, 102])

        return state

    def getSelectedAction(self, state,agent, minAllowedAction, maxAllowedAction):
        doubleValue = []
        maxDV = 0
        selectedAction = -1
        maxQ = -(sys.float_info.max)

        for action in range(minAllowedAction, maxAllowedAction):
            if agent.qTable[state][action] > maxQ:
                selectedAction = action
                maxQ = agent.qTable[state][action]
                doubleValue = []
                doubleValue.append(selectedAction)
                maxDV = 0

            elif self.qTable[state][action] == maxQ:
                maxDV += 1
                doubleValue.append(action)

        if maxDV > 0:
            randomIndex = randint(0, maxDV)
            selectedAction = doubleValue[randomIndex]

        return selectedAction

    def getMaxValuedAction(self, agent, state):
        self.action_holders = []
        action = 0
        while action < self.numActions:
            valueQ = self.qTable[state][action]
            self.action_holders.append(valueQ)
            action = action + 1

        maxActionIndex = self.action_holders.index(max(self.action_holders))

        return maxActionIndex


    def getMaxQValue(self,agent, state):
        maxIndex = self.getMaxValuedAction(agent, state)
        return maxIndex

    def getQTable(self):
        return self.qTable

    def savePower(self, Pn):
        self.powerArray.append(Pn)

    def savePnm(self, Pnm):
        self.Pnm = Pnm

    def getPnm(self):
        return self.Pnm


class Environment():

    global numActions
    global numAgents
    global _agents_
    global gamma
    global epsilon
    global alpha

    numActions = 101

    #numAgents = 42
    epsilon = 1.0
    gamma = 0.75
    alpha = 0.1
    _agents_ = []
    a = Agent()


    def __init__(self):
        self.PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

        self.B =[[0.000049, 0.000014, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017, 0.000018, 0.000019, 0.000020],
         [0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018],
         [0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016],
         [0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015],
         [0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016],
         [0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015],
         [0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018],
         [0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016],
         [0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019],
         [0.000020, 0.000018, 0.000016, 0.000015, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044]]

        self.numActions = 101
        self.epsilon = 1.0
        self.gamma = 0.75
        self.alpha = 0.1
        self.k = 0
        self.t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', self.t)
        self.rewardArray = []
        self.P1M_Array = []
        self.P1M_T_array = []
        self.minBatchSize = 64


    def quadraticEquation(self, a, b, c):

        root1 = (-b + math.sqrt((b ** 2) - 4 * a * c)) / (2 * a)
        root2 = (-b - math.sqrt((b ** 2) - 4 * a * c)) / (2 * a)

        return min(root1, root2)

    def createAgent(self, id):
        agent = Agent()
        agent_ = agent.agent(numActions, alpha, gamma, self.epsilon, id)
        return agent_

    def getP1M(self, Pnm, hour):

        PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                    1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for k in range(0, len(Pnm)):
            sum1 = sum1 + ((self.B[0][k + 1]) * Pnm[k])

        sum1 = (2 * sum1) - 1

        for i in range(0, len(Pnm)):
            for j in range(0, len(Pnm)):
                sum2 = sum2 + (self.B[j + 1][i + 1] * Pnm[i])

            sum3 = sum3 + Pnm[i]

        demand = PDM_hold[hour - 1]
        sum2 = sum2 + demand - sum3

        P1M = self.quadraticEquation(self.B[0][0], sum1, sum2)

        return P1M

    def getCost(self, Pnm, agentID, agent):

        val = agentID - 1
        id = agentID - 2

        cost = agent.UHolder[val][2] + (agent.UHolder[val][3] * Pnm[id]) + (
                    agent.UHolder[val][4] * (Pnm[id] ** 2)) + abs(
            agent.UHolder[val][5] + math.sin(agent.UHolder[val][6] * (agent.UHolder[val][0] - Pnm[id]))
        )
        return cost

    def getP1MCost(self, P1M, agent):

        cost = agent.UHolder[0][2] + (agent.UHolder[0][3] * P1M) + (
                agent.UHolder[0][4] * (P1M ** 2)) + abs(
            agent.UHolder[0][5] + math.sin(agent.UHolder[0][6] * (agent.UHolder[0][0] - P1M))
        )
        return cost


    def getEmissions(self,Pnm, agentID, agent):

        val = agentID - 1
        id = agentID - 2
        emissions = agent.UHolder[val][7] + (agent.UHolder[val][8] * Pnm[id]) + (
                    agent.UHolder[val][9] * (Pnm[id] ** 2)) + (
                            agent.UHolder[val][10] * math.exp(agent.UHolder[val][11] * Pnm[id]))
        emissions = 10 * emissions

        return emissions


    def getP1MEmissions(self, P1M, agent):

        emissions = agent.UHolder[0][7] + (agent.UHolder[0][8] * P1M) + (
                agent.UHolder[0][9] * (P1M ** 2)) + (
                agent.UHolder[0][10] * math.exp(agent.UHolder[0][11] * P1M))

        emissions = emissions * 10

        return emissions


    def calculateGlobalReward(self,x, i, _agents_, Pnm,  PDM, P1M, hour, scalarization):
        costReward = []
        emissionsReward = []
        previousPNM = []

        for agent in _agents_:
            a_id = agent.getAgentID()

            cost = self.getCost(Pnm, a_id, agent)
            costReward.append(cost)

            emissions = self.getEmissions(Pnm, a_id, agent)
            emissionsReward.append(emissions)

            if hour == 1:
                previousPNM.append((agent.UHolder[a_id - 1][1] - agent.UHolder[a_id - 1][0]) / float(2) + agent.UHolder[a_id - 1][0])
            else:
                previousPNM.append(agent.getPreviousAgentPower())

        P1M_cost = self.getP1MCost(P1M, agent)
        costReward.append(P1M_cost)

        P1M_emissions = self.getP1MEmissions(P1M, agent)
        emissionsReward.append(P1M_emissions)

        violationPenalty = self.getViolations(hour, Pnm, previousPNM, agent)

        if scalarization == "hypervolume":
            overallCostReward = -(sum(costReward))
            overallEmissionsReward = -(sum(emissionsReward))
            overallPenalty = -(violationPenalty)

        elif scalarization == "linear":
            overallCostReward = (sum(costReward)) * 0.225
            overallEmissionsReward = (sum(emissionsReward)) * 0.275
            overallPenalty = (violationPenalty) * 0.5

        reward = -(overallEmissionsReward + overallPenalty + overallCostReward)


        return reward, sum(costReward), sum(emissionsReward), overallPenalty

    def OverallReward(self, reward):
        self.rewardArray.append(reward)

    def getOverallReward(self):
        return self.rewardArray

    def setAgentPower(self, power):
        self.power = power


    def getMinAllowedAction(self, agent, minAllowedPosition, id):

        if minAllowedPosition < agent.UHolder[id - 1][0]:
            minAllowedPosition = int(agent.UHolder[id - 1][0] - 0)

        minPosLessOffset = minAllowedPosition - agent.genOffsets[id - 1]
        actionFraction = float(101) / (1 * (agent.genRanges[id - 1]))
        minAllowedAction_D = minPosLessOffset * actionFraction
        minAllowedAction = int(minAllowedAction_D)

        return minAllowedAction

    def getMaxAllowedAction(self, agent, maxAllowedPosition, id):

        if maxAllowedPosition > agent.UHolder[id - 1][1]:
            maxAllowedPosition = int(agent.UHolder[id - 1][1] - 0)

        maxPosLessOffset = maxAllowedPosition - agent.genOffsets[id - 1]
        actionFraction = float(101) / (1 * (agent.genRanges[id - 1]))
        maxAllowedAction_D = maxPosLessOffset * actionFraction
        maxAllowedAction = int(maxAllowedAction_D)

        return maxAllowedAction

    def getPowerDemand(self, hour):

        if hour == 1:
            Pdm_ = 0
        else:
            Pdm_ = self.PDM_hold[hour - 2]

        Pdm = self.PDM_hold[hour - 1]
        Pdm_delta = Pdm - Pdm_
        return Pdm, Pdm_, Pdm_delta

    def getPLM(self, Pnm, PDM, P1M):
        A_array = []
        B_array = []
        x = 0
        n = 2
        j = 2
        while x < len(Pnm):
            _b_ = self.B[0][x+1] * Pnm[x]
            B_array.append(_b_)
            x = x + 1
        B = sum(B_array)
        while n < (len(Pnm) +2):
            while j < (len(Pnm) +2):
                _a_ = (Pnm[n-2] * self.B[n-1][j-1] * Pnm[j-2]) + (2*P1M*B) + (self.B[0][0])*(P1M**2)
                A_array.append(_a_)
                j=j+1
            n=n+1

        PLM = sum(A_array)


        return PLM

    def getViolations(self, hour, Pnm, previousPNM, agent):

        violationMult = 1000000
        violation = 0
        diff = 0
        currentPSlack = self.getP1M(Pnm, hour)

        if hour > 1:
            previousPSlack = self.getP1M(previousPNM, hour - 1)
        else:
            previousPSlack = agent.UHolder[0][0] + (agent.UHolder[0][1] - agent.UHolder[0][0]) / float(2)

        if hour > 1:
            for i in range(0, len(Pnm)):
                diff = abs(Pnm[i] - previousPNM[i])
                if diff > agent.UHolder[i + 1][12]:
                    violation = violation + diff - agent.UHolder[i + 1][12]


        if currentPSlack > agent.UHolder[0][1]:
            violation = violation + currentPSlack - agent.UHolder[0][1]

        elif currentPSlack < agent.UHolder[0][0]:
            violation = violation + abs(currentPSlack - agent.UHolder[0][0])

        if hour > 1:
            diff = abs(currentPSlack - previousPSlack)

            if diff > agent.UHolder[0][12]:
                violation = violation + diff - agent.UHolder[0][12]

        if violation > 0:
            return (violation + 1) * violationMult
        else:
            return violation


    def getPNM(self, action, agent):

        val = agent.getAgentID() - 1
        PNM = action * 1 * (agent.genRanges[val] - 1) / 101 + agent.genOffsets[val]

        return PNM

    def calculateLocalReward(self,x, i, _agents_, Pnm, PDM, P1M, hour, agentID, scalarization):
        costReward = []
        emissionsReward = []
        previousPNM = []

        for agent in _agents_:
            a_id = agent.getAgentID()

            if hour == 1:
                previousPNM.append((agent.UHolder[a_id - 1][1] - agent.UHolder[a_id - 1][0]) / float(2) + agent.UHolder[a_id - 1][0])
            else:
                previousPNM.append(agent.getPreviousAgentPower())

            cost = self.getCost(Pnm, a_id, agent)
            costReward.append(cost)

            emissions = self.Emissions(Pnm, a_id, agent)
            emissionsReward.append(emissions)

        P1M_cost = self.getP1MCost(P1M, agent)
        costReward.append(P1M_cost)

        P1M_emissions = self.getP1MEmissions(P1M, agent)
        emissionsReward.append(P1M_emissions)

        cost = costReward[agentID.getAgentID() - 2]
        emissions = emissionsReward[agentID.getAgentID() - 2]

        violationPenalty = self.getViolations(hour, Pnm, previousPNM, agent)

        if scalarization == "hypervolume":
            reward = -(cost + emissions)

        elif scalarization == "linear":
            reward = -((cost * 0.225) + (emissions * 0.275))

        return reward, sum(costReward), sum(emissionsReward), violationPenalty


    def getMaxValueIndex(self, numbers, minAllowed, maxAllowed):

        actionHolder = []
        for i in range(minAllowed, maxAllowed):
            actionHolder.append(numbers[0][i])

        maxValueIndex = np.argmax(actionHolder) + minAllowed

        return maxValueIndex

    def calculateDifferenceReward(self, _agents_, Pnm, P1M, hour, scalarization, agent):
        costReward = []
        emissionsReward = []

        costReward_D = []
        emissionsReward_D = []

        previousPNM = []
        _PNM_ = []

        for agent in _agents_:
            a_id = agent.getAgentID()

            cost = self.getCost(Pnm, a_id, agent)
            costReward.append(cost)

            emissions = self.getEmissions(Pnm, a_id, agent)
            emissionsReward.append(emissions)

            if hour == 1:
                previousPNM.append((agent.UHolder[a_id - 1][1] - agent.UHolder[a_id - 1][0]) / float(2) + agent.UHolder[a_id - 1][0])
            else:
                previousPNM.append(agent.getPreviousAgentPower())

        previousAgentPower = previousPNM[agent.getAgentID() - 2]

        _PNM_ = Pnm[:]

        _PNM_[agent.getAgentID() - 2] = previousAgentPower



        for agent in _agents_:
            a_id = agent.getAgentID()

            cost_D = self.getCost(_PNM_, a_id, agent)
            costReward_D.append(cost_D)

            emissions_D = self.getEmissions(_PNM_, a_id, agent)
            emissionsReward_D.append(emissions_D)

        P1M_cost = self.getP1MCost(P1M, agent)
        costReward.append(P1M_cost)

        P1M_emissions = self.getP1MEmissions(P1M, agent)
        emissionsReward.append(P1M_emissions)

        P1M_D = self.getP1M(_PNM_, hour)

        P1M_D_cost = self.getP1MCost(P1M_D, agent)
        costReward_D.append(P1M_D_cost)
        P1M_D_emissions = self.getP1MEmissions(P1M_D, agent)
        emissionsReward_D.append(P1M_D_emissions)

        violationPenalty = self.getViolations(hour, Pnm, previousPNM, agent)
        violationPenalty_D = self.getViolations(hour, _PNM_, previousPNM, agent)

        totalCost = sum(costReward) - sum(costReward_D)
        totalEmissions = sum(emissionsReward) - sum(emissionsReward_D)
        totalViolations = violationPenalty - violationPenalty_D


        if scalarization == "hypervolume":
            reward = -(totalCost * totalEmissions * totalViolations)

        elif scalarization == "linear":
            reward = -((totalCost * 0.225) + (totalEmissions * 0.275) + (totalViolations * 0.5))

        return reward, sum(costReward), sum(emissionsReward), violationPenalty

    def timeStep(self, _agents_, j, rewardType, scalarization):
        hour = 1
        b = 0
        done = 0
        costTotal = []
        emissionTotal = []
        rewardTotal = []
        violationsTotal = []

        while hour <= 24:
            Pnm = []
            b = b+1
            self.k = self.k + 1
            CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
            for agent in _agents_:

                val = agent.getAgentID() - 1
                if hour == 1:
                    previousAgentPower = (agent.UHolder[val][1] - agent.UHolder[val][0]) / 2 + agent.UHolder[val][0]
                else:
                    previousAgentPower = agent.getPreviousAgentPower()

                currentState = agent.getStateMARL(hour, agent, previousAgentPower)
                currentStateArray = [currentState/100000000]

                minAllowedPosition = int(previousAgentPower - agent.UHolder[val][13])
                maxAllowedPosition = int(previousAgentPower + agent.UHolder[val][12])

                minAllowedAction = self.getMinAllowedAction(agent, minAllowedPosition, agent.getAgentID())
                maxAllowedAction = self.getMaxAllowedAction(agent, maxAllowedPosition, agent.getAgentID())

                check = random.uniform(0, 1)
                self.epsilon = 0.0
                if check < self.epsilon:
                    action = int(agent.selectrandomAction(minAllowedAction, maxAllowedAction - 1))
                else:
                    #action = self.getMaxValueIndex(agent.model.predict(np.array(currentStateArray)), minAllowedAction, maxAllowedAction)
                    action = agent.selectAction(currentState, agent, minAllowedAction, maxAllowedAction)

                Pn = self.getPNM(action, agent)
                Pnm.append(Pn)
                agent.saveAction(action)
                agent.setAgentPower(Pn)

            i = 0

            P1M = self.getP1M(Pnm, hour)
            #print(P1M)

            for agent in _agents_:

                action = agent.getAction()

                if rewardType == "Global":
                    reward, cost, emissions, violations = self.calculateGlobalReward(j, b, _agents_, Pnm,
                                                                                     CurrentPDM, P1M,
                                                                                     hour, scalarization)

                if rewardType == "Difference":
                    reward, cost, emissions,violations = self.calculateDifferenceReward(_agents_, Pnm, P1M, hour, scalarization, agent)

                if rewardType == "Local":
                    reward, cost, emissions, violations = self.calculateLocalReward(j, b, _agents_, Pnm,
                                                                             CurrentPDM, P1M,
                                                                             hour, agent, scalarization)

                if agent.getAgentID() == 2 and hour == 2:
                #    print("Hour 2 ::")
                #    print("Output :: ", agent.model.predict(np.array(currentStateArray)))
                #    print("Previous Agent Power :: ", agent.getPreviousAgentPower())
                #    print("Agent Power :: ", agent.getAgentPower())
                    #print("Reward :: ", reward)
                #    print("Current State :: ", previousState)
                #    print("Next State :: ", nextState)
                    print("Action :: ", action)

                val = agent.getAgentID() - 1
                if hour == 1:
                    previousAgentPower = (agent.UHolder[val][1] - agent.UHolder[val][0]) / 2 + agent.UHolder[val][0]
                else:
                    previousAgentPower = agent.getPreviousAgentPower()

                previousState = agent.getStateMARL(hour, agent, previousAgentPower)
                nextState = agent.getStateMARL(hour + 1, agent, agent.getAgentPower())

                agent.updateQTable(previousState, action, nextState, reward, agent)

                agent.setPreviousAgentPower(agent.getAgentPower())
                #agent.replay_memory.append([previousState/100000000, action, reward/1000000000000, nextState/100000000, done])

                i = i + 1


            emissionTotal.append(emissions)
            costTotal.append(cost)
            rewardTotal.append(reward)
            violationsTotal.append(violations)
            self.epsilon = self.epsilon * 0.9995
            hour += 1

        for agent in _agents_:
            agent.counter += 1

            #if agent.counter == 5:
            #    #print("*** Updating Weights ***")
            #    agent.target_model.set_weights(agent.model.get_weights())
            #    agent.counter = 0

            #if len(agent.replay_memory) > (self.minBatchSize):
            #    agent.train()



        totalCost = sum(costTotal)
        totalEmissions = sum(emissionTotal)
        totalReward = sum(rewardTotal)
        totalViolations = sum(violationsTotal)


        if self.epsilon < 0.1:
            self.epsilon = 0.0


        return totalCost, totalEmissions, totalReward, totalViolations



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def computeAverage(array):
    newArray = []
    for i in array:
        average = (sum(i)) / 200
        #print("Values: ", i)
        #print("Average: ", average)
        newArray.append(average)

    return newArray

def main():
    numEpisodes = 20000
    numAgents = 9
    _agentsGlobal_ = []
    global fileName

    inc = 1
    costArraySum = [0] * numEpisodes
    emissionsArraySum = [0] * numEpisodes
    violationsArraySum = [0] * numEpisodes

    outputData_Cost = pd.DataFrame()
    outputData_Emissions = pd.DataFrame()
    outputData_Violations = pd.DataFrame()


    while inc <= 1:
        j = 1
        env = Environment()

        starter = 1
        _agents_ = []
        _agentsDifference_ = []
        _agentsLocal_ = []
        costArray, emissionsArray, rewardArray, violationsArray = [], [], [], []

        while starter <= numAgents:
            agentRun = env.createAgent(starter + 1)
            starter = starter + 1
            _agents_.append(agentRun)

        print("*************** Run " + str(inc) + " ***************")
        while j <= numEpisodes:
            print("Episode:", j)
            rewardType = "Difference"
            scalarisation = "linear"
            cost, emissions, reward, violations = env.timeStep(_agents_, j, rewardType, scalarisation)

            costArray.append(cost)
            emissionsArray.append(emissions)
            rewardArray.append(reward)
            violationsArray.append(violations)

            cost = cost / 1000000
            emissions = emissions / 1000000
            violations = violations / 1000000
            print ("Cost :: ", str(cost))
            print ("Emissions :: ", str(emissions))
            print ("Violations :: " , str(violations))
            print("Epsilon :: ", str(env.epsilon))
            print(" ")

            j = j + 1

        rewardTypeCol = rewardType + str(inc)
        outputData_Cost[rewardTypeCol] = costArray
        outputData_Emissions[rewardTypeCol] = emissionsArray
        outputData_Violations[rewardTypeCol] = violationsArray


        costArraySum = [x + y for x, y in zip(costArraySum, costArray)]
        emissionsArraySum = [x + y for x, y in zip(emissionsArraySum, emissionsArray)]
        violationsArraySum = [x + y for x, y in zip(violationsArraySum, violationsArray)]
        inc = inc + 1

    myInt = 1
    outAvgCost = [y / myInt for y in costArraySum]
    outAvgViolations = [y / myInt for y in violationsArraySum]
    outAvgEmissions = [y / myInt for y in emissionsArraySum]

    scaleAvgCost = [j / 1000000 for j in outAvgCost]
    scaleAvgEmissions = [j / 1000000 for j in outAvgEmissions]
    scaleAvgViolations = [j / 1000000 for j in outAvgViolations]

    rewardCost_ = pd.DataFrame({rewardTypeCol + " (+)": scaleAvgCost})
    rewardEmissions_ = pd.DataFrame({rewardTypeCol + " (+)": scaleAvgEmissions})
    rewardViolations_ = pd.DataFrame({rewardTypeCol + " (+)": scaleAvgViolations})
    # rewardEmissions = pd.DataFrame({'global': AverageViolationsGlobal, 'difference': AverageViolationsDifference})

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    rewardCost_.to_csv(r'DEED_Problem_Cost_Result_Global' + timestamp + '.csv')
    rewardEmissions_.to_csv(r'DEED_Problem_Emissions_Result_Global' + timestamp + '.csv')
    rewardViolations_.to_csv(r'DEED_Problem_Violations_Result_Global' + timestamp + '.csv')

    outputData_Cost.to_csv(r'DEED_Linear_Problem_Cost_Result_Global_All_' + timestamp + '.csv')
    outputData_Emissions.to_csv(r'DEED_Linear_Problem_Emissions_Result_Global_All_' + timestamp + '.csv')
    outputData_Violations.to_csv(r'DEED_Linear_Problem_Violations_Result_Global_All_' + timestamp + '.csv')



if __name__ == "__main__":
    main()
