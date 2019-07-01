# Dynamic Economic Emissions Dispatch Problem implemenation with Q-learning.
# A V0.4 of this implementation for my MSC thesis in Data Analytics
# Code to be updated


import numpy as np
from numpy import array
import math
from random import randint
import time, os, fnmatch, shutil
from collections import Counter
import matplotlib.pyplot as plt
from random import seed
import random
from plotnine import *
import pandas as pd


class Agent:

    qTable = []
    actions = []

    U1 = [150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    U2 = [135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    U3 = [73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.038, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80]
    U4 = [60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50]
    U5 = [73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    U6 = [57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    U7 = [20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    U8 = [47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    U9 = [20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30]
    U10 = [10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30]

    PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

    def Agent(self, numStates, numActions, alpha, gamma, epsilon, id):
        self.id = int(id)
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.P1M_array = []
        self.qTable = self.initialiseQvalue(numStates, numActions, 4)
        self.VectorQTable = self.initialiseVectorQvalue(numStates, numActions, 4)
        #self.qTable = qTable
        self.selectedActions = []
        self.previousActions = []
        self.powerArray = []
        self.numStates = numStates
        self.numObjectives = 3
        self.returnState = 0
        self.qValues = []
        self.previousStates = []
        self.currentState = 0
        self.action = 0
        self.maxQ = 0
        self.stateVector = 0
        self.thresholds = self.initialiseThresholds(3, 0, -100, -2700000)
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
        self.cost = 0
        self.emissions = 0
        return self

    def decayEpsilon(self):
        self.epsilon = self.epsilon * 0.995
        return self.epsilon

    def decayAlpha(self):
        self.alpha = self.alpha * 0.99
        return self.alpha

    def setDifferenceReward(self, reward):
        self.dReward = reward

    def getDifferenceReward(self):

        return self.dReward

    def getCost(self):
        return self.cost

    def setCost(self, cost):
        self.cost = cost
        return self.cost

    def getEmissions(self):
        return self.emissions

    def setEmissions(self, emissions):
        self.emissions = emissions
        return self.emissions

    def setP1M_MinusD(self, P1M_Minus_D):
        self.P1M_Minus_D = P1M_Minus_D

    def getP1M_MinusD(self):
        return self.P1M_Minus_D

    def setP1M_Minus(self, P1M_Minus):
        self.P1M_Minus = P1M_Minus

    def getP1M_Minus(self):
        return self.P1M_Minus

    def initialiseQvalue(self, numStates, numActions, numObjectives):
        self.qTable = np.zeros((numStates, numActions, numObjectives))
        return self.qTable

    def initialiseVectorQvalue(self, numStates, numActions, numObjectives):
        self.VectorQTable = np.zeros((numStates, numActions, numObjectives), dtype=object)
        return self.VectorQTable

    def initialiseThresholds(self, numObjectives, objective1Threshold, objective2Threshold, objective3Threshold):
        self.thresholds = np.zeros((3))
        self.thresholds[0] = objective1Threshold
        self.thresholds[1] = objective2Threshold
        self.thresholds[2] = objective3Threshold
        #self.thresholds[3] = objective4Threshold
        return self.thresholds

    def getAgentID(self):

        return self.id


    def updateQTable(self,previousState, selectedAction, currentState, reward, agent, objective, cost, emissions, violation):
        emissions = float(emissions)
        cost = float(cost)
        violation = float(violation)
        oldQ = self.qTable[previousState][selectedAction][objective]
        maxQ = self.getMaxQValue(agent, currentState, objective)
        newQ = oldQ + self.alpha * (reward + self.gamma * maxQ) - oldQ
        self.qTable[previousState][selectedAction][objective] = newQ
        array = []
        #array.append(emissions)
        array.append(violation)
        array.append(cost)
        #array = np.array(array_, object)
        #print("Array: ", array)
        self.VectorQTable[previousState][selectedAction][objective] = array

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

    def find_bin(self, value, bins):
        for i in range(0, len(bins)):
            if bins[i][0] <= value < bins[i][1]:
                return i
        return -1

    def getNextState(self,hour,action, agent):
        base = [10, 10]
        #print(action)
        power = action[-1]
        #print(power)
        #power = action
        currentHour = hour - 1
        nextHour = hour
        if hour == 24:
            PDM = self.PDM_hold[0]
            PDM_ = self.PDM_hold[currentHour]
            PDM_delta = PDM - PDM_
        else:
            PDM = self.PDM_hold[nextHour]
            PDM_ = self.PDM_hold[currentHour]
            PDM_delta = PDM - PDM_

        PDM_rescale = (PDM_delta - (-296)) * (492 - 0) / (196 - (-296)) + 0
        bins_PDM = self.create_bins(lower_bound=0,
                                width=25,
                                quantity=20)

        if agent.getAgentID() == 2:
            power_rescale = ((power - 135) * (100 - 0)) / (470 - 135) + 0
        elif agent.getAgentID() == 3:
            power_rescale = ((power - 73) * (100 - 0)) / (340 - 73) + 0
        elif agent.getAgentID() == 4:
            power_rescale = ((power - 60) * (100 - 0)) / (300 - 60) + 0
        elif agent.getAgentID() == 5:
            power_rescale = ((power - 73) * (100 - 0)) / (243 - 73) + 0
        elif agent.getAgentID() == 6:
            power_rescale = ((power - 57) * (100 - 0)) / (160 - 57) + 0
        elif agent.getAgentID() == 7:
            power_rescale = ((power - 20) * (100 - 0)) / (130 - 20) + 0
        elif agent.getAgentID() == 8:
            power_rescale = ((power - 47) * (100 - 0)) / (120 - 47) + 0
        elif agent.getAgentID() == 9:
            power_rescale = ((power - 20) * (100 - 0)) / (80 - 20) + 0
        elif agent.getAgentID() == 10:
            power_rescale = ((power - 10) * (100 - 0)) / (55 - 10) + 0

        #print(power_rescale)
        bins_power = self.create_bins(lower_bound=0,
                                    width=5,
                                    quantity=20)

        bin_index1 = self.find_bin(PDM_rescale, bins_PDM)
        bin_index2 = self.find_bin(power_rescale, bins_power)
        currentState = [bin_index1, bin_index2]
        res = 0
        i = 0
        while i < 2:
            res = res * base[i] + currentState[i]
            i += 1

        state = res
        #print(state)
        return state


    def selectAction(self,hour, state, agent, objective):
        #check = random.uniform(0,100)
        check = random.uniform(0,1)
        #print(check)
        if check < self.epsilon:
            #print(check)
            selectedAction = self.selectrandomAction()
        elif objective == 0:
            selectedAction = self.getSelectedAction(hour, state, agent, objective)
        else:
            selectedAction = self.selectTLOAction(hour, state, agent, objective)

        return selectedAction

    def selectrandomAction(self):
        self.randomAction = round(randint(0,100))

        return self.randomAction

    def saveCurrentState(self, currentState):

        self.currentState = currentState
        return self.currentState

    def getState(self):

        return self.currentState

    def getSelectedAction(self,hour,state,agent, objective):
        action = 0

        self.action_holder = []
        self.action_ = []

        previousPowerOutput = agent.powerArray[-1]

        while action < self.numActions:
            testAction = Environment.getPNM(self, action, agent)

            if hour == 1:
                valueQ = agent.qTable[state][int(action)][objective]
                agent.action_holder.append(valueQ)
                agent.action_.append(action)

            elif agent.getAgentID() == 2:
                if testAction - previousPowerOutput <= self.U2[12] and previousPowerOutput - testAction <= self.U2[13]:
                        valueQ = agent.qTable[state][int(action)][objective]
                        agent.action_holder.append(valueQ)
                        agent.action_.append(action)

            elif agent.getAgentID() == 3:
                if testAction - previousPowerOutput <= self.U3[12] and previousPowerOutput - testAction <= self.U3[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 4:
                if testAction - previousPowerOutput <= self.U4[12] and previousPowerOutput - testAction <= self.U4[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 5:
                if testAction - previousPowerOutput <= self.U5[12] and previousPowerOutput - testAction <= self.U5[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 6:
                if testAction - previousPowerOutput <= self.U6[12] and previousPowerOutput - testAction <= self.U6[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 7:
                if testAction - previousPowerOutput <= self.U7[12] and previousPowerOutput - testAction <= self.U7[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 8:
                if testAction - previousPowerOutput <= self.U8[12] and previousPowerOutput - testAction <= self.U8[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 9:
                if testAction - previousPowerOutput <= self.U9[12] and previousPowerOutput - testAction <= self.U9[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 10:
                if testAction - previousPowerOutput <= self.U10[12] and previousPowerOutput - testAction <= self.U10[13]:
                        valueQ = self.qTable[state][int(action)][objective]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            action = action + 1
        maxIndex = self.action_holder.index(max(self.action_holder))
        maxActionIndex = self.action_[maxIndex]
        return int(maxActionIndex)

    def compare(self, a, b, thresholds, objective):
        #print("A: ", a)
        #print("B: ", b)
        #print("Threshold: ", thresholds[objective])

        thresholdA = min(a, thresholds[objective])
        thresholdB = min(b, thresholds[objective])

        #print("Threshold A: ", thresholdA)
        #print("Threshold B: ", thresholdB)

        if thresholdA > thresholdB:
            return 1
        elif thresholdA < thresholdB:
            return -1

        return 0

    def selectTLOAction(self, hour, state, agent, objective):
        action = 0

        self.action_holder = []
        self.action_ = []

        previousPowerOutput = agent.powerArray[-1]

        while action < self.numActions:
            testAction = Environment.getPNM(self, action, agent)
            if objective == 0:
                objective_ = objective
            else:
                objective_ = objective - 1

            if objective == 1:
                vectorObjective = 0
            elif objective == 2:
                vectorObjective = 1
            elif objective == 3:
                vectorObjective = 2

            if agent.getAgentID() == 2:
                if testAction - previousPowerOutput <= self.U2[12] and previousPowerOutput - testAction <= self.U2[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            #print(valueQ)
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 3:
                if testAction - previousPowerOutput <= self.U3[12] and previousPowerOutput - testAction <= self.U3[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 4:
                if testAction - previousPowerOutput <= self.U4[12] and previousPowerOutput - testAction <= self.U4[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 5:
                if testAction - previousPowerOutput <= self.U5[12] and previousPowerOutput - testAction <= self.U5[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 6:
                if testAction - previousPowerOutput <= self.U6[12] and previousPowerOutput - testAction <= self.U6[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 7:
                if testAction - previousPowerOutput <= self.U7[12] and previousPowerOutput - testAction <= self.U7[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 8:
                if testAction - previousPowerOutput <= self.U8[12] and previousPowerOutput - testAction <= self.U8[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 9:
                if testAction - previousPowerOutput <= self.U9[12] and previousPowerOutput - testAction <= self.U9[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            elif agent.getAgentID() == 10:
                if testAction - previousPowerOutput <= self.U10[12] and previousPowerOutput - testAction <= self.U10[13]:
                        valueQ = self.VectorQTable[state][int(action)][objective_]
                        if valueQ == 0:
                            self.action_holder.append(valueQ)
                            self.action_.append(action)
                        else:
                            valueQ = valueQ[vectorObjective]
                            self.action_holder.append(valueQ)
                            self.action_.append(action)

            action = action + 1
        maxIndex = self.action_holder.index(max(self.action_holder))
        maxActionIndex = self.action_[maxIndex]
        a = 0
        bestAction = []
        bestAction.append(0)
        while a < len(self.action_holder):
            #print(" ")
            #print("*************************************************")
            #print("Length of Action Holder: ", len(self.action_holder))
            #print("Action Holder: ", self.action_holder)
            index = bestAction[0]
            compareResult = self.compare(self.action_holder[a], self.action_holder[index], self.thresholds, objective)
            #print("Action Holder A: ", self.action_holder[a])
            #print("Aciton Index: ",self.action_holder[bestAction.index(0)])
            #print("Compare Result: ", compareResult)
            #print(self.action_[a])
            if compareResult > 0:
                bestAction = []
                bestAction.append(a)
            elif compareResult == 0:
                bestAction.append(a)
            elif compareResult < 0:
                bestAction.append(a)
            #print("A: ", a)
            #print("Best Action: ", bestAction)
            a = a + 1
        #print()

        if len(bestAction) > 1:
            return self.action_[bestAction[0]]
        elif len(bestAction) == 1:
            return self.action_[bestAction[0]]
        #return bestAction[0]

    def getMaxValuedAction(self, agent, state, objective):
        self.action_holders = []
        action = 0
        while action < self.numActions:
            valueQ = self.qTable[state][action][objective]
            self.action_holders.append(valueQ)
            action = action + 1

        maxActionIndex = self.action_holders.index(max(self.action_holders))

        return maxActionIndex

    def find_nearest(self, array, value):
        if len(array) == 1:
            return 0
        else:
            array = array[array != 0]
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
        return array[idx]


    def getMaxQValue(self,agent, state, objective):
        maxIndex = self.getMaxValuedAction(agent, state, objective)
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
    numEpisodes = 10000

    #numAgents = 42
    epsilon = 0.05
    gamma = 0.75
    alpha = 0.1
    _agents_ = []
    a = Agent()

    B = [[0.000049, 0.000015, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017, 0.000018, 0.000019, 0.000020],
         [0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018],
         [0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016],
         [0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015],
         [0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016],
         [0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015],
         [0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018],
         [0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016],
         [0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019],
         [0.000020, 0.000018, 0.000016, 0.000014, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044]]


    PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]
    U1 = [150, 470, 786.7998, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    U2 = [135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80]
    U3 = [73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.038, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80]
    U4 = [60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50]
    U5 = [73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    U6 = [57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50]
    U7 = [20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    U8 = [47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30]
    U9 = [20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30]
    U10 = [10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30]

    def __init__(self):
        self.numActions = 7
        self.numEpisodes = 10000
        self.epsilon = 0.05
        self.gamma = 0.75
        self.alpha = 0.1
        self.k = 0
        self.t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', self.t)
        self.rewardArray = []
        self.P1M_Array = []
        self.P1M_T_array = []


    def createAgent(self, numStates, id):
        agent = Agent()
        agent_ = agent.Agent(numStates, numActions, alpha, gamma, self.epsilon, id)
        return agent_

    def getP1M(self, Pnm, currentPDM):
        b_array = []
        c_array = []
        n = 2
        a = self.B[0][0]
        while n <= 10:
            holder = (self.B[0][n-1] * Pnm[n-2])
            b_array.append(holder)
            n = n + 1

        b = (2 * sum(b_array) - 1)
        #print("PNM: ", Pnm)
        nn = 2
        jj = 2
        while nn <= 10:
            while jj <= 10:
                cholder = (Pnm[nn-2] * self.B[nn-1][jj-1] * Pnm[nn-2])
                c_array.append(cholder)
                jj = jj + 1
            nn = nn + 1

        sumPNM = sum(Pnm)
        sumC_array = sum(c_array)

        c = currentPDM + sumC_array - sumPNM
        d = (b ** 2) - (4 * a * c)
        totalPLM = (-b - math.sqrt(d))/(2*a)

        return totalPLM



    def calculateGlobalReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour, scalarization, objective):
        costReward = []
        emissionsReward = []

        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2
            if agent.getAgentID() == 2:
                cost = self.U2[2] + (self.U2[3] * (Pnm[id])) + (self.U2[4] * (Pnm[id] ** 2)) + abs(
                    self.U2[5] * math.sin(self.U2[6] * (self.U2[0]-Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 3:
                cost = self.U3[2] + (self.U3[3] * (Pnm[id])) + (self.U3[4] * (Pnm[id] ** 2)) + abs(
                    self.U3[5] * math.sin(self.U3[6] * (self.U3[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 4:
                cost = self.U4[2] + (self.U4[3] * (Pnm[id])) + (self.U4[4] * (Pnm[id] ** 2)) + abs(
                    self.U4[5] * math.sin(self.U4[6] * (self.U4[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 5:
                cost = self.U5[2] + (self.U5[3] * (Pnm[id])) + (self.U5[4] * (Pnm[id] ** 2)) + abs(
                    self.U5[5] * math.sin(self.U5[6] * (self.U5[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 6:
                cost = self.U6[2] + (self.U6[3] * (Pnm[id])) + (self.U6[4] * (Pnm[id] ** 2)) + abs(
                    self.U6[5] * math.sin(self.U6[6] * (self.U6[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 7:
                cost = self.U7[2] + (self.U7[3] * (Pnm[id])) + (self.U7[4] * (Pnm[id] ** 2)) + abs(
                    self.U7[5] * math.sin(self.U7[6] * (self.U7[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 8:
                cost = self.U8[2] + (self.U8[3] * (Pnm[id])) + (self.U8[4] * (Pnm[id] ** 2)) + abs(
                    self.U8[5] * math.sin(self.U8[6] * (self.U8[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 9:
                cost = self.U9[2] + (self.U9[3] * (Pnm[id])) + (self.U9[4] * (Pnm[id] ** 2)) + abs(
                    self.U9[5] * math.sin(self.U9[6] * (self.U9[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 10:
                cost = self.U10[2] + (self.U10[3] * (Pnm[id])) + (self.U10[4] * (Pnm[id] ** 2)) + abs(
                    self.U10[5] * math.sin(self.U10[6] * (self.U10[0] - Pnm[id])))
                costReward.append(cost)

            if agent.getAgentID() == 2:
                E = 10
                #print("PNM 2:", Pnm[id])
                eqn2 = self.U2[7] + (self.U2[8] * Pnm[id]) + (self.U2[9] * (Pnm[id]**2)) + (
                            self.U2[10] * math.exp(self.U2[11] * Pnm[id]))
                emissions = E * eqn2
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 3:
                E = 10
                eqn3 = self.U3[7] + (self.U3[8] * Pnm[id]) + (self.U3[9] * (Pnm[id]**2)) + (
                            self.U3[10] * math.exp(self.U3[11] * Pnm[id]))
                emissions = E * eqn3
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 4:
                E = 10
                eqn4 = self.U4[7] + (self.U4[8] * Pnm[id]) + (self.U4[9] * (Pnm[id]**2)) + (
                            self.U4[10] * math.exp(self.U4[11] * Pnm[id]))
                emissions = E * eqn4
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 5:
                E = 10
                eqn5 = self.U5[7] + (self.U5[8] * Pnm[id]) + (self.U5[9] * (Pnm[id]**2)) + (
                            self.U5[10] * math.exp(self.U5[11] * Pnm[id]))
                emissions = E * eqn5
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 6:
                E = 10
                eqn6 = self.U6[7] + (self.U6[8] * Pnm[id]) + (self.U6[9] * (Pnm[id]**2)) + (
                            self.U6[10] * math.exp(self.U6[11] * Pnm[id]))
                emissions = E * eqn6
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 7:
                E = 10
                eqn7 = self.U7[7] + (self.U7[8] * Pnm[id]) + (self.U7[9] * (Pnm[id]**2)) + (
                            self.U7[10] * math.exp(self.U7[11] * Pnm[id]))
                emissions = E * eqn7
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 8:
                E = 10
                eqn8 = self.U8[7] + (self.U8[8] * Pnm[id]) + (self.U8[9] * (Pnm[id]**2)) + (
                            self.U8[10] * math.exp(self.U8[11] * Pnm[id]))
                emissions = E * eqn8
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 9:
                E = 10
                eqn9 = self.U9[7] + (self.U9[8] * Pnm[id]) + (self.U9[9] * (Pnm[id]**2)) + (
                            self.U9[10] * math.exp(self.U9[11] * Pnm[id]))
                emissions = E * eqn9
                #emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 10:
                E = 10
                eqn10 = self.U10[7] + (self.U10[8] * Pnm[id]) + (self.U10[9] * (Pnm[id]**2)) + (
                            self.U10[10] * math.exp(self.U10[11] * Pnm[id]))
                emissions = E * eqn10
                #emissions = eqn
                emissionsReward.append(emissions)

        P1M_cost = self.U1[2] + (self.U1[3] * (P1M)) + (self.U1[4] * (P1M ** 2)) + abs(
            self.U1[5] * math.sin(self.U1[6] * (self.U1[0] - P1M)))
        costReward.append(P1M_cost)

        E = 10
        eqn_ = self.U1[7] + (self.U1[8] * P1M) + (self.U1[9] * (P1M ** 2)) + (
                self.U1[10] * (math.exp(self.U1[11] * P1M)))
        P1M_emissions = E * eqn_
        #P1M_emissions = eqn_
        emissionsReward.append(P1M_emissions)

        # 1,000,000
        C = 100000
        if P1M > 470:
            h1 = P1M - 470
        elif P1M < 150:
            h1 = 150 - P1M
        else:
            h1 = 0


        if hour == 1 and x == 1:
            P1M_minus = 0
            self.P1M_Array.append(P1M)
        else:
            P1M_minus = self.P1M_Array[-1]
            self.P1M_Array.append(P1M)
        #print(P1M)
        #print(P1M_minus)
        #print(self.P1M_Array)
        if hour == 1 and x == 1:
            h2 = 0
        elif (P1M - P1M_minus) > 80:
            #print("H2 is true 1")
            h2 = (P1M - P1M_minus) - 80
        elif (P1M - P1M_minus) < (-80):
            #print("H2 is true 2")
            h2 = (P1M - P1M_minus) + 80
        else:
            h2 = 0

        if hour ==1 and x == 1:
            violationPenalty = 0
        elif h1 != 0 and h2 == 0:
            violationPenalty = (abs(h1 + 1)*self.U1[11]) * C
        elif h1 == 0 and h2 != 0:
            violationPenalty = (abs(h2 + 1)*self.U1[11]) * C
        elif h1 == 0 and h2 == 0:
            violationPenalty = 0
        elif h1 !=0 and h2 !=0:
            violationPenalty = (C * ((abs(h1 + 1)*self.U1[11]))) + (C * ((abs(h2 + 1) * self.U1[11])))

        if scalarization == "hypervolume":
            if objective == 1:
                overallCostReward = -(sum(costReward))
                overallEmissionsReward = -(sum(emissionsReward))
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            elif objective == 2:
                overallCostReward = -(sum(costReward))
                overallEmissionsReward = -(sum(emissionsReward))
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            elif objective ==3:
                overallCostReward = -(sum(costReward))
                overallEmissionsReward = -(sum(emissionsReward))
                overallPenalty = -violationPenalty
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            else:
                overallCostReward = -(sum(costReward))
                overallEmissionsReward = -(sum(emissionsReward))
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)

        elif scalarization == "linear":
            if objective == 1:
                overallCostReward = -(sum(costReward)) * 0.225
                overallEmissionsReward = -(sum(emissionsReward)) * 0.275
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            elif objective == 2:
                overallCostReward = -(sum(costReward)) * 0.225
                overallEmissionsReward = -(sum(emissionsReward)) * 0.275
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            elif objective ==3:
                overallCostReward = -(sum(costReward)) * 0.225
                overallEmissionsReward = -(sum(emissionsReward)) * 0.275
                overallPenalty = -violationPenalty
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)
            else:
                overallCostReward = -(sum(costReward)) * 0.225
                overallEmissionsReward = -(sum(emissionsReward)) * 0.275
                overallPenalty = -(violationPenalty)
                reward = (overallEmissionsReward + overallPenalty + overallCostReward)

        #reward = (overallEmissionsReward + overallPenalty + overallCostReward)

        fileName = ("DEED_Problem_Global_Reward_" + scalarization + "Scalarization_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(x) + ' ***********************'
        line2 = '***************** Hour: ' + str(i) + ' ***********************'
        line3 = "Reward: " + str(reward)
        line4 = "Power: " + str(PDM)
        line5 = "Agents Power: " + str(sum(Pnm))
        line6 = "Slack Generator Power: " + str(P1M)
        line7 = "Overall Power Output: " + str(sum(Pnm)+ P1M)
        line8 = " Agents Power: " + str(Pnm)
        line9 = "Cost: " + str(sum(costReward))
        line10 = "Emissions: " + str(sum(emissionsReward))
        line11 = "Violation Penalty: " + str(violationPenalty)
        line12 = "State: " + str(type)
        line13 = " "

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1,
            line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13))
        out.close()

        return reward, sum(costReward), sum(emissionsReward), -(violationPenalty),  sum(costReward), sum(emissionsReward)

    def OverallReward(self, reward):
        self.rewardArray.append(reward)

    def getOverallReward(self):
        return self.rewardArray

    def getPowerDemand(self, hour):

        if hour - 1 == 0:
            Pdm_ = self.PDM_hold[23]
        else:
            Pdm_ = self.PDM_hold[hour - 2]

        Pdm = self.PDM_hold[hour-1]
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


    def getPNM(self, action_, agent):

        if agent.getAgentID() == 2:
            PNM = self.U2[0] + (action_*((self.U2[1] - self.U2[0])/100))

        elif agent.getAgentID() == 3:
            PNM = self.U3[0] + (action_*((self.U3[1] - self.U3[0])/100))

        elif agent.getAgentID() == 4:
            PNM = self.U4[0] + (action_*((self.U4[1] - self.U4[0])/100))

        elif agent.getAgentID() == 5:
            PNM = self.U5[0] + (action_*((self.U5[1] - self.U5[0])/100))

        elif agent.getAgentID() == 6:
            PNM = self.U6[0] + (action_*((self.U6[1] - self.U6[0])/100))

        elif agent.getAgentID() == 7:
            PNM = self.U7[0] + (action_*((self.U7[1] - self.U7[0])/100))

        elif agent.getAgentID() == 8:
            PNM = self.U8[0] + (action_*((self.U8[1] - self.U8[0])/100))

        elif agent.getAgentID() == 9:
            PNM = self.U9[0] + (action_*((self.U9[1] - self.U9[0])/100))

        elif agent.getAgentID() == 10:
            PNM = self.U10[0] + (action_ * (self.U10[1] - self.U10[0]) / 100)

        return PNM

    def calculateLocalReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour, agentID, scalarization):
        costReward = []
        emissionsReward = []
        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2

            if agent.getAgentID() == 2:
                cost = self.U2[2] + (self.U2[3] * (Pnm[id])) + (self.U2[4] * (Pnm[id] ** 2)) + abs(
                    self.U2[5] * math.sin(self.U2[6] * (self.U2[0]-Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 3:
                cost = self.U3[2] + (self.U3[3] * (Pnm[id])) + (self.U3[4] * (Pnm[id] ** 2)) + abs(
                    self.U3[5] * math.sin(self.U3[6] * (self.U3[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 4:
                cost = self.U4[2] + (self.U4[3] * (Pnm[id])) + (self.U4[4] * (Pnm[id] ** 2)) + abs(
                    self.U4[5] * math.sin(self.U4[6] * (self.U4[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 5:
                cost = self.U5[2] + (self.U5[3] * (Pnm[id])) + (self.U5[4] * (Pnm[id] ** 2)) + abs(
                    self.U5[5] * math.sin(self.U5[6] * (self.U5[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 6:
                cost = self.U6[2] + (self.U6[3] * (Pnm[id])) + (self.U6[4] * (Pnm[id] ** 2)) + abs(
                    self.U6[5] * math.sin(self.U6[6] * (self.U6[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 7:
                cost = self.U7[2] + (self.U7[3] * (Pnm[id])) + (self.U7[4] * (Pnm[id] ** 2)) + abs(
                    self.U7[5] * math.sin(self.U7[6] * (self.U7[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 8:
                cost = self.U8[2] + (self.U8[3] * (Pnm[id])) + (self.U8[4] * (Pnm[id] ** 2)) + abs(
                    self.U8[5] * math.sin(self.U8[6] * (self.U8[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 9:
                cost = self.U9[2] + (self.U9[3] * (Pnm[id])) + (self.U9[4] * (Pnm[id] ** 2)) + abs(
                    self.U9[5] * math.sin(self.U9[6] * (self.U9[0] - Pnm[id])))
                costReward.append(cost)

            elif agent.getAgentID() == 10:
                cost = self.U10[2] + (self.U10[3] * (Pnm[id])) + (self.U10[4] * (Pnm[id] ** 2)) + abs(
                    self.U10[5] * math.sin(self.U10[6] * (self.U10[0] - Pnm[id])))
                costReward.append(cost)

            if agent.getAgentID() == 2:
                E = 10
                eqn2 = self.U2[7] + (self.U2[8] * Pnm[id]) + (self.U2[9] * (Pnm[id]**2)) + (
                            self.U2[10] * math.exp(self.U2[11] * Pnm[id]))
                emissions = E * eqn2
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 3:
                E = 10
                eqn3 = self.U3[7] + (self.U3[8] * Pnm[id]) + (self.U3[9] * (Pnm[id]**2)) + (
                            self.U3[10] * math.exp(self.U3[11] * Pnm[id]))
                emissions = E * eqn3
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 4:
                E = 10
                eqn4 = self.U4[7] + (self.U4[8] * Pnm[id]) + (self.U4[9] * (Pnm[id]**2)) + (
                            self.U4[10] * math.exp(self.U4[11] * Pnm[id]))
                emissions = E * eqn4
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 5:
                E = 10
                eqn5 = self.U5[7] + (self.U5[8] * Pnm[id]) + (self.U5[9] * (Pnm[id]**2)) + (
                            self.U5[10] * math.exp(self.U5[11] * Pnm[id]))
                emissions = E * eqn5
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 6:
                E = 10
                eqn6 = self.U6[7] + (self.U6[8] * Pnm[id]) + (self.U6[9] * (Pnm[id]**2)) + (
                            self.U6[10] * math.exp(self.U6[11] * Pnm[id]))
                emissions = E * eqn6
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 7:
                E = 10
                eqn7 = self.U7[7] + (self.U7[8] * Pnm[id]) + (self.U7[9] * (Pnm[id]**2)) + (
                            self.U7[10] * math.exp(self.U7[11] * Pnm[id]))
                emissions = E * eqn7
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 8:
                E = 10
                eqn8 = self.U8[7] + (self.U8[8] * Pnm[id]) + (self.U8[9] * (Pnm[id]**2)) + (
                            self.U8[10] * math.exp(self.U8[11] * Pnm[id]))
                emissions = E * eqn8
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 9:
                E = 10
                eqn9 = self.U9[7] + (self.U9[8] * Pnm[id]) + (self.U9[9] * (Pnm[id]**2)) + (
                            self.U9[10] * math.exp(self.U9[11] * Pnm[id]))
                emissions = E * eqn9
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 10:
                E = 10
                eqn10 = self.U10[7] + (self.U10[8] * Pnm[id]) + (self.U10[9] * (Pnm[id]**2)) + (
                            self.U10[10] * math.exp(self.U10[11] * Pnm[id]))
                emissions = E * eqn10
                emissionsReward.append(emissions)

            P1M_cost = self.U1[2] + (self.U1[3] * (P1M)) + (self.U1[4] * (P1M ** 2)) + abs(
                self.U1[5] * math.sin(self.U1[6] * (self.U1[0] - P1M)))
            costReward.append(P1M_cost)

            E = 10
            eqn_ = self.U1[7] + (self.U1[8] * P1M) + (self.U1[9] * (P1M ** 2)) + (
                    self.U1[10] * (math.exp(self.U1[11] * P1M)))
            P1M_emissions = E * eqn_
            emissionsReward.append(P1M_emissions)

        #print("Cost: ", cost)
        #print("Emissions: ", emissions)
        if scalarization == "hypervolume":
            reward = -(cost + emissions)

        elif scalarization == "linear":
            reward = -(cost * 0.225) + (emissions * 0.275)

        fileName = ("DEED_Problem_Local_Reward_" + scalarization + "_Scalarization_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(x) + ' ***********************'
        line2 = '***************** Hour: ' + str(i) + ' ***********************'
        line3 = "Reward: " + str(reward)
        line4 = "Power: " + str(PDM)
        line5 = "Agents Power: " + str(sum(Pnm))
        line6 = "Slack Generator Power: " + str(P1M)
        line7 = "Overall Power Output: " + str(sum(Pnm)+ P1M)
        line8 = " Agents Power: " + str(Pnm)
        line9 = "Cost: " + str(cost)
        line10 = "Emissions: " + str(emissions)
        line11 = "State: " + str(type)
        line12 = " "

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1,
            line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12))
        out.close()
        return reward, sum(costReward), sum(emissionsReward)

    def calculateDifferenceReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour,agentID, scalarization):
        costReward = []
        emissionsReward = []
        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2

            if agent.getAgentID() == 2:
                cost = self.U2[2] + (self.U2[3] * (Pnm[id])) + (self.U2[4] * (Pnm[id] ** 2)) + abs(
                    self.U2[5] * math.sin(self.U2[6] * (self.U2[0]-Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 3:
                cost = self.U3[2] + (self.U3[3] * (Pnm[id])) + (self.U3[4] * (Pnm[id] ** 2)) + abs(
                    self.U3[5] * math.sin(self.U3[6] * (self.U3[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 4:
                cost = self.U4[2] + (self.U4[3] * (Pnm[id])) + (self.U4[4] * (Pnm[id] ** 2)) + abs(
                    self.U4[5] * math.sin(self.U4[6] * (self.U4[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 5:
                cost = self.U5[2] + (self.U5[3] * (Pnm[id])) + (self.U5[4] * (Pnm[id] ** 2)) + abs(
                    self.U5[5] * math.sin(self.U5[6] * (self.U5[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 6:
                cost = self.U6[2] + (self.U6[3] * (Pnm[id])) + (self.U6[4] * (Pnm[id] ** 2)) + abs(
                    self.U6[5] * math.sin(self.U6[6] * (self.U6[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 7:
                cost = self.U7[2] + (self.U7[3] * (Pnm[id])) + (self.U7[4] * (Pnm[id] ** 2)) + abs(
                    self.U7[5] * math.sin(self.U7[6] * (self.U7[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 8:
                cost = self.U8[2] + (self.U8[3] * (Pnm[id])) + (self.U8[4] * (Pnm[id] ** 2)) + abs(
                    self.U8[5] * math.sin(self.U8[6] * (self.U8[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 9:
                cost = self.U9[2] + (self.U9[3] * (Pnm[id])) + (self.U9[4] * (Pnm[id] ** 2)) + abs(
                    self.U9[5] * math.sin(self.U9[6] * (self.U9[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            elif agent.getAgentID() == 10:
                cost = self.U10[2] + (self.U10[3] * (Pnm[id])) + (self.U10[4] * (Pnm[id] ** 2)) + abs(
                    self.U10[5] * math.sin(self.U10[6] * (self.U10[0] - Pnm[id])))
                costReward.append(cost)
                agent.setCost(cost)

            if agent.getAgentID() == 2:
                E = 10
                eqn2 = self.U2[7] + (self.U2[8] * Pnm[id]) + (self.U2[9] * (Pnm[id]**2)) + (
                            self.U2[10] * math.exp(self.U2[11] * Pnm[id]))

                emissions = E * eqn2
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 3:
                E = 10
                eqn3 = self.U3[7] + (self.U3[8] * Pnm[id]) + (self.U3[9] * (Pnm[id]**2)) + (
                            self.U3[10] * math.exp(self.U3[11] * Pnm[id]))
                emissions = E * eqn3
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 4:
                E = 10
                eqn4 = self.U4[7] + (self.U4[8] * Pnm[id]) + (self.U4[9] * (Pnm[id]**2)) + (
                            self.U4[10] * math.exp(self.U4[11] * Pnm[id]))
                emissions = E * eqn4
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 5:
                E = 10
                eqn5 = self.U5[7] + (self.U5[8] * Pnm[id]) + (self.U5[9] * (Pnm[id]**2)) + (
                            self.U5[10] * math.exp(self.U5[11] * Pnm[id]))
                emissions = E * eqn5
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 6:
                E = 10
                eqn6 = self.U6[7] + (self.U6[8] * Pnm[id]) + (self.U6[9] * (Pnm[id]**2)) + (
                            self.U6[10] * math.exp(self.U6[11] * Pnm[id]))
                emissions = E * eqn6
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 7:
                E = 10
                eqn7 = self.U7[7] + (self.U7[8] * Pnm[id]) + (self.U7[9] * (Pnm[id]**2)) + (
                            self.U7[10] * math.exp(self.U7[11] * Pnm[id]))
                emissions = E * eqn7
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 8:
                E = 10
                eqn8 = self.U8[7] + (self.U8[8] * Pnm[id]) + (self.U8[9] * (Pnm[id]**2)) + (
                            self.U8[10] * math.exp(self.U8[11] * Pnm[id]))
                emissions = E * eqn8
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 9:
                E = 10
                eqn9 = self.U9[7] + (self.U9[8] * Pnm[id]) + (self.U9[9] * (Pnm[id]**2)) + (
                            self.U9[10] * math.exp(self.U9[11] * Pnm[id]))
                emissions = E * eqn9
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

            elif agent.getAgentID() == 10:
                E = 10
                eqn10 = self.U10[7] + (self.U10[8] * Pnm[id]) + (self.U10[9] * (Pnm[id]**2)) + (
                            self.U10[10] * math.exp(self.U10[11] * Pnm[id]))
                emissions = E * eqn10
                emissionsReward.append(emissions)
                agent.setEmissions(emissions)

        P1M_cost = self.U1[2] + (self.U1[3] * (P1M)) + (self.U1[4] * (P1M ** 2)) + abs(
            self.U1[5] * math.sin(self.U1[6] * (self.U1[0] - P1M)))
        costReward.append(P1M_cost)

        E = 10
        eqn_ = self.U1[7] + (self.U1[8] * P1M) + (self.U1[9] * (P1M ** 2)) + (
                self.U1[10] * (math.exp(self.U1[11] * P1M)))
        P1M_emissions = E * eqn_
        emissionsReward.append(P1M_emissions)

        # 1,000,000
        C = 1000000
        if P1M > 470:
            h1 = P1M - 470
        elif P1M < 150:
            h1 = 150 - P1M
        else:
            h1 = 0

        P1M_minus = agentID.getP1M_Minus()

        if (P1M - P1M_minus) > 80:
           h2 = (P1M - P1M_minus) - 80
        elif (P1M - P1M_minus) < (-80):
            h2 = (P1M - P1M_minus) + 80
        else:
            h2 = 0

        agentID.setP1M_Minus(P1M)

        if hour == 1 and i == 1:
            violationPenalty = 0
        elif h1 != 0 and h2 == 0:
            violationPenalty = (abs(h1 + 1)*self.U1[11]) * C
        elif h1 == 0 and h2 != 0:
            violationPenalty = (abs(h2 + 1)*self.U2[11]) * C
        elif h1 == 0 and h2 == 0:
            violationPenalty = 0
        elif h1 !=0 and h2 !=0:
            violationPenalty = (C * ((abs(h1 + 1)*self.U1[11]))) + (C * ((abs(h2 + 1) * self.U2[11])))


        previousAgentCost = agentID.getPreviousAgentCost()
        previousAgentEmissions = agentID.getPreviousAgentEmissions()
        previousAgentPower = agentID.getPreviousAgentPower()

        agent_Cost = (sum(costReward) - agentID.getCost()) + previousAgentCost
        agentID.setPreviousAgentCost(agentID.getCost())
        global_cost = sum(costReward)
        global_emissions = sum(emissionsReward)
        global_penalty = violationPenalty
        G_z = global_cost + global_emissions + global_penalty
        agent_Emissions = (sum(emissionsReward) - agentID.getEmissions()) + previousAgentEmissions
        agentID.setPreviousAgentEmissions(agentID.getEmissions())

        _Pnm_ = Pnm.copy()
        _Pnm_[agentID.getAgentID()-2] = previousAgentPower
        P1M_D = self.getP1M(_Pnm_, PDM)
        agent_power = Pnm[agentID.getAgentID() - 2]
        agentID.setPreviousAgentPower(agent_power)

        C = 1000000
        if P1M_D > 470:
            h1 = P1M_D - 470
        elif P1M_D < 150:
            h1 = 150 - P1M_D
        else:
            h1 = 0

        P1M_minus_D = agentID.getP1M_MinusD()

        if (P1M_D - P1M_minus_D) > 80:
            h2 = (P1M_D - P1M_minus_D) - 80
        elif (P1M_D - P1M_minus_D) < (-80):
            h2 = (P1M_D - P1M_minus_D) + 80
        else:
            h2 = 0

        agentID.setP1M_MinusD(P1M_D)

        if hour == 1 and i == 1:
            violationPenalty_D = 0
        elif h1 != 0 and h2 == 0:
            violationPenalty_D = (abs(h1 + 1) * self.U1[11]) * C
        elif h1 == 0 and h2 != 0:
            violationPenalty_D = (abs(h2 + 1) * self.U1[11]) * C
        elif h1 == 0 and h2 == 0:
            violationPenalty_D = 0
        elif h1 != 0 and h2 != 0:
            violationPenalty_D = (C * ((abs(h1 + 1) * self.U1[11]))) + (C * ((abs(h2 + 1) * self.U1[11])))


        if scalarization == "hypervolume":
            overallCostReward = (sum(costReward) - agent_Cost)
            overallEmissionsReward = (sum(emissionsReward) - agent_Emissions)
            overallPenalty = (violationPenalty - violationPenalty_D)

        elif scalarization == "linear":
            overallCostReward = ((sum(costReward)) - (agent_Cost)) * 0.225
            overallEmissionsReward = ((sum(emissionsReward)) - (agent_Emissions)) * 0.275
            overallPenalty = (violationPenalty) - (violationPenalty_D) * 0.5

        G_z_i = overallCostReward + overallEmissionsReward + overallPenalty
        reward = -(G_z_i)

        fileName = ("DEED_Problem_Difference_Reward_" + scalarization + "Scalarization_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(x) + ' ***********************'
        line2 = '***************** Hour: ' + str(i) + ' ***********************'
        line3 = "Reward: " + str(reward)
        line4 = "Power: " + str(PDM)
        line5 = "Agents Power: " + str(sum(Pnm))
        line6 = "Slack Generator Power: " + str(P1M)
        line7 = "Overall Power Output: " + str(sum(Pnm)+ P1M)
        line8 = " Agents Power: " + str(Pnm)
        line9 = "Cost: " + str(sum(costReward))
        line10 = "Emissions: " + str(sum(emissionsReward))
        line11 = "Violation Penalty: " + str(violationPenalty)
        line12 = "State: " + str(type)
        line13 = " "

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1,
            line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13))
        out.close()

        return reward, sum(costReward), sum(emissionsReward)

    def timeStep(self, _agents_, j, rewardType, scalarization, objective):
        hour = 1
        b = 0
        costTotal = []
        emissionTotal = []
        rewardTotal = []
        while hour <= 24:
            Pnm = []
            b = b+1
            self.k = self.k + 1
            CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
            for agent in _agents_:
                if objective == 0:
                    if hour == 1 and j == 1:
                        action_ = 10
                        Pn = self.getPNM(action_, agent)
                        agent.powerArray.append(Pn)
                        Pnm.append(Pn)
                        agent.saveAction(action_)
                        currentState = 0
                        agent.saveCurrentState(currentState)
                        #print("Hello")

                    else:
                        currentState = agent.getState()
                        action_ = agent.selectAction(hour,currentState,agent, objective)
                        Pn = self.getPNM(action_, agent)
                        agent.powerArray.append(Pn)
                        Pnm.append(Pn)
                        agent.saveAction(action_)
                else:
                    if hour == 1 and j == 1:
                        currentState = 0
                        action_ = agent.selectAction(hour, currentState, agent, objective)
                        Pn = self.getPNM(action_, agent)
                        agent.powerArray.append(Pn)
                        Pnm.append(Pn)
                        agent.saveAction(action_)
                    else:
                        currentState = agent.getState()
                        action_ = agent.selectAction(hour, currentState, agent, objective)
                        Pn = self.getPNM(action_, agent)
                        agent.powerArray.append(Pn)
                        Pnm.append(Pn)
                        agent.saveAction(action_)

            i = 0
            for agent in _agents_:
                agent.savePnm(Pnm)

            P1M = self.getP1M(Pnm, CurrentPDM)
            if rewardType == "Global":
                reward, costReward, emissionsReward, violationReward, cost, emissions = self.calculateGlobalReward(j, b, _agents_, Pnm, currentState, CurrentPDM, P1M,
                                                                     hour, scalarization, objective)
                emissionTotal.append(emissions)
                costTotal.append(cost)
                rewardTotal.append(reward)

            for agent in _agents_:
                previousState = agent.getState()
                action = agent.getAction()
                if rewardType == "Difference":
                    reward, cost, emissions = self.calculateDifferenceReward(j, b, _agents_, Pnm, previousState, CurrentPDM, P1M,
                                                                        hour,agent, scalarization)
                if rewardType == "Local":
                    reward, cost, emissions = self.calculateLocalReward(j, b, _agents_, Pnm, previousState, CurrentPDM, P1M,
                                                                        hour,agent, scalarization)
                currentState = agent.getNextState(hour, agent.powerArray, agent)
                agent.saveCurrentState(currentState)
                #print(objective)
                #if rewardType == "Global" and objective == 0:
                #    reward = violationReward
                    #print("Reward: ", reward)
                    #agent.updateGlobalQTable(previousState, action, currentState, reward, agent, objective)
                #elif rewardType == "Global" and objective == 1:
                #    reward = costReward
                    #agent.updateGlobalQTable(previousState, action, currentState, reward, agent, objective)
                #elif rewardType == "Global" and objective == 2:
                #    reward = emissionsReward
                    #agent.updateGlobalQTable(previousState, action, currentState, reward, agent, objective)

                #print(costReward)
                #print(emissionsReward)
                #print(violationReward)
                agent.updateQTable(previousState, action, currentState, reward, agent, objective, -costReward, -emissionsReward, violationReward)
                i = i + 1

            hour = hour + 1
            if rewardType == "Difference":
                emissionTotal.append(emissions)
                costTotal.append(cost)
                rewardTotal.append(reward)

            if rewardType == "Local":
                emissionTotal.append(emissions)
                costTotal.append(cost)
                rewardTotal.append(reward)

        totalCost = sum(costTotal)
        totalEmissions = sum(emissionTotal)
        totalReward = sum(rewardTotal)


        fileName = ("DEED_Problem_Cost_Emissions" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(j) + ' ***********************'
        line2 = "Reward Type: " + str(rewardType)
        line3 = ("Total Emissions: " + str(totalEmissions))
        line4 = ("Total Cost: " + str(totalCost))
        line5 = ("Total Reward: " + str(totalReward))
        line6 = " "

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6))
        out.close()

        return totalCost, totalEmissions, totalReward


def costGraph(df):
    x_axis = np.array(range(0, len(df.index)))
    print(x_axis)
    df['counter'] = x_axis
    print(df[0:])
    x = df['counter']

    costG = (ggplot(df) +
            geom_line(aes(x='x', y=df['global']),alpha=0.5, size=0.5, color =  'green') +
            geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color='red') +
             geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color='blue') +
            scale_x_continuous(lim = (0, len(x_axis)), breaks= range(0,len(x_axis)+ 5000, 500)) +
            scale_y_continuous(lim = (2.5, max(df['global'])), breaks = np.arange(2.5, max(df['global']) + 0.2, 0.2)) +
            ylab(" Cost ($ x 10^6) ") +
            xlab(" Episode ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)))

def graph(df):
    x_axis = np.array(range(0, len(df.index)))
    print(x_axis)
    df['counter'] = x_axis
    print(df[0:])
    x = df['counter']
    # y = costDataframe

    cost = (ggplot(df) +
             geom_line(aes(x='x', y=df['plot']), alpha=0.5, size=0.5, color='green') +
             scale_x_continuous(lim=(0, len(x_axis)), breaks=range(0, len(x_axis) + 5000, 5000)) +
             scale_y_continuous(lim=(2.5, max(df['plot'])), breaks=np.arange(2.5, max(df['plot']) + 0.2, 0.2)) +
             ylab(" Cost ($ x 10^6) ") +
             xlab(" Episode ") +
             ggtitle(" ") +
             theme_matplotlib() +
             theme(axis_text_y=element_text(size=6)) +
             theme(axis_text_x=element_text(size=6)))

def main():
    numEpisodes = 2000
    numAgents = 9
    _agentsGlobal_ = []
    global fileName

    inc = 1
    costArraySumGlobal = [0] * numEpisodes
    emissionsArraySumGlobal = [0] * numEpisodes

    costArraySumDifference = [0] * numEpisodes
    emissionsArraySumDifference = [0] * numEpisodes

    costArraySumLocal = [0] * numEpisodes
    emissionsArraySumLocal = [0] * numEpisodes

    while inc <= 1:
        j = 1
        envGlobal = Environment()
        envDifference = Environment()
        envLocal = Environment()
        starter = 1
        _agentsGlobal_ = []
        _agentsDifference_ = []
        _agentsLocal_ = []
        costArrayGlobal, costArrayDifference, costArrayLocal = [], [], []
        emissionsArrayGlobal, emissionsArrayDifference, emissionsArrayLocal = [], [], []
        rewardArrayGlobal, rewardArrayDifference, rewardArrayLocal = [], [], []

        while starter <= numAgents:
            agentGlobal = envGlobal.createAgent((350), starter + 1)
            agentDifference = envDifference.createAgent((350), starter + 1)
            agentLocal = envLocal.createAgent((350), starter + 1)
            starter = starter + 1
            _agentsGlobal_.append(agentGlobal)
            _agentsDifference_.append(agentDifference)
            _agentsLocal_.append(agentLocal)
        print("*************** Run " + str(inc) + " ***************")
        objective = 0
        while objective < 3:
            j = 1
            print("******************************* Objective: " + str(objective) + " ***************************************")
            for agent in _agentsGlobal_:
                agent.epsilon = 0.05
                #print(agent.qTable)
            while j <= numEpisodes:
                print("Episode:", j)
                #print("Hello: ", int(objective))
                costGlobal, emissionsGlobal, rewardGlobal = envGlobal.timeStep(_agentsGlobal_, j, "Global", "linear", int(objective))
                #costDifference, emissionsDifference, rewardDifference = envDifference.timeStep(_agentsDifference_, j, "Difference", "linear", int(objective))
                #costLocal, emissionsLocal, rewardLocal = envLocal.timeStep(_agentsLocal_, j,"Local", "linear", int(objective))

                for agent in _agentsGlobal_:
                    agent.decayEpsilon()
                    # agent.decayAlpha()

                #for agent in _agentsDifference_:
                    #agent.decayEpsilon()
                    # agent.decayAlpha()

                # for agent in _agentsLocal_:
                    # agent.decayEpsilon()
                    # agent.decayAlpha()

                costArrayGlobal.append(costGlobal)
                emissionsArrayGlobal.append(emissionsGlobal)
                rewardArrayGlobal.append(rewardGlobal)

                #costArrayDifference.append(costDifference)
                #emissionsArrayDifference.append(emissionsDifference)
                #rewardArrayDifference.append(rewardDifference)

                #costArrayLocal.append(costLocal)
                #emissionsArrayLocal.append(emissionsLocal)
                #rewardArrayLocal.append(rewardLocal)

                j = j + 1
            objective = objective + 1

        costArraySumGlobal = [x + y for x, y in zip(costArraySumGlobal, costArrayGlobal)]
        emissionsArraySumGlobal = [x + y for x, y in zip(emissionsArraySumGlobal, emissionsArrayGlobal)]

        #costArraySumDifference = [x + y for x, y in zip(costArraySumDifference, costArrayDifference)]
        #emissionsArraySumDifference = [x + y for x, y in zip(emissionsArraySumDifference, emissionsArrayDifference)]

        #costArraySumLocal = [x + y for x, y in zip(costArraySumLocal, costArrayLocal)]
        #emissionsArraySumLocal = [x + y for x, y in zip(emissionsArraySumLocal, emissionsArrayLocal)]

        inc = inc + 1
    myInt = 1
    outAvgCostGlobal = [y / myInt for y in costArraySumGlobal]
    #outAvgEmissionsGlobal = [y / myInt for y in emissionsArraySumGlobal]

    #outAvgCostDifference = [x / myInt for x in costArraySumDifference]
    #outAvgEmissionsDifference = [x / myInt for x in emissionsArraySumDifference]

    #outAvgCostLocal = [x / myInt for x in costArraySumLocal]
    # outAvgEmissionsLocal = [x / myInt for x in emissionsArraySumLocal]

    scaleAvgCostGlobal = [j/1000000 for j in outAvgCostGlobal]
    #scaleAvgEmissionsGlobal = [j / 1000000 for j in outAvgEmissionsGlobal]

    #scaleAvgCostDifference = [j / 1000000 for j in outAvgCostDifference]
    #scaleAvgEmissionsDifference = [j / 1000000 for j in outAvgEmissionsDifference]
    #outAvgReward = [y / myInt for y in AvgReward]

    #scaleAvgCostLocal = [j / 1000000 for j in outAvgCostLocal]
    # scaleAvgEmissionsLocal = [j / 1000000 for j in outAvgEmissionsLocal]

    #rewardCost = pd.DataFrame({'global':scaleAvgCostGlobal , 'difference': scaleAvgCostDifference, 'local': scaleAvgCostLocal})
    #rewardEmissions = pd.DataFrame({'global': scaleAvgCostGlobal, 'difference': scaleAvgCostDifference, 'local': scaleAvgCostLocal})

    #print(rewardCost)
    #costGraph(rewardCost)
    reward1 = pd.DataFrame({'plot':scaleAvgCostGlobal})
    #reward2 = pd.DataFrame({'plot': scaleAvgCostDifference})
    #reward3 = pd.DataFrame({'plot': scaleAvgCostLocal})
    graph(reward1)
    #graph(reward2)
    #graph(reward3)
    #emissionsGraph(scaleAvgEmissions)
    #metric(outAvgReward)

if __name__ == "__main__":
    main()
