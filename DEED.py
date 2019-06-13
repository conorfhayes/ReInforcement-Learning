# Dynamic Economic Emissions Dispatch Problem implemenation with Q-learning.
# A V0.1 of this implementation for my MSC thesis in Data Analytics
# Code to be updated



import numpy as np
from numpy import array
import math
from random import randint
import time, os, fnmatch, shutil
from collections import Counter
import matplotlib.pyplot as plt

class Agent:

    qTable = []
    actions = []

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

    PDM_hold = [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776,
                1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184]

    def Agent(self, numStates, numActions, alpha, gamma, epsilon, id):
        self.id = int(id)
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        qTable = self.initialiseQvalue(numStates, numActions)
        self.qTable = qTable
        self.selectedActions = []
        self.previousActions = []
        self.powerArray = []
        self.numStates = numStates
        self.returnState = 0
        self.qValues = []
        self.previousStates = []
        self.currentState = 0
        self.action = 0
        self.maxQ = 0
        self.stateVector = 0
        self.action_holder = []
        self.action_ = []
        return self

    def decayEpsilon(self):
        self.epsilon = self.epsilon * 0.999
        return self.epsilon

    def decayAlpha(self):
        self.alpha = self.alpha * 0.999
        return self.alpha

    def initialiseQvalue(self, numStates, numActions):
        qTable = np.zeros((numStates, 101))
        return qTable

    def getAgentID(self):

        return self.id

    def updateQTable(self,previousState, selectedAction, currentState, reward):
        oldQ = self.qTable[previousState][selectedAction]
        maxQ = self.getMaxQValue(currentState)
        newQ = oldQ + self.alpha * (reward + self.gamma * maxQ) - oldQ
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

    def getNextState(self,hour,action):
        base = [10, 10]

        if hour == 24:
            PDM = self.PDM_hold[1]
            PDM_ = self.PDM_hold[hour-1]
            PDM_delta = PDM - PDM_
            power = action
        else:
            power = action
            PDM = self.PDM_hold[hour]
            PDM_ = self.PDM_hold[hour-1]
            PDM_delta = PDM - PDM_

        PDM_rescale = (PDM_delta - (-296)) * (492 - 0) / (196 - (-296)) + 0
        bins_PDM = self.create_bins(lower_bound=0,
                                width=25,
                                quantity=20)

        bin_index1 = self.find_bin(PDM_rescale, bins_PDM)
        currentState = array([bin_index1, power])

        res = 0
        i = 0
        while i < 2:
            res = res * base[i] + currentState[i]
            i += 1

        state = res

        return state


    def selectAction(self, state, previousPower, agent):
        check = np.random.random()
        if check < self.epsilon:
            selectedAction = self.selectrandomAction()
        else:
            selectedAction = self.getSelectedAction(state, previousPower, agent)

        return selectedAction

    def selectrandomAction(self):
        self.randomAction = round(randint(0,100))

        return self.randomAction

    def saveCurrentState(self, currentState):

        self.currentState = currentState
        return self.currentState

    def getState(self):

        return self.currentState

    def getSelectedAction(self,state, powerArray, agent):
        action = 0

        self.action_holder = []
        self.action_ = []

        previousPowerOutput = powerArray[-1]

        while action < self.numActions:
            testAction = Environment.getPNM(self, action, agent)

            if agent.getAgentID() == 2:
                if testAction - previousPowerOutput <= self.U2[12] and previousPowerOutput - testAction <= self.U2[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 3:
                if testAction - previousPowerOutput <= self.U3[12] and previousPowerOutput - testAction <= self.U3[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 4:
                if testAction - previousPowerOutput <= self.U4[12] and previousPowerOutput - testAction <= self.U4[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 5:
                if testAction - previousPowerOutput <= self.U5[12] and previousPowerOutput - testAction <= self.U5[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 6:
                if testAction - previousPowerOutput <= self.U6[12] and previousPowerOutput - testAction <= self.U6[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 7:
                if testAction - previousPowerOutput <= self.U7[12] and previousPowerOutput - testAction <= self.U7[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 8:
                if testAction - previousPowerOutput <= self.U8[12] and previousPowerOutput - testAction <= self.U8[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 9:
                if testAction - previousPowerOutput <= self.U9[12] and previousPowerOutput - testAction <= self.U9[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            elif agent.getAgentID() == 10:
                if testAction - previousPowerOutput <= self.U10[12] and previousPowerOutput - testAction <= self.U10[13]:
                        valueQ = self.qTable[state][int(action)]
                        self.action_holder.append(valueQ)
                        self.action_.append(action)

            action = action + 1
        maxIndex = self.action_holder.index(max(self.action_holder))
        maxActionIndex = self.action_[maxIndex]

        return int(maxActionIndex)

    def getMaxValuedAction(self, state):
        self.action_holders = []
        action = 0
        #
        while action < self.numActions:
            valueQ = self.qTable[state][action]
            self.action_holders.append(valueQ)
            action = action + 1

        maxActionIndex = self.action_holders.index(max(self.action_holders))

        return int(maxActionIndex)


    def getMaxQValue(self, state):
        maxIndex = self.getMaxValuedAction(state)
        return self.qTable[int(state)][int(maxIndex)]

    def getQTable(self):
        return self.qTable

    def savePower(self, Pn):
        self.powerArray.append(Pn)


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
        self.P1M_array = []

    def createAgent(self, numStates, id):
        agent = Agent()
        agent_ = agent.Agent(numStates, numActions, alpha, gamma, self.epsilon, id)
        return agent_

    def getNextAction(self,action):
        if action == 0:
            nextAction == 0
        elif action == 1:
            nextAction = 1
        elif action == 2:
            nextAction == 2
        elif action == 3:
            nextAction = 3
        elif action == 4:
            nextAction = 4
        elif action == 5:
            nextAction = 5
        elif action == 6:
            nextAction = 6

        return nextAction

    def getPLM(self, Pnm, currentPDM):
        sumOfArray = []
        extraN = 2
        b_array = []
        c_array = []
        while extraN <= 10:
            sumOf = self.B[1][extraN-1] * Pnm[extraN-2]
            sumOfArray.append(sumOf)
            extraN = extraN + 1
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
                cholder = (Pnm[nn-2] * self.B[nn-1][jj-1] * Pnm[nn-2]) #- sum(Pnm)
                c_array.append(cholder)
                jj = jj + 1
            nn = nn + 1

        sumPNM = sum(Pnm)
        sumC_array = sum(c_array)

        c = currentPDM + sumC_array - sumPNM
        d = (b ** 2) - (4 * a * c)
        totalPLM = (-b - math.sqrt(d))/(2*a)

        return totalPLM



    def calculateGlobalReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour):
        costReward = []
        emissionsReward = []
        #print("Pnm: ", Pnm)

        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2
            #print(Pnm)
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
                eqn = self.U2[7] + (self.U2[8] * Pnm[id]) + (self.U2[9] * (Pnm[id]**2)) + (
                            self.U2[10] * math.exp(self.U2[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 3:
                E = 10
                eqn = self.U3[7] + (self.U3[8] * Pnm[id]) + (self.U3[9] * (Pnm[id]**2)) + (
                            self.U3[10] * math.exp(self.U3[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 4:
                E = 10
                eqn = self.U4[7] + (self.U4[8] * Pnm[id]) + (self.U4[9] * (Pnm[id]**2)) + (
                            self.U4[10] * math.exp(self.U4[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 5:
                E = 10
                eqn = self.U5[7] + (self.U5[8] * Pnm[id]) + (self.U5[9] * (Pnm[id]**2)) + (
                            self.U5[10] * math.exp(self.U5[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 6:
                E = 10
                eqn = self.U6[7] + (self.U6[8] * Pnm[id]) + (self.U6[9] * (Pnm[id]**2)) + (
                            self.U6[10] * math.exp(self.U6[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 7:
                E = 10
                eqn = self.U7[7] + (self.U7[8] * Pnm[id]) + (self.U7[9] * (Pnm[id]**2)) + (
                            self.U7[10] * math.exp(self.U7[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 8:
                E = 10
                eqn = self.U8[7] + (self.U8[8] * Pnm[id]) + (self.U8[9] * (Pnm[id]**2)) + (
                            self.U8[10] * math.exp(self.U8[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 9:
                E = 10
                eqn = self.U9[7] + (self.U9[8] * Pnm[id]) + (self.U9[9] * (Pnm[id]**2)) + (
                            self.U9[10] * math.exp(self.U9[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

            elif agent.getAgentID() == 10:
                E = 10
                eqn = self.U10[7] + (self.U10[8] * Pnm[id]) + (self.U10[9] * (Pnm[id]**2)) + (
                            self.U10[10] * math.exp(self.U10[11] * Pnm[id]))
                #emissions = E * eqn
                emissions = eqn
                emissionsReward.append(emissions)

        P1M_cost = self.U1[2] + (self.U1[3] * (P1M)) + (self.U2[4] * (P1M ** 2)) + abs(
            self.U2[5] * math.sin(self.U2[6] * (self.U2[0] - P1M)))
        costReward.append(P1M_cost)

        E = 10
        eqn_ = self.U1[7] + (self.U1[8] * P1M) + (self.U1[9] * (P1M ** 2)) + (
                self.U1[10] * (math.exp(self.U1[11] * P1M)))
        #P1M_emissions = E * eqn_
        P1M_emissions = eqn_
        emissionsReward.append(P1M_emissions)

        # 1,000,000
        C = 1000000
        if P1M > 470:
            h1 = P1M - 470
        elif P1M < 150:
            h1 = 150 - P1M
        else:
            h1 = 0

        if hour == 1:
            self.P1M_array.append(P1M)
            P1M_minus = 0
        elif i == 2:
            P1M_minus = self.P1M_array[-1]
            self.P1M_array.append(P1M)
        else:
            P1M_minus = self.P1M_array[-1]
            self.P1M_array.append(P1M)

        if (P1M - P1M_minus) > 80:
           h2 = (P1M - P1M_minus) - 80
        elif (P1M - P1M_minus) < (-80):
            h2 = (P1M - P1M_minus) + 80
        else:
            h2 = 0

        if h1 != 0 and h2 == 0:
            violationPenalty = (abs(h1 + 1)*self.U1[11]) * C
        elif h1 == 0 and h2 != 0:
            violationPenalty = (abs(h2 + 1)*self.U1[11]) * C
        elif h1 == 0 and h2 == 0:
            violationPenalty = 0
        elif h1 !=0 and h2 !=0:
            violationPenalty = (C * ((abs(h1 + 1)*self.U1[11]))) + (C * ((abs(h2 + 1) * self.U1[11])))


        overallCostReward = -(sum(costReward) * 0.225)
        overallEmissionsReward = -(sum(emissionsReward) * 0.275)
        overallPenalty = -(violationPenalty * 0.5)
        reward = overallEmissionsReward + overallPenalty + overallCostReward

        fileName = ("DEED_Problem_Global_Reward_" + self.timestamp + ".txt")
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


    def calculateLocalReward(self, agent, action_, type, Pnm, j, PDM):


        if type == "Cost":

            if agent.getAgentID() == 2:
                sin_sqrt = math.sqrt(self.U2[1]-self.U2[0])
                val = self.U2[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U2[2] + self.U2[3]*(Pnm) + self.U2[4] * math.sqrt(Pnm) + self.U2[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 3:
                sin_sqrt = math.sqrt(self.U3[1]-self.U3[0])
                val = self.U3[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U3[2] + self.U3[3]*(Pnm) + self.U3[4] * math.sqrt(Pnm) + self.U3[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 4:
                sin_sqrt = math.sqrt(self.U4[1]-self.U4[0])
                val = self.U4[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U4[2] + self.U4[3]*(Pnm) + self.U4[4] * math.sqrt(Pnm) + self.U4[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 5:
                sin_sqrt = math.sqrt(self.U5[1]-self.U5[0])
                val = self.U5[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U5[2] + self.U5[3]*(Pnm) + self.U5[4] * math.sqrt(Pnm) + self.U5[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 6:
                sin_sqrt = math.sqrt(self.U6[1]-self.U6[0])
                val = self.U6[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U6[2] + self.U6[3]*(Pnm) + self.U6[4] * math.sqrt(Pnm) + self.U6[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 7:
                sin_sqrt = math.sqrt(self.U7[1]-self.U7[0])
                val = self.U7[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U7[2] + self.U7[3]*(Pnm) + self.U7[4] * math.sqrt(Pnm) + self.U7[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 8:
                sin_sqrt = math.sqrt(self.U8[1]-self.U8[0])
                val = self.U8[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U8[2] + self.U8[3]*(Pnm) + self.U8[4] * math.sqrt(Pnm) + self.U8[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 9:
                sin_sqrt = math.sqrt(self.U9[1]-self.U9[0])
                val = self.U9[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U9[2] + self.U9[3]*(Pnm) + self.U9[4] * math.sqrt(Pnm) + self.U9[5] * math.asin(tot_sqrt)

            elif agent.getAgentID() == 10:
                sin_sqrt = math.sqrt(self.U10[1]-self.U10[0])
                val = self.U10[6]
                tot_sqrt = sin_sqrt * val
                localReward = self.U10[2] + self.U10[3]*(Pnm) + self.U10[4] * math.sqrt(Pnm) + self.U10[5] * math.asin(tot_sqrt)

        if type == "Emissions":

            if agent.getAgentID() == 2:
                E = 10
                eqn = self.U2[7] + (self.U2[8]*Pnm) + (self.U2[9] * math.sqrt(Pnm)) + (self.U2[10] * math.exp(self.U2[11]*Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 3:
                E = 10
                eqn = self.U3[7] + (self.U3[8]*Pnm) + (self.U3[9]* math.sqrt(Pnm)) + (self.U3[10]* math.exp(self.U3[11]*Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 4:
                E = 10
                eqn = self.U4[7] + (self.U4[8] * Pnm) + (self.U4[9] * math.sqrt(Pnm)) + (self.U4[10] * math.exp(self.U4[11] * Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 5:
                E = 10
                eqn = self.U5[7] + (self.U5[8] * Pnm) + (self.U5[9] * math.sqrt(Pnm)) + (self.U5[10] * math.exp(self.U5[11] * Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 6:
                E = 10
                eqn = self.U6[7] + (self.U6[8] * Pnm) + (self.U6[9] * math.sqrt(Pnm)) + (self.U6[10] * math.exp(self.U6[11] * Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 7:
                E = 10
                eqn = self.U7[7] + (self.U7[8] * Pnm) + (self.U7[9] * math.sqrt(Pnm)) + (self.U7[10] * math.exp(self.U7[11] + Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 8:
                E = 10
                eqn = self.U8[7] + (self.U8[8] * Pnm) + (self.U8[9] * math.sqrt(Pnm)) + (self.U8[10] * math.exp(self.U8[11] + Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 9:
                E = 10
                eqn = self.U9[7] + (self.U9[8] * Pnm) + (self.U9[9] * math.sqrt(Pnm)) + (self.U9[10] * math.exp(self.U9[11] + Pnm))
                localReward = E * eqn

            elif agent.getAgentID() == 10:
                E = 10
                eqn = self.U6[7] + (self.U10[8] * Pnm[ag]) + (self.U10[9] * math.sqrt(Pnm)) + (self.U10[10] * math.exp(self.U10[11] * Pnm))
                localReward = E * eqn


        fileName = ("DEED_Problem_Local_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(j) + ' ***********************'
        line2 = 'Agent: ' + str(agent.getAgentID())
        line3 = 'Agent ' + str(agent.getAgentID()) + " Action: " + str(action_)
        line4 = str(type) + " Reward: " + str(localReward)
        line5 = "Overall Power: " + str(PDM)
        line6 = "Agent Power: " + str(Pnm)
        line7 = 'Q-Table: ' + str(agent.getQTable())

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6, line7))
        out.close()

        return -localReward * 0.45

    def OverallReward(self, reward):
        self.rewardArray.append(reward)

    def getOverallReward(self):
        return self.rewardArray

    def getPowerDemand(self, hour):

        if hour - 1 == 0:
            Pdm_ = 0
        else:
            Pdm_ = self.PDM_hold[hour - 2]

        Pdm = self.PDM_hold[hour-1]
        Pdm_delta = Pdm - Pdm_
        return Pdm, Pdm_, Pdm_delta


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


    def calculateDifferenceReward(self,action_, pubCrowd, i, agent):
        c = 6
        totalReward = []

        for people in pubCrowd:
            rewardPerDay1 = (math.exp(-(people/c))) * people
            rewardPerDay2 = (math.exp(-((people - 1)/c))) * (people - 1)
            rewardPerDay = rewardPerDay1 - rewardPerDay2
            totalReward.append(rewardPerDay)

        if action_ == 0:
            reward = totalReward[0]
        elif action_ == 1:
            reward = totalReward[1]
        elif action_ == 2:
            reward = totalReward[2]
        elif action_ == 3:
            reward = totalReward[3]
        elif action_ == 4:
            reward = totalReward[4]
        elif action_ == 5:
            reward = totalReward[5]
        elif action_ == 6:
            reward = totalReward[6]

        #print("Reward for Each Agent", reward)
        fileName = ("DEED_Problem_Difference_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(self.j) + ' ***********************'
        line2 = 'Agent: ' + str(i)
        line3 = 'Agent ' + str(i) + " Action: " + str(action_)
        line4 = "People Per Night " + str(pubCrowd)
        line5 = "Reward Per Night " + str(totalReward)
        line6 = "Reward for Each Agent " + str(reward)
        line7 = "Q Table " + str(agent.getQTable())

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6, line7))
        out.close()
        self.OverallReward(reward)
        return reward

    def timeStep(self, _agents_, j):
        hour = 1
        b = 0
        costTotal = []
        emissionTotal = []
        rewardTotal = []
        while hour <= 24:
            Pnm = []
            b = b+1
            self.k = self.k + 1
            for agent in _agents_:
                if hour == 1 and j == 1:
                    CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
                    action_ = 50
                    Pn = self.getPNM(action_, agent)
                    agent.powerArray.append(Pn)
                    currentState = 0
                    Pnm.append(Pn)
                    agent.saveAction(action_)

                elif hour == 1:
                    CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
                    currentState = agent.getState()
                    agent.saveCurrentState(currentState)
                    action_ = agent.selectAction(currentState, agent.powerArray, agent)
                    Pn = self.getPNM(action_, agent)
                    agent.powerArray.append(Pn)
                    Pnm.append(Pn)
                    agent.saveAction(action_)
                else:
                    CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
                    currentState = agent.getState()
                    action_ = agent.selectAction(currentState, agent.powerArray, agent)
                    Pn = self.getPNM(action_, agent)
                    agent.powerArray.append(Pn)
                    Pnm.append(Pn)
                    agent.saveAction(action_)

            i = 0
            P1M = self.getPLM(Pnm, CurrentPDM)
            reward, cost, emissions = self.calculateGlobalReward(j, b, _agents_, Pnm, currentState, CurrentPDM, P1M, hour)
            emissionTotal.append(emissions)
            costTotal.append(cost)
            rewardTotal.append(reward)
            #print(PNM)
            for agent in _agents_:
                previousState = agent.getState()
                action = agent.getAction()
                currentState= agent.getNextState(hour, action)
                agent.saveCurrentState(currentState)
                agent.updateQTable(previousState, action,currentState,reward)
                i = i + 1

            hour = hour + 1

        totalCost = sum(costTotal)
        totalEmissions = sum(emissionTotal)/10
        totalReward = sum(rewardTotal)

        fileName = ("DEED_Problem_Cost_Emissions" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(j) + ' ***********************'
        line2 = ("Total Emissions: " + str(totalEmissions))
        line3 = ("Total Cost: " + str(totalCost))
        line4 = " "

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4))
        out.close()

        return totalCost, totalEmissions, totalReward


def metric(array):
    plt.subplot(1, 1, 1)
    plt.axhline(y=15.6, color='black', linewidth=1, linestyle='dashed', label="Optimal")
    plt.plot(array, color='olive', linewidth=1, label="Global Rewards")
    plt.title('Reward Over Epochs')
    plt.ylabel('Performance - G(z)')
    plt.xlabel('Epochs')
    plt.xlim(0, len(array))
    plt.ylim(min(array), max(array))
    plt.legend()
    plt.show()

def main():
    numEpisodes = 5000
    numAgents = 9
    costArray = []
    emissionsArray = []
    rewardArray = []
    avgCost = []
    avgEmissions = []
    avgReward = []
    _agents_ = []
    starter = 1



    global fileName

    inc = 1
    while inc <= 1:
        j = 1
        env = Environment()
        starter = 1
        _agents_ = []
        while starter <= numAgents:
            agent = env.createAgent((300), starter + 1)
            starter = starter + 1
            _agents_.append(agent)
        print("*************** Run " + str(inc) +  " ***************")
        while j <= numEpisodes:
            print("Episode:", j)
            cost, emissions, reward = env.timeStep(_agents_, j)

            costArray.append(cost)
            emissionsArray.append(emissions)
            rewardArray.append(reward)

            j = j + 1

        avgCost.append(costArray)
        avgEmissions.append(emissionsArray)
        avgReward.append(rewardArray)

        inc = inc + 1

    AvgCost = [sum(x) for x in zip(*avgCost)]
    AvgEmissions = [sum(x) for x in zip(*avgEmissions)]
    AvgReward = [sum(x) for x in zip(*avgReward)]
    myInt = 1
    outAvgCost = [y / myInt for y in AvgCost]
    outAvgEmissions = [y / myInt for y in AvgEmissions]
    outAvgReward = [y / myInt for y in AvgReward]

    metric(outAvgCost)
    metric(outAvgEmissions)
    metric(outAvgReward)

if __name__ == "__main__":
    main()
