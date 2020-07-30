# Dynamic Economic Emissions Dispatch Problem implemenation with Q-learning.




import numpy as np
from numpy import array
import math
from random import randint
import time, os, fnmatch, shutil
from collections import Counter
from random import seed
import random
import pandas as pd


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

    def Agent(self, numStates, numActions, alpha, gamma, epsilon, id):
        self.UHolder
        self.id = int(id)
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.P1M_array = []
        self.qTable = self.initialiseQvalue(numStates, numActions)
        #self.qTable = qTable
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
        self.dReward = 0
        self.P1M_array_D = []
        self.Pnm = []
        self.previousAgentCost = 0
        self.previousAgentEmissions = 0
        self.previousAgentPower = 0
        self.P1M_Minus_D = 0
        self.P1M_Minus = 0
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
        #currentS = bin_index1
        res = 0
        i = 0
        while i < 2:
            res = res * base[i] + currentState[i]
            i += 1

        state = res

        return state


    def selectAction(self,hour, state, previousPower, agent):
        #check = random.uniform(0,100)
        check = random.uniform(0,1)
        #print(check)
        if check < 0:
            #print(check)
            selectedAction = self.selectrandomAction()
        else:
            selectedAction = self.getSelectedAction(hour, state, agent)

        return selectedAction

    def selectrandomAction(self):
        self.randomAction = round(randint(0,100))

        return self.randomAction

    def saveCurrentState(self, currentState):

        self.currentState = currentState
        return self.currentState

    def getState(self):

        return self.currentState

    def getSelectedAction(self,hour,state,agent):
        action = 0

        self.action_holder = []
        self.action_ = []

        previousPowerOutput = agent.powerArray[-1]

        while action < self.numActions:
            testAction = Environment.getPNM(self, action, agent)

            if hour == 1:
                valueQ = agent.qTable[state][int(action)]
                agent.action_holder.append(valueQ)
                agent.action_.append(action)

            else:
                val = agent.getAgentID() - 1
                if testAction - previousPowerOutput <= self.UHolder[val][12] and previousPowerOutput - testAction <= self.UHolder[val][13]:
                    valueQ = agent.qTable[state][int(action)]
                    agent.action_holder.append(valueQ)
                    agent.action_.append(action)

            action = action + 1
        maxIndex = self.action_holder.index(max(self.action_holder))
        maxActionIndex = self.action_[maxIndex]
        return int(maxActionIndex)

    def getMaxValuedAction(self, agent, state):
        self.action_holders = []
        action = 0
        while action < self.numActions:
            valueQ = self.qTable[state][action]
            self.action_holders.append(valueQ)
            action = action + 1

        maxActionIndex = self.action_holders.index(max(self.action_holders))

        return maxActionIndex

    def find_nearest(self, array, value):
        if len(array) == 1:
            return 0
        else:
            array = array[array !=0]
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
        return array[idx]


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



    def calculateGlobalReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour, scalarization):
        costReward = []
        emissionsReward = []
        E = 10
        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2

            val = agent.getAgentID() - 1
            cost = agent.UHolder[val][2] + (agent.UHolder[val][3] * Pnm[id]) + (agent.UHolder[val][4] * (Pnm[id] ** 2)) + abs(
                agent.UHolder[val][5] + math.sin(agent.UHolder[val][6] * (agent.UHolder[val][0] - Pnm[id]))
            )
            costReward.append(cost)

            emissions = agent.UHolder[val][7] + (agent.UHolder[val][8] * Pnm[id]) + (agent.UHolder[val][9] * (Pnm[id]**2)) + (
                        agent.UHolder[val][10] * math.exp(agent.UHolder[val][11] * Pnm[id]))
            emissions = E * emissions
            emissionsReward.append(emissions)

        P1M_cost = agent.UHolder[0][2] + (agent.UHolder[0][3] * Pnm[id]) + (
                    agent.UHolder[0][4] * (Pnm[id] ** 2)) + abs(
            agent.UHolder[0][5] + math.sin(agent.UHolder[0][6] * (agent.UHolder[0][0] - Pnm[id]))
        )
        costReward.append(P1M_cost)

        P1M_emissions = agent.UHolder[0][7] + (agent.UHolder[0][8] * Pnm[id]) + (
                    agent.UHolder[0][9] * (Pnm[id] ** 2)) + (
                            agent.UHolder[0][10] * math.exp(agent.UHolder[0][11] * Pnm[id]))

        P1M_emissions = E * P1M_emissions
        emissionsReward.append(P1M_emissions)

        # 1,000,000
        C = 1000000

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

        val = agent.getAgentID() - 1
        PNM = agent.UHolder[val][0] + (action_*((agent.UHolder[val][1] - agent.UHolder[val][0])/100))

        return PNM

    def calculateLocalReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour, agentID, scalarization):
        costReward = []
        emissionsReward = []
        #print("Number of Agents: ", len(_agents_))
        E = 10
        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2

            val = agent.getAgentID() - 1
            cost = agent.UHolder[val][2] + (agent.UHolder[val][3] * Pnm[id]) + (
                        agent.UHolder[val][4] * (Pnm[id] ** 2)) + abs(
                agent.UHolder[val][5] + math.sin(agent.UHolder[val][6] * (agent.UHolder[val][0] - Pnm[id]))
            )
            costReward.append(cost)

            emissions = agent.UHolder[val][7] + (agent.UHolder[val][8] * Pnm[id]) + (
                        agent.UHolder[val][9] * (Pnm[id] ** 2)) + (
                                agent.UHolder[val][10] * math.exp(agent.UHolder[val][11] * Pnm[id]))
            emissions = E * emissions
            emissionsReward.append(emissions)

        P1M_cost = agent.UHolder[0][2] + (agent.UHolder[0][3] * Pnm[id]) + (
                agent.UHolder[0][4] * (Pnm[id] ** 2)) + abs(
            agent.UHolder[0][5] + math.sin(agent.UHolder[0][6] * (agent.UHolder[0][0] - Pnm[id]))
        )
        costReward.append(P1M_cost)

        P1M_emissions = agent.UHolder[0][7] + (agent.UHolder[0][8] * Pnm[id]) + (
                agent.UHolder[0][9] * (Pnm[id] ** 2)) + (
                                agent.UHolder[0][10] * math.exp(agent.UHolder[0][11] * Pnm[id]))
        P1M_emissions = E * P1M_emissions
        emissionsReward.append(P1M_emissions)

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
            violationPenalty = (abs(h1 + 1) * self.U1[11]) * C
        elif h1 == 0 and h2 != 0:
            violationPenalty = (abs(h2 + 1) * self.U2[11]) * C
        elif h1 == 0 and h2 == 0:
            violationPenalty = 0
        elif h1 != 0 and h2 != 0:
            violationPenalty = (C * ((abs(h1 + 1) * self.U1[11]))) + (C * ((abs(h2 + 1) * self.U2[11])))

        cost = costReward[agentID.getAgentID() - 2]
        emissions = emissionsReward[agentID.getAgentID() - 2]
        #print(agentID.getAgentID())
        #print(costReward)
        #print("Length of Cost Reward: ", len(costReward))
        #print("Supposed Cost: ", cost)
        #print(" ")
        #print("Cost: ", cost)
        #print("Emissions: ", emissions)
        if scalarization == "hypervolume":
            reward = -(cost + emissions)

        elif scalarization == "linear":
            reward = -((cost * 0.225) + (emissions * 0.275))

        return reward, sum(costReward), sum(emissionsReward), violationPenalty


    def calculateDifferenceReward(self,x, i, _agents_, Pnm, type, PDM, P1M, hour,agentID, scalarization):
        costReward = []
        emissionsReward = []
        E = 10
        for agent in _agents_:
            a_id = agent.getAgentID()
            id = a_id - 2

            val = agent.getAgentID() - 1
            cost = agent.UHolder[val][2] + (agent.UHolder[val][3] * Pnm[id]) + (
                        agent.UHolder[val][4] * (Pnm[id] ** 2)) + abs(
                agent.UHolder[val][5] + math.sin(agent.UHolder[val][6] * (agent.UHolder[val][0] - Pnm[id]))
            )
            costReward.append(cost)

            emissions = agent.UHolder[val][7] + (agent.UHolder[val][8] * Pnm[id]) + (
                        agent.UHolder[val][9] * (Pnm[id] ** 2)) + (
                                agent.UHolder[val][10] * math.exp(agent.UHolder[val][11] * Pnm[id]))
            emissions = E * emissions
            emissionsReward.append(emissions)

        P1M_cost = agent.UHolder[0][2] + (agent.UHolder[0][3] * Pnm[id]) + (
                agent.UHolder[0][4] * (Pnm[id] ** 2)) + abs(
            agent.UHolder[0][5] + math.sin(agent.UHolder[0][6] * (agent.UHolder[0][0] - Pnm[id]))
        )
        costReward.append(P1M_cost)

        P1M_emissions = agent.UHolder[0][7] + (agent.UHolder[0][8] * Pnm[id]) + (
                agent.UHolder[0][9] * (Pnm[id] ** 2)) + (
                                agent.UHolder[0][10] * math.exp(agent.UHolder[0][11] * Pnm[id]))
        P1M_emissions = E * P1M_emissions
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

        agent_Cost = (sum(costReward) - costReward[agentID.getAgentID() - 2]) + previousAgentCost
        agentID.setPreviousAgentCost(costReward[agentID.getAgentID() - 2])
        global_cost = sum(costReward)
        global_emissions = sum(emissionsReward)
        global_penalty = violationPenalty
        G_z = global_cost + global_emissions + global_penalty
        agent_Emissions = (sum(emissionsReward) - emissionsReward[agentID.getAgentID() - 2]) + previousAgentEmissions
        agentID.setPreviousAgentEmissions(emissionsReward[agentID.getAgentID() - 2])

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
            overallCostReward = (sum(costReward) - agent_Cost) * 0.225
            overallEmissionsReward = (sum(emissionsReward) - agent_Emissions) * 0.275
            overallPenalty = (violationPenalty - violationPenalty_D) * 0.5

        G_z_i = overallCostReward + overallEmissionsReward + overallPenalty
        reward = -(G_z_i)

        #if agentID.getAgentID() == 2:
        #    print("Current PDM", PDM)
        #    print("Hour: ", hour)
        #    print("Previous Agent Cost: ", previousAgentCost)
        #    print("Current Agent Cost: ", costReward[agentID.getAgentID() - 2])
        #    print("Current Agent Cost: ", costReward[agentID.getAgentID() - 2])
        #    print("P1M G: ", P1M)
        #    print("P1M D: ", P1M_D)
        #    print("P1M Minus D: ", P1M_minus_D)
        #    print("Violation Penalty G: ", violationPenalty)
        #    print("Violation Penalty D:", violationPenalty_D)
        #    print("Cost: ", overallCostReward)
        #    print("Emissions: ", overallEmissionsReward)
        #    print("Penalty: ", overallPenalty)
        #    print("Reward: ",reward)
        #print(reward)
        return reward, sum(costReward), sum(emissionsReward), overallPenalty

    def timeStep(self, _agents_, j, rewardType, scalarization):
        hour = 1
        b = 0
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
                    #CurrentPDM, PreviousPDM, PDM_delta = self.getPowerDemand(hour)
                    currentState = agent.getState()
                    action_ = agent.selectAction(hour, currentState, agent.powerArray, agent)
                    Pn = self.getPNM(action_, agent)
                    agent.powerArray.append(Pn)
                    Pnm.append(Pn)
                    agent.saveAction(action_)
            i = 0
            for agent in _agents_:
                agent.savePnm(Pnm)

            P1M = self.getP1M(Pnm, CurrentPDM)



            for agent in _agents_:
                #print("PNM::: ", self.getPnm())
                previousState = agent.getState()
                action = agent.getAction()

                if rewardType == "Global":
                    reward, cost, emissions, violations = self.calculateGlobalReward(j, b, _agents_, Pnm, currentState,
                                                                                     CurrentPDM, P1M,
                                                                                     hour, scalarization)


                if rewardType == "Difference":
                    reward, cost, emissions,violations = self.calculateDifferenceReward(j, b, _agents_, Pnm, previousState, CurrentPDM, P1M,
                                                                        hour,agent, scalarization)
                    #print("Difference hello: ", cost)

                if rewardType == "Local":
                    reward, cost, emissions, violations = self.calculateLocalReward(j, b, _agents_, Pnm, previousState,
                                                                             CurrentPDM, P1M,
                                                                             hour, agent, scalarization)
                    #print("Local1: ", reward)
                currentState = agent.getNextState(hour, agent.powerArray, agent)
                agent.saveCurrentState(currentState)
                agent.updateQTable(previousState, action, currentState, reward, agent)

                i = i + 1


                #print("Difference: ", cost)
                emissionTotal.append(emissions)
                costTotal.append(cost)
                rewardTotal.append(reward)
                violationsTotal.append(violations)

            hour += 1



        totalCost = sum(costTotal)
        totalEmissions = sum(emissionTotal)
        totalReward = sum(rewardTotal)
        totalViolations = sum(violationsTotal)

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

    outputDataDifference_Cost = pd.DataFrame()
    outputDataDifference_Emissions = pd.DataFrame()
    outputDataDifference_Violations = pd.DataFrame()

    outputDataLocal_Cost = pd.DataFrame()
    outputDataLocal_Emissions = pd.DataFrame()
    outputDataLocal_Violations = pd.DataFrame()

    while inc <= 1:
        j = 1
        env = Environment()

        starter = 1
        _agents_ = []
        _agentsDifference_ = []
        _agentsLocal_ = []
        costArray, emissionsArray, rewardArray, violationsArray = [], [], [], []

        while starter <= numAgents:
            agentRun = env.createAgent((250), starter + 1)
            starter = starter + 1
            _agents_.append(agentRun)

        print("*************** Run " + str(inc) + " ***************")
        while j <= numEpisodes:
            print("Episode:", j)
            rewardType = "Global"
            scalarisation = "linear"
            cost, emissions, reward, violations = env.timeStep(_agents_, j, rewardType, scalarisation)

            costArray.append(cost)
            emissionsArray.append(emissions)
            rewardArray.append(reward)
            violationsArray.append(violations)

            print("Cost :: ", cost)
            print("Emissions :: ", emissions)
            print("Violations :: " , violations)

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
