import numpy as np
import math
from random import randint
import time, os, fnmatch, shutil
import matplotlib.pyplot as plt
from plotnine import *
import pandas as pd
import random


class Agent:
    qTable = []
    actions = []  

    def Agent(self, numActions, alpha, gamma, epsilon):
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        qTable= self.initialiseQvalue(numActions)
        self.qTable = qTable
        #self.old_qTable = old_qTable
        self.selectedActions = []
        self.previousActions = []
        self.qValues = []
        self.action = 0
        return self

    def decayEpsilon(self):
        self.epsilon = self.epsilon * 0.99
        return self.epsilon

    def decayAlpha(self):
        self.alpha = self.alpha * 0.99
        return self.alpha

    def setEpsilon(self):
        self.epsilon = 0.05

    def setAlpha(self):
        self.alpha = 0.1

    def initialiseQvalue(self, numActions):
        qTable = np.zeros((numActions))


        return qTable

    # def updateQTable(self, previousState, selectedAction,reward):

    # PM: What does "i" do in this function? Episode number?
    def updateQTable(self, selectedAction, reward, i):

        oldQ = self.qTable[selectedAction]
        newQ = oldQ + self.alpha * (reward + self.gamma * 0 - oldQ)
        self.qTable[selectedAction] = newQ

        return self

    def selectAction(self):
        check = random.uniform(0,1)
        if check < self.epsilon:
            selectedAction = self.selectrandomAction()
            # self.decayEpsilon(self.epsilon)
        else:
            selectedAction = self.getMaxValuedAction()

        return selectedAction

    def selectrandomAction(self):
        self.randomAction = round(randint(0, 6))

        return self.randomAction

    def previousAction(self):
        previousAction = getMostRecentPreviousAction()
        self.previousActionValue = self.old_qTable[previousAction]

        return self.previousActionValue

    def previousActionTaken(self):
        self.previousAction = self.actions[len(self.actions) - 2]

        return self.previousAction

    def getMaxValuedAction(self):
        self.maxIndex = np.argmax(self.qTable)

        return self.maxIndex

    def getMaxQValue(self):
        maxIndex = self.getMaxValuedAction()

        return self.qTable[maxIndex]

    def getQTable(self):

        return self.qTable

    def getOldQTable(self):

        return self.old_qTable

    def savePreviousActions(self, action):
        self.previousActions.append(action)
        return self.previousActions

    def saveSelectedActions(self, action):
        self.selectedActions.append(action)
        return self.selectedActions

    def saveAction(self, action):

        self.action = action

    def getAction(self):

        return self.action

    def getMostRecentAction(self):
        self.mostRecentAction = self.selectedActions[len(self.selectedActions) - 1]
        return self.mostRecentAction

    def getMostRecentPreviousAction(self, k):
        if k == 0:
            return 0
        elif k == 1:
            return 0
        else:
            self.mostRecentPreviousAction = self.previousActions[len(self.previousActions) - 2]
        return self.mostRecentPreviousAction


class Environment():
    global numActions
    global numAgents
    global _agents_
    global gamma
    global epsilon
    global alpha

    numActions = 7
    numEpisodes = 4000
    numAgents = 42
    epsilon = 0.1
    gamma = 1
    alpha = 0.1
    _agents_ = []
    a = Agent()

    def __init__(self):
        self.numActions = 7
        self.numEpisodes = 4000
        self.numAgents = 42
        self.epsilon = 0.05
        self.gamma = 1
        self.alpha = 0.1
        self.k = 0
        self.t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', self.t)
        self.rewardArray = []
        self._agents_ = []

    def createAgents(self, numAgents):
        for i in range(numAgents):
            self.agent = Agent()
            self.agent_ = self.agent.Agent(numActions, alpha, gamma, epsilon)
            self._agents_.append(self.agent_)

        return self._agents_

    def getNextAction(self, action):
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

    def getBarPopulation(self, action, crowd):
        if action == 0:
            crowd[0] = crowd[0] + 1
        elif action == 1:
            crowd[1] = crowd[1] + 1
        elif action == 2:
            crowd[2] = crowd[2] + 1
        elif action == 3:
            crowd[3] = crowd[3] + 1
        elif action == 4:
            crowd[4] = crowd[4] + 1
        elif action == 5:
            crowd[5] = crowd[5] + 1
        elif action == 6:
            crowd[6] = crowd[6] + 1

        return crowd

    def calculateGlobalReward(self, pubCrowd, i):
        c = 6
        totalReward = []
        #print("Global Crowd: ", pubCrowd)
        for people in pubCrowd:
            rewardPerDay = people * (math.exp(((-people) / c)))
            #a = (-people) / c
            #b = math.exp(a)
            #rewardPerDay = people * b
            totalReward.append(rewardPerDay)
        #print("Global: ", totalReward)
        reward = sum(totalReward)

        #print(reward)
        fileName = ("Multi_Bar_Problem_Global_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(i) + ' ***********************'
        line2 = "People Per Night " + str(pubCrowd)
        line3 = "Reward Per Night " + str(totalReward)
        line4 = "Reward for Each Agent " + str(reward)

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4))
        out.close()

        return reward

    def calculateLocalReward(self, action_, pubCrowd, i, agent):
        c = 6
        totalReward = []

        for people in pubCrowd:
            rewardPerDay = people * (math.exp(((-people) / c)))
            totalReward.append(rewardPerDay)

        # print("People Per Night", pubCrowd)
        # print("Reward Per Night", totalReward)

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

        if i == 1:
            # print("Reward for Each Agent", reward)
            fileName = ("Multi_Bar_Problem_Local_Reward_" + self.timestamp + ".txt")
            line1 = '***************** Episode: ' + str(self.k) + ' ***********************'
            line2 = 'Agent: ' + str(i)
            line3 = 'Agent ' + str(i) + " Action: " + str(action_)
            line4 = "People Per Night " + str(pubCrowd)
            line5 = "Reward Per Night " + str(totalReward)
            # line6 = "Reward for Each Agent " + str(reward)
            # line7 = "Q Table " + str(agent.getQTable())

            with open(fileName, 'a') as out:
                out.write('{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5))
            out.close()

        performanceOut = sum(totalReward)
        #print("Total Local Reward: ", sum(totalReward))
        return reward, performanceOut

    def OverallReward(self, reward):
        self.rewardArray.append(reward)

    def getOverallReward(self):
        return self.rewardArray

    def calculateDifferenceReward(self, action_, pubCrowd, i, agent):
        c = 6
        totalReward = []
        totalPerformance = []

        for people in pubCrowd:
            calc = (-(people - 1))
            calc_ = calc / c
            rewardPerDay1 = math.exp((-people) / c) * people
            rewardPerDay2 = (people - 1) * math.exp(calc_)
            rewardPerDay = rewardPerDay1 - rewardPerDay2
            totalReward.append(rewardPerDay)

        for people in pubCrowd:
            performance = people * (math.exp(((-people) / c)))
            totalPerformance.append(performance)
            # print(totalReward)
        # print("People Per Night", pubCrowd)
        # print("Reward Per Night", totalReward)

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

        if i == 1:
            # print("Reward for Each Agent", reward)
            fileName = ("Multi_Bar_Problem_Difference_Reward_" + self.timestamp + ".txt")
            line1 = '***************** Episode: ' + str(self.k) + ' ***********************'
            line2 = 'Agent: ' + str(i)
            line3 = 'Agent ' + str(i) + " Action: " + str(action_)
            line4 = "People Per Night " + str(pubCrowd)
            line5 = "Reward Per Night " + str(totalReward)
            # line6 = "Reward for Each Agent " + str(reward)
            # line7 = "Q Table " + str(agent.getQTable())

            with open(fileName, 'a') as out:
                out.write('{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5))
            out.close()

        self.OverallReward(reward)
        performanceOut = sum(totalPerformance)
        return reward, performanceOut

    def globalStep(self, _agents_, i):
        crowd = np.zeros(numActions)
        selectedActions = []
        self.k = self.k + 1
        for agent in _agents_:
            action_ = agent.selectAction()
            # ("Action: ", action_)
            agent.saveSelectedActions(action_)
            selectedActions.append(action_)
            previousAction = agent.getMostRecentPreviousAction(self.k)
            agent.savePreviousActions(previousAction)

        for action in selectedActions:
            population = self.getBarPopulation(action, crowd)


        reward = self.calculateGlobalReward(population, i)
        totPerformance = reward
        #print("Global Reward: ", reward)

        x = 0
        rewards_ = []
        performance_ = []
        for agent in _agents_:
            action_ = agent.getAction()
            agent.getQTable = agent.getOldQTable
            # agent.updateQTable(int(agent.getMostRecentPreviousAction(self.k)), int(agent.getMostRecentAction()), reward)
            agent.updateQTable(int(agent.getMostRecentAction()), reward, i)
            x = x + 1

        return totPerformance

    def localStep(self, _agents_, i):
        crowd = np.zeros(numActions)
        selectedActions = []
        self.k = self.k + 1
        for agent in _agents_:
            action_ = agent.selectAction()
            # ("Action: ", action_)
            agent.saveAction(action_)
            selectedActions.append(action_)
            previousAction = agent.getMostRecentPreviousAction(self.k)
            agent.savePreviousActions(previousAction)

        for action in selectedActions:
            population = self.getBarPopulation(action, crowd)

        x = 0
        rewards_ = []
        performance_ = []
        for agent in _agents_:
            action_ = agent.getAction()

            # Uncomment Line Below to Use Local Reward
            reward, performance = self.calculateLocalReward(action_, population, x, agent)
            #print("Local Reward: ", reward)

            #agent.getQTable = agent.getOldQTable
            # agent.updateQTable(int(agent.getMostRecentPreviousAction(self.k)), int(agent.getMostRecentAction()), reward)
            agent.updateQTable(int(action_), reward, i)
            x = x + 1
            rewards_.append(reward)
            performance_.append(performance)

        rewards = sum(rewards_)
        totPerformance = (performance)
        #print("TotPerformance: ", totPerformance)
        return totPerformance

    def differenceStep(self, _agents_, i):
        crowd = np.zeros(numActions)
        selectedActions = []
        self.k = self.k + 1
        for agent in _agents_:
            action_ = agent.selectAction()
            # ("Action: ", action_)
            agent.saveAction(action_)
            selectedActions.append(action_)
            previousAction = agent.getMostRecentPreviousAction(self.k)
            agent.savePreviousActions(previousAction)

        for action in selectedActions:
            population = self.getBarPopulation(action, crowd)

        x = 0
        rewards_ = []
        performance_ = []
        for agent in _agents_:
            action_ = agent.getAction()

            # Uncomment Line Below to Use Difference Reward
            reward, performance = self.calculateDifferenceReward(action_, population, x, agent)

            agent.getQTable = agent.getOldQTable
            # agent.updateQTable(int(agent.getMostRecentPreviousAction(self.k)), int(agent.getMostRecentAction()), reward)
            agent.updateQTable(int(agent.getAction()), reward, i)
            x = x + 1
            rewards_.append(reward)
            performance_.append(performance)

        totPerformance = (sum(performance_)) / 42
        return totPerformance


def metric(globalRewardOverEpisode, localRewardOverEpisode, differenceRewardOverEpisode, numEpisodes):
    plt.subplot(1, 1, 1)
    #print(differenceRewardOverEpisode)
    plt.axhline(y=15.6, color='black', linewidth=1, linestyle='dashed', label="Optimal")
    plt.plot(globalRewardOverEpisode, color='olive', linewidth=1, label="Global Rewards")
    plt.plot(localRewardOverEpisode, color='red', linewidth=1, label="Local Rewards")
    plt.plot(differenceRewardOverEpisode, color='blue', linewidth=1, label="Difference Rewards")
    plt.title('Reward Over Epochs')
    plt.ylabel('Performance - G(z)')
    plt.xlabel('Epochs')
    plt.xlim(1, numEpisodes)
    plt.ylim(0, max(differenceRewardOverEpisode))
    plt.legend()
    plt.show()

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def computeAverage(array):
    newArray = []
    for i in array:
        average = (sum(i)) / 10
        #print("Values: ", i)
        #print("Average: ", average)
        newArray.append(average)

    return newArray

def graph(df, numEpisodes):
    x_axis = np.arange(0, numEpisodes, 10)
    #print(x_axis)
    df['Counter'] = x_axis
    print(df[0:])
    x = df['Counter']
    df1 = df.melt(id_vars=["Counter"], var_name="Reward", value_name="Value")
    #print(df1)

    costG = (ggplot(df1) +
            aes(x=df1['Counter'], y=df1['Value'], color = df1['Reward']) +
            geom_line(alpha=1, size=0.75) +
            #geom_line(aes(x='x', y=df['difference']), alpha=0.5, size=0.5, color= df['difference']) +
            #geom_line(aes(x='x', y=df['local']), alpha=0.5, size=0.5, color= df['local']) +
            #scale_x_continuous(lim = (0, max(x_axis)), breaks= range(0,len(x_axis)+ 500, 500)) +
            scale_y_continuous(lim = (0, 15), breaks = np.arange(0, max(df1['Value']) + 2, 2)) +
            ylab(" Performance - G(z) ") +
            xlab(" Episode ") +
            ggtitle(" ") +
            theme_matplotlib() +
            theme(axis_text_y = element_text(size =6)) +
            theme(axis_text_x=element_text(size=6)) +
            theme(legend_position="top",
            legend_text=element_text(size=8),
            legend_key=element_rect(colour="white", fill="white"),
            legend_title=element_blank()))

    print(costG)


def main():
    numEpisodes = 4000
    localnumAgents = 42
    globalnumAgents = 42
    differencenumAgents = 42
    # PM: some other parameters could be specified from here, e.g. capacity, alpha_decay_rate, epsilon_decay_rate
    inc = 1

    ArraySumGlobal = [0] * numEpisodes
    ArraySumDifference = [0] * numEpisodes
    ArraySumLocal = [0] * numEpisodes

    while inc <= 10:
        print("*************** Run " + str(inc) + " *********************")
        localEnv = Environment()
        globalEnv = Environment()
        differenceEnv = Environment()

        _localAgents_ = localEnv.createAgents(localnumAgents)
        _globalAgents_ = globalEnv.createAgents(globalnumAgents)
        _differenceAgents_ = differenceEnv.createAgents(differencenumAgents)
        #print(localnumAgents)
        #print(len(_localAgents_))

        global fileName
        localRewardOverEpisode = []
        differenceRewardOverEpisode = []
        globalRewardOverEpisode = []
        episodes = []

        for agent in _localAgents_:
            agent.setAlpha()
            agent.setEpsilon()

        for agent in _globalAgents_:
            agent.setAlpha()
            agent.setEpsilon()

        for agent in _differenceAgents_:
            agent.setAlpha()
            agent.setEpsilon()

        i = 1
        while i <= numEpisodes:
            print("Episode:", i)
            globalReward = globalEnv.globalStep(_globalAgents_, i)
            localReward = localEnv.localStep(_localAgents_, i)
            differenceReward = differenceEnv.differenceStep(_differenceAgents_, i)
            globalRewardOverEpisode.append(globalReward)
            localRewardOverEpisode.append(localReward)
            differenceRewardOverEpisode.append(differenceReward)

            episodes.append(i)
            #for agent in _localAgents_:
            #    agent.decayAlpha()
            #    agent.decayEpsilon()

            #for agent in _globalAgents_:
            #    agent.decayAlpha()
            #    agent.decayEpsilon()

            #for agent in _differenceAgents_:
            #    agent.decayAlpha()
            #    agent.decayEpsilon()

            i = i + 1

        ArraySumGlobal = [x + y for x, y in zip(ArraySumGlobal, globalRewardOverEpisode)]
        ArraySumDifference = [x + y for x, y in zip(ArraySumDifference, differenceRewardOverEpisode)]
        ArraySumLocal = [x + y for x, y in zip(ArraySumLocal, localRewardOverEpisode)]

        inc = inc + 1

    myInt = 10
    outAvgDifference = [x / myInt for x in ArraySumDifference]
    outAvgLocal = [x / myInt for x in ArraySumLocal]
    outAvgGlobal = [x / myInt for x in ArraySumGlobal]

    span = len(outAvgDifference) / 10
    pointAverageDifference = list(split(outAvgDifference, int(span)))
    pointAverageLocal = list(split(outAvgLocal, int(span)))
    pointAverageGlobal = list(split(outAvgGlobal, int(span)))

    AverageDifference = computeAverage(pointAverageDifference)
    AverageLocal = computeAverage(pointAverageLocal)
    AverageGlobal = computeAverage(pointAverageGlobal)

    reward = pd.DataFrame({'Global': AverageGlobal, 'Difference': AverageDifference, 'Local': AverageLocal})

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    reward.to_csv(r'El_Farol_Bar_Problem_Results_' + timestamp + '.csv')

    graph(reward, numEpisodes)
    #metric(globalRewardOverEpisode, localRewardOverEpisode, differenceRewardOverEpisode, numEpisodes)


if __name__ == "__main__":
    main()
